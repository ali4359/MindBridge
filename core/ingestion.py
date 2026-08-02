"""Config-driven document loading and chunking for the MindBridge platform."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import UseCaseConfig

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _tag_page_documents(
    pages: list[Document],
    *,
    source: str,
    doc_type: str,
) -> list[Document]:
    """Normalize per-page metadata using the configured document-type tag."""
    tagged: list[Document] = []
    for doc in pages:
        page_index = doc.metadata.get("page", 0)
        if not isinstance(page_index, int):
            raise ValueError(f"Expected integer page index in metadata, got {page_index!r}")
        tagged.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source": source,
                    "doc_type": doc_type,
                    "page_number": page_index + 1,
                },
            )
        )
    return tagged


def load_pdf(
    pdf_path: Path,
    *,
    doc_type: str,
) -> list[Document]:
    """Load one PDF as one LangChain Document per page with metadata."""
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    loader = PyMuPDFLoader(str(pdf_path), mode="page")
    pages = loader.load()
    documents = _tag_page_documents(
        pages,
        source=pdf_path.name,
        doc_type=doc_type,
    )
    logger.info("Loaded %s — %d page(s) [%s]", pdf_path.name, len(documents), doc_type)
    return documents


def load_all_pdfs(config: UseCaseConfig, *, base_dir: Path = REPO_ROOT) -> list[Document]:
    """Load every PDF under each configured data source folder."""
    data_dir = config.sources_dir(base=base_dir)
    all_documents: list[Document] = []
    pdf_count = 0

    for source in config.data_sources:
        source_dir = data_dir / source.folder
        if not source_dir.is_dir():
            logger.warning("Skipping missing directory %s", source_dir)
            continue

        pdf_paths = sorted(source_dir.glob("*.pdf"))
        if not pdf_paths:
            logger.info("No PDFs in %s", source_dir)
            continue

        for pdf_path in pdf_paths:
            all_documents.extend(load_pdf(pdf_path, doc_type=source.doc_type))
            pdf_count += 1

    logger.info(
        "Loaded %d page document(s) from %d PDF(s)",
        len(all_documents),
        pdf_count,
    )
    return all_documents


def create_splitter(config: UseCaseConfig) -> RecursiveCharacterTextSplitter:
    """Build a text splitter from ``config.chunking``."""
    chunking = config.chunking
    return RecursiveCharacterTextSplitter(
        chunk_size=chunking.chunk_size,
        chunk_overlap=chunking.chunk_overlap,
        separators=chunking.separators,
        is_separator_regex=chunking.is_separator_regex,
    )


def chunk_documents(
    documents: Iterable[Document],
    config: UseCaseConfig,
) -> list[Document]:
    """Split documents into chunks while preserving source metadata."""
    docs = list(documents)
    if not docs:
        return []

    splitter = create_splitter(config)
    chunks = splitter.split_documents(docs)

    by_source: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "<unknown>")
        by_source[source] = by_source.get(source, 0) + 1

    for source, count in sorted(by_source.items()):
        logger.info("%s — %d chunk(s)", source, count)
    logger.info("Total: %d chunk(s) from %d document(s)", len(chunks), len(docs))
    return chunks


@dataclass(frozen=True)
class IngestionPipeline:
    """Load and chunk documents for a single use-case profile."""

    config: UseCaseConfig
    base_dir: Path = REPO_ROOT

    def load_documents(self) -> list[Document]:
        return load_all_pdfs(self.config, base_dir=self.base_dir)

    def chunk_documents(self, documents: Iterable[Document]) -> list[Document]:
        return chunk_documents(documents, self.config)

    def run(self) -> list[Document]:
        """Load all PDFs and return chunked documents."""
        return self.chunk_documents(self.load_documents())


def build_ingestion_pipeline(
    config: UseCaseConfig,
    *,
    base_dir: Optional[Path] = None,
) -> IngestionPipeline:
    """Factory for a config-bound ingestion pipeline."""
    return IngestionPipeline(config=config, base_dir=base_dir or REPO_ROOT)


def index_corpus(
    config: UseCaseConfig,
    *,
    base_dir: Path = REPO_ROOT,
    reset: bool = True,
):
    """Load, chunk, embed, and persist the corpus for a use-case profile."""
    from core.vectorstore import build_vectorstore

    pipeline = build_ingestion_pipeline(config, base_dir=base_dir)
    chunks = pipeline.run()
    if not chunks:
        raise SystemExit(
            f"No chunks produced for {config.name!r}. "
            f"Add PDFs under {config.sources_dir(base=base_dir)}"
        )
    return build_vectorstore(chunks, config, base_dir=base_dir, reset=reset)


def _session_doc_type(config: UseCaseConfig) -> str:
    """Resolve the session-note doc_type from the active profile."""
    for source in config.data_sources:
        if "session" in source.folder.lower() or "session" in source.doc_type.lower():
            return source.doc_type
    mapping = config.entities.category_to_doc_type
    if "session_notes" in mapping:
        return mapping["session_notes"]
    for doc_type in config.entities.doc_types:
        if "session" in doc_type.lower():
            return doc_type
    return "session_note"


def chunk_session_note(
    note_text: str,
    config: UseCaseConfig,
    *,
    entity_id: str,
    session_number: int,
    session_date: str,
    session_id: Optional[str] = None,
) -> list[Document]:
    """Chunk a session note string into Documents with entity metadata."""
    text = note_text.strip()
    if not text:
        return []

    resolved_session_id = session_id or f"{entity_id}-session-{session_number}"
    doc_type = _session_doc_type(config)
    source = f"session_note:{resolved_session_id}"

    document = Document(
        page_content=text,
        metadata={
            "source": source,
            "doc_type": doc_type,
            "page_number": 1,
            "entity_id": entity_id,
            "session_number": session_number,
            "session_date": session_date,
            "session_id": resolved_session_id,
        },
    )
    return chunk_documents([document], config)


def count_extracted_entities(entities: dict) -> int:
    """Count non-empty extracted entity values across schema keys."""
    total = 0
    for value in entities.values():
        if isinstance(value, list):
            total += sum(1 for item in value if str(item).strip())
        elif value is not None and str(value).strip():
            total += 1
    return total


@dataclass(frozen=True)
class DualIndexResult:
    """Outcome of dual indexing a session note into Chroma and Neo4j."""

    chunks_indexed: int
    entities_extracted: int
    entities: dict
    session_id: str
    status: str = "ok"


SESSION_NOTE_FILENAME_RE = re.compile(r"^synthetic_session_(\d{2})_.*\.txt$", re.IGNORECASE)


def strip_note_headers(text: str) -> str:
    """Remove leading ``#`` comment lines from generated session-note files."""
    lines = text.splitlines()
    while lines and lines[0].strip().startswith("#"):
        lines.pop(0)
    return "\n".join(lines).strip()


def parse_session_note_number(path: Path) -> int:
    """Extract the session ordinal from a synthetic note filename."""
    match = SESSION_NOTE_FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(
            f"Cannot parse session number from {path.name!r} "
            "(expected synthetic_session_NN_*.txt)"
        )
    return int(match.group(1))


def session_notes_dir(config: UseCaseConfig, *, base_dir: Path = REPO_ROOT) -> Path:
    """Resolve the directory containing generated ``.txt`` session notes."""
    return base_dir / config.data.session_notes.output_dir


def load_session_note_files(
    config: UseCaseConfig,
    *,
    base_dir: Path = REPO_ROOT,
) -> list[tuple[int, Path, str]]:
    """Load all ``.txt`` session notes sorted by session number."""
    notes_dir = session_notes_dir(config, base_dir=base_dir)
    if not notes_dir.is_dir():
        raise FileNotFoundError(f"Session notes directory not found: {notes_dir}")

    paths = sorted(notes_dir.glob("synthetic_session_*.txt"))
    if not paths:
        raise FileNotFoundError(f"No synthetic_session_*.txt files in {notes_dir}")

    loaded: list[tuple[int, Path, str]] = []
    for path in paths:
        session_number = parse_session_note_number(path)
        note_text = strip_note_headers(path.read_text(encoding="utf-8"))
        if not note_text:
            logger.warning("Skipping empty session note %s", path.name)
            continue
        loaded.append((session_number, path, note_text))

    loaded.sort(key=lambda item: item[0])
    logger.info("Loaded %d session note file(s) from %s", len(loaded), notes_dir)
    return loaded


def ingest_session_note_vector_only(
    *,
    entity_id: str,
    session_number: int,
    session_date: str,
    note_text: str,
    config: UseCaseConfig,
    base_dir: Path = REPO_ROOT,
    session_id: Optional[str] = None,
) -> int:
    """Chunk and embed a session note into Chroma without graph extraction."""
    from core.vectorstore import append_documents

    resolved_session_id = session_id or f"{entity_id}-session-{session_number}"
    chunks = chunk_session_note(
        note_text,
        config,
        entity_id=entity_id,
        session_number=session_number,
        session_date=session_date,
        session_id=resolved_session_id,
    )
    return append_documents(chunks, config, base_dir=base_dir)


GraphIngestMode = Literal["off", "shell", "full"]


def _existing_session_numbers(
    config: UseCaseConfig,
    entity_id: str,
    *,
    base_dir: Path = REPO_ROOT,
) -> set[int]:
    """Return session ordinals already indexed for ``entity_id`` in Chroma."""
    from core.vectorstore import load_vectorstore, lookup_entity_documents

    try:
        vectorstore = load_vectorstore(config, base_dir=base_dir)
    except FileNotFoundError:
        return set()

    numbers: set[int] = set()
    for doc in lookup_entity_documents(vectorstore, entity_id):
        session_number = doc.metadata.get("session_number")
        if isinstance(session_number, int):
            numbers.add(session_number)
    return numbers


def _is_rate_limit_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "429" in message or "rate_limit" in message or "rate limit" in message


@dataclass(frozen=True)
class BulkSessionIngestResult:
    """Outcome of bulk-ingesting demo session notes."""

    files_processed: int
    chunks_indexed: int
    entities_extracted: int
    entity_id: str
    session_ids: list[str]
    graph_mode: GraphIngestMode
    files_skipped: int = 0


def ingest_all_session_notes(
    config: UseCaseConfig,
    *,
    entity_id: str = "demo-entity",
    base_dir: Path = REPO_ROOT,
    graph_mode: GraphIngestMode = "full",
    vector_only: bool = False,
    llm=None,
    driver=None,
    session_date_prefix: str = "2026-01",
    resume: bool = False,
    graph_only: bool = False,
    pause_seconds: float = 0.0,
    max_rate_limit_retries: int = 5,
) -> BulkSessionIngestResult:
    """Ingest every ``.txt`` session note under the profile's session_notes dir."""
    if vector_only:
        graph_mode = "off"

    if graph_mode != "off" and config.graph_schema is None:
        raise ValueError(
            f"Profile {config.name!r} has no graph_schema; use graph_mode='off'"
        )

    notes = load_session_note_files(config, base_dir=base_dir)
    existing_sessions = _existing_session_numbers(config, entity_id, base_dir=base_dir) if resume else set()
    total_chunks = 0
    total_entities = 0
    session_ids: list[str] = []
    skipped = 0

    if graph_mode in {"shell", "full"}:
        if driver is None:
            from core.graph import get_graph_driver

            driver = get_graph_driver()
        if graph_mode == "full" and llm is None:
            from core.llm import build_llm

            llm = build_llm(config)

    import time

    for session_number, path, note_text in notes:
        if resume and session_number in existing_sessions:
            logger.info("Skipping session %d — already indexed for %s", session_number, entity_id)
            skipped += 1
            continue

        session_date = f"{session_date_prefix}-{session_number:02d}"
        resolved_session_id = f"{entity_id}-session-{session_number}"
        logger.info(
            "Ingesting session %d from %s → %s",
            session_number,
            path.name,
            resolved_session_id,
        )

        if graph_mode == "off":
            chunks_indexed = ingest_session_note_vector_only(
                entity_id=entity_id,
                session_number=session_number,
                session_date=session_date,
                note_text=note_text,
                config=config,
                base_dir=base_dir,
                session_id=resolved_session_id,
            )
            total_chunks += chunks_indexed
        elif graph_only:
            use_llm = graph_mode == "full"
            for attempt in range(max_rate_limit_retries + 1):
                try:
                    result = ingest_session_note_graph_only(
                        entity_id=entity_id,
                        session_number=session_number,
                        session_date=session_date,
                        note_text=note_text,
                        config=config,
                        llm=llm,
                        driver=driver,
                        session_id=resolved_session_id,
                        use_llm_extraction=use_llm,
                    )
                    break
                except Exception as exc:
                    if use_llm and _is_rate_limit_error(exc) and attempt < max_rate_limit_retries:
                        wait_seconds = min(60 * (attempt + 1), 300)
                        logger.warning(
                            "Groq rate limit on session %d (attempt %d/%d); waiting %ds",
                            session_number,
                            attempt + 1,
                            max_rate_limit_retries,
                            wait_seconds,
                        )
                        time.sleep(wait_seconds)
                        continue
                    raise
            total_entities += result.entities_extracted
            if pause_seconds > 0 and use_llm:
                time.sleep(pause_seconds)
        else:
            use_llm = graph_mode == "full"
            for attempt in range(max_rate_limit_retries + 1):
                try:
                    result = ingest_session_note(
                        entity_id=entity_id,
                        session_number=session_number,
                        session_date=session_date,
                        note_text=note_text,
                        config=config,
                        llm=llm,
                        driver=driver,
                        base_dir=base_dir,
                        session_id=resolved_session_id,
                        use_llm_extraction=use_llm,
                    )
                    break
                except Exception as exc:
                    if use_llm and _is_rate_limit_error(exc) and attempt < max_rate_limit_retries:
                        wait_seconds = min(60 * (attempt + 1), 300)
                        logger.warning(
                            "Groq rate limit on session %d (attempt %d/%d); waiting %ds",
                            session_number,
                            attempt + 1,
                            max_rate_limit_retries,
                            wait_seconds,
                        )
                        time.sleep(wait_seconds)
                        continue
                    raise

            total_chunks += result.chunks_indexed
            total_entities += result.entities_extracted
            if pause_seconds > 0 and use_llm:
                time.sleep(pause_seconds)

        session_ids.append(resolved_session_id)

    logger.info(
        "Bulk ingest complete: %d file(s), %d skipped, %d chunk(s), entity=%s, mode=%s",
        len(session_ids),
        skipped,
        total_chunks,
        entity_id,
        graph_mode,
    )
    return BulkSessionIngestResult(
        files_processed=len(session_ids),
        chunks_indexed=total_chunks,
        entities_extracted=total_entities,
        entity_id=entity_id,
        session_ids=session_ids,
        graph_mode=graph_mode,
        files_skipped=skipped,
    )


def ingest_session_note_graph_only(
    *,
    entity_id: str,
    session_number: int,
    session_date: str,
    note_text: str,
    config: UseCaseConfig,
    llm,
    driver,
    session_id: Optional[str] = None,
    use_llm_extraction: bool = True,
) -> DualIndexResult:
    """Write session entities to Neo4j without appending to Chroma."""
    from core.graph import empty_extraction_payload, extract_entities, write_to_graph

    resolved_session_id = session_id or f"{entity_id}-session-{session_number}"
    if use_llm_extraction:
        entities = extract_entities(note_text, config, llm)
    else:
        entities = empty_extraction_payload(config)
    write_to_graph(entities, entity_id, resolved_session_id, config, driver)

    return DualIndexResult(
        chunks_indexed=0,
        entities_extracted=count_extracted_entities(entities),
        entities=entities,
        session_id=resolved_session_id,
        status="ok",
    )


def ingest_session_note(
    *,
    entity_id: str,
    session_number: int,
    session_date: str,
    note_text: str,
    config: UseCaseConfig,
    llm,
    driver,
    base_dir: Path = REPO_ROOT,
    session_id: Optional[str] = None,
    use_llm_extraction: bool = True,
) -> DualIndexResult:
    """Chunk/embed into Chroma and extract/write entities into Neo4j."""
    from core.graph import empty_extraction_payload, extract_entities, write_to_graph
    from core.vectorstore import append_documents

    resolved_session_id = session_id or f"{entity_id}-session-{session_number}"
    chunks = chunk_session_note(
        note_text,
        config,
        entity_id=entity_id,
        session_number=session_number,
        session_date=session_date,
        session_id=resolved_session_id,
    )
    chunks_indexed = append_documents(chunks, config, base_dir=base_dir)

    if use_llm_extraction:
        entities = extract_entities(note_text, config, llm)
    else:
        entities = empty_extraction_payload(config)
    write_to_graph(entities, entity_id, resolved_session_id, config, driver)

    return DualIndexResult(
        chunks_indexed=chunks_indexed,
        entities_extracted=count_extracted_entities(entities),
        entities=entities,
        session_id=resolved_session_id,
        status="ok",
    )


def ingest_trajectory_document(
    *,
    entity_id: str,
    note_text: str,
    config: UseCaseConfig,
    base_dir: Path = REPO_ROOT,
) -> int:
    """Chunk and append a trajectory narrative with ``doc_type='trajectory'``."""
    from core.vectorstore import append_documents

    text = note_text.strip()
    if not text:
        return 0

    document = Document(
        page_content=text,
        metadata={
            "source": f"trajectory:{entity_id}",
            "doc_type": "trajectory",
            "page_number": 1,
            "entity_id": entity_id,
            "session_id": f"{entity_id}-trajectory",
        },
    )
    chunks = chunk_documents([document], config)
    return append_documents(chunks, config, base_dir=base_dir)


def main() -> None:
    """CLI: ``USE_CASE_CONFIG=configs/<profile>.yaml python -m core.ingestion``"""
    from dotenv import load_dotenv

    from core.config import USE_CASE_CONFIG_ENV, load_active_config, resolve_config_path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv(REPO_ROOT / ".env")

    config = load_active_config()
    config_path = resolve_config_path()
    logger.info("Use case: %s (%s)", config.display_name, config_path)

    vectorstore = index_corpus(config)
    print(
        f"\nIndexed {vectorstore._collection.count()} vectors "
        f"→ {config.chroma_path(base=REPO_ROOT)} "
        f"[{config.data.collection}]"
    )
    print(f"Set {USE_CASE_CONFIG_ENV}={config_path!r} to reuse this profile.")


if __name__ == "__main__":
    main()
