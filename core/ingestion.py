"""Config-driven document loading and chunking for the MindBridge platform."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

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
