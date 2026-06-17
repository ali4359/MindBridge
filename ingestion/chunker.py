"""Split loaded documents with RecursiveCharacterTextSplitter."""

from __future__ import annotations

import logging
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Prefer paragraph and sentence boundaries before word splits for clinical prose.
CLINICAL_SEPARATORS: list[str] = [
    "\n\n",
    "\n",
    r"(?<=[.!?])\s+",
    " ",
    "",
]


def create_splitter(
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """Build a splitter tuned for guideline and session-note text."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CLINICAL_SEPARATORS,
        is_separator_regex=True,
    )


def chunk_documents(
    documents: Iterable[Document],
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into chunks while preserving source metadata."""
    docs = list(documents)
    if not docs:
        return []

    splitter = create_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    by_source: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "<unknown>")
        by_source[source] = by_source.get(source, 0) + 1

    for source, count in sorted(by_source.items()):
        logger.info("%s — %d chunk(s)", source, count)
    logger.info("Total: %d chunk(s) from %d document(s)", len(chunks), len(docs))
    return chunks


def _preview_chunks(chunks: list[Document], limit: int = 5) -> None:
    for index, chunk in enumerate(chunks[:limit], start=1):
        preview = chunk.page_content[:200].replace("\n", " ")
        if len(chunk.page_content) > 200:
            preview += "…"
        print(f"\n--- chunk {index} ---")
        print(f"metadata: {chunk.metadata}")
        print(f"text: {preview}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from ingestion.loader import load_all_pdfs

    documents = load_all_pdfs()
    chunks = chunk_documents(documents)

    if len(chunks) < 500:
        raise SystemExit(
            f"Expected at least 500 chunks, got {len(chunks)}. "
            "Lower chunk_size or add more source documents."
        )

    print(f"\nFirst 5 of {len(chunks)} chunks:")
    _preview_chunks(chunks)


if __name__ == "__main__":
    main()
