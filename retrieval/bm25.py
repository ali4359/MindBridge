"""BM25 keyword retriever over Week 1 document chunks."""

from __future__ import annotations

import logging

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

DEFAULT_K = 10

# Keyword-heavy queries where exact clinical terminology matters.
KEYWORD_TEST_QUERIES: tuple[str, ...] = (
    "DBT distress tolerance skills",
    "cognitive restructuring thought record",
    "NICE guideline antidepressant first line",
)


def load_indexed_chunks() -> list[Document]:
    """Load and chunk all Week 1 PDFs (same corpus as the Chroma index)."""
    from ingestion.chunker import chunk_documents
    from ingestion.loader import load_all_pdfs

    documents = load_all_pdfs()
    chunks = chunk_documents(documents)
    if not chunks:
        raise ValueError("No chunks produced from Week 1 PDFs — run ingestion/download_sources first")
    return chunks


def create_bm25_retriever(
    chunks: list[Document] | None = None,
    *,
    k: int = DEFAULT_K,
) -> BM25Retriever:
    """Build a BM25 retriever from document chunks."""
    docs = chunks if chunks is not None else load_indexed_chunks()
    logger.info("Initialising BM25 retriever over %d chunk(s), k=%d", len(docs), k)
    return BM25Retriever.from_documents(docs, k=k)


def _preview_results(documents: list[Document], *, query: str) -> None:
    print(f"\nTop {len(documents)} BM25 results for: {query!r}\n")
    for index, doc in enumerate(documents, start=1):
        preview = doc.page_content[:250].replace("\n", " ")
        if len(doc.page_content) > 250:
            preview += "…"
        print(f"--- result {index} ---")
        print(f"metadata: {doc.metadata}")
        print(f"text: {preview}\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    chunks = load_indexed_chunks()
    retriever = create_bm25_retriever(chunks, k=DEFAULT_K)

    for query in KEYWORD_TEST_QUERIES:
        results = retriever.invoke(query)
        if not results:
            raise SystemExit(f"BM25 returned no results for {query!r}")
        _preview_results(results, query=query)

    logger.info("BM25 keyword retrieval checks passed (%d queries)", len(KEYWORD_TEST_QUERIES))


if __name__ == "__main__":
    main()
