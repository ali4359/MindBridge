"""Build and persist the MindBridge Chroma vector store."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
CHROMA_PERSIST_DIR = REPO_ROOT / "data" / "chromadb"
COLLECTION_NAME = "mindbridge"

SIMILARITY_TEST_QUERY = "CBT for depression"


def build_vectorstore(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
    *,
    persist_directory: Path = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
    reset: bool = True,
) -> Chroma:
    """Embed chunks and persist them to disk."""
    persist_directory = Path(persist_directory)
    if reset and persist_directory.exists():
        shutil.rmtree(persist_directory)
        logger.info("Cleared existing index at %s", persist_directory)

    persist_directory.mkdir(parents=True, exist_ok=True)
    logger.info("Indexing %d chunk(s) into %s", len(chunks), persist_directory)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_directory),
    )
    logger.info("Persisted %d vector(s)", vectorstore._collection.count())
    return vectorstore


def load_vectorstore(
    embeddings: HuggingFaceEmbeddings,
    *,
    persist_directory: Path = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """Reload a persisted Chroma store from disk."""
    persist_directory = Path(persist_directory)
    if not persist_directory.is_dir():
        raise FileNotFoundError(f"No persisted index at {persist_directory}")

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )
    logger.info(
        "Reloaded %d vector(s) from %s",
        vectorstore._collection.count(),
        persist_directory,
    )
    return vectorstore


def _preview_results(documents: list[Document], *, query: str) -> None:
    print(f"\nTop {len(documents)} results for: {query!r}\n")
    for index, doc in enumerate(documents, start=1):
        preview = doc.page_content[:250].replace("\n", " ")
        if len(doc.page_content) > 250:
            preview += "…"
        print(f"--- result {index} ---")
        print(f"metadata: {doc.metadata}")
        print(f"text: {preview}\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from ingestion.chunker import chunk_documents
    from ingestion.embeddings import create_embeddings
    from ingestion.loader import load_all_pdfs

    embeddings = create_embeddings()
    chunks = chunk_documents(load_all_pdfs())
    expected_count = len(chunks)

    build_vectorstore(chunks, embeddings)

    # Reload from disk with a fresh client to verify persistence.
    reloaded = load_vectorstore(embeddings)
    stored_count = reloaded._collection.count()
    if stored_count != expected_count:
        raise SystemExit(
            f"Persistence check failed: expected {expected_count} vectors, found {stored_count}"
        )

    results = reloaded.similarity_search(SIMILARITY_TEST_QUERY, k=5)
    if not results:
        raise SystemExit("similarity_search returned no results")

    _preview_results(results, query=SIMILARITY_TEST_QUERY)
    logger.info("Persistence and similarity search checks passed")


if __name__ == "__main__":
    main()
