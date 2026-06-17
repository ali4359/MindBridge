"""Hybrid retrieval: BM25 + Chroma fused with EnsembleRetriever."""

from __future__ import annotations

import logging

from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from retrieval.bm25 import DEFAULT_K, create_bm25_retriever, load_indexed_chunks

logger = logging.getLogger(__name__)

BM25_WEIGHT = 0.4
VECTOR_WEIGHT = 0.6

# Same query exercised across BM25-only, vector-only, and hybrid retrievers.
HYBRID_TEST_QUERY = "DBT distress tolerance skills"


def create_chroma_retriever(*, k: int = DEFAULT_K) -> BaseRetriever:
    """Build a similarity retriever over the persisted Chroma index."""
    from ingestion.embeddings import create_embeddings
    from ingestion.indexer import load_vectorstore

    vectorstore = load_vectorstore(create_embeddings())
    return vectorstore.as_retriever(search_kwargs={"k": k})


def create_ensemble_retriever(
    chunks: list[Document] | None = None,
    *,
    k: int = DEFAULT_K,
    bm25_weight: float = BM25_WEIGHT,
    vector_weight: float = VECTOR_WEIGHT,
) -> EnsembleRetriever:
    """Fuse BM25 and vector search with weighted reciprocal rank fusion."""
    docs = chunks if chunks is not None else load_indexed_chunks()
    bm25 = create_bm25_retriever(docs, k=k)
    chroma = create_chroma_retriever(k=k)
    logger.info(
        "Initialising hybrid retriever (BM25=%.1f, vector=%.1f), k=%d",
        bm25_weight,
        vector_weight,
        k,
    )
    return EnsembleRetriever(
        retrievers=[bm25, chroma],
        weights=[bm25_weight, vector_weight],
    )


def _preview_results(
    documents: list[Document],
    *,
    label: str,
    query: str,
) -> None:
    print(f"\n{'=' * 72}")
    print(f"{label} — top {len(documents)} for: {query!r}")
    print("=" * 72)
    for index, doc in enumerate(documents, start=1):
        preview = doc.page_content[:250].replace("\n", " ")
        if len(doc.page_content) > 250:
            preview += "…"
        print(f"\n--- result {index} ---")
        print(f"metadata: {doc.metadata}")
        print(f"text: {preview}")


def compare_retrievers(
    query: str,
    *,
    k: int = DEFAULT_K,
    chunks: list[Document] | None = None,
) -> dict[str, list[Document]]:
    """Run the same query through BM25-only, vector-only, and hybrid retrievers."""
    docs = chunks if chunks is not None else load_indexed_chunks()
    bm25 = create_bm25_retriever(docs, k=k)
    chroma = create_chroma_retriever(k=k)
    hybrid = create_ensemble_retriever(docs, k=k)

    results = {
        "BM25-only": bm25.invoke(query),
        "Vector-only": chroma.invoke(query),
        "Hybrid (EnsembleRetriever)": hybrid.invoke(query),
    }

    for label, documents in results.items():
        if not documents:
            raise ValueError(f"{label} returned no results for {query!r}")
        _preview_results(documents, label=label, query=query)

    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    compare_retrievers(HYBRID_TEST_QUERY, k=DEFAULT_K)
    logger.info(
        "Hybrid retrieval comparison passed (weights: BM25=%.1f, vector=%.1f)",
        BM25_WEIGHT,
        VECTOR_WEIGHT,
    )


if __name__ == "__main__":
    main()
