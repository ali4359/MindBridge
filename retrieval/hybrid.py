"""Hybrid retrieval: BM25 + Chroma fused with EnsembleRetriever."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_groq import ChatGroq
from langsmith.run_helpers import get_current_run_tree

from retrieval.bm25 import DEFAULT_K, create_bm25_retriever, load_indexed_chunks

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

BM25_WEIGHT = 0.4
VECTOR_WEIGHT = 0.6
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
NUM_QUERY_VARIANTS = 3

CLINICAL_MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are assisting retrieval for a mental-health clinical knowledge base
(coverage includes CBT, DBT, psychotherapy notes, and treatment guidelines).
Generate {num_variants} different versions of the user question to improve recall over
keyword and semantic search. Use clinical terminology and synonyms where appropriate.
Provide each alternative on its own line. Original question: {{question}}""".format(
        num_variants=NUM_QUERY_VARIANTS
    ),
)

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


class _LangSmithMultiQueryRetriever(MultiQueryRetriever):
    """Records generated sub-queries as LangSmith run metadata."""

    def _record_sub_queries(self, original: str, queries: list[str]) -> None:
        logger.info("MultiQuery sub-queries for %r: %s", original, queries)
        run = get_current_run_tree()
        if run is not None:
            run.add_metadata(
                {
                    "original_query": original,
                    "generated_sub_queries": queries,
                    "sub_query_count": len(queries),
                }
            )

    def generate_queries(
        self,
        question: str,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[str]:
        queries = super().generate_queries(question, run_manager)
        self._record_sub_queries(question, queries)
        return queries

    async def agenerate_queries(
        self,
        question: str,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[str]:
        queries = await super().agenerate_queries(question, run_manager)
        self._record_sub_queries(question, queries)
        return queries


def create_groq_llm(*, model: str = DEFAULT_GROQ_MODEL, temperature: float = 0.0) -> ChatGroq:
    """Build a ChatGroq client for query expansion (reads GROQ_API_KEY from .env)."""
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in .env")
    return ChatGroq(model=model, temperature=temperature, groq_api_key=api_key)


def create_multi_query_retriever(
    chunks: list[Document] | None = None,
    *,
    k: int = DEFAULT_K,
    bm25_weight: float = BM25_WEIGHT,
    vector_weight: float = VECTOR_WEIGHT,
    groq_llm: ChatGroq | None = None,
) -> MultiQueryRetriever:
    """Wrap hybrid ensemble retrieval with LLM-generated query variants for higher recall."""
    ensemble = create_ensemble_retriever(
        chunks,
        k=k,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
    )
    llm = groq_llm if groq_llm is not None else create_groq_llm()
    logger.info(
        "Initialising MultiQueryRetriever (%d variants) over hybrid ensemble",
        NUM_QUERY_VARIANTS,
    )
    return _LangSmithMultiQueryRetriever.from_llm(
        retriever=ensemble,
        llm=llm,
        prompt=CLINICAL_MULTI_QUERY_PROMPT,
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
