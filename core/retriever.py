"""Config-driven retriever factory — no domain-specific retrieval logic."""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever, MultiQueryRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever

from core.config import UseCaseConfig

logger = logging.getLogger(__name__)


def _resolve_chunks(config: UseCaseConfig, chunks: Optional[list[Document]]) -> list[Document]:
    if chunks is not None:
        return chunks
    from core.ingestion import build_ingestion_pipeline

    return build_ingestion_pipeline(config).run()


def create_vector_retriever(vectorstore: Any, config: UseCaseConfig) -> BaseRetriever:
    """Build a similarity retriever over the supplied vector store."""
    k = config.retrieval.top_k
    return vectorstore.as_retriever(search_kwargs={"k": k})


def create_bm25_retriever(chunks: list[Document], config: UseCaseConfig) -> BM25Retriever:
    """Build a BM25 retriever from in-memory document chunks."""
    k = config.retrieval.top_k
    logger.info("Initialising BM25 retriever over %d chunk(s), k=%d", len(chunks), k)
    return BM25Retriever.from_documents(chunks, k=k)


def create_ensemble_retriever(
    vectorstore: Any,
    chunks: list[Document],
    config: UseCaseConfig,
) -> EnsembleRetriever:
    """Fuse BM25 and vector search with weighted reciprocal rank fusion."""
    retrieval = config.retrieval
    bm25 = create_bm25_retriever(chunks, config)
    vector = create_vector_retriever(vectorstore, config)
    logger.info(
        "Initialising hybrid retriever (BM25=%.1f, vector=%.1f), k=%d",
        retrieval.bm25_weight,
        retrieval.vector_weight,
        retrieval.top_k,
    )
    return EnsembleRetriever(
        retrievers=[bm25, vector],
        weights=[retrieval.bm25_weight, retrieval.vector_weight],
    )


def create_multi_query_retriever(
    base_retriever: BaseRetriever,
    config: UseCaseConfig,
    llm: Any,
) -> MultiQueryRetriever:
    """Wrap a base retriever with LLM-generated query variants."""
    template = config.prompts.multi_query_template
    if not template:
        raise ValueError("multi_query retrieval requires prompts.multi_query_template in config")

    prompt = PromptTemplate(
        input_variables=["question"],
        template=template.replace(
            "{num_variants}",
            str(config.retrieval.multi_query_variants),
        ),
    )
    logger.info(
        "Initialising MultiQueryRetriever (%d variants)",
        config.retrieval.multi_query_variants,
    )
    return MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        prompt=prompt,
    )


def apply_reranker(base_retriever: BaseRetriever, config: UseCaseConfig) -> BaseRetriever:
    """Re-score retrieved chunks with FlashRank and return only the top matches."""
    retrieval = config.retrieval
    logger.info(
        "Initialising FlashrankRerank model=%r top_n=%d",
        retrieval.rerank_model,
        retrieval.rerank_top_n,
    )
    compressor = FlashrankRerank(
        model=retrieval.rerank_model,
        top_n=retrieval.rerank_top_n,
    )
    return ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor,
    )


def _should_rerank(config: UseCaseConfig) -> bool:
    retrieval = config.retrieval
    return retrieval.enable_rerank or retrieval.mode == "hybrid_rerank"


def build_retriever(
    config: UseCaseConfig,
    vectorstore: Any,
    *,
    chunks: Optional[list[Document]] = None,
    llm: Optional[Any] = None,
) -> BaseRetriever:
    """Factory for a config-bound retriever over the given vector store."""
    retrieval = config.retrieval
    mode = retrieval.mode

    if mode == "bm25":
        docs = _resolve_chunks(config, chunks)
        retriever: BaseRetriever = create_bm25_retriever(docs, config)
    elif mode == "vector":
        retriever = create_vector_retriever(vectorstore, config)
    elif mode in ("hybrid", "hybrid_rerank"):
        docs = _resolve_chunks(config, chunks)
        retriever = create_ensemble_retriever(vectorstore, docs, config)
    elif mode == "multi_query":
        if llm is None:
            raise ValueError("multi_query retrieval requires an llm argument")
        docs = _resolve_chunks(config, chunks)
        hybrid = create_ensemble_retriever(vectorstore, docs, config)
        retriever = create_multi_query_retriever(hybrid, config, llm)
    else:
        raise ValueError(f"Unsupported retrieval mode: {mode!r}")

    if retrieval.enable_multi_query and mode == "hybrid":
        if llm is None:
            raise ValueError("enable_multi_query requires an llm argument")
        retriever = create_multi_query_retriever(retriever, config, llm)

    if _should_rerank(config):
        retriever = apply_reranker(retriever, config)

    return retriever
