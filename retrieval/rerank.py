"""FlashRank cross-encoder re-ranking over retrieved document chunks."""

from __future__ import annotations

import logging

from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.retrievers import BaseRetriever, RetrieverLike

logger = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"
DEFAULT_RERANK_TOP_N = 5


def create_flashrank_compressor(
    *,
    model: str = DEFAULT_RERANK_MODEL,
    top_n: int = DEFAULT_RERANK_TOP_N,
) -> FlashrankRerank:
    """Build a local CPU cross-encoder compressor (no API key required)."""
    logger.info("Initialising FlashrankRerank model=%r top_n=%d", model, top_n)
    return FlashrankRerank(model=model, top_n=top_n)


def wrap_with_reranker(
    base_retriever: RetrieverLike,
    *,
    model: str = DEFAULT_RERANK_MODEL,
    top_n: int = DEFAULT_RERANK_TOP_N,
) -> BaseRetriever:
    """Re-score retrieved chunks with FlashRank and return only the top matches."""
    compressor = create_flashrank_compressor(model=model, top_n=top_n)
    return ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor,
    )
