"""MindBridge retrieval: keyword (BM25), vector search, and hybrid fusion."""

__all__ = [
    "create_bm25_retriever",
    "create_chroma_retriever",
    "create_ensemble_retriever",
    "create_flashrank_compressor",
    "create_groq_llm",
    "create_multi_query_retriever",
    "create_reranked_ensemble_retriever",
    "create_reranked_multi_query_retriever",
    "compare_retrievers",
    "load_indexed_chunks",
    "wrap_with_reranker",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name in {"create_bm25_retriever", "load_indexed_chunks"}:
        from retrieval import bm25

        return getattr(bm25, name)

    if name in {"create_flashrank_compressor", "wrap_with_reranker"}:
        from retrieval import rerank

        return getattr(rerank, name)

    from retrieval import hybrid

    return getattr(hybrid, name)
