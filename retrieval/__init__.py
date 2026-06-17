"""MindBridge retrieval: keyword (BM25), vector search, and hybrid fusion."""

__all__ = [
    "create_bm25_retriever",
    "create_chroma_retriever",
    "create_ensemble_retriever",
    "create_groq_llm",
    "create_multi_query_retriever",
    "compare_retrievers",
    "load_indexed_chunks",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name in {"create_bm25_retriever", "load_indexed_chunks"}:
        from retrieval import bm25

        return getattr(bm25, name)

    from retrieval import hybrid

    return getattr(hybrid, name)
