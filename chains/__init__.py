"""LangChain QA chains and RAG pipeline logic."""

from typing import TYPE_CHECKING

__all__ = [
    "CLINICAL_GROQ_MODEL",
    "CLINICAL_RAG_PROMPT",
    "CONTEXT_CHUNK_TEMPLATE",
    "SAMPLE_THERAPIST_QUERIES",
    "create_groq_llm",
    "create_rag_chain",
    "format_retrieved_context",
    "response_appears_cited",
]

if TYPE_CHECKING:
    from chains.llm import CLINICAL_GROQ_MODEL, create_groq_llm
    from chains.prompts import (
        CLINICAL_RAG_PROMPT,
        CONTEXT_CHUNK_TEMPLATE,
        format_retrieved_context,
    )
    from chains.rag import (
        SAMPLE_THERAPIST_QUERIES,
        create_rag_chain,
        response_appears_cited,
    )


def __getattr__(name: str):
    if name in __all__:
        if name in ("CLINICAL_GROQ_MODEL", "create_groq_llm"):
            from chains import llm

            return getattr(llm, name)
        if name in ("SAMPLE_THERAPIST_QUERIES", "create_rag_chain", "response_appears_cited"):
            from chains import rag

            return getattr(rag, name)
        from chains import prompts

        return getattr(prompts, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
