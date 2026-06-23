"""LangChain QA chains and RAG pipeline logic."""

from typing import TYPE_CHECKING

__all__ = [
    "CLINICAL_GROQ_MODEL",
    "CLINICAL_RAG_PROMPT",
    "CONTEXT_CHUNK_TEMPLATE",
    "create_groq_llm",
    "format_retrieved_context",
]

if TYPE_CHECKING:
    from chains.llm import CLINICAL_GROQ_MODEL, create_groq_llm
    from chains.prompts import (
        CLINICAL_RAG_PROMPT,
        CONTEXT_CHUNK_TEMPLATE,
        format_retrieved_context,
    )


def __getattr__(name: str):
    if name in __all__:
        if name in ("CLINICAL_GROQ_MODEL", "create_groq_llm"):
            from chains import llm

            return getattr(llm, name)
        from chains import prompts

        return getattr(prompts, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
