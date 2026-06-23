"""Groq LLM initialisation for the clinical RAG chain."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

REPO_ROOT = Path(__file__).resolve().parents[1]

# llama-3.1-70b-versatile was decommissioned; Groq recommends llama-3.3-70b-versatile.
CLINICAL_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024


def create_groq_llm(
    *,
    model: str = CLINICAL_GROQ_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> ChatGroq:
    """Build a ChatGroq client for grounded clinical responses (reads ``GROQ_API_KEY`` from ``.env``)."""
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in .env")
    return ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        groq_api_key=api_key,
    )


def _smoke_test() -> None:
    """Quick invoke() smoke test when run as ``python -m chains.llm``."""
    llm = create_groq_llm()
    response = llm.invoke([HumanMessage(content="Reply with exactly: OK")])
    print(response.content)


if __name__ == "__main__":
    _smoke_test()
