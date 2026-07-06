"""Config-driven Groq LLM factory."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from core.config import UseCaseConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_llm(config: UseCaseConfig) -> ChatGroq:
    """Build a ChatGroq client from ``config.llm`` (reads ``GROQ_API_KEY`` from ``.env``)."""
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in .env")

    llm_cfg = config.llm
    return ChatGroq(
        model=llm_cfg.model,
        temperature=llm_cfg.temperature,
        max_tokens=llm_cfg.max_tokens,
        groq_api_key=api_key,
    )
