"""Tests for chains.llm Groq initialisation."""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from pathlib import Path

from chains.llm import CLINICAL_GROQ_MODEL, DEFAULT_MAX_TOKENS, create_groq_llm

REPO_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(REPO_ROOT / ".env")
_has_groq_key = bool(os.environ.get("GROQ_API_KEY"))


def test_create_groq_llm_raises_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("chains.llm.load_dotenv", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        create_groq_llm()


@pytest.mark.skipif(not _has_groq_key, reason="GROQ_API_KEY not set in .env")
def test_groq_llm_invoke() -> None:
    llm = create_groq_llm()
    assert llm.model_name == CLINICAL_GROQ_MODEL
    assert abs(llm.temperature) < 1e-6
    assert llm.max_tokens == DEFAULT_MAX_TOKENS

    response = llm.invoke([HumanMessage(content="Reply with exactly: OK")])
    assert response.content
