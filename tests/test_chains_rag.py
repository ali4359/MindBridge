"""Tests for the LCEL clinical RAG chain."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.retrievers import BaseRetriever

from chains.rag import (
    SAMPLE_THERAPIST_QUERIES,
    create_rag_chain,
    response_appears_cited,
)
from retrieval.hybrid import create_reranked_ensemble_retriever

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

_has_groq_key = bool(os.environ.get("GROQ_API_KEY"))
CHROMA_DIR = REPO_ROOT / "data" / "chromadb"
_has_chroma = CHROMA_DIR.is_dir()


class _StaticRetriever(BaseRetriever):
    """Return fixed documents for deterministic chain tests."""

    documents: list[Document]

    def _get_relevant_documents(self, query: str) -> list[Document]:
        return self.documents

    async def _aget_relevant_documents(self, query: str) -> list[Document]:
        return self.documents


@pytest.fixture
def static_retriever() -> _StaticRetriever:
    return _StaticRetriever(
        documents=[
            Document(
                page_content="SSRIs are recommended as first-line antidepressants for adults.",
                metadata={
                    "source": "nice-ng222-depression.pdf",
                    "page_number": 14,
                    "doc_type": "guideline",
                },
            )
        ]
    )


def test_response_appears_cited_detects_pdf_and_page() -> None:
    assert response_appears_cited("Per nice-ng222-depression.pdf, p. 14, SSRIs are first-line.")
    assert not response_appears_cited("SSRIs are first-line with no citation.")
    assert not response_appears_cited("See nice-ng222-depression.pdf for details.")


def test_rag_chain_with_fake_llm(static_retriever: _StaticRetriever) -> None:
    fake_llm = FakeListChatModel(
        responses=[
            "Per nice-ng222-depression.pdf, p. 14, SSRIs are recommended as first-line treatment."
        ]
    )
    chain = create_rag_chain(retriever=static_retriever, llm=fake_llm)
    answer = chain.invoke("What is first-line treatment for depression?")

    assert "nice-ng222-depression.pdf" in answer
    assert response_appears_cited(answer)


@pytest.mark.skipif(not _has_chroma, reason=f"No Chroma index at {CHROMA_DIR}")
@pytest.mark.skipif(not _has_groq_key, reason="GROQ_API_KEY not set in .env")
@pytest.mark.parametrize("query", SAMPLE_THERAPIST_QUERIES)
def test_rag_chain_live_grounded_cited(query: str) -> None:
    """End-to-end: hybrid retrieval + Groq answer includes source citations."""
    chain = create_rag_chain(retriever=create_reranked_ensemble_retriever())
    answer = chain.invoke(query)

    assert answer.strip()
    assert response_appears_cited(answer), f"Expected PDF + page citation in: {answer!r}"
