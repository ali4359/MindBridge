"""Tests for the config-driven core RAG chain."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.retrievers import BaseRetriever

from core import load_config
from core.chain import (
    build_chain,
    build_prompt,
    format_retrieved_context,
    response_appears_cited,
)


class _StaticRetriever(BaseRetriever):
    documents: list[Document]

    def _get_relevant_documents(self, query: str) -> list[Document]:
        return self.documents

    async def _aget_relevant_documents(self, query: str) -> list[Document]:
        return self.documents


def test_build_prompt_uses_config_headers() -> None:
    config = load_config("configs/mental_health.yaml")
    prompt = build_prompt(config)
    messages = prompt.format_messages(
        entity_context="(No patient profile provided.)",
        context="[1] Source: nice-ng222.pdf | Page: 5 | Type: guideline\nCBT overview.",
        question="What does NICE recommend for depression?",
    )
    system_text = messages[0].content
    human_text = messages[1].content

    assert "NOT a clinician" in system_text
    assert "## Patient context" in human_text
    assert "## Retrieved context" in human_text
    assert "## Therapist question" in human_text
    assert "source name and page" in human_text.lower()


def test_format_retrieved_context_uses_config_template() -> None:
    config = load_config("configs/mental_health.yaml")
    docs = [
        Document(
            page_content="First chunk.",
            metadata={"source": "a.pdf", "page_number": 1, "doc_type": "guideline"},
        ),
    ]
    context = format_retrieved_context(docs, config)
    assert "[1] Source: a.pdf | Page: 1" in context
    assert format_retrieved_context([], config) == config.output_schema.empty_context_message


def test_response_appears_cited_reads_output_schema() -> None:
    config = load_config("configs/mental_health.yaml")
    assert response_appears_cited("Per nice-ng222-depression.pdf, p. 14, SSRIs are first-line.", config)
    assert not response_appears_cited("SSRIs are first-line with no citation.", config)


def test_build_chain_with_fake_llm() -> None:
    config = load_config("configs/mental_health.yaml")
    retriever = _StaticRetriever(
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
    fake_llm = FakeListChatModel(
        responses=[
            "Per nice-ng222-depression.pdf, p. 14, SSRIs are recommended as first-line treatment."
        ]
    )
    chain = build_chain(config, retriever, fake_llm)
    answer = chain.invoke("What is first-line treatment for depression?")

    assert "nice-ng222-depression.pdf" in answer
    assert response_appears_cited(answer, config)
