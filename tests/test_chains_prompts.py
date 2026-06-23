"""Tests for clinical RAG prompt templates."""

from __future__ import annotations

from langchain_core.documents import Document

from chains.prompts import (
    CLINICAL_RAG_PROMPT,
    CONTEXT_CHUNK_TEMPLATE,
    format_retrieved_context,
)


def test_context_chunk_template_includes_source_metadata() -> None:
    rendered = CONTEXT_CHUNK_TEMPLATE.format(
        index=1,
        source="nice-ng222.pdf",
        page_number=12,
        doc_type="guideline",
        content="CBT is recommended as a first-line treatment.",
    )
    assert "nice-ng222.pdf" in rendered
    assert "Page: 12" in rendered
    assert "guideline" in rendered
    assert "CBT is recommended" in rendered


def test_format_retrieved_context_joins_multiple_chunks() -> None:
    docs = [
        Document(
            page_content="First chunk.",
            metadata={"source": "a.pdf", "page_number": 1, "doc_type": "guideline"},
        ),
        Document(
            page_content="Second chunk.",
            metadata={"source": "b.pdf", "page_number": 3, "doc_type": "session_note"},
        ),
    ]
    context = format_retrieved_context(docs)
    assert "[1] Source: a.pdf | Page: 1" in context
    assert "[2] Source: b.pdf | Page: 3" in context
    assert "First chunk." in context
    assert "Second chunk." in context


def test_format_retrieved_context_empty() -> None:
    assert "No relevant context" in format_retrieved_context([])


def test_clinical_rag_prompt_has_context_and_question_variables() -> None:
    assert CLINICAL_RAG_PROMPT.input_variables == ["context", "question"]


def test_clinical_rag_prompt_renders_all_three_sections() -> None:
    messages = CLINICAL_RAG_PROMPT.format_messages(
        context="[1] Source: nice-ng222.pdf | Page: 5 | Type: guideline\nCBT overview.",
        question="What does NICE recommend for depression?",
    )
    system_text = messages[0].content
    human_text = messages[1].content

    assert "NOT a clinician" in system_text
    assert "cite" in system_text.lower()

    assert "## Retrieved context" in human_text
    assert "nice-ng222.pdf" in human_text
    assert "## Therapist question" in human_text
    assert "What does NICE recommend" in human_text
    assert "source name and page" in human_text.lower()
