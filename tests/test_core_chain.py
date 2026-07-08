"""Tests for the config-driven core RAG chain."""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.retrievers import BaseRetriever

from core import load_config
from core.chain import (
    build_chain,
    build_prompt,
    format_entity_graph,
    format_retrieved_context,
    response_appears_cited,
)


class _StaticRetriever(BaseRetriever):
    documents: list[Document]

    def _get_relevant_documents(self, query: str) -> list[Document]:
        return self.documents

    async def _aget_relevant_documents(self, query: str) -> list[Document]:
        return self.documents


class _PromptEchoChatModel(BaseChatModel):
    """Echoes human-message content so tests can inspect prompt injection."""

    @property
    def _llm_type(self) -> str:
        return "prompt-echo"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        human = next((m.content for m in messages if m.type == "human"), "")
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=str(human)))]
        )


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


def test_format_entity_graph_uses_config_labels() -> None:
    config = load_config("configs/mental_health.yaml")
    text = format_entity_graph(
        {
            "Session": ["session-1"],
            "Diagnosis": ["Generalized Anxiety Disorder"],
            "Intervention": ["thought record"],
            "Symptom": [],
            "Medication": ["sertraline 50mg"],
            "Homework": ["daily thought log"],
        },
        config,
    )
    assert "- Diagnosis: Generalized Anxiety Disorder" in text
    assert "- Medication: sertraline 50mg" in text
    assert "- Symptom: (none)" in text


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


def test_build_chain_injects_patient_a_graph_into_llm_prompt() -> None:
    """Query about Patient A — graph data must appear in the LLM-facing prompt/response."""
    config = load_config("configs/mental_health.yaml")
    patient_a_graph = {
        "Session": ["patient-A-session-1", "patient-A-session-2"],
        "Diagnosis": ["Major Depressive Disorder"],
        "Intervention": ["behavioral activation", "thought record"],
        "Symptom": ["anhedonia", "low mood"],
        "Medication": ["escitalopram 10mg"],
        "Homework": ["activity scheduling"],
    }
    retriever = _StaticRetriever(
        documents=[
            Document(
                page_content="Behavioral activation is effective for depression in adults.",
                metadata={
                    "source": "nice-ng222-depression.pdf",
                    "page_number": 22,
                    "doc_type": "guideline",
                },
            )
        ]
    )
    echo_llm = _PromptEchoChatModel()
    driver = MagicMock()

    with patch(
        "core.graph.read_entity_graph",
        return_value=patient_a_graph,
    ) as read_mock:
        chain = build_chain(
            config,
            retriever,
            echo_llm,
            entity_id="patient-A",
            driver=driver,
        )
        answer = chain.invoke(
            "What interventions has Patient A responded to, and what medication is prescribed?"
        )

    read_mock.assert_called_once_with("patient-A", config, driver)
    assert "## Patient context" in answer
    assert "Major Depressive Disorder" in answer
    assert "behavioral activation" in answer
    assert "escitalopram 10mg" in answer
    assert "activity scheduling" in answer
    assert "## Retrieved context" in answer
    assert "nice-ng222-depression.pdf" in answer
    assert "Patient A" in answer


def test_build_chain_without_entity_id_uses_empty_context() -> None:
    config = load_config("configs/mental_health.yaml")
    retriever = _StaticRetriever(documents=[])
    echo_llm = _PromptEchoChatModel()

    with patch("core.graph.read_entity_graph") as read_mock:
        chain = build_chain(config, retriever, echo_llm)
        answer = chain.invoke("Any patient-specific history?")

    read_mock.assert_not_called()
    assert config.prompts.empty_entity_context in answer
