"""Tests for the domain-agnostic platform FastAPI endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda

REPO_ROOT = Path(__file__).resolve().parents[1]


class _StaticRetriever(BaseRetriever):
    documents: list[Document]

    def _get_relevant_documents(self, query: str) -> list[Document]:
        return self.documents

    async def _aget_relevant_documents(self, query: str) -> list[Document]:
        return self.documents


@pytest.fixture
def light_client(monkeypatch):
    monkeypatch.setenv("USE_CASE_CONFIG", str(REPO_ROOT / "configs" / "mental_health.yaml"))
    monkeypatch.setenv("MINDBRIDGE_LIGHT_STARTUP", "1")

    from backend.main import create_app

    with TestClient(create_app()) as client:
        yield client


def test_health_and_use_case(light_client) -> None:
    health = light_client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert body["model_name"] == "llama-3.3-70b-versatile"
    assert body["vector_store_chunk_count"] == 0
    assert body["neo4j_status"] == "disconnected"
    assert body["uptime_seconds"] >= 0

    summary = light_client.get("/use-case")
    assert summary.status_code == 200
    payload = summary.json()
    assert payload["use_case_name"] == "MindBridge Clinical Copilot"
    assert payload["domain"] == "mental_health"
    assert payload["langsmith_project"] == "mental_health-traces"
    assert payload["entity_name"] == "patient"
    assert "subjective" in payload["output_schema_fields"]
    assert payload["chunk_count"] == 0
    assert payload["graph_node_count"] > 0
    assert payload["data_sources_loaded"]


def test_metrics_returns_missing_when_no_csv(light_client) -> None:
    response = light_client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"missing", "empty", "invalid", "ok"}
    assert body["source"].endswith("ragas_results.csv")


def test_metrics_reads_ragas_csv(light_client, tmp_path) -> None:
    csv_path = tmp_path / "ragas_results.csv"
    csv_path.write_text(
        "timestamp,faithfulness,answer_relevancy,context_precision,context_recall\n"
        "2026-07-08T15:00:00Z,0.81,0.77,0.72,0.70\n",
        encoding="utf-8",
    )
    with patch("backend.routes.platform.RAGAS_RESULTS_CSV", csv_path):
        response = light_client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["scores"]["faithfulness"] == 0.81


def test_query_endpoint_applies_safety(light_client) -> None:
    docs = [
        Document(
            page_content="SSRIs are first-line for depression in adults.",
            metadata={
                "source": "nice-ng222-depression.pdf",
                "page_number": 12,
                "doc_type": "guideline",
            },
        )
    ]
    fake_llm = FakeListChatModel(
        responses=[
            "SSRIs are first-line (nice-ng222-depression.pdf, p. 12).",
            json.dumps({"safe": True, "violated_rules": [], "explanation": "ok"}),
        ]
    )
    light_client.app.state.llm = fake_llm
    light_client.app.state.retriever = _StaticRetriever(documents=docs)
    light_client.app.state.vectorstore = MagicMock()

    response = light_client.post(
        "/query",
        json={"question": "What does NICE recommend first-line for depression?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["safety_status"] == "safe"
    assert body["session_id"]
    assert "SSRIs" in body["answer"]
    assert body["citations"][0]["source"] == "nice-ng222-depression.pdf"


def test_generate_output_returns_schema_fields(light_client) -> None:
    structured_payload = {
        "subjective": "Client reports worry.",
        "objective": "Affect anxious.",
        "assessment": "GAD.",
        "plan": "Continue CBT.",
    }
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = RunnableLambda(
        lambda _: structured_payload
    )
    light_client.app.state.llm = fake_llm
    light_client.app.state.graph_driver = None

    with patch(
        "backend.routes.output.load_entity_context",
        return_value="(No patient profile provided.)",
    ):
        response = light_client.post(
            "/generate-output",
            json={
                "entity_id": "patient-1",
                "notes_text": "SOAP note text here for testing.",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["subjective"] == "Client reports worry."
    assert set(body) == {"subjective", "objective", "assessment", "plan"}


def test_generate_output_legal_profile_returns_legal_brief(monkeypatch) -> None:
    monkeypatch.setenv("USE_CASE_CONFIG", str(REPO_ROOT / "configs" / "legal.yaml"))
    monkeypatch.setenv("MINDBRIDGE_LIGHT_STARTUP", "1")

    structured_payload = {
        "issue": "Whether breach is proven.",
        "rule": "Plaintiff must prove contract, breach, and damages.",
        "analysis": "The evidence supports each element under cited authorities.",
        "conclusion": "Breach claim is likely viable.",
    }
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = RunnableLambda(
        lambda _: structured_payload
    )

    from backend.main import create_app

    with TestClient(create_app()) as client:
        client.app.state.llm = fake_llm
        client.app.state.graph_driver = None
        with patch(
            "backend.routes.output.load_entity_context",
            return_value="(No client profile provided.)",
        ):
            response = client.post(
                "/generate-output",
                json={
                    "entity_id": "client-1",
                    "notes_text": "Matter intake and legal analysis notes.",
                },
            )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"issue", "rule", "analysis", "conclusion"}
    assert body["conclusion"] == "Breach claim is likely viable."


def test_entity_graph_requires_schema(light_client) -> None:
    light_client.app.state.graph_driver = MagicMock()
    with (
        patch(
            "backend.routes.entity.read_entity_graph",
            return_value={
                "Session": ["patient-1-session-1"],
                "Diagnosis": ["GAD"],
                "Intervention": ["thought record"],
                "Symptom": ["worry"],
                "Medication": ["sertraline 50mg"],
                "Homework": ["daily thought log"],
            },
        ),
        patch(
            "backend.routes.entity.find_similar_entities",
            return_value=["patient-7", "patient-9"],
        ),
    ):
        response = light_client.get("/entity/patient-1/graph")
    assert response.status_code == 200
    body = response.json()
    assert body["entity_id"] == "patient-1"
    assert body["diagnoses"] == ["GAD"]
    assert body["interventions_tried"] == ["thought record"]
    assert body["similar_entities"] == ["patient-7", "patient-9"]
    assert body["subgraph"]["Diagnosis"] == ["GAD"]


def test_query_stream_emits_sse_tokens(light_client) -> None:
    docs = [
        Document(
            page_content="TIPP skills calm the body during crisis.",
            metadata={
                "source": "dbt-skills-workbook-mckay.pdf",
                "page_number": 40,
                "doc_type": "research",
            },
        )
    ]
    fake_llm = FakeListChatModel(
        responses=[
            "TIPP: Temperature, Intense exercise.",
            json.dumps({"safe": True, "violated_rules": [], "explanation": "ok"}),
        ]
    )
    light_client.app.state.llm = fake_llm
    light_client.app.state.retriever = _StaticRetriever(documents=docs)

    with light_client.stream(
        "POST",
        "/query/stream",
        json={"question": "What are TIPP skills?"},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert "data:" in body
    assert "TIPP" in body or "done" in body
    assert "session_id" in body
    assert "safety_status" in body


def test_initialize_platform_stores_state(monkeypatch) -> None:
    monkeypatch.setenv("USE_CASE_CONFIG", str(REPO_ROOT / "configs" / "mental_health.yaml"))
    monkeypatch.delenv("MINDBRIDGE_LIGHT_STARTUP", raising=False)

    fake_collection = MagicMock()
    fake_collection.count.return_value = 743
    fake_vs = MagicMock()
    fake_vs._collection = fake_collection

    with (
        patch("backend.main.validate_config"),
        patch("backend.main.get_embeddings", return_value=MagicMock()),
        patch("backend.main.load_vectorstore", return_value=fake_vs),
        patch("backend.main.documents_from_vectorstore", return_value=[]),
        patch("backend.main.build_llm", return_value=MagicMock()),
        patch("backend.main.build_retriever", return_value=MagicMock()),
        patch("backend.main.build_chain", return_value=MagicMock()),
        patch("backend.main._connect_neo4j", return_value=MagicMock()),
    ):
        from backend.main import create_app, initialize_platform

        application = create_app()
        count = initialize_platform(application)

    assert count == 743
    assert application.state.chunk_count == 743
    assert application.state.config.name == "mental_health"


def test_initialize_platform_initialises_agent(monkeypatch, caplog) -> None:
    monkeypatch.setenv("USE_CASE_CONFIG", str(REPO_ROOT / "configs" / "mental_health.yaml"))
    monkeypatch.delenv("MINDBRIDGE_LIGHT_STARTUP", raising=False)

    fake_collection = MagicMock()
    fake_collection.count.return_value = 0
    fake_vs = MagicMock()
    fake_vs._collection = fake_collection

    with (
        patch("backend.main.validate_config"),
        patch("backend.main.get_embeddings", return_value=MagicMock()),
        patch("backend.main.load_vectorstore", return_value=fake_vs),
        patch("backend.main.documents_from_vectorstore", return_value=[]),
        patch("backend.main.build_llm", return_value=MagicMock()),
        patch("backend.main.build_retriever", return_value=MagicMock()),
        patch("backend.main.build_chain", return_value=MagicMock()),
        patch("backend.main._connect_neo4j", return_value=MagicMock()),
    ):
        from backend.main import create_app, initialize_platform

        application = create_app()
        with caplog.at_level("INFO", logger="backend.main"):
            initialize_platform(application)

    assert application.state.source_system is not None
    # mental_health.yaml declares source_system.adapter — resolve via ADAPTER_REGISTRY
    assert application.state.source_system.adapter is not None
    assert application.state.source_system.adapter.__class__.__name__ == "RauhaAdapter"
    assert application.state.cache is not None
    assert application.state.agent is not None
    assert application.state.router is not None
    assert any("Agent initialised with 6 tools" in record.message for record in caplog.records)


def test_entity_profile_from_vector_lookup(light_client) -> None:
    docs = [
        Document(
            page_content="session note",
            metadata={
                "entity_id": "patient-42",
                "session_id": "patient-42-session-1",
                "session_number": 1,
                "primary_diagnosis": "GAD",
            },
        ),
        Document(
            page_content="session note 2",
            metadata={
                "entity_id": "patient-42",
                "session_id": "patient-42-session-2",
                "session_number": 2,
                "primary_diagnosis": "GAD",
            },
        ),
    ]
    with patch(
        "backend.routes.entity.lookup_entity_documents",
        return_value=docs,
    ):
        light_client.app.state.vectorstore = MagicMock()
        response = light_client.get("/entity/patient-42")

    assert response.status_code == 200
    body = response.json()
    assert body["session_count"] == 2
    assert body["profile"]["primary_diagnosis"] == "GAD"
