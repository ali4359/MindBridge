"""Tests for router-driven /query behaviour (backend/routes/query.py)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.retrievers import BaseRetriever

REPO_ROOT = Path(__file__).resolve().parents[1]


class _StaticRetriever(BaseRetriever):
    documents: list[Document]

    def _get_relevant_documents(self, query: str) -> list[Document]:
        return self.documents


class _FakeCache:
    def __init__(self, hits: dict[str, str] | None = None) -> None:
        self.hits = hits or {}
        self.stored: dict[tuple[str, str], str] = {}

    def get_cached(self, entity_id: str, module: str):
        return self.hits.get(module)

    def store_cache(self, entity_id: str, module: str, text: str) -> None:
        self.stored[(entity_id, module)] = text


class _FakeSourceSystem:
    def get_sessions(self, entity_id: str, **kwargs) -> str:
        return "fresh-sessions"

    def get_profile(self, entity_id: str) -> str:
        return "fresh-profile"

    def get_goals(self, entity_id: str) -> str:
        return "fresh-goals"


@pytest.fixture
def light_client(monkeypatch):
    monkeypatch.setenv("USE_CASE_CONFIG", str(REPO_ROOT / "configs" / "mental_health.yaml"))
    monkeypatch.setenv("MINDBRIDGE_LIGHT_STARTUP", "1")

    from backend.main import create_app

    with TestClient(create_app()) as client:
        yield client


def test_rag_route_used_for_question_without_entity_id(light_client) -> None:
    docs = [
        Document(
            page_content="SSRIs are first-line for depression in adults.",
            metadata={"source": "nice-ng222.pdf", "page_number": 12, "doc_type": "guideline"},
        )
    ]
    light_client.app.state.llm = FakeListChatModel(responses=["SSRIs are first-line."])
    light_client.app.state.retriever = _StaticRetriever(documents=docs)

    response = light_client.post(
        "/query",
        json={"question": "What does the guideline recommend for depression?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["route_used"] == "rag"


def test_agent_route_cache_hit_skips_tool_round_trip(light_client) -> None:
    light_client.app.state.cache = _FakeCache(
        hits={"sessions": "cached-sessions", "profile": "cached-profile", "goals": "cached-goals"}
    )
    light_client.app.state.source_system = _FakeSourceSystem()
    light_client.app.state.llm = FakeListChatModel(responses=["Source: entity_sessions -- cached answer."])
    light_client.app.state.agent = AsyncMock()  # must not be called on a full cache hit

    response = light_client.post(
        "/query",
        json={"question": "Schedule a follow-up next week", "entity_id": "e1"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["route_used"] == "agent"
    assert "cached answer" in body["answer"]
    light_client.app.state.agent.ainvoke.assert_not_called()


def test_agent_route_cache_miss_uses_agent_executor(light_client) -> None:
    light_client.app.state.cache = _FakeCache(hits={"sessions": "cached-sessions"})  # profile/goals miss
    light_client.app.state.source_system = _FakeSourceSystem()
    light_client.app.state.llm = FakeListChatModel(responses=["unused"])
    fake_agent = AsyncMock()
    fake_agent.ainvoke.return_value = {"output": "Source: entity_sessions -- fresh answer."}
    light_client.app.state.agent = fake_agent

    response = light_client.post(
        "/query",
        json={"question": "Schedule a follow-up next week", "entity_id": "e1"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["route_used"] == "agent"
    assert "fresh answer" in body["answer"]
    fake_agent.ainvoke.assert_called_once()


def test_agent_answer_with_risk_signal_returns_fallback(light_client) -> None:
    light_client.app.state.cache = _FakeCache(
        hits={"sessions": "cached-sessions", "profile": "cached-profile", "goals": "cached-goals"}
    )
    light_client.app.state.source_system = _FakeSourceSystem()
    light_client.app.state.llm = FakeListChatModel(
        responses=["I diagnose the patient with depression and recommend treatment."]
    )
    light_client.app.state.agent = AsyncMock()

    response = light_client.post(
        "/query",
        json={"question": "Schedule a follow-up next week", "entity_id": "e1"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["safety_status"] == "blocked"
    assert body["answer"] == light_client.app.state.config.safety.fallback_message
    assert body["citations"] == []
