"""B9 — Agent integration tests via httpx + LangSmith run-collector callbacks.

Counts real LLM / tool runs from ``RunCollectorCallbackHandler`` (the LangSmith
local run collector) rather than mocking call counts.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Iterator, List, Optional
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.retrievers import BaseRetriever
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from pydantic import Field

from core.agent import build_agent
from core.tools.cache import HybridCache
from core.tools.source_system import SourceSystemInterface
from tests import test_domain_separation as domain_separation

REPO_ROOT = Path(__file__).resolve().parents[1]

# Soft upper bound for the no-tool RAG path (no network, fake LLM).
_RAG_FAST_SECONDS = 1.0


class _StaticRetriever(BaseRetriever):
    documents: list[Document]

    def _get_relevant_documents(self, query: str) -> list[Document]:
        return self.documents

    async def _aget_relevant_documents(self, query: str) -> list[Document]:
        return self.documents


class _FakeCache:
    def __init__(self, hits: dict[str, str] | None = None) -> None:
        self.hits = hits or {}
        self.stored: dict[tuple[str, str], str] = {}

    def get_cached(self, entity_id: str, module: str) -> str | None:
        return self.hits.get(module)

    def store_cache(self, entity_id: str, module: str, text: str) -> None:
        self.stored[(entity_id, module)] = text


class _FakeSourceSystem:
    def get_sessions(self, entity_id: str, **kwargs: Any) -> str:
        return "fresh-sessions"

    def get_profile(self, entity_id: str) -> str:
        return "fresh-profile"

    def get_goals(self, entity_id: str) -> str:
        return "fresh-goals"


class _StubAdapter:
    def fetch_sessions(self, entity_id: str, **kwargs: Any) -> str:
        return f"raw-sessions-{entity_id}"

    def compile_sessions(self, raw: str) -> str:
        return raw

    def fetch_profile(self, entity_id: str) -> str:
        return f"raw-profile-{entity_id}"

    def compile_profile(self, raw: str) -> str:
        return raw

    def fetch_goals(self, entity_id: str) -> str:
        return f"raw-goals-{entity_id}"

    def compile_goals(self, raw: str) -> str:
        return raw

    def fetch_module(self, entity_id: str, module: str) -> str:
        return f"raw-{module}-{entity_id}"

    def compile_module(self, raw: str, module: str) -> str:
        return raw


class _ScriptedToolCallingChatModel(BaseChatModel):
    """First call requests a tool; second call returns the final cited answer."""

    calls: List[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-calling"

    def bind_tools(self, tools: Any, *, tool_choice: Optional[str] = None, **kwargs: Any):
        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(messages)
        if len(self.calls) == 1:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_entity_sessions",
                        "args": {"entity_id": "e1"},
                        "id": "call_1",
                    }
                ],
            )
        else:
            message = AIMessage(content="Source: entity_sessions -- fresh text.")
        return ChatResult(generations=[ChatGeneration(message=message)])


def _iter_runs(runs: list) -> Iterator[Any]:
    for run in runs:
        yield run
        yield from _iter_runs(getattr(run, "child_runs", None) or [])


def _count_runs(collector: RunCollectorCallbackHandler, run_type: str) -> int:
    return sum(1 for run in _iter_runs(collector.traced_runs) if run.run_type == run_type)


@pytest.fixture
async def agent_client(monkeypatch: pytest.MonkeyPatch):
    """httpx ASGI client against a light-startup FastAPI app."""
    monkeypatch.setenv("USE_CASE_CONFIG", str(REPO_ROOT / "configs" / "mental_health.yaml"))
    monkeypatch.setenv("MINDBRIDGE_LIGHT_STARTUP", "1")

    from backend.main import create_app

    app = create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client, app


@pytest.mark.asyncio
async def test_clinical_question_without_entity_id_routes_to_rag_zero_tools(
    agent_client,
) -> None:
    client, app = agent_client
    collector = RunCollectorCallbackHandler()
    docs = [
        Document(
            page_content="SSRIs are first-line for depression in adults.",
            metadata={"source": "nice-ng222.pdf", "page_number": 12, "doc_type": "guideline"},
        )
    ]
    app.state.llm = FakeListChatModel(responses=["SSRIs are first-line."]).with_config(
        callbacks=[collector]
    )
    app.state.retriever = _StaticRetriever(documents=docs)
    app.state.agent = AsyncMock()

    started = time.perf_counter()
    response = await client.post(
        "/query",
        json={"question": "What does the guideline recommend for depression?"},
    )
    elapsed = time.perf_counter() - started

    assert response.status_code == 200
    body = response.json()
    assert body["route_used"] == "rag"
    assert _count_runs(collector, "tool") == 0
    assert elapsed < _RAG_FAST_SECONDS
    app.state.agent.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_entity_question_cache_hit_one_llm_call(agent_client) -> None:
    client, app = agent_client
    collector = RunCollectorCallbackHandler()
    app.state.cache = _FakeCache(
        hits={"sessions": "cached-sessions", "profile": "cached-profile", "goals": "cached-goals"}
    )
    app.state.source_system = _FakeSourceSystem()
    app.state.llm = FakeListChatModel(
        responses=["Source: entity_sessions -- cached text."]
    ).with_config(callbacks=[collector])
    app.state.agent = AsyncMock()

    response = await client.post(
        "/query",
        json={"question": "Schedule a follow-up next week", "entity_id": "e1"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["route_used"] == "agent"
    assert "cached text" in body["answer"]
    assert _count_runs(collector, "llm") == 1
    assert _count_runs(collector, "tool") == 0
    app.state.agent.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_entity_question_cache_miss_two_llm_calls(agent_client) -> None:
    client, app = agent_client
    collector = RunCollectorCallbackHandler()

    persist_directory = tempfile.mkdtemp()
    vectorstore = Chroma(
        collection_name="agent-integration",
        embedding_function=FakeEmbeddings(size=8),
        persist_directory=persist_directory,
    )
    cache = HybridCache(vectorstore)
    source_system = SourceSystemInterface(_StubAdapter())
    llm = _ScriptedToolCallingChatModel()
    agent = build_agent(app.state.config, llm, source_system, cache)

    app.state.cache = _FakeCache(hits={})  # full miss → AgentExecutor path
    app.state.source_system = _FakeSourceSystem()
    app.state.llm = llm
    app.state.agent = agent.with_config(callbacks=[collector])

    response = await client.post(
        "/query",
        json={"question": "Schedule a follow-up next week", "entity_id": "e1"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["route_used"] == "agent"
    assert "fresh text" in body["answer"]
    assert _count_runs(collector, "llm") == 2
    assert _count_runs(collector, "tool") >= 1


@pytest.mark.asyncio
async def test_agent_answer_with_risk_signal_returns_fallback(agent_client) -> None:
    client, app = agent_client
    collector = RunCollectorCallbackHandler()
    app.state.cache = _FakeCache(
        hits={"sessions": "cached-sessions", "profile": "cached-profile", "goals": "cached-goals"}
    )
    app.state.source_system = _FakeSourceSystem()
    app.state.llm = FakeListChatModel(
        responses=["I diagnose the patient with depression and recommend treatment."]
    ).with_config(callbacks=[collector])
    app.state.agent = AsyncMock()

    response = await client.post(
        "/query",
        json={"question": "Schedule a follow-up next week", "entity_id": "e1"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["route_used"] == "agent"
    assert body["safety_status"] == "blocked"
    assert body["answer"] == app.state.config.safety.fallback_message
    assert body["citations"] == []
    assert _count_runs(collector, "llm") == 1


def test_domain_separation_still_passes_after_agent_files() -> None:
    """Regression gate: domain vocabulary must stay out of core/ and backend/."""
    domain_separation.test_no_domain_words_in_core_or_backend()
