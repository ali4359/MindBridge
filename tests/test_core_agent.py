"""Tests for the tool-calling AgentExecutor factory (core/agent.py)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, List, Optional

import pytest
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from core import load_config
from core.agent import build_agent
from core.tools.cache import HybridCache
from core.tools.source_system import SourceSystemInterface

REPO_ROOT = Path(__file__).resolve().parents[1]
MENTAL_HEALTH_CONFIG = REPO_ROOT / "configs" / "mental_health.yaml"


class _ScriptedToolCallingChatModel(BaseChatModel):
    """First call requests get_entity_sessions; second call returns a cited answer."""

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
                tool_calls=[{"name": "get_entity_sessions", "args": {"entity_id": "e1"}, "id": "call_1"}],
            )
        else:
            message = AIMessage(
                content="Source: entity_sessions -- entity e1 has attended 3 sessions this quarter."
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


class _StubAdapter:
    def fetch_sessions(self, entity_id: str, **kwargs: Any) -> str:
        return f"raw-sessions-{entity_id}"

    def compile_sessions(self, raw: str) -> str:
        return raw


@pytest.fixture()
def cache() -> HybridCache:
    persist_directory = tempfile.mkdtemp()
    vectorstore = Chroma(
        collection_name="agent-test",
        embedding_function=FakeEmbeddings(size=8),
        persist_directory=persist_directory,
    )
    return HybridCache(vectorstore)


def test_build_agent_requires_system_prompt(cache: HybridCache) -> None:
    config = load_config(str(MENTAL_HEALTH_CONFIG))
    config.agent.system_prompt = ""
    source_system = SourceSystemInterface(_StubAdapter())

    with pytest.raises(ValueError, match="agent.system_prompt"):
        build_agent(config, _ScriptedToolCallingChatModel(), source_system, cache)


def test_agent_calls_get_entity_sessions_and_returns_cited_answer(cache: HybridCache) -> None:
    config = load_config(str(MENTAL_HEALTH_CONFIG))
    source_system = SourceSystemInterface(_StubAdapter())
    llm = _ScriptedToolCallingChatModel()

    executor = build_agent(config, llm, source_system, cache)
    result = executor.invoke({"input": "What sessions has entity e1 attended?"})

    tool_names_called = [action.tool for action, _observation in result["intermediate_steps"]]
    assert "get_entity_sessions" in tool_names_called
    assert "Source:" in result["output"]
