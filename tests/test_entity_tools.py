"""Tests for the domain-blind agent tools (core/tools/entity_tools.py)."""

from __future__ import annotations

import tempfile

import pytest
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from core.tools import entity_tools
from core.tools.cache import HybridCache
from core.tools.source_system import SourceSystemInterface


class _StubAdapter:
    def fetch_sessions(self, entity_id, **kwargs):
        return f"raw-sessions-{entity_id}-limit{kwargs.get('limit')}"

    def compile_sessions(self, raw):
        return raw

    def fetch_profile(self, entity_id):
        return f"raw-profile-{entity_id}"

    def compile_profile(self, raw):
        return raw

    def fetch_goals(self, entity_id):
        return f"raw-goals-{entity_id}"

    def compile_goals(self, raw):
        return raw

    def fetch_module(self, entity_id, module):
        return f"raw-{module}-{entity_id}"

    def compile_module(self, raw):
        return raw


class _StaticRetriever(BaseRetriever):
    documents: list[Document] = []

    def _get_relevant_documents(self, query: str) -> list[Document]:
        return self.documents


@pytest.fixture()
def wired_tools():
    persist_directory = tempfile.mkdtemp()
    vectorstore = Chroma(
        collection_name="entity-tools-test",
        embedding_function=FakeEmbeddings(size=8),
        persist_directory=persist_directory,
    )
    cache = HybridCache(vectorstore)
    source_system = SourceSystemInterface(_StubAdapter())
    retriever = _StaticRetriever(
        documents=[Document(page_content="hybrid fusion detail", metadata={"source": "doc.pdf", "page_number": 3})]
    )
    entity_tools.init_tools(source_system, cache, retriever)
    yield entity_tools
    entity_tools.init_tools(None, None, None)


def test_get_entity_sessions_calls_source_system(wired_tools) -> None:
    result = wired_tools.get_entity_sessions.invoke({"entity_id": "e1", "limit": 3})
    assert result == "raw-sessions-e1-limit3"


def test_get_entity_profile_uses_cache_on_second_call(wired_tools) -> None:
    first = wired_tools.get_entity_profile.invoke({"entity_id": "e1"})
    assert first == "raw-profile-e1"

    # swap the adapter so a cache miss would return a different value
    wired_tools._source_system = SourceSystemInterface(_RaisingAdapter())
    second = wired_tools.get_entity_profile.invoke({"entity_id": "e1"})
    assert second == "raw-profile-e1"


class _RaisingAdapter:
    def fetch_profile(self, entity_id):
        raise AssertionError("should not be called on a cache hit")

    def compile_profile(self, raw):
        return raw


def test_get_entity_goals(wired_tools) -> None:
    assert wired_tools.get_entity_goals.invoke({"entity_id": "e1"}) == "raw-goals-e1"


def test_get_entity_notes(wired_tools) -> None:
    assert wired_tools.get_entity_notes.invoke({"entity_id": "e1"}) == "raw-notes-e1"


def test_get_source_module_generic_fallback(wired_tools) -> None:
    assert wired_tools.get_source_module.invoke({"entity_id": "e1", "module": "assessments"}) == "raw-assessments-e1"


def test_search_knowledge_base_uses_injected_retriever(wired_tools) -> None:
    result = wired_tools.search_knowledge_base.invoke({"query": "fusion"})
    assert "hybrid fusion detail" in result
    assert "doc.pdf" in result


def test_entity_tools_list_has_six_tools() -> None:
    assert len(entity_tools.ENTITY_TOOLS) == 6
    assert [t.name for t in entity_tools.ENTITY_TOOLS] == [
        "get_entity_sessions",
        "get_entity_profile",
        "get_entity_goals",
        "get_entity_notes",
        "get_source_module",
        "search_knowledge_base",
    ]
