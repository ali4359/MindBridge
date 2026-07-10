"""Tests for the Chroma-backed HybridCache."""

from __future__ import annotations

import tempfile

import pytest
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Chroma

from core.tools.cache import CACHE_TTL, HybridCache


@pytest.fixture()
def cache() -> HybridCache:
    persist_directory = tempfile.mkdtemp()
    vectorstore = Chroma(
        collection_name="cache-test",
        embedding_function=FakeEmbeddings(size=8),
        persist_directory=persist_directory,
    )
    return HybridCache(vectorstore)


def test_get_cached_returns_none_when_absent(cache: HybridCache) -> None:
    assert cache.get_cached("entity-1", "sessions") is None


def test_store_then_get_within_ttl_is_a_hit(cache: HybridCache, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.tools.cache.time.time", lambda: 1_000.0)

    cache.store_cache("entity-1", "sessions", "cached session text")

    monkeypatch.setattr("core.tools.cache.time.time", lambda: 1_000.0 + CACHE_TTL["sessions"] - 1)
    assert cache.get_cached("entity-1", "sessions") == "cached session text"


def test_get_after_ttl_expiry_is_a_miss(cache: HybridCache, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.tools.cache.time.time", lambda: 1_000.0)

    cache.store_cache("entity-1", "sessions", "cached session text")

    monkeypatch.setattr("core.tools.cache.time.time", lambda: 1_000.0 + CACHE_TTL["sessions"] + 1)
    assert cache.get_cached("entity-1", "sessions") is None


def test_store_cache_overwrites_previous_entry(cache: HybridCache, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.tools.cache.time.time", lambda: 1_000.0)

    cache.store_cache("entity-1", "profile", "first")
    cache.store_cache("entity-1", "profile", "second")

    assert cache.get_cached("entity-1", "profile") == "second"


def test_cache_is_scoped_by_entity_and_module(cache: HybridCache, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.tools.cache.time.time", lambda: 1_000.0)

    cache.store_cache("entity-1", "goals", "entity-1 goals")
    cache.store_cache("entity-2", "goals", "entity-2 goals")
    cache.store_cache("entity-1", "notes", "entity-1 notes")

    assert cache.get_cached("entity-1", "goals") == "entity-1 goals"
    assert cache.get_cached("entity-2", "goals") == "entity-2 goals"
    assert cache.get_cached("entity-1", "notes") == "entity-1 notes"
