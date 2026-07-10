"""Tests for async parallel entity-context fetch (core/tools/parallel_fetch.py)."""

from __future__ import annotations

import time

import pytest

from core.tools.parallel_fetch import fetch_entity_context
from core.tools.source_system import SourceSystemInterface

FETCH_LATENCY = 0.2


class _SlowAdapter:
    """Adapter whose fetch_X calls simulate blocking I/O latency."""

    def fetch_sessions(self, entity_id, **kwargs):
        time.sleep(FETCH_LATENCY)
        return f"sessions-for-{entity_id}"

    def compile_sessions(self, raw):
        return raw

    def fetch_profile(self, entity_id):
        time.sleep(FETCH_LATENCY)
        return f"profile-for-{entity_id}"

    def compile_profile(self, raw):
        return raw

    def fetch_goals(self, entity_id):
        time.sleep(FETCH_LATENCY)
        return f"goals-for-{entity_id}"

    def compile_goals(self, raw):
        return raw


class _FlakyAdapter(_SlowAdapter):
    def fetch_profile(self, entity_id):
        time.sleep(FETCH_LATENCY)
        raise RuntimeError("upstream profile lookup failed")


async def test_fetch_entity_context_combines_sections() -> None:
    source_system = SourceSystemInterface(_SlowAdapter())

    text = await fetch_entity_context("e1", source_system)

    assert text.count("---") == 2
    assert "SESSIONS\nsessions-for-e1" in text
    assert "PROFILE\nprofile-for-e1" in text
    assert "GOALS\ngoals-for-e1" in text


async def test_fetch_entity_context_one_failure_does_not_break_others() -> None:
    source_system = SourceSystemInterface(_FlakyAdapter())

    text = await fetch_entity_context("e1", source_system)

    assert "sessions-for-e1" in text
    assert "goals-for-e1" in text
    assert "error fetching profile" in text.lower()
    assert "upstream profile lookup failed" in text


async def test_fetch_entity_context_is_faster_than_sequential() -> None:
    source_system = SourceSystemInterface(_SlowAdapter())

    start = time.perf_counter()
    await fetch_entity_context("e1", source_system)
    parallel_duration = time.perf_counter() - start

    start = time.perf_counter()
    source_system.get_sessions("e1")
    source_system.get_profile("e1")
    source_system.get_goals("e1")
    sequential_duration = time.perf_counter() - start

    assert parallel_duration < sequential_duration / 2, (
        f"expected parallel fetch to be markedly faster: "
        f"parallel={parallel_duration:.3f}s sequential={sequential_duration:.3f}s"
    )


async def test_fetch_entity_context_uses_cache_on_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.tools.cache import HybridCache

    calls: list[str] = []

    class _CountingAdapter(_SlowAdapter):
        def fetch_sessions(self, entity_id, **kwargs):
            calls.append("sessions")
            return super().fetch_sessions(entity_id, **kwargs)

    class _StubCache:
        def get_cached(self, entity_id, module):
            if module == "sessions":
                return "cached-sessions"
            return None

        def store_cache(self, entity_id, module, text):
            pass

    source_system = SourceSystemInterface(_CountingAdapter())
    text = await fetch_entity_context("e1", source_system, cache=_StubCache())

    assert "SESSIONS\ncached-sessions" in text
    assert calls == []  # cache hit skipped the external fetch
