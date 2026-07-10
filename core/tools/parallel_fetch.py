"""Parallel fetch of an entity's context across source-system modules.

Sessions, profile, and goals are independent lookups, so they are fetched
concurrently instead of sequentially. Each one checks the hybrid cache first —
only a miss reaches the source system — and a failure in one fetch is
captured inline rather than aborting the others.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

from core.tools.cache import HybridCache
from core.tools.source_system import SourceSystemInterface

_SECTIONS: tuple[tuple[str, str], ...] = (
    ("SESSIONS", "sessions"),
    ("PROFILE", "profile"),
    ("GOALS", "goals"),
)


async def _cached_or_fetch(
    cache: Optional[HybridCache],
    module: str,
    entity_id: str,
    fetch: Callable[[], Any],
) -> str:
    if cache is not None:
        hit = cache.get_cached(entity_id, module)
        if hit is not None:
            return hit

    result = str(await asyncio.to_thread(fetch))

    if cache is not None:
        cache.store_cache(entity_id, module, result)

    return result


async def fetch_entity_context(
    entity_id: str,
    source_system: SourceSystemInterface,
    *,
    cache: Optional[HybridCache] = None,
) -> str:
    """Fetch sessions, profile, and goals in parallel and combine them into one text block.

    Uses ``asyncio.gather(..., return_exceptions=True)`` so a single failing
    fetch does not prevent the other two from completing — the failure is
    rendered inline as an error note in that section instead.
    """
    fetchers = {
        "SESSIONS": lambda: source_system.get_sessions(entity_id),
        "PROFILE": lambda: source_system.get_profile(entity_id),
        "GOALS": lambda: source_system.get_goals(entity_id),
    }
    results = await asyncio.gather(
        *(
            _cached_or_fetch(cache, module, entity_id, fetchers[label])
            for label, module in _SECTIONS
        ),
        return_exceptions=True,
    )

    sections: list[str] = []
    for (label, _module), result in zip(_SECTIONS, results):
        body = f"(error fetching {label.lower()}: {result})" if isinstance(result, Exception) else result
        sections.append(f"{label}\n{body}")

    return "\n---\n".join(sections)
