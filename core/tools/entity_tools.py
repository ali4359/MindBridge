"""Domain-blind LangChain tools the agent calls to reach entity context and the knowledge base.

Tool names are all ``entity_*`` — never a domain-specific noun. Each tool talks
only to the injected :class:`SourceSystemInterface` / :class:`HybridCache` /
retriever, never a vendor adapter directly. The source system, cache, and
retriever are wired once at startup via :func:`init_tools`.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool

from core.tools.cache import HybridCache
from core.tools.source_system import SourceSystemInterface

_source_system: Optional[SourceSystemInterface] = None
_cache: Optional[HybridCache] = None
_retriever: Optional[BaseRetriever] = None


def init_tools(
    source_system: SourceSystemInterface,
    cache: HybridCache,
    retriever: Optional[BaseRetriever] = None,
) -> None:
    """Inject the source system, cache, and retriever the tool functions call into."""
    global _source_system, _cache, _retriever
    _source_system = source_system
    _cache = cache
    _retriever = retriever


def _get_or_fetch(module: str, entity_id: str, fetch: Callable[[], Any]) -> str:
    """Check the cache for ``module``/``entity_id`` before falling back to ``fetch``."""
    if _cache is not None:
        hit = _cache.get_cached(entity_id, module)
        if hit is not None:
            return hit

    result = str(fetch())

    if _cache is not None:
        _cache.store_cache(entity_id, module, result)

    return result


@tool
def get_entity_sessions(entity_id: str, limit: int = 5) -> str:
    """Return the most recent session records for an entity from the source system.

    Args:
        entity_id: The source-system record identifier.
        limit: Maximum number of sessions to return.
    """
    return _get_or_fetch(
        "sessions",
        entity_id,
        lambda: _source_system.get_sessions(entity_id, limit=limit),
    )


@tool
def get_entity_profile(entity_id: str) -> str:
    """Return the profile record for an entity from the source system.

    Args:
        entity_id: The source-system record identifier.
    """
    return _get_or_fetch("profile", entity_id, lambda: _source_system.get_profile(entity_id))


@tool
def get_entity_goals(entity_id: str) -> str:
    """Return the active goals recorded for an entity from the source system.

    Args:
        entity_id: The source-system record identifier.
    """
    return _get_or_fetch("goals", entity_id, lambda: _source_system.get_goals(entity_id))


@tool
def get_entity_notes(entity_id: str) -> str:
    """Return free-text notes recorded for an entity from the source system.

    Args:
        entity_id: The source-system record identifier.
    """
    return _get_or_fetch("notes", entity_id, lambda: _source_system.get_module(entity_id, "notes"))


@tool
def get_source_module(entity_id: str, module: str) -> str:
    """Return any named module of source-system data for an entity.

    Generic fallback for modules not covered by the other entity_* tools
    (for example, assessments).

    Args:
        entity_id: The source-system record identifier.
        module: Name of the source-system module to fetch.
    """
    return _get_or_fetch(module, entity_id, lambda: _source_system.get_module(entity_id, module))


@tool
def search_knowledge_base(query: str) -> str:
    """Search the indexed knowledge base for passages relevant to a query.

    Args:
        query: Free-text search query.
    """
    if _retriever is None:
        raise RuntimeError("search_knowledge_base called before init_tools() wired a retriever")

    documents = _retriever.invoke(query)
    if not documents:
        return "(no matching passages found)"

    blocks = []
    for index, doc in enumerate(documents, start=1):
        meta = doc.metadata or {}
        blocks.append(
            f"[{index}] Source: {meta.get('source', 'unknown')} | "
            f"Page: {meta.get('page_number', '?')}\n{doc.page_content.strip()}"
        )
    return "\n\n".join(blocks)


ENTITY_TOOLS = [
    get_entity_sessions,
    get_entity_profile,
    get_entity_goals,
    get_entity_notes,
    get_source_module,
    search_knowledge_base,
]
