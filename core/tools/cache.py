"""Hybrid cache for agent tool output, backed by the existing Chroma vector store.

Reuses the persisted Chroma collection as a cache store instead of standing up
separate infrastructure: cache entries are ordinary documents tagged with
``is_cache=True`` metadata, keyed by entity + module, and expire by TTL.
"""

from __future__ import annotations

import time
from typing import Optional

CACHE_TTL: dict[str, int] = {
    "sessions": 60 * 60,
    "profile": 24 * 60 * 60,
    "goals": 2 * 60 * 60,
    "notes": 60 * 60,
    "assessments": 24 * 60 * 60,
}


class HybridCache:
    """Read/write TTL-bound cache entries in an injected Chroma vectorstore."""

    def __init__(self, vectorstore: object) -> None:
        self.vectorstore = vectorstore

    def _cache_id(self, entity_id: str, module: str) -> str:
        return f"cache::{entity_id}::{module}"

    def get_cached(self, entity_id: str, module: str) -> Optional[str]:
        result = self.vectorstore._collection.get(
            where={
                "$and": [
                    {"is_cache": True},
                    {"entity_id": entity_id},
                    {"cache_module": module},
                ]
            },
            include=["documents", "metadatas"],
        )
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        if not documents:
            return None

        cached_at = (metadatas[0] or {}).get("cached_at")
        ttl_seconds = CACHE_TTL.get(module, 0)
        if cached_at is None or time.time() - float(cached_at) > ttl_seconds:
            return None

        return documents[0]

    def store_cache(self, entity_id: str, module: str, text: str) -> None:
        cache_id = self._cache_id(entity_id, module)
        self.vectorstore._collection.delete(ids=[cache_id])
        self.vectorstore.add_texts(
            texts=[text],
            metadatas=[
                {
                    "is_cache": True,
                    "entity_id": entity_id,
                    "cache_module": module,
                    "cached_at": time.time(),
                }
            ],
            ids=[cache_id],
        )
