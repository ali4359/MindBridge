"""Generic interface for reaching an external source system.

Agent tools call only this interface, never a vendor adapter directly. The
adapter is injected at startup (see ``source_system.adapter`` in the use-case
config) and must expose a ``fetch_X``/``compile_X`` pair for each method
below: ``fetch_X`` retrieves raw records, ``compile_X`` shapes them for the
agent. This module stays domain-blind — no vocabulary from any one deployment
belongs here.
"""

from __future__ import annotations

from typing import Any


class SourceSystemInterface:
    """Domain-blind facade over an injected source-system adapter."""

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter

    def get_sessions(self, entity_id: str, **kwargs: Any) -> Any:
        raw = self.adapter.fetch_sessions(entity_id, **kwargs)
        return self.adapter.compile_sessions(raw)

    def get_profile(self, entity_id: str) -> Any:
        raw = self.adapter.fetch_profile(entity_id)
        return self.adapter.compile_profile(raw)

    def get_goals(self, entity_id: str) -> Any:
        raw = self.adapter.fetch_goals(entity_id)
        return self.adapter.compile_goals(raw)

    def get_module(self, entity_id: str, module: str) -> Any:
        raw = self.adapter.fetch_module(entity_id, module)
        return self.adapter.compile_module(raw)
