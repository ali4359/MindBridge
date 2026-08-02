"""Abstract source-system adapter contract and shared payload model.

Concrete vendor adapters live under ``adapters/<vendor>/``. Agent tools never
import those modules — they talk only to :class:`~core.tools.source_system.SourceSystemInterface`,
which receives an injected adapter at startup.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class MindBridgePayload(BaseModel):
    """Canonical document shape produced by adapter ``transform()`` for ingest."""

    model_config = ConfigDict(extra="forbid")

    entity_id: str
    entity_type: str
    session_number: int
    session_date: str
    note_text: str
    use_case: str
    signals: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None


class BaseAdapter(ABC):
    """Full fetch / compile / transform interface. Subclasses must implement all
    abstract methods; ``compile_generic`` ships a JSON→text default.
    """

    #: Session ordinals that should trigger a trajectory document after ingest.
    trajectory_trigger_sessions: frozenset[int] = frozenset()

    # --- Ingest -------------------------------------------------------------

    @abstractmethod
    def transform(self, raw_document: Any) -> MindBridgePayload:
        """Map a vendor-native document into a :class:`MindBridgePayload`."""

    def build_trajectory(self, entity_id: str, sessions: list) -> Optional[str]:
        """Optional trajectory narrative. Default: not supported."""
        return None

    # --- Fetch --------------------------------------------------------------

    @abstractmethod
    def fetch_sessions(self, entity_id: str, **kwargs: Any) -> Any:
        """Retrieve raw session records for ``entity_id``."""

    @abstractmethod
    def fetch_profile(self, entity_id: str) -> Any:
        """Retrieve the raw entity profile."""

    @abstractmethod
    def fetch_goals(self, entity_id: str) -> Any:
        """Retrieve raw goals for ``entity_id``."""

    @abstractmethod
    def fetch_module(self, entity_id: str, module: str) -> Any:
        """Retrieve an arbitrary named module for ``entity_id``."""

    # --- Compile ------------------------------------------------------------

    @abstractmethod
    def compile_sessions(self, raw: Any) -> str:
        """Shape raw session records into agent-facing text."""

    @abstractmethod
    def compile_profile(self, raw: Any) -> str:
        """Shape a raw profile into agent-facing text."""

    @abstractmethod
    def compile_goals(self, raw: Any) -> str:
        """Shape raw goals into agent-facing text."""

    def compile_generic(self, raw: Any) -> str:
        """Default: serialise arbitrary JSON-compatible data to readable text.

        Adapters only need to override when a module has a richer shape.
        """
        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw
        try:
            return json.dumps(raw, indent=2, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(raw)

    def compile_module(self, raw: Any) -> str:
        """Alias for :meth:`compile_generic` used by ``SourceSystemInterface``."""
        return self.compile_generic(raw)


class GenericAdapter(BaseAdapter):
    """No-op / passthrough adapter for profiles without an external source system."""

    def transform(self, raw_document: Any) -> MindBridgePayload:
        if isinstance(raw_document, MindBridgePayload):
            return raw_document
        if isinstance(raw_document, dict):
            note_text = raw_document.get("note_text") or self.compile_generic(raw_document)
            return MindBridgePayload(
                entity_id=str(raw_document.get("entity_id", "")),
                entity_type=str(raw_document.get("entity_type", "entity")),
                session_number=int(raw_document.get("session_number", 0)),
                session_date=str(raw_document.get("session_date", "")),
                note_text=note_text,
                use_case=str(raw_document.get("use_case", "")),
                signals=raw_document.get("signals"),
                metadata=raw_document.get("metadata"),
            )
        return MindBridgePayload(
            entity_id="",
            entity_type="entity",
            session_number=0,
            session_date="",
            note_text=self.compile_generic(raw_document),
            use_case="",
        )

    def fetch_sessions(self, entity_id: str, **kwargs: Any) -> Any:
        return []

    def fetch_profile(self, entity_id: str) -> Any:
        return {}

    def fetch_goals(self, entity_id: str) -> Any:
        return {}

    def fetch_module(self, entity_id: str, module: str) -> Any:
        return {}

    def compile_sessions(self, raw: Any) -> str:
        return self.compile_generic(raw)

    def compile_profile(self, raw: Any) -> str:
        return self.compile_generic(raw)

    def compile_goals(self, raw: Any) -> str:
        return self.compile_generic(raw)
