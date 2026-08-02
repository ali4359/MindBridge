"""Vendor adapters for external source systems.

Domain-specific adapters live here so ``core/`` and ``backend/`` stay free of
vendor vocabulary. Look up an adapter by name via :data:`ADAPTER_REGISTRY`.
"""

from __future__ import annotations

from typing import Any, Optional, Type

from adapters.base_adapter import BaseAdapter, GenericAdapter, MindBridgePayload
from adapters.rauha import RauhaAdapter

ADAPTER_REGISTRY: dict[str, Type[BaseAdapter]] = {
    "rauha": RauhaAdapter,
    "generic": GenericAdapter,
}


def create_adapter(name: str, **kwargs: Any) -> Optional[BaseAdapter]:
    """Instantiate the adapter registered under ``name``, or ``None`` if unknown."""
    if not name:
        return None
    adapter_cls = ADAPTER_REGISTRY.get(name)
    if adapter_cls is None:
        return None
    return adapter_cls(**kwargs)


__all__ = [
    "ADAPTER_REGISTRY",
    "BaseAdapter",
    "GenericAdapter",
    "MindBridgePayload",
    "RauhaAdapter",
    "create_adapter",
]
