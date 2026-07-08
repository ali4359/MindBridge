"""Config-aware FastAPI entrypoint — re-exports the platform app from ``backend``."""

from __future__ import annotations

from backend.main import app, create_app, lifespan

__all__ = ["app", "create_app", "lifespan"]
