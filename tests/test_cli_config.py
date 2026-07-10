"""Tests for USE_CASE_CONFIG resolution."""

from __future__ import annotations

import pytest

from core.config import (
    load_active_config,
    resolve_config_path,
)


def test_resolve_config_path_requires_explicit_selection(monkeypatch) -> None:
    monkeypatch.delenv("USE_CASE_CONFIG", raising=False)
    with pytest.raises(RuntimeError):
        resolve_config_path()


def test_resolve_config_path_explicit() -> None:
    assert resolve_config_path("configs/legal.yaml") == "configs/legal.yaml"


def test_load_active_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("USE_CASE_CONFIG", "configs/legal.yaml")
    config = load_active_config()
    assert config.name == "legal"
    assert config.output_schema.model_name == "LegalBrief"
