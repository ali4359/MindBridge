"""Tests for runtime config validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from core import load_config
from core.config import UseCaseConfig
from core.validation import ConfigValidationError, validate_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_validate_config_passes_for_legal_profile() -> None:
    config = load_config("configs/legal.yaml")
    validate_config(config, base_dir=REPO_ROOT)


def test_validate_config_passes_for_mental_health_profile() -> None:
    config = load_config("configs/mental_health.yaml")
    validate_config(config, base_dir=REPO_ROOT)


def test_validate_config_missing_data_source_dir(tmp_path: Path) -> None:
    sources = tmp_path / "corpus"
    sources.mkdir()
    (sources / "case_law").mkdir()

    config = UseCaseConfig.model_validate(
        {
            "name": "broken",
            "display_name": "Broken",
            "data_sources": [
                {"folder": "case_law", "doc_type": "case_law"},
                {"folder": "contracts", "doc_type": "contract"},
            ],
            "data": {"sources_dir": str(sources)},
            "output_schema": {"fields": ["issue", "rule"]},
            "safety": {"enabled": True, "flag_patterns": ["(?i)test"]},
            "retrieval": {"mode": "hybrid"},
        }
    )

    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(config, base_dir=tmp_path)

    assert "data_sources[1].folder missing" in str(exc_info.value)


def test_validate_config_output_schema_requires_two_fields(tmp_path: Path) -> None:
    sources = tmp_path / "data"
    (sources / "docs").mkdir(parents=True)

    config = UseCaseConfig.model_validate(
        {
            "name": "broken",
            "display_name": "Broken",
            "data_sources": [{"folder": "docs", "doc_type": "doc"}],
            "data": {"sources_dir": str(sources)},
            "output_schema": {"fields": ["only_one"]},
            "safety": {"enabled": True, "flag_patterns": ["(?i)test"]},
            "retrieval": {"mode": "hybrid"},
        }
    )

    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(config, base_dir=tmp_path)

    assert "at least 2 entries" in str(exc_info.value)


def test_validate_config_safety_patterns_required_when_enabled(tmp_path: Path) -> None:
    sources = tmp_path / "data"
    (sources / "docs").mkdir(parents=True)

    config = UseCaseConfig.model_validate(
        {
            "name": "broken",
            "display_name": "Broken",
            "data_sources": [{"folder": "docs", "doc_type": "doc"}],
            "data": {"sources_dir": str(sources)},
            "output_schema": {"fields": ["a", "b"]},
            "safety": {"enabled": True, "flag_patterns": []},
            "retrieval": {"mode": "hybrid"},
        }
    )

    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(config, base_dir=tmp_path)

    assert "safety.flag_patterns must be non-empty" in str(exc_info.value)


def test_fastapi_startup_fails_on_invalid_config(tmp_path: Path, monkeypatch) -> None:
    broken_yaml = tmp_path / "broken.yaml"
    broken_yaml.write_text(
        "name: broken\ndisplay_name: Broken\n"
        "data_sources:\n  - folder: missing\n    doc_type: x\n"
        "data:\n  sources_dir: data\n"
        "output_schema:\n  fields: [a]\n"
        "safety:\n  enabled: true\n  flag_patterns: []\n"
        "retrieval:\n  mode: hybrid\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("USE_CASE_CONFIG", str(broken_yaml))

    from fastapi.testclient import TestClient

    from app.main import app

    with pytest.raises(ConfigValidationError):
        with TestClient(app):
            pass
