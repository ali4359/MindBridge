"""Config profile validation — catch broken YAML before runtime."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from core import load_config
from core.config import UseCaseConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
MENTAL_HEALTH_CONFIG = REPO_ROOT / "configs" / "mental_health.yaml"

ALLOWED_RETRIEVAL_MODES = frozenset({"hybrid", "vector", "bm25"})


@pytest.fixture(scope="module")
def mental_health_config() -> UseCaseConfig:
    return load_config(str(MENTAL_HEALTH_CONFIG))


def test_mental_health_yaml_exists() -> None:
    assert MENTAL_HEALTH_CONFIG.is_file(), f"Missing config: {MENTAL_HEALTH_CONFIG}"


def test_mental_health_loads_via_pydantic(mental_health_config: UseCaseConfig) -> None:
    """Pydantic validation succeeds for the reference profile."""
    assert isinstance(mental_health_config, UseCaseConfig)


def test_mental_health_required_fields_present(mental_health_config: UseCaseConfig) -> None:
    cfg = mental_health_config
    assert cfg.name
    assert cfg.display_name
    assert cfg.data_sources
    assert cfg.data.sources_dir
    assert cfg.data.collection
    assert cfg.chunking.chunk_size
    assert cfg.retrieval.mode
    assert cfg.entities.name
    assert cfg.prompts.system
    assert cfg.prompts.knowledge_header
    assert cfg.output_schema.fields
    assert cfg.safety.disclaimer
    assert cfg.evaluation.sample_queries
    assert cfg.llm.model


def test_mental_health_chunk_size_positive_int(mental_health_config: UseCaseConfig) -> None:
    chunk_size = mental_health_config.chunking.chunk_size
    assert isinstance(chunk_size, int)
    assert chunk_size > 0


def test_mental_health_retrieval_mode_allowed(mental_health_config: UseCaseConfig) -> None:
    assert mental_health_config.retrieval.mode in ALLOWED_RETRIEVAL_MODES


def test_mental_health_output_schema_has_fields(mental_health_config: UseCaseConfig) -> None:
    assert len(mental_health_config.output_schema.fields) >= 1


def test_mental_health_safety_enabled_is_bool(mental_health_config: UseCaseConfig) -> None:
    assert isinstance(mental_health_config.safety.enabled, bool)


def test_invalid_config_rejected_by_pydantic(tmp_path: Path) -> None:
    broken = tmp_path / "broken.yaml"
    broken.write_text("name: test\nchunking:\n  chunk_size: not-an-int\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(str(broken))
