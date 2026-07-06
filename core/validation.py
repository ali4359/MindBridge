"""Runtime validation for loaded use-case profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from core.config import RetrievalMode, UseCaseConfig

VALID_RETRIEVAL_MODES: frozenset[str] = frozenset(
    {"bm25", "vector", "hybrid", "multi_query", "hybrid_rerank"}
)


class ConfigValidationError(ValueError):
    """Raised when a profile fails platform startup checks."""

    def __init__(self, violations: Iterable[str]) -> None:
        self.violations = list(violations)
        message = "Config validation failed:\n" + "\n".join(
            f"  - {item}" for item in self.violations
        )
        super().__init__(message)


def validate_config(
    config: UseCaseConfig,
    *,
    base_dir: Path,
) -> None:
    """Verify a profile is runnable; raise ``ConfigValidationError`` on failure."""
    violations: list[str] = []

    sources_root = config.sources_dir(base=base_dir)
    if not sources_root.is_dir():
        violations.append(
            f"data.sources_dir does not exist on disk: {sources_root}"
        )
    else:
        for index, source in enumerate(config.data_sources):
            folder_path = sources_root / source.folder
            if not folder_path.is_dir():
                violations.append(
                    f"data_sources[{index}].folder missing on disk: {folder_path} "
                    f"(doc_type={source.doc_type!r})"
                )

    field_count = len(config.output_schema.fields)
    if field_count < 2:
        violations.append(
            f"output_schema.fields must contain at least 2 entries, got {field_count}"
        )

    if config.safety.enabled and not config.safety.flag_patterns:
        violations.append(
            "safety.flag_patterns must be non-empty when safety.enabled is true"
        )

    mode: RetrievalMode = config.retrieval.mode
    if mode not in VALID_RETRIEVAL_MODES:
        violations.append(
            f"retrieval.mode {mode!r} is invalid; "
            f"expected one of {sorted(VALID_RETRIEVAL_MODES)}"
        )

    if config.graph_schema is not None:
        entity_name = config.entities.name.strip().lower()
        graph_entity = config.graph_schema.entity_node.strip().lower()
        if entity_name != graph_entity:
            violations.append(
                "graph_schema.entity_node must align with entities.name "
                f"({config.graph_schema.entity_node!r} vs {config.entities.name!r})"
            )

    if violations:
        raise ConfigValidationError(violations)
