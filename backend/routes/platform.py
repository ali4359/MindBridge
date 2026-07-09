"""Use-case summary, health, and metrics endpoints."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Request

from backend.schemas import HealthResponse, MetricsResponse, UseCaseSummary

router = APIRouter(tags=["platform"])
REPO_ROOT = Path(__file__).resolve().parents[2]
RAGAS_RESULTS_CSV = REPO_ROOT / "ragas_results.csv"


def _parse_float_map(row: dict[str, str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for key, value in row.items():
        if key.lower() in {"timestamp", "created_at", "run_id"}:
            continue
        if value is None:
            continue
        raw = str(value).strip()
        if not raw:
            continue
        try:
            scores[key] = float(raw)
        except ValueError:
            continue
    return scores


def _read_latest_ragas_scores(path: Path) -> tuple[str, str, Optional[dict[str, float]]]:
    if not path.is_file():
        return ("missing", str(path), None)

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return ("empty", str(path), None)

    scores = _parse_float_map(rows[-1])
    if not scores:
        return ("invalid", str(path), None)
    return ("ok", str(path), scores)


@router.get("/use-case", response_model=UseCaseSummary)
def use_case_summary(request: Request) -> UseCaseSummary:
    """Return a domain-agnostic summary of the active config profile."""
    state = request.app.state
    config = state.config
    return UseCaseSummary(
        use_case_name=config.display_name,
        domain=config.name,
        langsmith_project=config.langsmith_project,
        entity_name=config.entities.name,
        output_schema_fields=list(config.output_schema.fields),
        data_sources_loaded=[
            {"folder": source.folder, "doc_type": source.doc_type}
            for source in config.data_sources
        ],
        chunk_count=getattr(state, "chunk_count", 0),
        graph_node_count=len(config.graph_schema.nodes) if config.graph_schema else 0,
    )


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Liveness check — only reachable when lifespan initialisation succeeded."""
    state = request.app.state
    config = state.config
    started_at = float(getattr(state, "started_at", time.time()))
    uptime_seconds = max(0.0, time.time() - started_at)
    neo4j_status = "connected" if getattr(state, "graph_driver", None) is not None else "disconnected"
    return HealthResponse(
        status="ok",
        model_name=config.llm.model,
        vector_store_chunk_count=getattr(state, "chunk_count", 0),
        neo4j_status=neo4j_status,
        uptime_seconds=uptime_seconds,
    )


@router.get("/metrics", response_model=MetricsResponse)
def metrics(request: Request) -> MetricsResponse:
    """Read ``ragas_results.csv`` and return the latest score row as JSON."""
    status, source, scores = _read_latest_ragas_scores(RAGAS_RESULTS_CSV)
    return MetricsResponse(
        status=status,
        source=source,
        scores=scores,
    )
