"""Use-case summary, health, and metrics endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

from backend.schemas import HealthResponse, MetricsResponse, UseCaseSummary

router = APIRouter(tags=["platform"])


@router.get("/use-case", response_model=UseCaseSummary)
def use_case_summary(request: Request) -> UseCaseSummary:
    """Return a domain-agnostic summary of the active config profile."""
    state = request.app.state
    config = state.config
    return UseCaseSummary(
        name=config.name,
        display_name=config.display_name,
        description=config.description,
        entity_name=config.entities.name,
        collection=config.data.collection,
        retrieval_mode=config.retrieval.mode,
        chunk_count=getattr(state, "chunk_count", 0),
        output_schema=config.output_schema.model_name or "",
        output_fields=list(config.output_schema.fields),
        graph_enabled=config.graph_schema is not None,
        sample_queries=list(config.evaluation.sample_queries[:5]),
    )


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Liveness check — only reachable when lifespan initialisation succeeded."""
    state = request.app.state
    config = state.config
    return HealthResponse(
        status="ok",
        profile=config.name,
        display_name=config.display_name,
        chunk_count=getattr(state, "chunk_count", 0),
    )


@router.get("/metrics", response_model=MetricsResponse)
def metrics(request: Request) -> MetricsResponse:
    """Return RAGAS target scores from config (plus any cached evaluation results)."""
    state = request.app.state
    config = state.config
    return MetricsResponse(
        profile=config.name,
        ragas_target_scores=dict(config.evaluation.ragas_target_scores),
        ragas_scores=getattr(state, "ragas_scores", None),
        chunk_count=getattr(state, "chunk_count", 0),
    )
