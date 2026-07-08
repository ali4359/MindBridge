"""Shared request/response models for the platform API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """RAG query payload — optional entity and metadata filter."""

    question: str = Field(..., min_length=1)
    entity_id: Optional[str] = None
    doc_type_filter: Optional[str] = None


class Citation(BaseModel):
    """Source reference extracted from retrieved context."""

    source: str
    page_number: Optional[Any] = None
    doc_type: Optional[str] = None


class QueryResponse(BaseModel):
    """Grounded answer with citations and safety verdict."""

    answer: str
    citations: list[Citation]
    safety_status: str
    session_id: str


class GenerateOutputRequest(BaseModel):
    """Structured output generation from free-text notes."""

    entity_id: str = Field(..., min_length=1)
    notes_text: str = Field(..., min_length=1)


class EntityProfileResponse(BaseModel):
    """Entity summary assembled from vector-store metadata."""

    entity_id: str
    profile: dict[str, Any]
    session_count: int


class UseCaseSummary(BaseModel):
    """Active profile summary for operator tooling."""

    name: str
    display_name: str
    description: str
    entity_name: str
    collection: str
    retrieval_mode: str
    chunk_count: int
    output_schema: str
    output_fields: list[str]
    graph_enabled: bool
    sample_queries: list[str]


class HealthResponse(BaseModel):
    """Liveness probe for the loaded platform process."""

    status: str
    profile: str
    display_name: str
    chunk_count: int


class MetricsResponse(BaseModel):
    """Evaluation targets (and optional cached RAGAS scores)."""

    profile: str
    ragas_target_scores: dict[str, float]
    ragas_scores: Optional[dict[str, float]] = None
    chunk_count: int
