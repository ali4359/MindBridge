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
    route_used: str


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

    use_case_name: str
    domain: str
    langsmith_project: str
    entity_name: str
    output_schema_fields: list[str]
    data_sources_loaded: list[dict[str, str]]
    chunk_count: int
    graph_node_count: int


class HealthResponse(BaseModel):
    """Liveness probe for the loaded platform process."""

    status: str
    model_name: str
    vector_store_chunk_count: int
    neo4j_status: str
    uptime_seconds: float


class MetricsResponse(BaseModel):
    """RAGAS metrics read from CSV when available."""

    status: str
    source: str
    scores: Optional[dict[str, float]] = None
