"""Webhook endpoint for dual indexing session notes into Chroma and Neo4j."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from core.graph import get_graph_driver
from core.ingestion import ingest_session_note
from core.llm import build_llm

REPO_ROOT = Path(__file__).resolve().parents[2]

router = APIRouter(tags=["ingest"])


class IngestRequest(BaseModel):
    """Payload from the external session-note system."""

    entity_id: str = Field(..., min_length=1, description="Entity/patient identifier")
    session_number: int = Field(..., ge=1, description="Session ordinal for this entity")
    session_date: str = Field(..., min_length=1, description="Session date (ISO or freeform)")
    note_text: str = Field(..., min_length=1, description="Full session note text")


class IngestResponse(BaseModel):
    """Dual-index outcome returned to the webhook caller."""

    chunks_indexed: int
    entities_extracted: int
    status: str


@router.post("/ingest", response_model=IngestResponse)
def ingest_note(payload: IngestRequest, request: Request) -> IngestResponse:
    """Chunk + embed into Chroma and extract + write entities into Neo4j."""
    config = request.app.state.config
    if config.graph_schema is None:
        raise HTTPException(
            status_code=400,
            detail=f"Profile {config.name!r} has no graph_schema; cannot dual-index",
        )

    llm = build_llm(config)
    driver = get_graph_driver()
    try:
        result = ingest_session_note(
            entity_id=payload.entity_id,
            session_number=payload.session_number,
            session_date=payload.session_date,
            note_text=payload.note_text,
            config=config,
            llm=llm,
            driver=driver,
            base_dir=REPO_ROOT,
        )
    except Exception as exc:  # noqa: BLE001 — surface dual-index failures as 500
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        driver.close()

    return IngestResponse(
        chunks_indexed=result.chunks_indexed,
        entities_extracted=result.entities_extracted,
        status=result.status,
    )
