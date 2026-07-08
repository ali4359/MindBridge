"""Webhook endpoint for dual indexing session notes into Chroma and Neo4j."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from core.graph import get_graph_driver
from core.ingestion import ingest_session_note

REPO_ROOT = Path(__file__).resolve().parents[2]

router = APIRouter(tags=["ingest"])


class IngestRequest(BaseModel):
    """Payload from the external session-note system."""

    entity_id: str = Field(..., min_length=1, description="Entity identifier")
    note_text: str = Field(..., min_length=1, description="Full session note text")
    date: Optional[str] = Field(
        default=None,
        description="Session date (ISO or freeform); alias of session_date",
    )
    session_date: Optional[str] = Field(
        default=None,
        description="Session date (preferred alias kept for existing clients)",
    )
    session_number: Optional[int] = Field(
        default=None,
        ge=1,
        description="Session ordinal; defaults to next inferred count",
    )


class IngestResponse(BaseModel):
    """Dual-index outcome returned to the webhook caller."""

    chunks_indexed: int
    entities_extracted: int
    status: str


def _resolve_session_date(payload: IngestRequest) -> str:
    value = payload.session_date or payload.date
    if not value:
        raise HTTPException(
            status_code=422,
            detail="Provide 'date' or 'session_date' in the request body",
        )
    return value


def _resolve_session_number(payload: IngestRequest, request: Request, entity_id: str) -> int:
    if payload.session_number is not None:
        return payload.session_number

    state = request.app.state
    config = state.config
    vectorstore = getattr(state, "vectorstore", None)
    if vectorstore is None:
        return 1

    from core.vectorstore import lookup_entity_documents
    from backend.services import session_count_from_documents

    docs = lookup_entity_documents(vectorstore, entity_id)
    return session_count_from_documents(docs) + 1


@router.post("/ingest", response_model=IngestResponse)
def ingest_note(payload: IngestRequest, request: Request) -> IngestResponse:
    """Chunk + embed into Chroma and extract + write entities into Neo4j."""
    config = request.app.state.config
    if config.graph_schema is None:
        raise HTTPException(
            status_code=400,
            detail=f"Profile {config.name!r} has no graph_schema; cannot dual-index",
        )

    session_date = _resolve_session_date(payload)
    session_number = _resolve_session_number(payload, request, payload.entity_id)

    llm = getattr(request.app.state, "llm", None)
    if llm is None:
        from core.llm import build_llm

        llm = build_llm(config)

    owns_driver = False
    driver = getattr(request.app.state, "graph_driver", None)
    if driver is None:
        driver = get_graph_driver()
        owns_driver = True

    try:
        result = ingest_session_note(
            entity_id=payload.entity_id,
            session_number=session_number,
            session_date=session_date,
            note_text=payload.note_text,
            config=config,
            llm=llm,
            driver=driver,
            base_dir=REPO_ROOT,
        )
    except Exception as exc:  # noqa: BLE001 — surface dual-index failures as 500
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if owns_driver:
            driver.close()

    return IngestResponse(
        chunks_indexed=result.chunks_indexed,
        entities_extracted=result.entities_extracted,
        status=result.status,
    )
