"""Webhook endpoint for dual indexing session notes into Chroma and Neo4j.

Accepts either a pre-compiled payload (``note_text`` + ``entity_id``) or a raw
vendor document that is transformed via the adapter registry.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from adapters import ADAPTER_REGISTRY, create_adapter, is_raw_vendor_document
from adapters.base_adapter import BaseAdapter, MindBridgePayload
from core.graph import get_graph_driver
from core.ingestion import DualIndexResult, ingest_session_note, ingest_trajectory_document

REPO_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)

router = APIRouter(tags=["ingest"])

DEFAULT_SOURCE_SYSTEM = "generic"


class IngestRequest(BaseModel):
    """Payload from an external session-note system or a pre-compiled note."""

    model_config = ConfigDict(extra="allow")

    entity_id: Optional[str] = Field(default=None, description="Entity identifier")
    note_text: Optional[str] = Field(default=None, description="Full session note text")
    date: Optional[str] = Field(
        default=None,
        description="Session date (ISO or freeform); alias of session_date",
    )
    session_date: Optional[str] = Field(
        default=None,
        description="Session date (preferred alias kept for existing callers)",
    )
    session_number: Optional[int] = Field(
        default=None,
        ge=0,
        description="Session ordinal; defaults to next inferred count",
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Caller metadata; ``source_system`` selects the adapter",
    )


class IngestResponse(BaseModel):
    """Dual-index outcome returned to the webhook caller."""

    chunks_indexed: int
    entities_extracted: int
    status: str
    trajectory_indexed: int = 0


def _source_system_name(body: dict[str, Any]) -> str:
    metadata = body.get("metadata")
    if isinstance(metadata, dict):
        name = metadata.get("source_system")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return DEFAULT_SOURCE_SYSTEM


def _has_direct_fields(body: dict[str, Any]) -> bool:
    note_text = body.get("note_text")
    entity_id = body.get("entity_id")
    return bool(
        isinstance(note_text, str)
        and note_text.strip()
        and isinstance(entity_id, str)
        and entity_id.strip()
    )


def _resolve_via_adapter(body: dict[str, Any], source_system: str) -> MindBridgePayload:
    if source_system not in ADAPTER_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown source_system adapter {source_system!r}",
        )
    adapter = create_adapter(source_system)
    if adapter is None:
        raise HTTPException(
            status_code=400,
            detail=f"Could not instantiate adapter {source_system!r}",
        )
    try:
        return adapter.transform(body)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=422,
            detail=f"Adapter transform failed: {exc}",
        ) from exc


def _resolve_session_date(
    *,
    session_date: Optional[str],
    date: Optional[str],
    required: bool = True,
) -> str:
    value = session_date or date or ""
    if not value and required:
        raise HTTPException(
            status_code=422,
            detail="Provide 'date' or 'session_date' in the request body",
        )
    return str(value)


def _resolve_session_number(
    explicit: Optional[int],
    request: Request,
    entity_id: str,
) -> int:
    if explicit is not None:
        return explicit

    state = request.app.state
    vectorstore = getattr(state, "vectorstore", None)
    if vectorstore is None:
        return 1

    from backend.services import session_count_from_documents
    from core.vectorstore import lookup_entity_documents

    docs = lookup_entity_documents(vectorstore, entity_id)
    return session_count_from_documents(docs) + 1


def _normalize_session_list(raw: Any) -> list[dict]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [s for s in raw if isinstance(s, dict)]
    if isinstance(raw, dict):
        for key in ("sessions", "data", "items", "results"):
            nested = raw.get(key)
            if isinstance(nested, list):
                return [s for s in nested if isinstance(s, dict)]
        return [raw]
    return []


def _maybe_index_trajectory(
    *,
    adapter: Optional[BaseAdapter],
    entity_id: str,
    session_number: int,
    current_raw: Optional[dict[str, Any]],
    config: Any,
) -> int:
    """Index a trajectory document when the adapter marks this session as a trigger."""
    if adapter is None:
        return 0
    triggers = getattr(adapter, "trajectory_trigger_sessions", frozenset()) or frozenset()
    if session_number not in triggers:
        return 0

    sessions: list[dict] = []
    try:
        raw = adapter.fetch_sessions(entity_id)
        sessions = _normalize_session_list(raw)
    except Exception as exc:  # noqa: BLE001 — trajectory is best-effort
        logger.warning("Trajectory session fetch failed for %s: %s", entity_id, exc)

    if not sessions and current_raw:
        sessions = [current_raw]

    try:
        text = adapter.build_trajectory(entity_id, sessions)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Trajectory build failed for %s: %s", entity_id, exc)
        return 0

    if not text:
        return 0

    try:
        return ingest_trajectory_document(
            entity_id=entity_id,
            note_text=text,
            config=config,
            base_dir=REPO_ROOT,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Trajectory index failed for %s: %s", entity_id, exc)
        return 0


@router.post("/ingest", response_model=IngestResponse)
async def ingest_note(request: Request) -> IngestResponse:
    """Chunk + embed into Chroma and extract + write entities into Neo4j.

    Resolution order:
    1. If ``note_text`` + ``entity_id`` are present → process directly.
    2. Else look up ``metadata.source_system`` in the adapter registry, call
       ``transform()``, then process the resulting :class:`MindBridgePayload`.
    """
    try:
        body = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail="Request body must be JSON") from exc

    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="Request body must be a JSON object")

    # Validate known fields while preserving extras for adapter.transform().
    try:
        parsed = IngestRequest.model_validate(body)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    source_system = _source_system_name(body)
    adapter: Optional[BaseAdapter] = None
    raw_for_trajectory: Optional[dict[str, Any]] = None

    if _has_direct_fields(body):
        entity_id = str(parsed.entity_id).strip()
        note_text = str(parsed.note_text).strip()
        session_date = _resolve_session_date(
            session_date=parsed.session_date,
            date=parsed.date,
            required=True,
        )
        session_number = _resolve_session_number(parsed.session_number, request, entity_id)
        if source_system in ADAPTER_REGISTRY:
            adapter = create_adapter(source_system)
    elif is_raw_vendor_document(body) or source_system != DEFAULT_SOURCE_SYSTEM:
        mb_payload = _resolve_via_adapter(body, source_system)
        adapter = create_adapter(source_system)
        raw_for_trajectory = body
        entity_id = mb_payload.entity_id
        note_text = mb_payload.note_text
        session_number = mb_payload.session_number
        session_date = mb_payload.session_date or _resolve_session_date(
            session_date=parsed.session_date,
            date=parsed.date,
            required=False,
        )
        if not entity_id or not note_text:
            raise HTTPException(
                status_code=422,
                detail="Adapter transform produced empty entity_id or note_text",
            )
    else:
        raise HTTPException(
            status_code=422,
            detail=(
                "Provide note_text + entity_id, or a raw vendor document with "
                "metadata.source_system set to a registered adapter"
            ),
        )

    config = request.app.state.config
    if config.graph_schema is None:
        raise HTTPException(
            status_code=400,
            detail=f"Profile {config.name!r} has no graph_schema; cannot dual-index",
        )

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
        result: DualIndexResult = ingest_session_note(
            entity_id=entity_id,
            session_number=session_number,
            session_date=session_date,
            note_text=note_text,
            config=config,
            llm=llm,
            driver=driver,
            base_dir=REPO_ROOT,
        )
        trajectory_indexed = _maybe_index_trajectory(
            adapter=adapter,
            entity_id=entity_id,
            session_number=session_number,
            current_raw=raw_for_trajectory,
            config=config,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — surface dual-index failures as 500
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if owns_driver:
            driver.close()

    # Refresh in-memory vectorstore handle when the platform is fully booted.
    vectorstore = getattr(request.app.state, "vectorstore", None)
    if vectorstore is not None:
        try:
            from core.vectorstore import load_vectorstore

            request.app.state.vectorstore = load_vectorstore(config, base_dir=REPO_ROOT)
            request.app.state.chunk_count = request.app.state.vectorstore._collection.count()
        except Exception as exc:  # noqa: BLE001 — non-fatal; next restart reloads
            logger.warning("Could not reload vectorstore after ingest: %s", exc)

    return IngestResponse(
        chunks_indexed=result.chunks_indexed,
        entities_extracted=result.entities_extracted,
        status=result.status,
        trajectory_indexed=trajectory_indexed,
    )
