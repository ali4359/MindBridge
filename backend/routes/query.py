"""RAG query endpoints — synchronous and SSE streaming."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from backend.schemas import QueryRequest, QueryResponse
from backend.services import extract_citations, resolve_retriever
from core.chain import build_chain
from core.safety import apply_safety

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, request: Request) -> QueryResponse:
    """Run RAG + safety over the active use-case profile."""
    state = request.app.state
    config = state.config

    retriever = resolve_retriever(
        request_app_state=state,
        config=config,
        metadata_filter=payload.filter,
    )
    chain = build_chain(
        config,
        retriever,
        state.llm,
        entity_id=payload.entity_id,
        driver=getattr(state, "graph_driver", None),
    )

    try:
        documents = retriever.invoke(payload.question)
        answer = chain.invoke(payload.question)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    safety = apply_safety(answer, config, llm=state.llm)
    citations = extract_citations(safety["text"] if safety["safe"] else answer, documents)
    if not safety["safe"]:
        citations = []

    return QueryResponse(
        answer=safety["text"],
        citations=citations,
        safe=bool(safety["safe"]),
    )


async def _token_event_stream(
    *,
    chain: Any,
    question: str,
) -> AsyncIterator[str]:
    """Yield Server-Sent Events for each streamed chain token."""
    try:
        async for chunk in chain.astream(question):
            if chunk is None:
                continue
            token = chunk if isinstance(chunk, str) else str(chunk)
            if not token:
                continue
            payload = json.dumps({"token": token})
            yield f"data: {payload}\n\n"
        yield "data: {\"done\": true}\n\n"
    except Exception as exc:  # noqa: BLE001
        error = json.dumps({"error": str(exc)})
        yield f"data: {error}\n\n"


@router.post("/query/stream")
async def query_stream(payload: QueryRequest, request: Request) -> StreamingResponse:
    """Stream RAG answer tokens via Server-Sent Events."""
    state = request.app.state
    config = state.config

    retriever = resolve_retriever(
        request_app_state=state,
        config=config,
        metadata_filter=payload.filter,
    )
    chain = build_chain(
        config,
        retriever,
        state.llm,
        entity_id=payload.entity_id,
        driver=getattr(state, "graph_driver", None),
    )

    return StreamingResponse(
        _token_event_stream(chain=chain, question=payload.question),
        media_type="text/event-stream",
    )
