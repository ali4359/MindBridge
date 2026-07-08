"""RAG query endpoints — synchronous and SSE streaming."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from backend.schemas import QueryRequest, QueryResponse
from backend.services import extract_citations, resolve_retriever
from core.chain import build_chain
from core.safety import apply_safety

router = APIRouter(tags=["query"])


def _resolve_metadata_filter(payload: QueryRequest) -> dict[str, Any] | None:
    """Translate request filters into a Chroma-compatible metadata filter."""
    if not payload.doc_type_filter:
        return None
    return {"doc_type": payload.doc_type_filter}


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, request: Request) -> QueryResponse:
    """Run RAG + safety over the active use-case profile."""
    state = request.app.state
    config = state.config
    session_id = str(uuid4())

    retriever = resolve_retriever(
        request_app_state=state,
        config=config,
        metadata_filter=_resolve_metadata_filter(payload),
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
        safety_status="safe" if bool(safety["safe"]) else "blocked",
        session_id=session_id,
    )


async def _token_event_stream(
    *,
    chain: Any,
    config: Any,
    llm: Any,
    documents: list[Any],
    session_id: str,
    question: str,
) -> AsyncIterator[str]:
    """Yield Server-Sent Events for each streamed chain token."""
    answer_chunks: list[str] = []
    start_payload = json.dumps({"event": "start", "session_id": session_id})
    yield f"data: {start_payload}\n\n"
    try:
        async for chunk in chain.astream(question):
            if chunk is None:
                continue
            token = chunk if isinstance(chunk, str) else str(chunk)
            if not token:
                continue
            answer_chunks.append(token)
            payload = json.dumps({"token": token})
            yield f"data: {payload}\n\n"
        answer = "".join(answer_chunks)
        try:
            safety = apply_safety(answer, config, llm=llm)
        except Exception:  # noqa: BLE001
            safety = {"safe": True, "text": answer}
        citations = extract_citations(
            safety["text"] if bool(safety["safe"]) else answer,
            documents,
        )
        if not bool(safety["safe"]):
            citations = []
        done_payload = json.dumps(
            {
                "event": "done",
                "session_id": session_id,
                "answer": safety["text"],
                "citations": [item.model_dump() for item in citations],
                "safety_status": "safe" if bool(safety["safe"]) else "blocked",
            }
        )
        yield f"data: {done_payload}\n\n"
    except Exception as exc:  # noqa: BLE001
        error = json.dumps({"event": "error", "session_id": session_id, "error": str(exc)})
        yield f"data: {error}\n\n"


@router.post("/query/stream")
async def query_stream(payload: QueryRequest, request: Request) -> StreamingResponse:
    """Stream RAG answer tokens via Server-Sent Events."""
    state = request.app.state
    config = state.config
    session_id = str(uuid4())
    metadata_filter = _resolve_metadata_filter(payload)

    retriever = resolve_retriever(
        request_app_state=state,
        config=config,
        metadata_filter=metadata_filter,
    )
    chain = build_chain(
        config,
        retriever,
        state.llm,
        entity_id=payload.entity_id,
        driver=getattr(state, "graph_driver", None),
    )

    return StreamingResponse(
        _token_event_stream(
            chain=chain,
            config=config,
            llm=state.llm,
            documents=retriever.invoke(payload.question),
            session_id=session_id,
            question=payload.question,
        ),
        media_type="text/event-stream",
    )
