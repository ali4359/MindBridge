"""RAG query endpoints — synchronous and SSE streaming.

Each request is routed (zero LLM cost) between the RAG chain and the agent by
``core.router.route_query``. The hybrid cache lets a repeat entity query skip
the external source system, and (on a full cache hit) skip the agent's tool
round trip entirely — see ``core.agent.answer_with_context``.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from backend.schemas import QueryRequest, QueryResponse
from backend.services import extract_citations, resolve_retriever
from core.agent import answer_with_context
from core.chain import build_chain
from core.config import UseCaseConfig
from core.router import route_query
from core.safety import apply_safety
from core.tools.parallel_fetch import fetch_entity_context

router = APIRouter(tags=["query"])

_ENTITY_CONTEXT_CACHE_MODULE = "profile"
_FULL_CONTEXT_MODULES = ("sessions", "profile", "goals")


def _resolve_metadata_filter(payload: QueryRequest) -> dict[str, Any] | None:
    """Translate request filters into a Chroma-compatible metadata filter."""
    if not payload.doc_type_filter:
        return None
    return {"doc_type": payload.doc_type_filter}


async def _run_rag(payload: QueryRequest, state: Any, config: UseCaseConfig) -> tuple[str, list[Any]]:
    """Run the RAG chain — one LLM call, using a cached entity context on a hit."""
    retriever = resolve_retriever(
        request_app_state=state,
        config=config,
        metadata_filter=_resolve_metadata_filter(payload),
    )

    entity_context = None
    cache = getattr(state, "cache", None)
    if payload.entity_id and cache is not None:
        entity_context = cache.get_cached(payload.entity_id, _ENTITY_CONTEXT_CACHE_MODULE)

    chain = build_chain(
        config,
        retriever,
        state.llm,
        entity_id=payload.entity_id,
        driver=getattr(state, "graph_driver", None),
        entity_context=entity_context,
    )

    documents = retriever.invoke(payload.question)
    answer = chain.invoke(payload.question)
    return answer, documents


async def _run_agent(payload: QueryRequest, state: Any, config: UseCaseConfig) -> tuple[str, list[Any]]:
    """Run the agent path: a full cache hit skips tool calls (1 LLM call); a miss uses the agent (2)."""
    cache = state.cache
    source_system = state.source_system
    entity_id = payload.entity_id

    fully_cached = all(
        cache.get_cached(entity_id, module) is not None for module in _FULL_CONTEXT_MODULES
    )
    context_text = await fetch_entity_context(entity_id, source_system, cache=cache)

    if fully_cached:
        answer = await answer_with_context(config, state.llm, payload.question, context_text)
        return answer, []

    agent_input = (
        f"[entity_id: {entity_id}]\n\n"
        f"## Pre-loaded entity context\n{context_text}\n\n"
        f"## Question\n{payload.question}"
    )
    result = await state.agent.ainvoke({"input": agent_input})
    return result["output"], []


async def _route_and_answer(payload: QueryRequest, state: Any, config: UseCaseConfig) -> tuple[str, list[Any], str]:
    """Route the query and run the matching path. Returns (answer, documents, route)."""
    route_fn = getattr(state, "router", route_query)
    route = route_fn(payload.question, payload.entity_id, config)

    if route == "agent":
        answer, documents = await _run_agent(payload, state, config)
    else:
        answer, documents = await _run_rag(payload, state, config)

    return answer, documents, route


@router.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest, request: Request) -> QueryResponse:
    """Route the query (RAG vs agent) and run safety over the answer."""
    state = request.app.state
    config = state.config
    session_id = str(uuid4())

    try:
        answer, documents, route = await _route_and_answer(payload, state, config)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    safety = apply_safety(answer, config)
    citations = extract_citations(safety["text"] if safety["safe"] else answer, documents)
    if not safety["safe"]:
        citations = []

    return QueryResponse(
        answer=safety["text"],
        citations=citations,
        safety_status="safe" if bool(safety["safe"]) else "blocked",
        session_id=session_id,
        route_used=route,
    )


async def _token_event_stream(
    *,
    state: Any,
    config: Any,
    payload: QueryRequest,
    documents: list[Any],
    session_id: str,
    route: str,
) -> AsyncIterator[str]:
    """Yield Server-Sent Events for each streamed chain token."""
    answer_chunks: list[str] = []
    start_payload = json.dumps({"event": "start", "session_id": session_id, "route_used": route})
    yield f"data: {start_payload}\n\n"
    try:
        if route == "agent":
            answer, documents = await _run_agent(payload, state, config)
            answer_chunks.append(answer)
            yield f"data: {json.dumps({'token': answer})}\n\n"
        else:
            retriever = resolve_retriever(
                request_app_state=state,
                config=config,
                metadata_filter=_resolve_metadata_filter(payload),
            )
            entity_context = None
            cache = getattr(state, "cache", None)
            if payload.entity_id and cache is not None:
                entity_context = cache.get_cached(payload.entity_id, _ENTITY_CONTEXT_CACHE_MODULE)
            chain = build_chain(
                config,
                retriever,
                state.llm,
                entity_id=payload.entity_id,
                driver=getattr(state, "graph_driver", None),
                entity_context=entity_context,
            )
            async for chunk in chain.astream(payload.question):
                if chunk is None:
                    continue
                token = chunk if isinstance(chunk, str) else str(chunk)
                if not token:
                    continue
                answer_chunks.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"

        answer = "".join(answer_chunks)
        try:
            safety = apply_safety(answer, config)
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
                "route_used": route,
            }
        )
        yield f"data: {done_payload}\n\n"
    except Exception as exc:  # noqa: BLE001
        error = json.dumps({"event": "error", "session_id": session_id, "error": str(exc)})
        yield f"data: {error}\n\n"


@router.post("/query/stream")
async def query_stream(payload: QueryRequest, request: Request) -> StreamingResponse:
    """Stream a RAG or agent answer via Server-Sent Events, using the same router logic."""
    state = request.app.state
    config = state.config
    session_id = str(uuid4())

    route_fn = getattr(state, "router", route_query)
    route = route_fn(payload.question, payload.entity_id, config)

    documents: list[Any] = []
    if route != "agent":
        retriever = resolve_retriever(
            request_app_state=state,
            config=config,
            metadata_filter=_resolve_metadata_filter(payload),
        )
        documents = retriever.invoke(payload.question)

    return StreamingResponse(
        _token_event_stream(
            state=state,
            config=config,
            payload=payload,
            documents=documents,
            session_id=session_id,
            route=route,
        ),
        media_type="text/event-stream",
    )
