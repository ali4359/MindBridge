"""Structured output generation from domain notes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from backend.schemas import GenerateOutputRequest
from core.chain import load_entity_context
from core.output_chain import build_output_chain, parse_output_result
from core.output_parser import build_output_schema

router = APIRouter(tags=["output"])


@router.post("/generate-output")
def generate_output(payload: GenerateOutputRequest, request: Request) -> dict:
    """Fill ``config.output_schema`` fields from notes using the active LLM."""
    state = request.app.state
    config = state.config

    try:
        schema_model = build_output_schema(config)
        entity_context = load_entity_context(
            entity_id=payload.entity_id,
            config=config,
            driver=getattr(state, "graph_driver", None),
        )
        chain = build_output_chain(config, state.llm)
        raw = chain.invoke(
            {
                "entity_id": payload.entity_id,
                "notes_text": payload.notes_text,
                "entity_context": entity_context,
            }
        )
        return parse_output_result(raw, schema_model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
