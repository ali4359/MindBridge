"""Entity profile and graph traversal endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from backend.schemas import EntityProfileResponse
from backend.services import profile_from_documents, session_count_from_documents
from core.graph import find_similar_entities, read_entity_graph
from core.vectorstore import lookup_entity_documents

router = APIRouter(tags=["entity"])


@router.get("/entity/{entity_id}", response_model=EntityProfileResponse)
def get_entity(entity_id: str, request: Request) -> EntityProfileResponse:
    """Look up entity profile fields and session count from the vector store."""
    state = request.app.state
    config = state.config
    vectorstore = getattr(state, "vectorstore", None)
    if vectorstore is None:
        documents: list = []
    else:
        documents = lookup_entity_documents(vectorstore, entity_id)
    if documents:
        return EntityProfileResponse(
            entity_id=entity_id,
            profile=profile_from_documents(documents, config),
            session_count=session_count_from_documents(documents),
        )

    # Fall back to Neo4j session list when notes are graph-only.
    if config.graph_schema is None or getattr(state, "graph_driver", None) is None:
        raise HTTPException(status_code=404, detail=f"Entity {entity_id!r} not found")

    try:
        graph = read_entity_graph(entity_id, config, state.graph_driver)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sessions = graph.get(config.graph_schema.session_node, [])
    if not sessions:
        raise HTTPException(status_code=404, detail=f"Entity {entity_id!r} not found")

    return EntityProfileResponse(
        entity_id=entity_id,
        profile={field: None for field in config.entities.profile_fields},
        session_count=len(sessions),
    )


@router.get("/entity/{entity_id}/graph")
def get_entity_graph(entity_id: str, request: Request) -> dict:
    """Return enriched graph traversal payload for ``entity_id``."""
    state = request.app.state
    config = state.config
    if config.graph_schema is None:
        raise HTTPException(
            status_code=400,
            detail=f"Profile {config.name!r} has no graph_schema",
        )
    driver = getattr(state, "graph_driver", None)
    if driver is None:
        raise HTTPException(status_code=503, detail="Neo4j driver is not connected")

    try:
        graph = read_entity_graph(entity_id, config, driver)
        schema = config.graph_schema
        session_key = schema.session_node

        diagnoses_key = next(
            (
                node.label
                for node in schema.nodes
                if "diagnos" in node.extraction_key.lower()
                or "issue" in node.extraction_key.lower()
            ),
            None,
        )
        interventions_key = next(
            (
                node.label
                for node in schema.nodes
                if "intervention" in node.extraction_key.lower()
                or "argument" in node.extraction_key.lower()
                or "precedent" in node.extraction_key.lower()
            ),
            None,
        )
        medications_key = next(
            (
                node.label
                for node in schema.nodes
                if "medication" in node.extraction_key.lower()
            ),
            None,
        )
        symptoms_key = next(
            (
                node.label
                for node in schema.nodes
                if "symptom" in node.extraction_key.lower()
            ),
            None,
        )
        worked_key = next(
            (
                node.label
                for node in schema.nodes
                if "homework" in node.extraction_key.lower()
                or "outcome" in node.extraction_key.lower()
                or "response" in node.extraction_key.lower()
            ),
            None,
        )

        similar_entities = find_similar_entities(entity_id, config, driver)
        sessions = [str(item) for item in graph.get(session_key, [])]
        symptoms = [str(item) for item in graph.get(symptoms_key or "", [])]

        return {
            "entity_id": entity_id,
            "diagnoses": [str(item) for item in graph.get(diagnoses_key or "", [])],
            "sessions": sessions,
            "interventions_tried": [
                str(item) for item in graph.get(interventions_key or "", [])
            ],
            "what_worked": [str(item) for item in graph.get(worked_key or "", [])],
            "medications": [str(item) for item in graph.get(medications_key or "", [])],
            "symptom_trajectory": {
                "sessions": sessions,
                "symptoms": symptoms,
            },
            "similar_entities": similar_entities,
            "subgraph": graph,
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
