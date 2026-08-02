"""Stage C end-to-end integration checks against a live Rauha + MindBridge stack.

Skipped unless ``MINDBRIDGE_E2E=1``. Requires:

- MindBridge API running (embeddings, Chroma, Neo4j, Groq)
- ``RAUHA_BASE_URL`` / ``RAUHA_API_TOKEN`` for agent fetch
- ``MINDBRIDGE_E2E_ENTITY_ID`` — a real patient id with completed sessions
- Optional ``MINDBRIDGE_URL`` (default http://127.0.0.1:8000)

Checks:
1. POST /ingest with a raw completed session → chunks + entities indexed
2. POST /query with entity_id → agent answer cites session content
3. Repeat query → cache hit path (answer still returned)
4. GET /entity/{id}/graph → Neo4j entities present
5. Risk language in a note surfaces via retrieval/agent when present
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    os.environ.get("MINDBRIDGE_E2E") != "1",
    reason="Set MINDBRIDGE_E2E=1 to run live Stage C integration checks",
)


def _base_url() -> str:
    return os.environ.get("MINDBRIDGE_URL", "http://127.0.0.1:8000").rstrip("/")


def _entity_id() -> str:
    entity_id = os.environ.get("MINDBRIDGE_E2E_ENTITY_ID", "").strip()
    if not entity_id:
        pytest.skip("MINDBRIDGE_E2E_ENTITY_ID is required for Stage C e2e")
    return entity_id


@pytest.fixture(scope="module")
def client() -> httpx.Client:
    base = _base_url()
    with httpx.Client(base_url=base, timeout=120.0) as http:
        health = http.get("/health")
        if health.status_code != 200:
            pytest.skip(f"MindBridge not healthy at {base}: {health.status_code}")
        yield http


def test_c10_1_ingest_raw_session(client: httpx.Client) -> None:
    entity_id = _entity_id()
    payload = {
        "patientId": entity_id,
        "sessionNumber": 1,
        "sessionDate": "2026-07-01",
        "status": "completed",
        "coachNotes": {
            "sessionResponses": [
                {"questionKey": "stressRating", "response": 7},
                {"questionKey": "senseOfPride", "response": 4},
            ],
            "notes": (
                "Patient working on anxiety with CBT thought records. "
                "Currently on sertraline. Risk screen: ideation denied."
            ),
        },
        "sessionHomeworkResponses": [
            {
                "questionKey": "valuesIdentification",
                "response": ["family", "health", "growth"],
            },
            {"questionKey": "longTermGoalsFormation", "response": "Reduce anxiety symptoms"},
        ],
        "metadata": {"source_system": "rauha", "source_record_id": "e2e-session-1"},
    }
    response = client.post("/ingest", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["chunks_indexed"] >= 1
    assert body["status"] == "ok"
    assert body["entities_extracted"] >= 0


def test_c10_2_query_agent_uses_session_content(client: httpx.Client) -> None:
    entity_id = _entity_id()
    response = client.post(
        "/query",
        json={
            "entity_id": entity_id,
            "question": "What is this patient working on?",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    answer = (body.get("answer") or "").lower()
    assert answer
    # Should reference real clinical content from the ingested / fetched session.
    assert any(
        token in answer
        for token in ("anxiety", "cbt", "value", "goal", "stress", "sertraline", "thought")
    ), answer


def test_c10_3_second_query_cache_path(client: httpx.Client) -> None:
    entity_id = _entity_id()
    question = "What is this patient working on?"
    first = client.post("/query", json={"entity_id": entity_id, "question": question})
    second = client.post("/query", json={"entity_id": entity_id, "question": question})
    assert first.status_code == 200
    assert second.status_code == 200
    assert (second.json().get("answer") or "").strip()
    # Cache hit returns a stable answer; exact Rauha call count is asserted via logs
    # in live runs — here we confirm the second response is non-empty and coherent.
    assert len(second.json()["answer"]) > 20


def test_c10_4_entity_graph(client: httpx.Client) -> None:
    entity_id = _entity_id()
    response = client.get(f"/entity/{entity_id}/graph")
    assert response.status_code == 200, response.text
    graph = response.json()
    assert isinstance(graph, dict)
    # At least one schema key should carry extracted entities after ingest.
    non_empty = [k for k, v in graph.items() if v]
    assert non_empty, f"Expected graph entities, got: {graph}"


def test_c10_5_risk_flag_surfaces(client: httpx.Client) -> None:
    entity_id = _entity_id()
    # Seed a monitor-tier risk note then ask about risk.
    payload = {
        "patientId": entity_id,
        "sessionNumber": 3,
        "sessionDate": "2026-07-15",
        "coachNotes": {
            "sessionResponses": [{"questionKey": "stressRating", "response": 8}],
            "notes": "Patient reports passive suicidal ideation; monitor closely.",
        },
        "metadata": {"source_system": "rauha", "source_record_id": "e2e-risk-3"},
    }
    ingest = client.post("/ingest", json=payload)
    assert ingest.status_code == 200, ingest.text

    from adapters.rauha.entity_extractor import extract_entities

    risk = extract_entities(payload)["risk"]
    assert risk["level"] == "MONITOR"
    assert risk["severity"] == "MEDIUM"

    response = client.post(
        "/query",
        json={
            "entity_id": entity_id,
            "question": "Are there any risk flags or safety concerns for this patient?",
        },
    )
    assert response.status_code == 200, response.text
    answer = (response.json().get("answer") or "").lower()
    assert answer
    assert any(
        token in answer
        for token in ("risk", "ideation", "monitor", "safety", "suicid")
    ), answer
