"""Tests for POST /ingest dual-index webhook."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from core.config import load_config
from core.ingestion import (
    DualIndexResult,
    chunk_session_note,
    count_extracted_entities,
    ingest_session_note,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

SYNTHETIC_NOTE = """
SOAP Note — Session 3

Client reports persistent worry and difficulty sleeping.
Assessment: Generalized Anxiety Disorder.
Plan: Used thought record. Continued sertraline 50mg. Assigned daily thought log.
"""


@pytest.fixture
def mental_health_config():
    return load_config("configs/mental_health.yaml")


def test_chunk_session_note_attaches_metadata(mental_health_config) -> None:
    chunks = chunk_session_note(
        SYNTHETIC_NOTE,
        mental_health_config,
        entity_id="patient-42",
        session_number=3,
        session_date="2026-07-08",
    )

    assert len(chunks) >= 1
    meta = chunks[0].metadata
    assert meta["entity_id"] == "patient-42"
    assert meta["session_number"] == 3
    assert meta["session_date"] == "2026-07-08"
    assert meta["doc_type"] == "session_note"
    assert meta["session_id"] == "patient-42-session-3"


def test_count_extracted_entities() -> None:
    assert (
        count_extracted_entities(
            {
                "diagnoses": ["GAD"],
                "interventions": ["thought record", ""],
                "symptoms": [],
                "medications": ["sertraline 50mg"],
                "homework": ["daily thought log"],
            }
        )
        == 4
    )


def test_ingest_session_note_dual_writes(mental_health_config, tmp_path) -> None:
    fake_payload = {
        "diagnoses": ["Generalized Anxiety Disorder"],
        "interventions": ["thought record"],
        "symptoms": ["worry", "difficulty sleeping"],
        "medications": ["sertraline 50mg"],
        "homework": ["daily thought log"],
    }
    llm = FakeListChatModel(responses=[json.dumps(fake_payload)])
    driver = MagicMock()

    with (
        patch(
            "core.vectorstore.append_documents",
            return_value=2,
        ) as append_mock,
        patch("core.graph.write_to_graph") as write_mock,
    ):
        result = ingest_session_note(
            entity_id="patient-42",
            session_number=3,
            session_date="2026-07-08",
            note_text=SYNTHETIC_NOTE,
            config=mental_health_config,
            llm=llm,
            driver=driver,
            base_dir=tmp_path,
        )

    assert isinstance(result, DualIndexResult)
    assert result.status == "ok"
    assert result.chunks_indexed == 2
    assert result.entities_extracted == 6
    assert result.session_id == "patient-42-session-3"
    append_mock.assert_called_once()
    write_mock.assert_called_once()
    written_entities = write_mock.call_args.args[0]
    assert written_entities["diagnoses"] == ["Generalized Anxiety Disorder"]


def test_post_ingest_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("USE_CASE_CONFIG", str(REPO_ROOT / "configs" / "mental_health.yaml"))

    fake_result = DualIndexResult(
        chunks_indexed=3,
        entities_extracted=5,
        entities={"diagnoses": ["GAD"]},
        session_id="patient-1-session-2",
        status="ok",
    )

    with (
        patch("backend.routes.ingest.build_llm", return_value=MagicMock()),
        patch("backend.routes.ingest.get_graph_driver") as driver_factory,
        patch(
            "backend.routes.ingest.ingest_session_note",
            return_value=fake_result,
        ) as ingest_mock,
    ):
        driver = MagicMock()
        driver_factory.return_value = driver

        from app.main import app

        with TestClient(app) as client:
            response = client.post(
                "/ingest",
                json={
                    "entity_id": "patient-1",
                    "session_number": 2,
                    "session_date": "2026-07-08",
                    "note_text": SYNTHETIC_NOTE,
                },
            )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "chunks_indexed": 3,
        "entities_extracted": 5,
        "status": "ok",
    }
    ingest_mock.assert_called_once()
    driver.close.assert_called_once()


def test_post_ingest_validation_error(monkeypatch) -> None:
    monkeypatch.setenv("USE_CASE_CONFIG", str(REPO_ROOT / "configs" / "mental_health.yaml"))

    from app.main import app

    with TestClient(app) as client:
        response = client.post(
            "/ingest",
            json={
                "entity_id": "patient-1",
                "session_number": 2,
                "session_date": "2026-07-08",
            },
        )

    assert response.status_code == 422
