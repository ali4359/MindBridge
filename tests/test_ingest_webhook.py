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
    ingest_all_session_notes,
    ingest_session_note,
    load_session_note_files,
    parse_session_note_number,
    strip_note_headers,
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


def test_strip_note_headers_removes_comment_lines() -> None:
    raw = "# header\n# another\n\nSOAP body here."
    assert strip_note_headers(raw) == "SOAP body here."


def test_parse_session_note_number_from_filename() -> None:
    assert parse_session_note_number(Path("synthetic_session_10_session-10.txt")) == 10
    assert parse_session_note_number(Path("synthetic_session_02_session-2.txt")) == 2


def test_load_session_note_files_from_demo_dir(mental_health_config) -> None:
    notes = load_session_note_files(mental_health_config, base_dir=REPO_ROOT)
    assert len(notes) == 20
    assert notes[0][0] == 1
    assert notes[-1][0] == 20
    assert "Subjective" in notes[9][2]  # session 10 — DBT/TIPP note


def test_ingest_all_session_notes_vector_only(mental_health_config, tmp_path) -> None:
    notes_dir = tmp_path / "data" / "session_notes"
    notes_dir.mkdir(parents=True)
    (notes_dir / "synthetic_session_01_first-session.txt").write_text(
        "# comment\n\nClient reports worry.\nPlan: breathing homework.\n",
        encoding="utf-8",
    )

    config = mental_health_config.model_copy(
        update={
            "data": mental_health_config.data.model_copy(
                update={
                    "chroma_path": "data/chromadb-test",
                    "session_notes": mental_health_config.data.session_notes.model_copy(
                        update={"output_dir": "data/session_notes"}
                    ),
                }
            ),
        }
    )

    with patch("core.vectorstore.append_documents", return_value=1) as append_mock:
        result = ingest_all_session_notes(
            config,
            entity_id="patient-demo",
            base_dir=tmp_path,
            vector_only=True,
        )

    assert result.files_processed == 1
    assert result.chunks_indexed == 1
    assert result.graph_mode == "off"
    assert result.session_ids == ["patient-demo-session-1"]
    append_mock.assert_called_once()


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
    monkeypatch.setenv("MINDBRIDGE_LIGHT_STARTUP", "1")

    fake_result = DualIndexResult(
        chunks_indexed=3,
        entities_extracted=5,
        entities={"diagnoses": ["GAD"]},
        session_id="patient-1-session-2",
        status="ok",
    )

    with (
        patch("backend.routes.ingest.get_graph_driver") as driver_factory,
        patch(
            "backend.routes.ingest.ingest_session_note",
            return_value=fake_result,
        ) as ingest_mock,
    ):
        driver = MagicMock()
        driver_factory.return_value = driver

        from backend.main import create_app

        with TestClient(create_app()) as client:
            # Inject LLM so ingest does not require GROQ in light startup.
            client.app.state.llm = MagicMock()
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
        "trajectory_indexed": 0,
    }
    ingest_mock.assert_called_once()
    driver.close.assert_called_once()


def test_post_ingest_validation_error(monkeypatch) -> None:
    monkeypatch.setenv("USE_CASE_CONFIG", str(REPO_ROOT / "configs" / "mental_health.yaml"))
    monkeypatch.setenv("MINDBRIDGE_LIGHT_STARTUP", "1")

    from backend.main import create_app

    with TestClient(create_app()) as client:
        response = client.post(
            "/ingest",
            json={
                "entity_id": "patient-1",
                "session_number": 2,
                "session_date": "2026-07-08",
            },
        )

    assert response.status_code == 422


def test_post_ingest_accepts_date_alias(monkeypatch) -> None:
    monkeypatch.setenv("USE_CASE_CONFIG", str(REPO_ROOT / "configs" / "mental_health.yaml"))
    monkeypatch.setenv("MINDBRIDGE_LIGHT_STARTUP", "1")

    fake_result = DualIndexResult(
        chunks_indexed=1,
        entities_extracted=2,
        entities={"diagnoses": ["GAD"]},
        session_id="patient-9-session-1",
        status="ok",
    )

    with (
        patch("backend.routes.ingest.get_graph_driver", return_value=MagicMock()),
        patch(
            "backend.routes.ingest.ingest_session_note",
            return_value=fake_result,
        ) as ingest_mock,
    ):
        from backend.main import create_app

        with TestClient(create_app()) as client:
            client.app.state.llm = MagicMock()
            response = client.post(
                "/ingest",
                json={
                    "entity_id": "patient-9",
                    "date": "2026-07-08",
                    "note_text": SYNTHETIC_NOTE,
                },
            )

    assert response.status_code == 200
    kwargs = ingest_mock.call_args.kwargs
    assert kwargs["session_date"] == "2026-07-08"
    assert kwargs["session_number"] == 1


def test_post_ingest_transforms_raw_vendor_session(monkeypatch) -> None:
    """Raw PatientSession JSON with metadata.source_system uses the adapter registry."""
    monkeypatch.setenv("USE_CASE_CONFIG", str(REPO_ROOT / "configs" / "mental_health.yaml"))
    monkeypatch.setenv("MINDBRIDGE_LIGHT_STARTUP", "1")

    raw_session = {
        "patientId": "pt-rauha-42",
        "sessionNumber": 2,
        "sessionDate": "2026-03-15",
        "coachNotes": {
            "sessionResponses": [
                {"questionKey": "stressRating", "response": 6},
            ],
            "notes": "Discussed CBT thought record for anxiety.",
        },
        "sessionHomeworkResponses": [
            {"questionKey": "valuesIdentification", "response": ["family", "health"]},
        ],
        "metadata": {"source_system": "rauha", "source_record_id": "abc123"},
    }

    fake_result = DualIndexResult(
        chunks_indexed=2,
        entities_extracted=4,
        entities={"diagnoses": ["anxiety"], "interventions": ["cbt"]},
        session_id="pt-rauha-42-session-2",
        status="ok",
    )

    with (
        patch("backend.routes.ingest.get_graph_driver", return_value=MagicMock()),
        patch(
            "backend.routes.ingest.ingest_session_note",
            return_value=fake_result,
        ) as ingest_mock,
        patch(
            "backend.routes.ingest._maybe_index_trajectory",
            return_value=0,
        ),
    ):
        from backend.main import create_app

        with TestClient(create_app()) as client:
            client.app.state.llm = MagicMock()
            response = client.post("/ingest", json=raw_session)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["chunks_indexed"] == 2
    assert body["entities_extracted"] == 4
    assert body["status"] == "ok"

    kwargs = ingest_mock.call_args.kwargs
    assert kwargs["entity_id"] == "pt-rauha-42"
    assert kwargs["session_number"] == 2
    assert "Stress Rating" in kwargs["note_text"]
    assert "stressRating" not in kwargs["note_text"]
    assert "Core Values" in kwargs["note_text"]


def test_post_ingest_indexes_trajectory_on_session_8(monkeypatch) -> None:
    monkeypatch.setenv("USE_CASE_CONFIG", str(REPO_ROOT / "configs" / "mental_health.yaml"))
    monkeypatch.setenv("MINDBRIDGE_LIGHT_STARTUP", "1")

    raw_session = {
        "patientId": "pt-traj",
        "sessionNumber": 8,
        "sessionDate": "2026-06-01",
        "coachNotes": {
            "sessionResponses": [{"questionKey": "stressRating", "response": 3}],
        },
        "metadata": {"source_system": "rauha"},
    }

    fake_result = DualIndexResult(
        chunks_indexed=1,
        entities_extracted=1,
        entities={},
        session_id="pt-traj-session-8",
        status="ok",
    )

    with (
        patch("backend.routes.ingest.get_graph_driver", return_value=MagicMock()),
        patch("backend.routes.ingest.ingest_session_note", return_value=fake_result),
        patch(
            "backend.routes.ingest.ingest_trajectory_document",
            return_value=3,
        ) as traj_mock,
        patch(
            "adapters.rauha.adapter.RauhaAdapter.fetch_sessions",
            return_value=[raw_session],
        ),
    ):
        from backend.main import create_app

        with TestClient(create_app()) as client:
            client.app.state.llm = MagicMock()
            response = client.post("/ingest", json=raw_session)

    assert response.status_code == 200, response.text
    assert response.json()["trajectory_indexed"] == 3
    traj_mock.assert_called_once()
    assert traj_mock.call_args.kwargs["entity_id"] == "pt-traj"
    assert "trajectory" in traj_mock.call_args.kwargs["note_text"].lower() or "Stress" in traj_mock.call_args.kwargs["note_text"]
