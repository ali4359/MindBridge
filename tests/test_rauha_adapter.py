"""Unit tests for the Rauha adapter stack (C1–C5)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from adapters import ADAPTER_REGISTRY, create_adapter
from adapters.base_adapter import BaseAdapter, GenericAdapter, MindBridgePayload
from adapters.rauha.adapter import RauhaAdapter
from adapters.rauha.entity_extractor import extract_entities
from adapters.rauha.note_compiler import (
    QUESTION_LABELS,
    SESSION_CONTEXT,
    compile_session_note,
)
from adapters.rauha.trajectory import build_trajectory_note


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_SESSION = {
    "patientId": "pt-001",
    "sessionNumber": 2,
    "sessionDate": "2026-03-15",
    "coachNotes": {
        "sessionResponses": [
            {"questionKey": "stressRating", "response": 7},
            {"questionKey": "senseOfPride", "response": 4},
        ],
        "notes": "Patient engaged well. Discussed CBT thought record.",
    },
    "sessionHomeworkResponses": [
        {
            "questionKey": "valuesIdentification",
            "response": ["family", "health", "growth"],
        },
        {"questionKey": "confidenceRating", "response": 6},
    ],
    "midWeekResponses": [
        {"questionKey": "confidenceRating", "response": 5},
        {"questionKey": "moodRating", "response": 6},
    ],
}


# ---------------------------------------------------------------------------
# C1 — BaseAdapter / MindBridgePayload / registry
# ---------------------------------------------------------------------------


def test_mindbridge_payload_fields() -> None:
    payload = MindBridgePayload(
        entity_id="e1",
        entity_type="patient",
        session_number=1,
        session_date="2026-01-01",
        note_text="hello",
        use_case="mental_health",
        signals={"risk": {"level": "DENIED"}},
        metadata={"source": "rauha"},
    )
    assert payload.entity_id == "e1"
    assert payload.signals["risk"]["level"] == "DENIED"


def test_base_adapter_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseAdapter()  # type: ignore[abstract]


def test_compile_generic_default_json() -> None:
    adapter = GenericAdapter()
    text = adapter.compile_generic({"a": 1, "b": ["x"]})
    assert '"a": 1' in text
    assert "x" in text


def test_adapter_registry_contains_rauha_and_generic() -> None:
    assert ADAPTER_REGISTRY["rauha"] is RauhaAdapter
    assert ADAPTER_REGISTRY["generic"] is GenericAdapter
    assert isinstance(create_adapter("generic"), GenericAdapter)
    assert isinstance(create_adapter("rauha"), RauhaAdapter)
    assert create_adapter("unknown") is None


# ---------------------------------------------------------------------------
# C2 — note_compiler
# ---------------------------------------------------------------------------


def test_question_labels_key_mappings() -> None:
    assert QUESTION_LABELS["stressRating"] == "Stress Rating"
    assert QUESTION_LABELS["valuesIdentification"] == "Core Values"
    assert QUESTION_LABELS["confidenceRating"] == "Homework Confidence"
    assert QUESTION_LABELS["senseOfPride"] == "Pride in Goals"


def test_session_context_covers_0_to_10() -> None:
    assert set(SESSION_CONTEXT) == set(range(11))
    assert "intake" in SESSION_CONTEXT[0].lower() or "goal" in SESSION_CONTEXT[0].lower()
    assert "completion" in SESSION_CONTEXT[8].lower()
    assert "follow-up" in SESSION_CONTEXT[10].lower() or "follow up" in SESSION_CONTEXT[10].lower()


def test_compile_session_note_prose_no_raw_keys() -> None:
    text = compile_session_note(SAMPLE_SESSION)
    assert "Session 2" in text
    assert "Stress Rating: 7" in text
    assert "Core Values: family, health, growth" in text
    assert "Homework Confidence" in text or "confidence" in text.lower()
    assert "stressRating" not in text
    assert "questionKey" not in text
    assert "{" not in text


# ---------------------------------------------------------------------------
# C3 — entity_extractor (rule-based risk)
# ---------------------------------------------------------------------------


def test_extract_entities_structured_fields() -> None:
    result = extract_entities(SAMPLE_SESSION)
    assert result["stress_rating"] == 7
    assert result["values"] == ["family", "health", "growth"]
    assert result["pride_rating"] == 4
    assert "cbt" in result["interventions"]


def test_risk_passive_ideation_is_monitor() -> None:
    doc = {
        "coachNotes": {
            "sessionResponses": [],
            "notes": "Patient reports passive suicidal ideation this week.",
        }
    }
    risk = extract_entities(doc)["risk"]
    assert risk["level"] == "MONITOR"
    assert risk["severity"] == "MEDIUM"


def test_risk_ideation_denied_is_none() -> None:
    doc = {
        "coachNotes": {
            "notes": "Risk screen: ideation denied. No self-harm.",
        }
    }
    risk = extract_entities(doc)["risk"]
    assert risk["level"] == "DENIED"
    assert risk["severity"] == "NONE"


def test_risk_active_is_high() -> None:
    doc = {"coachNotes": {"notes": "Suicidal ideation present with plan to harm."}}
    risk = extract_entities(doc)["risk"]
    assert risk["level"] == "ACTIVE"
    assert risk["severity"] == "HIGH"


def test_extract_medications_and_diagnoses() -> None:
    doc = {
        "coachNotes": {
            "notes": "History of depression and GAD. Currently on sertraline and quetiapine."
        }
    }
    result = extract_entities(doc)
    assert "sertraline" in result["medications"]
    assert "quetiapine" in result["medications"]
    assert "depression" in result["diagnoses"]
    assert "gad" in result["diagnoses"]


# ---------------------------------------------------------------------------
# C4 — trajectory
# ---------------------------------------------------------------------------


def test_build_trajectory_note() -> None:
    sessions = [
        {
            "sessionNumber": 0,
            "coachNotes": {"sessionResponses": [{"questionKey": "stressRating", "response": 8}]},
            "sessionHomeworkResponses": [
                {"questionKey": "valuesIdentification", "response": ["family"]},
                {"questionKey": "longTermGoalsFormation", "response": "Reduce anxiety"},
            ],
            "midWeekResponses": [{"questionKey": "confidenceRating", "response": 3}],
        },
        {
            "sessionNumber": 4,
            "coachNotes": {"sessionResponses": [{"questionKey": "stressRating", "response": 5}]},
            "midWeekResponses": [{"questionKey": "confidenceRating", "response": 6}],
        },
        {
            "sessionNumber": 8,
            "coachNotes": {"sessionResponses": [{"questionKey": "stressRating", "response": 3}]},
            "midWeekResponses": [{"questionKey": "confidenceRating", "response": 8}],
        },
    ]
    text = build_trajectory_note("pt-001", sessions)
    assert "pt-001" in text
    assert "Baseline" in text
    assert "8" in text and "3" in text
    assert "Percentage improvement" in text
    assert "family" in text
    assert "Reduce anxiety" in text
    assert "confidence" in text.lower()


# ---------------------------------------------------------------------------
# C5 — RauhaAdapter fetch / compile / transform
# ---------------------------------------------------------------------------


def test_transform_returns_payload() -> None:
    adapter = RauhaAdapter(base_url="https://example.test", token="t")
    payload = adapter.transform(SAMPLE_SESSION)
    assert isinstance(payload, MindBridgePayload)
    assert payload.entity_id == "pt-001"
    assert payload.session_number == 2
    assert "Stress Rating" in payload.note_text
    assert payload.signals is not None
    assert payload.signals["stress_rating"] == 7


def test_compile_sessions_joins_with_separator() -> None:
    adapter = RauhaAdapter(base_url="https://example.test", token="t")
    s0 = {**SAMPLE_SESSION, "sessionNumber": 0}
    s1 = {**SAMPLE_SESSION, "sessionNumber": 1}
    text = adapter.compile_sessions([s0, s1])
    assert "---" in text
    assert "Session 0" in text
    assert "Session 1" in text


def test_compile_goals_formats_status() -> None:
    adapter = RauhaAdapter(base_url="https://example.test", token="t")
    text = adapter.compile_goals(
        {
            "shortTerm": [{"text": "Walk daily", "status": "active"}],
            "longTerm": [{"goal": "Return to work", "status": "in_progress"}],
        }
    )
    assert "Short-term" in text
    assert "Walk daily" in text
    assert "active" in text
    assert "Return to work" in text


def test_fetch_sessions_calls_rauha_api() -> None:
    adapter = RauhaAdapter(base_url="https://rauha.test", token="secret")
    fake_response = MagicMock()
    fake_response.json.return_value = [SAMPLE_SESSION]
    fake_response.raise_for_status = MagicMock()

    with patch("adapters.rauha.adapter.httpx.Client") as client_cls:
        client = MagicMock()
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        client.get.return_value = fake_response
        client_cls.return_value = client

        raw = adapter.fetch_sessions("pt-001", limit=5)

    assert raw == [SAMPLE_SESSION]
    client.get.assert_called_once()
    args, kwargs = client.get.call_args
    assert args[0] == "https://rauha.test/api/patients/pt-001/sessions"
    assert kwargs["headers"]["Authorization"] == "Bearer secret"
    assert kwargs["params"] == {"limit": 5}


def test_fetch_profile_goals_module_paths() -> None:
    adapter = RauhaAdapter(base_url="https://rauha.test", token="t")
    fake_response = MagicMock()
    fake_response.json.return_value = {"ok": True}
    fake_response.raise_for_status = MagicMock()

    with patch("adapters.rauha.adapter.httpx.Client") as client_cls:
        client = MagicMock()
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        client.get.return_value = fake_response
        client_cls.return_value = client

        adapter.fetch_profile("pt-001")
        adapter.fetch_goals("pt-001")
        adapter.fetch_module("pt-001", "notes")

    paths = [c.args[0] for c in client.get.call_args_list]
    assert paths == [
        "https://rauha.test/api/patients/pt-001/profile",
        "https://rauha.test/api/patients/pt-001/goals",
        "https://rauha.test/api/patients/pt-001/notes",
    ]
