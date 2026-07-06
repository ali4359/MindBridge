"""Tests for dynamic output schema generation."""

from __future__ import annotations

from pydantic import BaseModel

from core import load_config
from core.config import UseCaseConfig
from core.output_parser import build_output_schema


def test_mental_health_builds_soap_note_model() -> None:
    config = load_config("configs/mental_health.yaml")
    model = build_output_schema(config)

    assert model.__name__ == "SOAPNote"
    assert set(model.model_fields) == {"subjective", "objective", "assessment", "plan"}

    note = model(
        subjective="Client reports low mood.",
        objective="Affect flat, speech normal rate.",
        assessment="Major depressive episode, mild.",
        plan="Continue CBT; review in one week.",
    )
    assert note.subjective.startswith("Client")
    assert isinstance(note, BaseModel)


def test_legal_builds_legal_brief_model() -> None:
    config = UseCaseConfig.model_validate(
        {
            "name": "legal",
            "display_name": "Legal Assistant",
            "output_schema": {
                "model_name": "LegalBrief",
                "fields": ["issue", "rule", "analysis", "conclusion"],
            },
        }
    )
    model = build_output_schema(config)

    assert model.__name__ == "LegalBrief"
    assert set(model.model_fields) == {"issue", "rule", "analysis", "conclusion"}

    brief = model(
        issue="Whether a non-compete clause is enforceable.",
        rule="Courts apply a reasonableness test.",
        analysis="The duration and geography appear narrow.",
        conclusion="The clause is likely enforceable.",
    )
    assert brief.issue.startswith("Whether")


def test_build_output_schema_requires_fields() -> None:
    config = UseCaseConfig.model_validate(
        {
            "name": "empty",
            "display_name": "Empty",
            "output_schema": {"fields": []},
        }
    )
    try:
        build_output_schema(config)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "fields must not be empty" in str(exc)
