"""Tests for config-driven entity extraction chain."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from core.config import load_config
from core.graph import (
    build_extraction_chain,
    build_extraction_schema,
    empty_extraction_payload,
    extract_entities,
)
from core.llm import build_llm

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")
_has_groq_key = bool(os.environ.get("GROQ_API_KEY"))

SYNTHETIC_SESSION_NOTE = """
SOAP Note — Session 3

Subjective:
Client reports persistent worry about work performance and difficulty sleeping.
Denies suicidal or homicidal ideation.

Objective:
Affect anxious but cooperative. Engaged throughout session.

Assessment:
Generalized Anxiety Disorder. Client responded positively to psychoeducation.

Plan:
Used breathing reframe and cognitive restructuring (thought record).
Continued sertraline 50mg. Assigned daily thought log for homework.
"""


@pytest.fixture
def mental_health_config():
    return load_config("configs/mental_health.yaml")


def test_build_extraction_schema_matches_graph_nodes(mental_health_config) -> None:
    model = build_extraction_schema(mental_health_config)
    assert set(model.model_fields) == set(empty_extraction_payload(mental_health_config))


def test_extract_entities_returns_schema_keys(mental_health_config) -> None:
    expected_keys = set(empty_extraction_payload(mental_health_config))
    fake_payload = {
        "diagnoses": ["Generalized Anxiety Disorder"],
        "interventions": ["breathing reframe", "thought record", "cognitive restructuring"],
        "symptoms": ["worry", "difficulty sleeping"],
        "medications": ["sertraline 50mg"],
        "homework": ["daily thought log"],
    }
    llm = FakeListChatModel(responses=[json.dumps(fake_payload)])

    result = extract_entities(SYNTHETIC_SESSION_NOTE, mental_health_config, llm)

    assert set(result.keys()) == expected_keys
    assert result["diagnoses"] == ["Generalized Anxiety Disorder"]
    assert "thought record" in result["interventions"]
    assert result["homework"] == ["daily thought log"]


def test_build_extraction_chain_is_lcel_runnable(mental_health_config) -> None:
    llm = FakeListChatModel(
        responses=[
            json.dumps(
                {
                    "diagnoses": [],
                    "interventions": [],
                    "symptoms": [],
                    "medications": [],
                    "homework": [],
                }
            )
        ]
    )
    chain = build_extraction_chain(mental_health_config, llm)
    result = chain.invoke(SYNTHETIC_SESSION_NOTE)

    assert set(result.keys()) == set(empty_extraction_payload(mental_health_config))


@pytest.mark.skipif(not _has_groq_key, reason="GROQ_API_KEY not set in .env")
def test_extract_entities_live_groq(mental_health_config) -> None:
    llm = build_llm(mental_health_config)
    result = extract_entities(SYNTHETIC_SESSION_NOTE, mental_health_config, llm)

    expected_keys = set(empty_extraction_payload(mental_health_config))
    assert set(result.keys()) == expected_keys
    for key, values in result.items():
        assert isinstance(values, list)
        assert all(isinstance(item, str) for item in values)

    assert len(result["diagnoses"]) >= 1
    assert len(result["interventions"]) >= 1
    assert len(result["homework"]) >= 1
