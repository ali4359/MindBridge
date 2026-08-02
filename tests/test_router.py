"""Tests for the config-driven SmartQueryRouter (core/router.py)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from core import load_config
from core.config import UseCaseConfig
from core.router import route_query

REPO_ROOT = Path(__file__).resolve().parents[1]
MENTAL_HEALTH_CONFIG = REPO_ROOT / "configs" / "mental_health.yaml"
LEGAL_CONFIG = REPO_ROOT / "configs" / "legal.yaml"


@pytest.fixture(scope="module")
def mental_health_config() -> UseCaseConfig:
    return load_config(str(MENTAL_HEALTH_CONFIG))


@pytest.fixture(scope="module")
def legal_config() -> UseCaseConfig:
    return load_config(str(LEGAL_CONFIG))


# (question, entity_id, expected_route)
MENTAL_HEALTH_CASES = [
    # rule 1: no entity_id -> always rag, even with an agent signal present
    ("Schedule a follow-up for next week", None, "rag"),
    ("What does the guideline recommend for anxiety treatment?", None, "rag"),
    # rule 2: rag signal + entity_id -> rag
    ("What does the guideline recommend for anxiety treatment?", "e1", "rag"),
    ("What evidence supports this intervention?", "e1", "rag"),
    ("Which workbook has skills for distress tolerance?", "e1", "rag"),
    # rule 2 wins over rule 3 when both signals present
    ("Log this session and tell me what does the guideline recommend", "e1", "rag"),
    # rule 3: agent signal + entity_id, no rag signal -> agent
    ("Schedule a follow-up for next week", "e1", "agent"),
    ("Remind me to update their goals", "e1", "agent"),
    ("Add a note about today's session", "e1", "agent"),
    # rule 4: entity_id present, no signal at all -> default agent
    ("Summarize recent progress", "e1", "agent"),
]

LEGAL_CASES = [
    # rule 1: no entity_id -> always rag, even with an agent signal present
    ("File a motion for extension", None, "rag"),
    ("Is there relevant case law for this claim?", None, "rag"),
    # rule 2: rag signal + entity_id -> rag
    ("Is there relevant case law for this claim?", "c1", "rag"),
    ("What are the elements of breach of contract?", "c1", "rag"),
    ("Is this clause enforceable?", "c1", "rag"),
    ("What does the statute say about liability limits?", "c1", "rag"),
    # rule 2 wins over rule 3 when both signals present
    ("Draft an email citing relevant precedent", "c1", "rag"),
    # rule 3: agent signal + entity_id, no rag signal -> agent
    ("Add this to my calendar for Friday", "c1", "agent"),
    ("Remind me to follow up next week", "c1", "agent"),
    # rule 4: entity_id present, no signal at all -> default agent
    ("Summarize the situation", "c1", "agent"),
]


@pytest.mark.parametrize(("question", "entity_id", "expected"), MENTAL_HEALTH_CASES)
def test_route_query_mental_health(
    mental_health_config: UseCaseConfig, question: str, entity_id: str | None, expected: str
) -> None:
    assert route_query(question, entity_id, mental_health_config) == expected


@pytest.mark.parametrize(("question", "entity_id", "expected"), LEGAL_CASES)
def test_route_query_legal(
    legal_config: UseCaseConfig, question: str, entity_id: str | None, expected: str
) -> None:
    assert route_query(question, entity_id, legal_config) == expected


def test_route_query_logs_every_decision(mental_health_config: UseCaseConfig, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="core.router"):
        route_query("What does the guideline recommend?", "e1", mental_health_config)

    assert any("Route: rag | Q: What does the guideline recommend?" in record.message for record in caplog.records)
