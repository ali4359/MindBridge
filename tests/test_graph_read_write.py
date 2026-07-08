"""Tests for Neo4j graph write/read helpers."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv

from core.config import load_config
from core.graph import (
    get_graph_driver,
    read_entity_graph,
    write_to_graph,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

_has_neo4j = bool(os.environ.get("NEO4J_URI"))


class _FakeRecord(dict):
    def __getitem__(self, key):
        return super().__getitem__(key)


class _FakeResult:
    def __init__(self, record: _FakeRecord | None) -> None:
        self._record = record

    def single(self) -> _FakeRecord | None:
        return self._record


class _FakeGraphSession:
    def __init__(self, read_responses: list[_FakeRecord]) -> None:
        self.read_responses = list(read_responses)
        self.read_index = 0
        self.write_queries: list[str] = []

    def run(self, query: str, **params):
        if "MERGE" in query:
            self.write_queries.append(query)
            return _FakeResult(None)

        record = (
            self.read_responses[self.read_index]
            if self.read_index < len(self.read_responses)
            else _FakeRecord(values=[])
        )
        self.read_index += 1
        return _FakeResult(record)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeDriver:
    def __init__(self, session: _FakeGraphSession) -> None:
        self._session = session

    def session(self) -> _FakeGraphSession:
        return self._session


@pytest.fixture
def mental_health_config():
    return load_config("configs/mental_health.yaml")


def test_write_to_graph_uses_driver_session(mental_health_config) -> None:
    fake_session = _FakeGraphSession(read_responses=[])
    driver = _FakeDriver(fake_session)
    entities = {
        "diagnoses": ["Generalized Anxiety Disorder"],
        "interventions": ["thought record"],
        "symptoms": ["worry"],
        "medications": ["sertraline 50mg"],
        "homework": ["daily thought log"],
    }

    write_to_graph(
        entities,
        entity_id="patient-1",
        session_id="session-3",
        config=mental_health_config,
        driver=driver,
    )

    queries = fake_session.write_queries
    assert any("MERGE (e:Patient" in query for query in queries)
    assert any("MERGE (s:Session" in query for query in queries)
    assert any("[:HAD_SESSION]" in query for query in queries)
    assert any("MERGE (b:Diagnosis" in query for query in queries)
    assert any("[:HAS_DIAGNOSIS]" in query for query in queries)
    assert any("[:PRESCRIBED]" in query for query in queries)


def test_read_entity_graph_returns_config_label_keys(mental_health_config) -> None:
    read_responses = [
        _FakeRecord(session_ids=["session-1", "session-3"]),
        _FakeRecord(values=["Generalized Anxiety Disorder"]),
        _FakeRecord(values=["thought record"]),
        _FakeRecord(values=["breathing reframe"]),
        _FakeRecord(values=["worry"]),
        _FakeRecord(values=["sertraline 50mg"]),
        _FakeRecord(values=["daily thought log"]),
    ]
    fake_session = _FakeGraphSession(read_responses=read_responses)
    driver = _FakeDriver(fake_session)

    graph = read_entity_graph("patient-1", mental_health_config, driver)

    assert set(graph.keys()) == {
        "Session",
        "Diagnosis",
        "Intervention",
        "Symptom",
        "Medication",
        "Homework",
    }
    assert graph["Session"] == ["session-1", "session-3"]
    assert graph["Diagnosis"] == ["Generalized Anxiety Disorder"]
    assert "thought record" in graph["Intervention"]
    assert graph["Homework"] == ["daily thought log"]


def test_read_entity_graph_legal_profile_uses_client_labels() -> None:
    config = load_config("configs/legal.yaml")
    read_responses = [
        _FakeRecord(session_ids=["case-42"]),
        _FakeRecord(values=["UCC Article 2"]),
        _FakeRecord(values=["Hadley v. Baxendale"]),
        _FakeRecord(values=["summary judgment granted"]),
    ]
    fake_session = _FakeGraphSession(read_responses=read_responses)
    driver = _FakeDriver(fake_session)

    graph = read_entity_graph("client-9", config, driver)

    assert set(graph.keys()) == {"Case", "Statute", "Precedent", "Argument", "Outcome"}
    assert graph["Case"] == ["case-42"]
    assert graph["Statute"] == ["UCC Article 2"]


def test_write_to_graph_requires_graph_schema() -> None:
    config = load_config("configs/mental_health.yaml").model_copy(update={"graph_schema": None})
    driver = MagicMock()

    with pytest.raises(ValueError, match="no graph_schema"):
        write_to_graph({}, "patient-1", "session-1", config, driver)


@pytest.mark.skipif(not _has_neo4j, reason="NEO4J_URI not configured")
def test_write_and_read_entity_graph_round_trip(mental_health_config) -> None:
    driver = get_graph_driver()
    entities = {
        "diagnoses": ["Generalized Anxiety Disorder"],
        "interventions": ["breathing reframe", "thought record"],
        "symptoms": ["worry", "insomnia"],
        "medications": ["sertraline 50mg"],
        "homework": ["daily thought log"],
    }
    entity_id = "test-patient-roundtrip"
    session_id = "test-session-roundtrip"

    try:
        write_to_graph(
            entities,
            entity_id=entity_id,
            session_id=session_id,
            config=mental_health_config,
            driver=driver,
        )
        write_to_graph(
            entities,
            entity_id=entity_id,
            session_id=session_id,
            config=mental_health_config,
            driver=driver,
        )

        graph = read_entity_graph(entity_id, mental_health_config, driver)

        assert session_id in graph["Session"]
        assert "Generalized Anxiety Disorder" in graph["Diagnosis"]
        assert "thought record" in graph["Intervention"]
        assert "daily thought log" in graph["Homework"]
    finally:
        with driver.session() as session:
            session.run(
                """
                MATCH (p:Patient {id: $entity_id})
                OPTIONAL MATCH (p)-[*]->(n)
                DETACH DELETE p, n
                """,
                entity_id=entity_id,
            )
        driver.close()
