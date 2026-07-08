"""Tests for config-driven graph schema loading and helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.config import GraphRelationshipConfig, GraphSchemaConfig, load_config
from core.graph import (
    build_extraction_prompt,
    empty_extraction_payload,
    write_entity_to_graph,
)


def test_mental_health_graph_schema_loads() -> None:
    config = load_config("configs/mental_health.yaml")

    assert config.graph_schema is not None
    schema = config.graph_schema
    assert schema.entity_node == "Patient"
    assert schema.session_node == "Session"
    assert schema.entity_session_relationship == "HAD_SESSION"
    assert [node.label for node in schema.nodes] == [
        "Diagnosis",
        "Intervention",
        "Symptom",
        "Medication",
        "Homework",
    ]
    assert schema.extraction_keys()["diagnoses"] == "Diagnosis"
    assert schema.extraction_keys()["homework"] == "Homework"


def test_legal_graph_schema_loads() -> None:
    config = load_config("configs/legal.yaml")

    assert config.graph_schema is not None
    schema = config.graph_schema
    assert schema.entity_node == "Client"
    assert schema.session_node == "Case"
    assert schema.entity_session_relationship == "INVOLVED_IN"
    rel_types = [rel.type for rel in schema.relationships]
    assert "CITES" in rel_types
    assert "GOVERNED_BY" in rel_types


def test_build_extraction_prompt_uses_config_keys() -> None:
    config = load_config("configs/mental_health.yaml")
    prompt = build_extraction_prompt(config)

    assert "diagnoses" in prompt
    assert "interventions" in prompt
    assert "homework" in prompt
    assert "homeworks" not in prompt
    assert "diagnosiss" not in prompt


def test_empty_extraction_payload_shape() -> None:
    config = load_config("configs/legal.yaml")
    payload = empty_extraction_payload(config)

    assert payload == {
        "statutes": [],
        "precedents": [],
        "arguments": [],
        "outcomes": [],
    }


def test_write_entity_to_graph_uses_config_labels() -> None:
    config = load_config("configs/mental_health.yaml")
    neo4j_session = MagicMock()
    extracted = {
        "diagnoses": ["Generalized Anxiety Disorder"],
        "interventions": ["thought record"],
        "symptoms": ["worry"],
        "medications": [],
        "homework": ["daily thought log"],
    }

    write_entity_to_graph(
        neo4j_session,
        entity_id="patient-1",
        session_id="session-3",
        extracted=extracted,
        config=config,
    )

    queries = [call.args[0] for call in neo4j_session.run.call_args_list]
    assert any("MERGE (e:Patient" in query for query in queries)
    assert any("MERGE (s:Session" in query for query in queries)
    assert any("[:HAD_SESSION]" in query for query in queries)
    assert any("MERGE (b:Diagnosis" in query for query in queries)
    assert any("[:HAS_DIAGNOSIS]" in query for query in queries)
    assert any("[:USED_INTERVENTION]" in query for query in queries)
    assert any("[:ASSIGNED_HOMEWORK]" in query for query in queries)
    assert not any(":Client" in query for query in queries)


def test_graph_schema_rejects_unknown_relationship_endpoint() -> None:
    base = load_config("configs/mental_health.yaml").graph_schema
    assert base is not None

    with pytest.raises(ValueError, match="unknown from label"):
        GraphSchemaConfig(
            entity_node=base.entity_node,
            session_node=base.session_node,
            nodes=base.nodes,
            relationships=base.relationships
            + [
                GraphRelationshipConfig(
                    type="INVALID",
                    **{"from": "Clinician"},
                    to="Diagnosis",
                )
            ],
        )
