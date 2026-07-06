"""Config-driven knowledge-graph helpers — no hardcoded domain node labels."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from neo4j import Driver, GraphDatabase, Session

from core.config import GraphNodeConfig, GraphSchemaConfig, UseCaseConfig

_CYPHER_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _require_identifier(value: str, *, field: str) -> str:
    if not _CYPHER_IDENTIFIER.match(value):
        raise ValueError(f"{field} must be a valid Cypher identifier, got {value!r}")
    return value


def _validate_schema_identifiers(schema: GraphSchemaConfig) -> None:
    _require_identifier(schema.entity_node, field="graph_schema.entity_node")
    _require_identifier(schema.session_node, field="graph_schema.session_node")
    for node in schema.nodes:
        _require_identifier(node.label, field="graph_schema.nodes.label")
    for rel in schema.relationships:
        _require_identifier(rel.type, field="graph_schema.relationships.type")
        _require_identifier(rel.from_, field="graph_schema.relationships.from")
        _require_identifier(rel.to, field="graph_schema.relationships.to")


def get_graph_driver(
    *,
    uri: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Driver:
    """Return a Neo4j driver using environment variables by default."""
    resolved_uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    resolved_username = username or os.environ.get("NEO4J_USERNAME", "neo4j")
    resolved_password = password or os.environ.get("NEO4J_PASSWORD", "password")
    return GraphDatabase.driver(
        resolved_uri,
        auth=(resolved_username, resolved_password),
    )


def build_extraction_prompt(config: UseCaseConfig) -> str:
    """Build an LLM extraction prompt from the active graph schema."""
    if config.graph_schema is None:
        raise ValueError(
            f"Profile {config.name!r} has no graph_schema; cannot build extraction prompt"
        )

    schema = config.graph_schema
    template = {
        node.extraction_key: [] for node in schema.nodes
    }
    keys_json = json.dumps(template, indent=2)

    return f"""Extract entities from the following {config.display_name} note.

Return a JSON object with these keys:
{keys_json}

Only extract what is explicitly mentioned in the note. Return empty lists for keys with
no matches. Use short canonical phrases for each extracted value.

Return JSON only, with no markdown fences or other text."""


def empty_extraction_payload(config: UseCaseConfig) -> dict[str, list[str]]:
    """Return an empty extraction dict shaped for the active graph schema."""
    if config.graph_schema is None:
        return {}
    return {node.extraction_key: [] for node in config.graph_schema.nodes}


def _anchor_id(
    label: str,
    *,
    entity_id: str,
    session_id: str,
    schema: GraphSchemaConfig,
) -> str:
    if label == schema.entity_node:
        return entity_id
    if label == schema.session_node:
        return session_id
    raise ValueError(
        f"Cannot resolve anchor id for label {label!r}; "
        f"expected {schema.entity_node!r} or {schema.session_node!r}"
    )


def _merge_target_properties(node: GraphNodeConfig, value: str) -> dict[str, Any]:
    if node.identity == "id":
        return {"id": value}
    return {"name": value}


def write_entity_to_graph(
    session: Session,
    *,
    entity_id: str,
    session_id: str,
    extracted: dict[str, Any],
    config: UseCaseConfig,
) -> None:
    """Merge entity/session anchors and extracted relationships into Neo4j."""
    if config.graph_schema is None:
        return

    schema = config.graph_schema
    _validate_schema_identifiers(schema)

    entity_label = schema.entity_node
    session_label = schema.session_node
    entity_session_type = schema.entity_session_relationship
    if entity_session_type is None:
        raise ValueError("graph_schema.entity_session_relationship is not resolved")

    session.run(
        f"""
        MERGE (e:{entity_label} {{id: $entity_id}})
        MERGE (s:{session_label} {{id: $session_id}})
        MERGE (e)-[:{entity_session_type}]->(s)
        """,
        entity_id=entity_id,
        session_id=session_id,
    )

    nodes_by_label = schema.node_by_label()

    for rel in schema.relationships:
        if rel.from_ == entity_label and rel.to == session_label:
            continue

        target = nodes_by_label.get(rel.to)
        if target is None:
            continue

        values = extracted.get(target.extraction_key, [])
        if not isinstance(values, list):
            continue

        from_id = _anchor_id(
            rel.from_,
            entity_id=entity_id,
            session_id=session_id,
            schema=schema,
        )
        from_label = rel.from_
        to_label = rel.to

        for raw_value in values:
            if raw_value is None:
                continue
            value = str(raw_value).strip()
            if not value:
                continue

            props = _merge_target_properties(target, value)
            prop_name = next(iter(props))
            session.run(
                f"""
                MERGE (a:{from_label} {{id: $from_id}})
                MERGE (b:{to_label} {{{prop_name}: $value}})
                MERGE (a)-[:{rel.type}]->(b)
                """,
                from_id=from_id,
                value=props[prop_name],
            )
