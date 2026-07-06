"""Config-driven knowledge-graph helpers — no hardcoded domain node labels."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional, Type

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from neo4j import Driver, GraphDatabase, Session
from pydantic import BaseModel, Field, create_model

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


def build_extraction_schema(config: UseCaseConfig) -> Type[BaseModel]:
    """Create a Pydantic extraction model from ``config.graph_schema.nodes``."""
    if config.graph_schema is None:
        raise ValueError(
            f"Profile {config.name!r} has no graph_schema; cannot build extraction schema"
        )

    field_definitions = {
        node.extraction_key: (
            list[str],
            Field(
                default_factory=list,
                description=f"Extracted {node.label} entities mentioned in the note",
            ),
        )
        for node in config.graph_schema.nodes
    }
    model_name = "".join(part.capitalize() for part in config.name.split("_")) + "Extraction"
    return create_model(model_name, **field_definitions)


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
    keys_json = json.dumps(template, indent=2).replace("{", "{{").replace("}", "}}")
    node_labels = ", ".join(node.label for node in schema.nodes)

    return f"""Extract entities from the following {config.display_name} note.

Return a JSON object with these keys:
{keys_json}

Extract only explicitly mentioned entities for: {node_labels}.
Return empty lists for keys with no matches. Use short canonical phrases for each value.

Return JSON only, with no markdown fences or other text."""


def build_extraction_chain(
    config: UseCaseConfig,
    llm: BaseChatModel,
) -> Runnable[str, dict[str, list[str]]]:
    """Build an LCEL chain that extracts graph entities from session note text."""
    extraction_model = build_extraction_schema(config)
    parser = JsonOutputParser(pydantic_object=extraction_model)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", build_extraction_prompt(config) + "\n\n{format_instructions}"),
            ("human", "{text}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    chain: Runnable[str, dict[str, list[str]]] = (
        {"text": RunnablePassthrough()} | prompt | llm | parser
    )
    return chain


def extract_entities(
    text: str,
    config: UseCaseConfig,
    llm: BaseChatModel,
) -> dict[str, list[str]]:
    """Run the extraction chain and return a validated entity dictionary."""
    chain = build_extraction_chain(config, llm)
    result = chain.invoke(text)
    if isinstance(result, BaseModel):
        return result.model_dump()
    return dict(result)


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


def write_to_graph(
    entities: dict[str, Any],
    entity_id: str,
    session_id: str,
    config: UseCaseConfig,
    driver: Driver,
) -> None:
    """Write extracted entities to Neo4j using config-driven MERGE statements."""
    if config.graph_schema is None:
        raise ValueError(
            f"Profile {config.name!r} has no graph_schema; cannot write to graph"
        )

    with driver.session() as session:
        write_entity_to_graph(
            session,
            entity_id=entity_id,
            session_id=session_id,
            extracted=entities,
            config=config,
        )


def _empty_entity_graph(config: UseCaseConfig) -> dict[str, list[str]]:
    """Return an empty subgraph dict keyed by config node labels."""
    schema = config.graph_schema
    if schema is None:
        return {}

    graph: dict[str, list[str]] = {schema.session_node: []}
    for node in schema.nodes:
        graph[node.label] = []
    return graph


def _relationship_types_to(
    schema: GraphSchemaConfig,
    *,
    from_label: str,
    to_label: str,
) -> list[str]:
    return [
        rel.type
        for rel in schema.relationships
        if rel.from_ == from_label and rel.to == to_label
    ]


def _node_display_property(node: GraphNodeConfig) -> str:
    return "id" if node.identity == "id" else "name"


def _collect_node_values(
    session: Session,
    *,
    entity_id: str,
    schema: GraphSchemaConfig,
    node: GraphNodeConfig,
) -> list[str]:
    entity_label = schema.entity_node
    session_label = schema.session_node
    entity_session_type = schema.entity_session_relationship
    if entity_session_type is None:
        raise ValueError("graph_schema.entity_session_relationship is not resolved")

    node_label = node.label
    display_prop = _node_display_property(node)
    entity_rel_types = _relationship_types_to(
        schema, from_label=entity_label, to_label=node_label
    )
    session_rel_types = _relationship_types_to(
        schema, from_label=session_label, to_label=node_label
    )

    values: list[str] = []

    if entity_rel_types:
        record = session.run(
            f"""
            MATCH (e:{entity_label} {{id: $entity_id}})-[r]->(n:{node_label})
            WHERE type(r) IN $rel_types
            RETURN collect(DISTINCT n.{display_prop}) AS values
            """,
            entity_id=entity_id,
            rel_types=entity_rel_types,
        ).single()
        if record:
            values.extend(value for value in record["values"] if value)

    if session_rel_types:
        record = session.run(
            f"""
            MATCH (e:{entity_label} {{id: $entity_id}})
                  -[:{entity_session_type}]->(s:{session_label})-[r]->(n:{node_label})
            WHERE type(r) IN $rel_types
            RETURN collect(DISTINCT n.{display_prop}) AS values
            """,
            entity_id=entity_id,
            rel_types=session_rel_types,
        ).single()
        if record:
            values.extend(value for value in record["values"] if value)

    return list(dict.fromkeys(str(value) for value in values if value is not None))


def read_entity_graph(
    entity_id: str,
    config: UseCaseConfig,
    driver: Driver,
) -> dict[str, list[str]]:
    """Return the entity subgraph keyed by config node labels."""
    if config.graph_schema is None:
        raise ValueError(
            f"Profile {config.name!r} has no graph_schema; cannot read entity graph"
        )

    schema = config.graph_schema
    _validate_schema_identifiers(schema)
    entity_session_type = schema.entity_session_relationship
    if entity_session_type is None:
        raise ValueError("graph_schema.entity_session_relationship is not resolved")

    graph = _empty_entity_graph(config)

    with driver.session() as session:
        session_record = session.run(
            f"""
            MATCH (e:{schema.entity_node} {{id: $entity_id}})
                  -[:{entity_session_type}]->(s:{schema.session_node})
            RETURN collect(DISTINCT s.id) AS session_ids
            """,
            entity_id=entity_id,
        ).single()
        if session_record:
            graph[schema.session_node] = [
                str(value) for value in session_record["session_ids"] if value
            ]

        for node in schema.nodes:
            graph[node.label] = _collect_node_values(
                session,
                entity_id=entity_id,
                schema=schema,
                node=node,
            )

    return graph
