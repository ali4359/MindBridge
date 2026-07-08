"""Config-driven structured-output chain from ``output_schema`` fields."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from core.config import UseCaseConfig
from core.output_parser import build_output_schema


def build_output_chain(
    config: UseCaseConfig,
    llm: BaseChatModel,
) -> Runnable[dict[str, str], dict[str, Any]]:
    """Build an LCEL chain using ``with_structured_output(schema_model)``.

    Input keys: ``notes_text``, ``entity_id``, optional ``entity_context``.
    Output: object/dict keyed by ``config.output_schema.fields``.
    """
    schema_model = build_output_schema(config)
    field_names = ", ".join(config.output_schema.fields)
    structured_llm = llm.with_structured_output(schema_model)

    system = (
        f"You generate structured {schema_model.__name__} JSON for "
        f"{config.display_name}.\n"
        f"Populate these fields: {field_names}.\n"
        "Use only information supported by the notes. "
        "Return only the structured output."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Entity ID: {entity_id}\n\n"
                "Entity context:\n{entity_context}\n\n"
                "Notes:\n{notes_text}",
            ),
        ]
    )

    chain: Runnable[dict[str, str], dict[str, Any]] = (
        {
            "entity_id": lambda x: x["entity_id"],
            "notes_text": lambda x: x["notes_text"],
            "entity_context": lambda x: x.get(
                "entity_context",
                config.prompts.empty_entity_context,
            ),
        }
        | prompt
        | structured_llm
    )
    return chain


def parse_output_result(result: Any, schema_model: type[BaseModel]) -> dict[str, Any]:
    """Normalise chain output to a plain dict matching the schema fields."""
    if isinstance(result, BaseModel):
        return result.model_dump()
    if isinstance(result, dict):
        return {key: result.get(key, "") for key in schema_model.model_fields}
    raise TypeError(f"Unexpected structured output type: {type(result).__name__}")
