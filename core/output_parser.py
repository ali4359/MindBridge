"""Dynamic Pydantic output models driven by use-case config."""

from __future__ import annotations

from typing import Type

from pydantic import BaseModel, Field, create_model

from core.config import UseCaseConfig


def _resolve_model_name(config: UseCaseConfig) -> str:
    name = config.output_schema.model_name.strip()
    if name:
        return name
    return "".join(part.capitalize() for part in config.name.split("_"))


def build_output_schema(config: UseCaseConfig) -> Type[BaseModel]:
    """Create a Pydantic model at runtime from ``config.output_schema.fields``.

  The model name and field list are entirely config-driven: each use-case profile's
  ``output_schema.model_name`` and ``output_schema.fields`` determine the shape.
  """
    schema = config.output_schema
    if not schema.fields:
        raise ValueError("output_schema.fields must not be empty")

    model_name = _resolve_model_name(config)
    field_definitions = {
        field_name: (str, Field(description=f"{field_name} section"))
        for field_name in schema.fields
    }
    return create_model(model_name, **field_definitions)
