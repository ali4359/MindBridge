"""LangSmith tracing configuration — one project per use-case profile."""

from __future__ import annotations

import os

from core.config import UseCaseConfig

LANGCHAIN_PROJECT_ENV = "LANGCHAIN_PROJECT"


def langsmith_project_name(config: UseCaseConfig) -> str:
    """Return the LangSmith project name for a use-case profile."""
    return f"{config.name}-traces"


def configure_langsmith_project(config: UseCaseConfig) -> str:
    """Set ``LANGCHAIN_PROJECT`` so traces are isolated per use case."""
    project = langsmith_project_name(config)
    os.environ[LANGCHAIN_PROJECT_ENV] = project
    return project
