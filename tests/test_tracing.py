"""Tests for per-use-case LangSmith project isolation."""

from __future__ import annotations

import os

from core.config import load_config
from core.tracing import LANGCHAIN_PROJECT_ENV, configure_langsmith_project, langsmith_project_name


def test_langsmith_project_name_per_profile() -> None:
    mental_health = load_config("configs/mental_health.yaml")
    legal = load_config("configs/legal.yaml")

    assert langsmith_project_name(mental_health) == "mental_health-traces"
    assert langsmith_project_name(legal) == "legal-traces"
    assert mental_health.langsmith_project == "mental_health-traces"
    assert legal.langsmith_project == "legal-traces"


def test_load_config_sets_langchain_project_env() -> None:
    load_config("configs/legal.yaml")
    assert os.environ[LANGCHAIN_PROJECT_ENV] == "legal-traces"

    load_config("configs/mental_health.yaml")
    assert os.environ[LANGCHAIN_PROJECT_ENV] == "mental_health-traces"


def test_configure_langsmith_project_overwrites_env() -> None:
    config = load_config("configs/legal.yaml")
    project = configure_langsmith_project(config)
    assert project == "legal-traces"
    assert os.environ[LANGCHAIN_PROJECT_ENV] == "legal-traces"
