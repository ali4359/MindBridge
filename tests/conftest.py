"""Shared pytest fixtures for MindBridge integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = REPO_ROOT / "data" / "chromadb"


@pytest.fixture(scope="session")
def chroma_index_available() -> bool:
    return CHROMA_DIR.is_dir()


@pytest.fixture(scope="session")
def require_chroma_index(chroma_index_available: bool) -> None:
    if not chroma_index_available:
        pytest.skip(f"No persisted Chroma index at {CHROMA_DIR}")
