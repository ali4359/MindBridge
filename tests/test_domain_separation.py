"""CI gate: core/ and backend/ must contain zero hardcoded domain vocabulary.

Domain-specific words (clinical, legal, or vendor-specific) belong only in
configs/*.yaml. This walks every .py file in core/ and backend/ and reports
the exact file, line, and word the moment one leaks in.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ["core", "backend"]

DOMAIN_WORDS = [
    "patient",
    "therapy",
    "clinical",
    "mental_health",
    "rauha",
    "nice",
    "cbt",
    "dbt",
    "depression",
    "client",
    "legal",
    "statute",
    "case_law",
]


def _iter_py_files():
    for scan_dir in SCAN_DIRS:
        base = REPO_ROOT / scan_dir
        for dirpath, _dirnames, filenames in os.walk(base):
            for filename in filenames:
                if filename.endswith(".py"):
                    yield Path(dirpath) / filename


def test_no_domain_words_in_core_or_backend() -> None:
    violations: list[str] = []

    for path in _iter_py_files():
        rel_path = path.relative_to(REPO_ROOT)
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            lowered = line.lower()
            for word in DOMAIN_WORDS:
                if word in lowered:
                    violations.append(f"{rel_path}:{line_number}: {word!r} in {line.strip()!r}")

    if violations:
        report = "\n".join(violations)
        pytest.fail(f"Domain words leaked into core/ or backend/:\n{report}", pytrace=False)
