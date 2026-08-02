"""Rauha EHR adapter — HTTP fetch + compile into MindBridge payloads."""

from __future__ import annotations

import os
from typing import Any, Optional

import httpx

from adapters.base_adapter import BaseAdapter, MindBridgePayload
from adapters.rauha.entity_extractor import extract_entities
from adapters.rauha.note_compiler import compile_session_note
from adapters.rauha.trajectory import build_trajectory_note


class RauhaAdapter(BaseAdapter):
    """Adapter for the Rauha patient API.

    Reads ``RAUHA_BASE_URL`` / ``RAUHA_API_TOKEN`` from the environment when
    constructor args are omitted so domain-specific env names stay inside this
    package (never in ``backend/`` or ``core/``).
    """

    trajectory_trigger_sessions: frozenset[int] = frozenset({8, 10})

    def __init__(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        *,
        timeout: float = 30.0,
        use_case: str = "mental_health",
    ) -> None:
        self.base_url = (base_url or os.environ.get("RAUHA_BASE_URL", "")).rstrip("/")
        self.token = token if token is not None else os.environ.get("RAUHA_API_TOKEN", "")
        self.timeout = timeout
        self.use_case = use_case

    def build_trajectory(self, entity_id: str, sessions: list) -> Optional[str]:
        if not sessions:
            return None
        return build_trajectory_note(entity_id, sessions)

    # --- HTTP helpers -------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _get(self, path: str, **params: Any) -> Any:
        if not self.base_url:
            raise RuntimeError(
                "RAUHA_BASE_URL is not configured; cannot reach the Rauha API."
            )
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(url, headers=self._headers(), params=params or None)
            response.raise_for_status()
            return response.json()

    # --- Fetch --------------------------------------------------------------

    def fetch_sessions(self, entity_id: str, **kwargs: Any) -> Any:
        params = {k: v for k, v in kwargs.items() if v is not None}
        return self._get(f"/api/patients/{entity_id}/sessions", **params)

    def fetch_profile(self, entity_id: str) -> Any:
        return self._get(f"/api/patients/{entity_id}/profile")

    def fetch_goals(self, entity_id: str) -> Any:
        return self._get(f"/api/patients/{entity_id}/goals")

    def fetch_module(self, entity_id: str, module: str) -> Any:
        return self._get(f"/api/patients/{entity_id}/{module}")

    # --- Compile ------------------------------------------------------------

    def compile_sessions(self, raw: Any) -> str:
        sessions = _as_session_list(raw)
        if not sessions:
            return ""
        notes = [compile_session_note(s) for s in sessions]
        return "\n\n---\n\n".join(n for n in notes if n)

    def compile_profile(self, raw: Any) -> str:
        if not raw:
            return ""
        if isinstance(raw, str):
            return raw
        if not isinstance(raw, dict):
            return self.compile_generic(raw)

        lines: list[str] = ["Entity profile:"]
        preferred_order = [
            ("name", "Name"),
            ("fullName", "Name"),
            ("dateOfBirth", "Date of birth"),
            ("dob", "Date of birth"),
            ("sex", "Sex"),
            ("gender", "Gender"),
            ("email", "Email"),
            ("phone", "Phone"),
            ("primaryDiagnosis", "Primary diagnosis"),
            ("diagnoses", "Diagnoses"),
            ("medications", "Medications"),
            ("coach", "Coach"),
            ("therapist", "Therapist"),
            ("program", "Program"),
            ("status", "Status"),
            ("startDate", "Start date"),
            ("endDate", "End date"),
        ]
        seen: set[str] = set()
        for key, label in preferred_order:
            if key in raw and raw[key] is not None:
                lines.append(f"  {label}: {_fmt(raw[key])}")
                seen.add(key)
        for key, value in raw.items():
            if key in seen or value is None:
                continue
            label = _humanize(key)
            lines.append(f"  {label}: {_fmt(value)}")
        return "\n".join(lines)

    def compile_goals(self, raw: Any) -> str:
        if not raw:
            return ""
        if isinstance(raw, str):
            return raw

        short_term: list[Any] = []
        long_term: list[Any] = []

        if isinstance(raw, dict):
            short_term = (
                raw.get("shortTerm")
                or raw.get("short_term")
                or raw.get("shortTermGoals")
                or []
            )
            long_term = (
                raw.get("longTerm")
                or raw.get("long_term")
                or raw.get("longTermGoals")
                or []
            )
            if not short_term and not long_term and "goals" in raw:
                return self.compile_goals(raw["goals"])
        elif isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    short_term.append(item)
                    continue
                kind = str(item.get("type") or item.get("term") or "").lower()
                if "long" in kind:
                    long_term.append(item)
                else:
                    short_term.append(item)

        lines: list[str] = ["Goals:"]
        lines.append("  Short-term:")
        lines.extend(_format_goal_items(short_term) or ["    (none)"])
        lines.append("  Long-term:")
        lines.extend(_format_goal_items(long_term) or ["    (none)"])
        return "\n".join(lines)

    # --- Transform (ingest) -------------------------------------------------

    def transform(self, raw_document: Any) -> MindBridgePayload:
        if not isinstance(raw_document, dict):
            raise TypeError("RauhaAdapter.transform expects a session document dict")

        entity_id = str(
            raw_document.get("patientId")
            or raw_document.get("entity_id")
            or raw_document.get("entityId")
            or ""
        )
        session_number = raw_document.get("sessionNumber", raw_document.get("session_number", 0))
        try:
            session_number_int = int(session_number)
        except (TypeError, ValueError):
            session_number_int = 0

        session_date = str(
            raw_document.get("sessionDate")
            or raw_document.get("session_date")
            or raw_document.get("date")
            or ""
        )

        note_text = compile_session_note(raw_document)
        signals = extract_entities(raw_document)

        return MindBridgePayload(
            entity_id=entity_id,
            entity_type="patient",
            session_number=session_number_int,
            session_date=session_date,
            note_text=note_text,
            use_case=self.use_case,
            signals=signals,
            metadata={
                "source": "rauha",
                "session_id": raw_document.get("id") or raw_document.get("_id"),
            },
        )


def _as_session_list(raw: Any) -> list[dict]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [s for s in raw if isinstance(s, dict)]
    if isinstance(raw, dict):
        for key in ("sessions", "data", "items", "results"):
            if isinstance(raw.get(key), list):
                return [s for s in raw[key] if isinstance(s, dict)]
        # Single session document
        if any(k in raw for k in ("sessionNumber", "session_number", "coachNotes")):
            return [raw]
    return []


def _fmt(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return ", ".join(f"{k}: {v}" for k, v in value.items())
    return str(value)


def _humanize(key: str) -> str:
    parts: list[str] = []
    current: list[str] = []
    for ch in key:
        if ch in ("_", "-"):
            if current:
                parts.append("".join(current))
                current = []
            continue
        if ch.isupper() and current:
            parts.append("".join(current))
            current = [ch]
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return " ".join(p.capitalize() for p in parts) if parts else key


def _format_goal_items(items: list[Any]) -> list[str]:
    lines: list[str] = []
    for item in items:
        if isinstance(item, dict):
            text = item.get("text") or item.get("goal") or item.get("description") or _fmt(item)
            status = item.get("status")
            if status:
                lines.append(f"    - {_fmt(text)} [{status}]")
            else:
                lines.append(f"    - {_fmt(text)}")
        else:
            lines.append(f"    - {_fmt(item)}")
    return lines
