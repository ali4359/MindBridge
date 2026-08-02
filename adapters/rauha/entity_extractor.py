"""Rule-based entity extraction from Rauha session documents.

Zero LLM cost. Structured questionKey extraction plus free-text regex for
medications, diagnoses, interventions, and risk flags. Risk severity is ALWAYS
determined by rules — never by an LLM.
"""

from __future__ import annotations

import re
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Structured questionKey → output field
# ---------------------------------------------------------------------------

_STRUCTURED_KEYS: dict[str, str] = {
    "stressRating": "stress_rating",
    "valuesIdentification": "values",
    "confidenceRating": "confidence_rating",
    "senseOfPride": "pride_rating",
    "longTermGoalsFormation": "long_term_goals",
    "shortTermGoalsFormation": "short_term_goals",
    "moodRating": "mood_rating",
    "anxietyRating": "anxiety_rating",
}

# ---------------------------------------------------------------------------
# Free-text lexicon (case-insensitive)
# ---------------------------------------------------------------------------

_MEDICATIONS = [
    "sertraline",
    "lithium",
    "quetiapine",
    "fluoxetine",
    "escitalopram",
    "venlafaxine",
    "duloxetine",
    "bupropion",
    "mirtazapine",
    "aripiprazole",
    "olanzapine",
    "risperidone",
    "lamotrigine",
    "valproate",
    "clonazepam",
    "lorazepam",
    "alprazolam",
    "propranolol",
    "trazodone",
    "citalopram",
    "paroxetine",
]

_DIAGNOSES = [
    "depression",
    "major depressive disorder",
    "mdd",
    "anxiety",
    "generalized anxiety disorder",
    "gad",
    "ptsd",
    "post-traumatic stress",
    "posttraumatic stress",
    "bipolar",
    "bipolar disorder",
    "ocd",
    "panic disorder",
    "social anxiety",
    "borderline personality",
]

_INTERVENTIONS = [
    "cbt",
    "cognitive behavioural therapy",
    "cognitive behavioral therapy",
    "dbt",
    "dialectical behaviour therapy",
    "dialectical behavior therapy",
    "mindfulness",
    "grounding",
    "thought record",
    "behavioural activation",
    "behavioral activation",
    "exposure therapy",
    "safety planning",
    "motivational interviewing",
    "act",
    "acceptance and commitment",
]

# ---------------------------------------------------------------------------
# Risk flags — three severity tiers (ALWAYS rule-based)
# Order within a tier: more specific patterns first.
# Evaluation order across tiers: ACTIVE → DENIED → MONITOR (denied before
# monitor so "ideation denied" is not misclassified by a passive pattern).
# ---------------------------------------------------------------------------

_ACTIVE_PATTERNS = [
    r"\bsuicidal\s+ideation\s+present\b",
    r"\bactive\s+suicidal\s+ideation\b",
    r"\bplan\s+to\s+harm\b",
    r"\bplan\s+to\s+(?:kill|hurt)\s+(?:him|her|them)?self\b",
    r"\bself[- ]harm\s+(?:occurring|ongoing|present|active)\b",
    r"\b(?:occurring|ongoing|active)\s+self[- ]harm\b",
    r"\bintent\s+to\s+(?:die|suicide|kill\s+(?:him|her|them)?self)\b",
]

_DENIED_PATTERNS = [
    r"\bsuicidal\s+ideation\s+denied\b",
    r"\bideation\s+denied\b",
    r"\bdenies\s+(?:si|suicidal\s+ideation|self[- ]harm)\b",
    r"\bno\s+(?:si|suicidal\s+ideation|self[- ]harm)\b",
    r"\bdenied\s+(?:si|suicidal\s+ideation|self[- ]harm)\b",
    r"\bnegative\s+(?:for\s+)?(?:si|suicidal\s+ideation)\b",
]

_MONITOR_PATTERNS = [
    r"\bpassive\s+suicidal\s+ideation\b",
    r"\bpassive\s+ideation\b",
    r"\bthoughts?\s+of\s+death\b",
    r"\bdeath\s+wish(?:es)?\b",
    r"\bmonitor\s+closely\b",
    r"\bsuicidal\s+ideation\b",  # bare SI without denied/present → monitor
    r"\bself[- ]harm\s+ideation\b",
]


def _iter_responses(session_doc: dict) -> list[dict]:
    """Collect all response dicts from the three arrays."""
    items: list[dict] = []

    coach = session_doc.get("coachNotes") or session_doc.get("coach_notes") or {}
    if isinstance(coach, dict):
        for key in ("sessionResponses", "session_responses"):
            responses = coach.get(key)
            if isinstance(responses, list):
                items.extend(r for r in responses if isinstance(r, dict))

    for key in (
        "sessionHomeworkResponses",
        "session_homework_responses",
        "midWeekResponses",
        "mid_week_responses",
    ):
        responses = session_doc.get(key)
        if isinstance(responses, list):
            items.extend(r for r in responses if isinstance(r, dict))

    return items


def _response_value(item: dict) -> Any:
    for key in ("response", "value", "answer"):
        if key in item:
            return item[key]
    return None


def _flatten_text(session_doc: dict) -> str:
    """Concatenate all free-text strings in the session for regex scanning."""
    parts: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, str):
            parts.append(node)
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(session_doc)
    return "\n".join(parts)


def _find_lexicon(text: str, terms: list[str]) -> list[str]:
    found: list[str] = []
    lowered = text.lower()
    for term in terms:
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            found.append(term)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for item in found:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _classify_risk(text: str, structured_hint: Optional[str] = None) -> dict[str, str]:
    """Return ``{level, severity, evidence}`` from rule-based patterns.

    Levels: ACTIVE / MONITOR / DENIED  →  severities HIGH / MEDIUM / NONE
    """
    haystack = text.lower()
    if structured_hint:
        haystack = f"{haystack}\n{structured_hint.lower()}"

    for pattern in _ACTIVE_PATTERNS:
        match = re.search(pattern, haystack, flags=re.IGNORECASE)
        if match:
            return {
                "level": "ACTIVE",
                "severity": "HIGH",
                "evidence": match.group(0),
            }

    for pattern in _DENIED_PATTERNS:
        match = re.search(pattern, haystack, flags=re.IGNORECASE)
        if match:
            return {
                "level": "DENIED",
                "severity": "NONE",
                "evidence": match.group(0),
            }

    for pattern in _MONITOR_PATTERNS:
        match = re.search(pattern, haystack, flags=re.IGNORECASE)
        if match:
            return {
                "level": "MONITOR",
                "severity": "MEDIUM",
                "evidence": match.group(0),
            }

    return {"level": "DENIED", "severity": "NONE", "evidence": ""}


def _coerce_numeric(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return value


def extract_entities(session_doc: dict) -> dict:
    """Extract structured signals from a Rauha session document.

    Returns a dict with numeric/list fields from questionKeys plus
    ``medications``, ``diagnoses``, ``interventions``, and ``risk``.
    """
    if not isinstance(session_doc, dict):
        return {"risk": {"level": "DENIED", "severity": "NONE", "evidence": ""}}

    result: dict[str, Any] = {}

    # --- Structured questionKey pass (belt) ---------------------------------
    for item in _iter_responses(session_doc):
        key = item.get("questionKey") or item.get("question_key")
        if not key or key not in _STRUCTURED_KEYS:
            continue
        field = _STRUCTURED_KEYS[key]
        value = _response_value(item)
        if field in ("values", "long_term_goals", "short_term_goals"):
            if isinstance(value, list):
                result[field] = value
            elif value is not None and str(value).strip():
                # Split common delimiters into a list
                result[field] = [p.strip() for p in re.split(r"[,;|/]", str(value)) if p.strip()]
            else:
                result[field] = []
        else:
            result[field] = _coerce_numeric(value)

    # --- Free-text regex pass (braces) --------------------------------------
    text = _flatten_text(session_doc)
    result["medications"] = _find_lexicon(text, _MEDICATIONS)
    result["diagnoses"] = _find_lexicon(text, _DIAGNOSES)
    result["interventions"] = _find_lexicon(text, _INTERVENTIONS)

    # Structured risk screen response (if present) feeds the same classifier
    risk_hint = None
    for item in _iter_responses(session_doc):
        key = item.get("questionKey") or item.get("question_key")
        if key in ("riskScreen", "suicidalIdeation", "selfHarm"):
            risk_hint = str(_response_value(item) or "")
            break

    result["risk"] = _classify_risk(text, structured_hint=risk_hint)
    return result
