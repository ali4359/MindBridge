"""Build a clinical trajectory note from a patient's full session history.

Triggered automatically when session 8 (program completion) or session 10
(follow-up) is ingested. The resulting text is indexed as ``doc_type='trajectory'``.
"""

from __future__ import annotations

from typing import Any, Optional


def _session_number(session: dict) -> int:
    raw = session.get("sessionNumber", session.get("session_number", 0))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _response_value(item: dict) -> Any:
    for key in ("response", "value", "answer"):
        if key in item:
            return item[key]
    return None


def _find_response(responses: Any, question_key: str) -> Any:
    if not isinstance(responses, list):
        return None
    for item in responses:
        if not isinstance(item, dict):
            continue
        key = item.get("questionKey") or item.get("question_key")
        if key == question_key:
            return _response_value(item)
    return None


def _coach_responses(session: dict) -> Any:
    coach = session.get("coachNotes") or session.get("coach_notes") or {}
    if not isinstance(coach, dict):
        return None
    return coach.get("sessionResponses") or coach.get("session_responses")


def _homework_responses(session: dict) -> Any:
    return session.get("sessionHomeworkResponses") or session.get(
        "session_homework_responses"
    )


def _midweek_responses(session: dict) -> Any:
    return session.get("midWeekResponses") or session.get("mid_week_responses")


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_list(value: Any) -> str:
    if value is None:
        return "(not recorded)"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) if value else "(none)"
    text = str(value).strip()
    return text if text else "(not recorded)"


def build_trajectory_note(entity_id: str, sessions: list) -> str:
    """Build a rich clinical trajectory narrative for ``entity_id``.

    ``sessions`` must be PatientSession docs for one patient, preferably sorted
    by ``sessionNumber``. Stress trend, session-0 values/goals, and mid-week
    confidence trajectory are summarised for indexing as ``doc_type='trajectory'``.
    """
    if not sessions:
        return f"Trajectory for {entity_id}: no sessions available."

    ordered = sorted(
        (s for s in sessions if isinstance(s, dict)),
        key=_session_number,
    )
    if not ordered:
        return f"Trajectory for {entity_id}: no sessions available."

    # --- Stress rating trend from coachNotes.sessionResponses ---------------
    stress_points: list[tuple[int, float]] = []
    for session in ordered:
        rating = _find_response(_coach_responses(session), "stressRating")
        numeric = _as_float(rating)
        if numeric is not None:
            stress_points.append((_session_number(session), numeric))

    baseline = stress_points[0][1] if stress_points else None
    # Prefer session 0 specifically when present
    for num, value in stress_points:
        if num == 0:
            baseline = value
            break
    latest = stress_points[-1][1] if stress_points else None
    change = (latest - baseline) if baseline is not None and latest is not None else None
    pct_improvement: Optional[float] = None
    if baseline is not None and latest is not None and baseline != 0:
        # Improvement = reduction in stress
        pct_improvement = ((baseline - latest) / abs(baseline)) * 100.0

    # --- Session 0 values + long-term goals from homework -------------------
    session0 = next((s for s in ordered if _session_number(s) == 0), ordered[0])
    values = _find_response(_homework_responses(session0), "valuesIdentification")
    long_term_goals = _find_response(
        _homework_responses(session0), "longTermGoalsFormation"
    )

    # --- Mid-week confidence trend across all sessions ----------------------
    confidence_points: list[tuple[int, float]] = []
    for session in ordered:
        rating = _find_response(_midweek_responses(session), "confidenceRating")
        numeric = _as_float(rating)
        if numeric is not None:
            confidence_points.append((_session_number(session), numeric))

    # --- Assemble narrative -------------------------------------------------
    lines: list[str] = [
        f"Clinical trajectory for entity {entity_id}",
        f"Sessions covered: {len(ordered)} "
        f"(session {_session_number(ordered[0])} through "
        f"session {_session_number(ordered[-1])}).",
        "",
        "Stress rating trend:",
    ]

    if stress_points:
        trend_str = " → ".join(f"S{num}={val:g}" for num, val in stress_points)
        lines.append(f"  Series: {trend_str}")
        lines.append(
            f"  Baseline (session 0): {baseline:g}"
            if baseline is not None
            else "  Baseline: (not recorded)"
        )
        lines.append(
            f"  Latest: {latest:g}" if latest is not None else "  Latest: (not recorded)"
        )
        if change is not None:
            direction = "decrease" if change < 0 else "increase" if change > 0 else "no change"
            lines.append(f"  Change: {change:+g} ({direction})")
        if pct_improvement is not None:
            lines.append(f"  Percentage improvement: {pct_improvement:.1f}%")
    else:
        lines.append("  No stress ratings recorded across sessions.")

    lines.extend(
        [
            "",
            "Session 0 values and goals:",
            f"  Core values: {_format_list(values)}",
            f"  Long-term goals: {_format_list(long_term_goals)}",
            "",
            "Mid-week homework confidence trajectory:",
        ]
    )

    if confidence_points:
        conf_str = " → ".join(f"S{num}={val:g}" for num, val in confidence_points)
        lines.append(f"  Series: {conf_str}")
        first_c, last_c = confidence_points[0][1], confidence_points[-1][1]
        delta = last_c - first_c
        lines.append(f"  Change from first to last: {delta:+g}")
    else:
        lines.append("  No confidence ratings recorded across sessions.")

    return "\n".join(lines).strip()
