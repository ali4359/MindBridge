"""Compile a Rauha PatientSession document into clean clinical prose.

Handles the three response arrays on a session doc:
  - coachNotes.sessionResponses  (coach during session)
  - sessionHomeworkResponses     (patient after session)
  - midWeekResponses             (patient midweek)
"""

from __future__ import annotations

from typing import Any

# Maps every known Rauha questionKey → readable English label.
QUESTION_LABELS: dict[str, str] = {
    "stressRating": "Stress Rating",
    "valuesIdentification": "Core Values",
    "confidenceRating": "Homework Confidence",
    "senseOfPride": "Pride in Goals",
    "longTermGoalsFormation": "Long-Term Goals",
    "shortTermGoalsFormation": "Short-Term Goals",
    "moodRating": "Mood Rating",
    "anxietyRating": "Anxiety Rating",
    "sleepQuality": "Sleep Quality",
    "energyLevel": "Energy Level",
    "socialConnection": "Social Connection",
    "copingStrategies": "Coping Strategies",
    "barriersEncountered": "Barriers Encountered",
    "winsThisWeek": "Wins This Week",
    "sessionReflection": "Session Reflection",
    "homeworkCompletion": "Homework Completion",
    "motivationRating": "Motivation Rating",
    "progressTowardGoals": "Progress Toward Goals",
    "challengesFaced": "Challenges Faced",
    "supportNeeded": "Support Needed",
    "gratitudePractice": "Gratitude Practice",
    "mindfulnessPractice": "Mindfulness Practice",
    "physicalActivity": "Physical Activity",
    "medicationAdherence": "Medication Adherence",
    "riskScreen": "Risk Screen",
    "suicidalIdeation": "Suicidal Ideation",
    "selfHarm": "Self-Harm",
    "safetyPlanReview": "Safety Plan Review",
    "allianceRating": "Therapeutic Alliance",
    "sessionGoals": "Session Goals",
    "actionItems": "Action Items",
    "followUpPlan": "Follow-Up Plan",
    "intakePresentingConcern": "Presenting Concern",
    "intakeHistory": "Clinical History",
    "intakeMedications": "Current Medications",
    "intakeDiagnoses": "Diagnoses",
}

# Maps sessionNumber 0–10 → clinical program context.
SESSION_CONTEXT: dict[int, str] = {
    0: "Intake and goal-setting session — establishing baseline, values, and treatment direction.",
    1: "Early engagement — building therapeutic alliance and introducing core skills.",
    2: "Skill building — practising foundational coping and cognitive strategies.",
    3: "Deepening practice — applying skills to daily stressors and barriers.",
    4: "Mid-program review — assessing progress and refining short-term goals.",
    5: "Consolidation — strengthening gains and addressing emerging challenges.",
    6: "Advanced application — integrating skills across life domains.",
    7: "Preparation for completion — reviewing trajectory and relapse prevention.",
    8: "Program completion — final skills review, celebration of progress, and discharge planning.",
    9: "Post-program booster — reinforcing maintenance strategies.",
    10: "Follow-up — long-term outcome check and booster support.",
}


def _label_for(question_key: str) -> str:
    if question_key in QUESTION_LABELS:
        return QUESTION_LABELS[question_key]
    # Fallback: split camelCase into Title Case words.
    parts: list[str] = []
    current = []
    for ch in question_key:
        if ch.isupper() and current:
            parts.append("".join(current))
            current = [ch]
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return " ".join(p.capitalize() for p in parts) if parts else question_key


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value if v is not None and str(v).strip())
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value).strip()


def _format_responses(responses: Any, heading: str) -> list[str]:
    """Turn a list of {questionKey, response/value/answer} dicts into prose lines."""
    if not responses or not isinstance(responses, list):
        return []

    lines: list[str] = [f"{heading}:"]
    for item in responses:
        if not isinstance(item, dict):
            continue
        key = item.get("questionKey") or item.get("question_key") or ""
        raw_value = (
            item.get("response")
            if "response" in item
            else item.get("value")
            if "value" in item
            else item.get("answer")
        )
        formatted = _format_value(raw_value)
        if not key and not formatted:
            continue
        label = _label_for(str(key)) if key else "Response"
        if formatted:
            lines.append(f"  {label}: {formatted}")
        else:
            lines.append(f"  {label}: (no response)")
    return lines if len(lines) > 1 else []


def compile_session_note(session_doc: dict) -> str:
    """Compile a Rauha PatientSession dict into clean clinical prose.

    Output contains no JSON syntax, no camelCase keys, and no raw field names.
    """
    if not isinstance(session_doc, dict):
        return ""

    session_number = session_doc.get("sessionNumber", session_doc.get("session_number", 0))
    try:
        session_number_int = int(session_number)
    except (TypeError, ValueError):
        session_number_int = 0

    session_date = (
        session_doc.get("sessionDate")
        or session_doc.get("session_date")
        or session_doc.get("date")
        or ""
    )
    context = SESSION_CONTEXT.get(
        session_number_int,
        f"Session {session_number_int} of the clinical program.",
    )

    sections: list[str] = [
        f"Session {session_number_int}"
        + (f" ({session_date})" if session_date else ""),
        context,
    ]

    coach_notes = session_doc.get("coachNotes") or session_doc.get("coach_notes") or {}
    if isinstance(coach_notes, dict):
        coach_responses = coach_notes.get("sessionResponses") or coach_notes.get(
            "session_responses"
        )
        sections.extend(_format_responses(coach_responses, "Coach session notes"))
        free_text = coach_notes.get("notes") or coach_notes.get("freeText")
        if free_text:
            sections.append(f"Coach narrative:\n  {_format_value(free_text)}")

    homework = (
        session_doc.get("sessionHomeworkResponses")
        or session_doc.get("session_homework_responses")
    )
    sections.extend(_format_responses(homework, "Post-session homework responses"))

    midweek = session_doc.get("midWeekResponses") or session_doc.get("mid_week_responses")
    sections.extend(_format_responses(midweek, "Mid-week check-in responses"))

    return "\n\n".join(s for s in sections if s).strip()
