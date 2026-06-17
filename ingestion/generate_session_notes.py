"""Generate synthetic therapy session notes for the MindBridge demo corpus."""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "session_notes"

# Documented demo prompt — kept inline so the story is reproducible without extra files.
SESSION_NOTE_PROMPT = """You are assisting a clinical documentation demo for a mental-health RAG system.
Write one realistic but entirely fictional therapy session note.

Requirements:
- Format: SOAP (Subjective, Objective, Assessment, Plan) with clear section headers.
- Length: 250–400 words.
- Setting: outpatient CBT or DBT-informed psychotherapy.
- Include: presenting concern, interventions used (e.g. thought record, behavioral activation,
  distress tolerance, mindfulness), client response, risk screen (deny SI/HI unless scenario says otherwise),
  and homework.
- Use neutral clinical tone. No real names — use "Client" or initials only.
- Scenario #{session_number}: {scenario}

Output only the session note, no preamble."""

SCENARIOS: tuple[str, ...] = (
    "First session — generalized anxiety, psychoeducation and breathing reframe.",
    "Session 2 — depression, behavioral activation scheduling.",
    "Session 3 — panic attacks, interoceptive exposure planning.",
    "Session 4 — social anxiety, cognitive restructuring of mind-reading thoughts.",
    "Session 5 — insomnia comorbid with depression, sleep hygiene and stimulus control.",
    "Session 6 — workplace burnout, values clarification and boundary setting.",
    "Session 7 — grief — normalizing waves of emotion, continuing bonds framework.",
    "Session 8 — OCD intrusive thoughts — ERP hierarchy introduction.",
    "Session 9 — PTSD — grounding after trigger, safety planning review.",
    "Session 10 — DBT skills — distress tolerance with TIPP and self-soothe.",
    "Session 11 — DBT skills — emotion regulation, PLEASE skills check-in.",
    "Session 12 — relationship conflict — DEAR MAN interpersonal effectiveness practice.",
    "Session 13 — low self-esteem — core belief log and evidence column.",
    "Session 14 — medication adherence ambivalence — motivational interviewing.",
    "Session 15 — relapse prevention — early warning signs and coping card.",
    "Session 16 — adolescent transition (age 17) — parent involvement and autonomy.",
    "Session 17 — perinatal anxiety — catastrophic thinking about infant safety.",
    "Session 18 — substance use + depression — functional analysis of use episode.",
    "Session 19 — chronic pain and mood — pacing and acceptance strategies.",
    "Session 20 — termination session — review gains, booster plan, warm handoff.",
)

DEFAULT_MODEL = "llama-3.3-70b-versatile"


def _client() -> Groq:
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in .env")
    return Groq(api_key=api_key)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60]


def generate_note(client: Groq, session_number: int, scenario: str, model: str) -> str:
    prompt = SESSION_NOTE_PROMPT.format(session_number=session_number, scenario=scenario)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=900,
    )
    return (response.choices[0].message.content or "").strip()


def write_notes(
    output_dir: Path = OUTPUT_DIR,
    count: int = 20,
    model: str = DEFAULT_MODEL,
    pause_seconds: float = 0.5,
) -> list[Path]:
    """Generate `count` synthetic session notes and save as plain text files."""
    if count > len(SCENARIOS):
        raise ValueError(f"Only {len(SCENARIOS)} scenarios defined, requested {count}")

    output_dir.mkdir(parents=True, exist_ok=True)
    client = _client()
    written: list[Path] = []

    for index in range(1, count + 1):
        scenario = SCENARIOS[index - 1]
        logger.info("Generating session note %d/%d", index, count)
        note = generate_note(client, index, scenario, model=model)

        filename = f"synthetic_session_{index:02d}_{_slugify(scenario.split('—')[0])}.txt"
        path = output_dir / filename
        header = (
            f"# Synthetic therapy session note {index:02d}\n"
            f"# Scenario: {scenario}\n"
            f"# Generated for MindBridge demo — fictional, not real patient data\n\n"
        )
        path.write_text(header + note + "\n", encoding="utf-8")
        written.append(path)
        time.sleep(pause_seconds)

    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paths = write_notes()
    logger.info("Wrote %d session notes to %s", len(paths), OUTPUT_DIR)


if __name__ == "__main__":
    main()
