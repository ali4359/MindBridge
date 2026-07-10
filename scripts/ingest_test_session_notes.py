#!/usr/bin/env python3
"""TEST ONLY — bulk-ingest synthetic session notes into Chroma + Neo4j.

Loads every ``data/session_notes/synthetic_session_*.txt`` file and indexes it
using the same dual-write path as ``POST /ingest`` (vector store + knowledge graph).

Prerequisites:
  1. PDF corpus already indexed:  ``python -m core ingest``
  2. Neo4j running locally (see ``.env`` NEO4J_* vars) for graph modes

Usage:
  source .venv/bin/activate

  # Chroma only — no Groq, no Neo4j (fastest; fixes RAG retrieval)
  python scripts/ingest_test_session_notes.py --vector-only

  # Chroma + Neo4j Patient/Session nodes — no Groq (good for graph testing)
  python scripts/ingest_test_session_notes.py --graph-shell

  # Neo4j only when Chroma already has the notes (avoids duplicate vectors)
  python scripts/ingest_test_session_notes.py --graph-shell --graph-only

  # Full dual-index with LLM entity extraction (requires GROQ_API_KEY)
  python scripts/ingest_test_session_notes.py

  # Resume after a rate-limit failure (skips sessions already in Chroma)
  python scripts/ingest_test_session_notes.py --resume --pause 2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.config import USE_CASE_CONFIG_ENV, load_active_config, resolve_config_path
from core.graph import get_graph_driver
from core.ingestion import GraphIngestMode, ingest_all_session_notes
from core.llm import build_llm

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TEST ONLY: bulk-ingest demo session notes into Chroma and Neo4j.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=f"YAML profile path (or set ${USE_CASE_CONFIG_ENV}); required if neither is set",
    )
    parser.add_argument(
        "--entity-id",
        default="patient-demo",
        help="Patient/entity id stamped on every note (default: patient-demo)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--vector-only",
        action="store_true",
        help="Index into Chroma only; skip Neo4j",
    )
    mode.add_argument(
        "--graph-shell",
        action="store_true",
        help="Chroma + Neo4j Patient/Session nodes only; no Groq LLM calls",
    )
    parser.add_argument(
        "--graph-only",
        action="store_true",
        help="Write Neo4j only; do not append to Chroma (use when vectors already exist)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip sessions already indexed for this entity in Chroma",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Pause between sessions in full LLM mode (reduces Groq rate limits)",
    )
    return parser


def _resolve_graph_mode(args: argparse.Namespace) -> GraphIngestMode:
    if args.vector_only:
        return "off"
    if args.graph_shell:
        return "shell"
    return "full"


def main(argv: list[str] | None = None) -> int:
    load_dotenv(REPO_ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    args = build_parser().parse_args(argv)
    config = load_active_config(args.config)
    config_path = resolve_config_path(args.config)
    graph_mode = _resolve_graph_mode(args)

    driver = None
    llm = None
    try:
        if graph_mode in {"shell", "full"}:
            driver = get_graph_driver()
            driver.verify_connectivity()
            logger.info("Connected to Neo4j")
            if graph_mode == "full":
                llm = build_llm(config)

        result = ingest_all_session_notes(
            config,
            entity_id=args.entity_id,
            base_dir=REPO_ROOT,
            graph_mode=graph_mode,
            llm=llm,
            driver=driver,
            resume=args.resume,
            graph_only=args.graph_only,
            pause_seconds=args.pause,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 — surface ingest failures to the CLI user
        logger.error("Bulk session-note ingest failed: %s", exc)
        if _is_rate_limit(exc):
            logger.error(
                "Groq daily token limit hit. Retry later with:\n"
                "  python scripts/ingest_test_session_notes.py --graph-shell\n"
                "or resume full ingest after the limit resets:\n"
                "  python scripts/ingest_test_session_notes.py --resume --pause 2"
            )
        return 1
    finally:
        if driver is not None:
            driver.close()

    mode_labels = {
        "off": "vector only (Chroma)",
        "shell": "vector + Neo4j shell (no Groq)",
        "full": "vector + Neo4j + LLM extraction",
    }
    print(f"\n=== TEST session-note ingest: {config.display_name} ===")
    print(f"Config: {Path(config_path).resolve()}")
    print(f"Mode: {mode_labels[result.graph_mode]}")
    print(f"Entity: {result.entity_id}")
    print(f"Files ingested: {result.files_processed}")
    if result.files_skipped:
        print(f"Files skipped (resume): {result.files_skipped}")
    print(f"Chunks indexed: {result.chunks_indexed}")
    if result.graph_mode == "full":
        print(f"Entities extracted: {result.entities_extracted}")
    print(f"Chroma: {config.chroma_path(base=REPO_ROOT)}")
    print("\nSample query:")
    print(
        '  curl -s -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" '
        f'-d \'{{"question":"What TIPP skills were practiced in session 10?",'
        f'"doc_type_filter":"session_note","entity_id":"{result.entity_id}"}}\''
    )
    return 0


def _is_rate_limit(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "429" in message or "rate_limit" in message or "rate limit" in message


if __name__ == "__main__":
    sys.exit(main())
