#!/usr/bin/env python3
"""RAGAS evaluation CLI — run golden-set eval for any use-case profile."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from core.chain import build_chain
from core.config import load_config
from core.evaluation import (
    format_score_table,
    format_target_warnings,
    load_eval_dataset,
    run_evaluation,
)
from core.llm import build_llm
from core.retriever import build_retriever
from core.vectorstore import load_vectorstore

REPO_ROOT = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run config-driven RAGAS evaluation against a use-case golden set.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="YAML profile path (e.g. configs/mental_health.yaml)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress RAGAS progress output",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Evaluate only the first N golden-set cases (reduces API load)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help=(
            "Override configured RAGAS metrics for this run "
            "(e.g. faithfulness answer_relevancy)"
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv(REPO_ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    args = build_parser().parse_args(argv)
    if args.max_cases is not None and args.max_cases <= 0:
        logger.error("--max-cases must be a positive integer")
        return 1

    config_path = Path(args.config)
    if not config_path.is_file():
        logger.error("Config file not found: %s", config_path)
        return 1

    try:
        config = load_config(str(config_path))
        cases = load_eval_dataset(config)
        vectorstore = load_vectorstore(config)
        llm = build_llm(config)
        retriever = build_retriever(config, vectorstore, llm=llm)
        chain = build_chain(config, retriever, llm)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    print(f"\n=== RAGAS evaluation: {config.display_name} ===")
    print(f"Config: {config_path.resolve()}")
    selected_cases = min(len(cases), args.max_cases) if args.max_cases else len(cases)
    selected_metrics = args.metrics or config.evaluation.metrics
    print(f"Cases: {selected_cases} (of {len(cases)})")
    print(f"Metrics: {', '.join(selected_metrics)}\n")

    try:
        result = run_evaluation(
            config,
            chain,
            retriever=retriever,
            show_progress=not args.quiet,
            max_cases=args.max_cases,
            metric_names_override=args.metrics,
        )
    except Exception as exc:  # noqa: BLE001 — surface eval failures to the CLI user
        logger.error("RAGAS evaluation failed: %s", exc)
        return 1

    targets = {
        metric_name: threshold
        for metric_name, threshold in config.evaluation.target_scores.items()
        if metric_name in selected_metrics
    }
    print(format_score_table(result.scores, targets, result.pass_fail))
    print(f"\nResults saved to: {result.csv_path.resolve()}")

    warnings = format_target_warnings(result.scores, targets, result.pass_fail)
    if warnings:
        print(f"\n{warnings}")
        return 1

    print("\nAll metrics met target thresholds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
