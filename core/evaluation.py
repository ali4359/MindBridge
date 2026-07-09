"""Config-driven RAGAS evaluation runner."""

from __future__ import annotations

import csv
import importlib
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from core.config import UseCaseConfig
from core.embeddings import get_embeddings
from core.llm import REPO_ROOT, build_llm

logger = logging.getLogger(__name__)

DEFAULT_METRICS: tuple[str, ...] = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)

METRIC_BY_NAME: dict[str, Any] = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall,
}


def load_eval_dataset(config: UseCaseConfig) -> list[dict[str, Any]]:
    """Load golden Q&A cases from ``evaluation/{name}_eval.py``."""
    module_name = f"evaluation.{config.name}_eval"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise FileNotFoundError(
            f"Evaluation dataset module not found: {module_name}. "
            f"Expected evaluation/{config.name}_eval.py"
        ) from exc

    cases = getattr(module, "EVAL_CASES", None)
    if not cases:
        raise ValueError(f"{module_name} must define a non-empty EVAL_CASES list")
    return list(cases)


def resolve_metrics(metric_names: list[str]) -> list[Any]:
    """Map configured metric names to RAGAS metric objects."""
    if not metric_names:
        metric_names = list(DEFAULT_METRICS)

    resolved: list[Any] = []
    unknown: list[str] = []
    for name in metric_names:
        metric = METRIC_BY_NAME.get(name)
        if metric is None:
            unknown.append(name)
            continue
        resolved.append(metric)

    if unknown:
        raise ValueError(
            f"Unknown RAGAS metrics: {unknown}. "
            f"Supported: {sorted(METRIC_BY_NAME)}"
        )
    if not resolved:
        raise ValueError("At least one RAGAS metric must be configured")
    return resolved


def build_judge_llm(config: UseCaseConfig) -> LangchainLLMWrapper:
    """Configure RAGAS with a Groq-backed LangChain LLM judge."""
    groq_llm: ChatGroq = build_llm(config)
    return LangchainLLMWrapper(groq_llm)


def ragas_results_path(config: UseCaseConfig, *, base: Path = REPO_ROOT) -> Path:
    """Return the per-use-case CSV path for RAGAS scores."""
    return base / "evaluation" / f"{config.name}_ragas_results.csv"


def _mean_score(values: list[Any]) -> float:
    numeric = [
        float(value)
        for value in values
        if value is not None and not (isinstance(value, float) and math.isnan(value))
    ]
    if not numeric:
        return float("nan")
    return sum(numeric) / len(numeric)


def aggregate_scores(result: Any) -> dict[str, float]:
    """Extract mean scores per metric from a RAGAS ``EvaluationResult``."""
    scores: dict[str, float] = {}
    for metric_name in result._scores_dict:
        scores[metric_name] = _mean_score(result[metric_name])
    return scores


def compare_to_targets(
    scores: dict[str, float],
    targets: dict[str, float],
) -> dict[str, bool]:
    """Return pass/fail per metric against configured target thresholds."""
    results: dict[str, bool] = {}
    for metric_name, target in targets.items():
        actual = scores.get(metric_name)
        if actual is None or math.isnan(actual):
            results[metric_name] = False
            continue
        results[metric_name] = actual >= target
    return results


def write_results_csv(
    path: Path,
    scores: dict[str, float],
    *,
    timestamp: Optional[str] = None,
) -> None:
    """Append a score row to the per-use-case RAGAS results CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row_timestamp = timestamp or datetime.now(timezone.utc).isoformat()

    metric_names = sorted(scores)
    fieldnames = ["timestamp", *metric_names]

    file_exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = {"timestamp": row_timestamp}
        row.update({name: f"{scores[name]:.4f}" for name in metric_names})
        writer.writerow(row)


def build_ragas_samples(
    config: UseCaseConfig,
    chain: Runnable[str, str],
    cases: list[dict[str, Any]],
    *,
    retriever: Optional[BaseRetriever] = None,
) -> list[dict[str, Any]]:
    """Run the RAG chain over golden questions and assemble RAGAS rows."""
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        question = case.get("question") or case.get("user_input")
        ground_truth = case.get("ground_truth") or case.get("reference")
        if not question or not ground_truth:
            raise ValueError(
                f"Eval case {index} must include question and ground_truth fields"
            )

        answer = chain.invoke(question)
        if retriever is not None:
            documents = retriever.invoke(question)
            contexts = [doc.page_content for doc in documents]
        else:
            contexts = list(case.get("contexts") or case.get("retrieved_contexts") or [])

        rows.append(
            {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": ground_truth,
            }
        )
    return rows


def run_evaluation(
    config: UseCaseConfig,
    chain: Runnable[str, str],
    *,
    retriever: Optional[BaseRetriever] = None,
    base_dir: Path = REPO_ROOT,
) -> dict[str, bool]:
    """Run RAGAS for the active use case and return pass/fail per metric.

    Reads golden Q&A from ``evaluation/{name}_eval.py``, configures RAGAS with
    ``LangchainLLMWrapper(ChatGroq(...))`` as the LLM judge, compares scores to
    ``config.evaluation.target_scores``, and writes
    ``evaluation/{name}_ragas_results.csv``.
    """
    cases = load_eval_dataset(config)
    metric_names = config.evaluation.metrics or list(DEFAULT_METRICS)
    metrics = resolve_metrics(metric_names)

    samples = build_ragas_samples(
        config,
        chain,
        cases,
        retriever=retriever,
    )
    dataset = EvaluationDataset.from_list(samples)

    judge = build_judge_llm(config)
    embeddings = LangchainEmbeddingsWrapper(get_embeddings(config))

    logger.info(
        "Running RAGAS for %s (%d cases, metrics=%s)",
        config.name,
        len(samples),
        metric_names,
    )
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=judge,
        embeddings=embeddings,
        show_progress=False,
        raise_exceptions=False,
    )

    scores = aggregate_scores(result)
    targets = config.evaluation.target_scores
    pass_fail = compare_to_targets(scores, targets)

    output_path = ragas_results_path(config, base=base_dir)
    write_results_csv(output_path, scores)

    logger.info("RAGAS scores for %s: %s", config.name, scores)
    logger.info("RAGAS pass/fail for %s: %s", config.name, pass_fail)
    return pass_fail
