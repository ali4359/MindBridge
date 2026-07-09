"""Tests for the config-driven RAGAS evaluation runner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda

from core.config import load_config
from core.evaluation import (
    aggregate_scores,
    build_ragas_samples,
    compare_to_targets,
    load_eval_dataset,
    ragas_results_path,
    resolve_metrics,
    run_evaluation,
    write_results_csv,
)


class _StaticRetriever(BaseRetriever):
    documents: list[Document]

    def _get_relevant_documents(self, query: str) -> list[Document]:
        return self.documents

    async def _aget_relevant_documents(self, query: str) -> list[Document]:
        return self.documents


def test_load_eval_dataset_mental_health() -> None:
    config = load_config("configs/mental_health.yaml")
    cases = load_eval_dataset(config)
    assert len(cases) == 20
    assert "question" in cases[0]
    assert "ground_truth" in cases[0]


def test_load_eval_dataset_legal() -> None:
    config = load_config("configs/legal.yaml")
    cases = load_eval_dataset(config)
    assert len(cases) == 10
    assert "question" in cases[0]
    assert "ground_truth" in cases[0]
    topics = " ".join(case["ground_truth"] for case in cases)
    assert "Hadley" in topics
    assert "UCC" in topics


def test_resolve_metrics_defaults() -> None:
    metrics = resolve_metrics(
        ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    )
    assert len(metrics) == 4


def test_resolve_metrics_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown RAGAS metrics"):
        resolve_metrics(["faithfulness", "not_a_metric"])


def test_build_ragas_samples_invokes_chain_and_retriever() -> None:
    config = load_config("configs/mental_health.yaml")
    cases = load_eval_dataset(config)[:1]
    retriever = _StaticRetriever(
        documents=[
            Document(
                page_content="SSRIs are first-line for depression.",
                metadata={"source": "nice-ng222-depression.pdf"},
            )
        ]
    )
    chain = RunnableLambda(lambda question: f"Answer to: {question}")

    rows = build_ragas_samples(config, chain, cases, retriever=retriever)
    assert rows[0]["user_input"] == cases[0]["question"]
    assert rows[0]["response"].startswith("Answer to:")
    assert rows[0]["retrieved_contexts"] == ["SSRIs are first-line for depression."]
    assert rows[0]["reference"] == cases[0]["ground_truth"]


def test_compare_to_targets_pass_fail() -> None:
    targets = {
        "faithfulness": 0.80,
        "answer_relevancy": 0.75,
    }
    passing = compare_to_targets(
        {"faithfulness": 0.81, "answer_relevancy": 0.76},
        targets,
    )
    assert passing == {"faithfulness": True, "answer_relevancy": True}

    failing = compare_to_targets(
        {"faithfulness": 0.79, "answer_relevancy": 0.76},
        targets,
    )
    assert failing["faithfulness"] is False
    assert failing["answer_relevancy"] is True


def test_write_results_csv_appends_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "mental_health_ragas_results.csv"
    write_results_csv(
        csv_path,
        {
            "faithfulness": 0.81,
            "answer_relevancy": 0.77,
        },
        timestamp="2026-07-09T10:00:00+00:00",
    )
    write_results_csv(
        csv_path,
        {
            "faithfulness": 0.83,
            "answer_relevancy": 0.78,
        },
        timestamp="2026-07-09T11:00:00+00:00",
    )

    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "timestamp,answer_relevancy,faithfulness"
    assert "0.8100" in lines[1]
    assert "0.8300" in lines[2]


def test_aggregate_scores_averages_per_metric() -> None:
    result = MagicMock()
    result._scores_dict = {
        "faithfulness": [0.8, 1.0],
        "answer_relevancy": [0.6, 0.8],
    }
    result.__getitem__ = lambda _self, key: result._scores_dict[key]

    scores = aggregate_scores(result)
    assert scores["faithfulness"] == pytest.approx(0.9)
    assert scores["answer_relevancy"] == pytest.approx(0.7)


def test_run_evaluation_writes_csv_and_returns_pass_fail(tmp_path: Path) -> None:
    config = load_config("configs/mental_health.yaml")
    cases = load_eval_dataset(config)[:2]
    retriever = _StaticRetriever(
        documents=[
            Document(page_content="Clinical context.", metadata={"source": "demo.pdf"}),
        ]
    )
    chain = RunnableLambda(lambda question: f"Grounded answer for {question[:20]}")

    mock_result = MagicMock()
    mock_result._scores_dict = {
        "faithfulness": [0.85, 0.82],
        "answer_relevancy": [0.80, 0.78],
        "context_precision": [0.75, 0.72],
        "context_recall": [0.74, 0.71],
    }
    mock_result.__getitem__ = lambda _self, key: mock_result._scores_dict[key]

    output_path = ragas_results_path(config, base=tmp_path)

    with (
        patch("core.evaluation.load_eval_dataset", return_value=cases),
        patch("core.evaluation.evaluate", return_value=mock_result),
        patch("core.evaluation.build_judge_llm") as judge_mock,
        patch("core.evaluation.LangchainEmbeddingsWrapper") as emb_mock,
    ):
        judge_mock.return_value = MagicMock()
        emb_mock.return_value = MagicMock()
        pass_fail = run_evaluation(
            config,
            chain,
            retriever=retriever,
            base_dir=tmp_path,
        )

    assert pass_fail["faithfulness"] is True
    assert pass_fail["answer_relevancy"] is True
    assert pass_fail["context_precision"] is True
    assert pass_fail["context_recall"] is True
    assert output_path.is_file()
    assert "faithfulness" in output_path.read_text(encoding="utf-8")


def test_evaluation_config_exposes_target_scores_alias() -> None:
    config = load_config("configs/mental_health.yaml")
    assert config.evaluation.target_scores == config.evaluation.ragas_target_scores
    assert "faithfulness" in config.evaluation.metrics
