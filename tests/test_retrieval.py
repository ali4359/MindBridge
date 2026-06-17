"""Retrieval precision baseline and doc-type filter tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from retrieval.filters import (
    GUIDELINE_DOC_TYPE,
    RESEARCH_DOC_TYPE,
    SESSION_DOC_TYPE,
    filter_by_doc_type,
)
from retrieval.hybrid import (
    create_reranked_ensemble_retriever,
    guideline_retriever,
    research_retriever,
    session_retriever,
)

# Week 5 RAGAS comparison baseline — update after pipeline changes.
RETRIEVAL_PRECISION_THRESHOLD = 8
RETRIEVAL_CASE_COUNT = 10


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    expected_source: str


# Manually curated (query, expected_source) pairs over the Week 1 demo corpus.
RETRIEVAL_BASELINE_CASES: tuple[RetrievalCase, ...] = (
    RetrievalCase(
        "DBT distress tolerance TIPP skills",
        "dbt-skills-workbook-mckay.pdf",
    ),
    RetrievalCase(
        "NICE guideline antidepressant first line treatment",
        "nice-ng222-depression.pdf",
    ),
    RetrievalCase(
        "generalized anxiety disorder stepped care NICE",
        "nice-cg113-anxiety.pdf",
    ),
    RetrievalCase(
        "mhGAP depression intervention primary care",
        "who-mhgap-intervention-guide-v2.pdf",
    ),
    RetrievalCase(
        "problem solving worksheet getselfhelp",
        "cbt-worksheets-getselfhelp.pdf",
    ),
    RetrievalCase(
        "Think CBT workbook behavioral experiments",
        "think-cbt-workbook.pdf",
    ),
    RetrievalCase(
        "dialectical behavior therapy emotion regulation",
        "dbt-skills-workbook-mckay.pdf",
    ),
    RetrievalCase(
        "NICE CG113 benzodiazepine anxiety",
        "nice-cg113-anxiety.pdf",
    ),
    RetrievalCase(
        "WHO mhGAP epilepsy management",
        "who-mhgap-intervention-guide-v2.pdf",
    ),
    RetrievalCase(
        "NICE NG222 SSRIs depression adults",
        "nice-ng222-depression.pdf",
    ),
)


def test_filter_by_doc_type_returns_chroma_where_clause() -> None:
    assert filter_by_doc_type(GUIDELINE_DOC_TYPE) == {"doc_type": "guideline"}
    assert filter_by_doc_type(SESSION_DOC_TYPE) == {"doc_type": "session_note"}
    assert filter_by_doc_type(RESEARCH_DOC_TYPE) == {"doc_type": "research"}


def test_baseline_case_count() -> None:
    assert len(RETRIEVAL_BASELINE_CASES) == RETRIEVAL_CASE_COUNT


@pytest.mark.parametrize("case", RETRIEVAL_BASELINE_CASES, ids=lambda c: c.expected_source)
def test_retrieval_precision_baseline(case: RetrievalCase, require_chroma_index: None) -> None:
    """Expected source must appear in top-3 for hybrid + FlashRank retrieval."""
    retriever = create_reranked_ensemble_retriever()
    results = retriever.invoke(case.query)
    assert results, f"No results for query: {case.query!r}"

    top_sources = [doc.metadata.get("source") for doc in results[:3]]
    assert case.expected_source in top_sources, (
        f"Query {case.query!r}: expected {case.expected_source!r} in top-3, got {top_sources!r}"
    )


def test_retrieval_precision_meets_threshold(require_chroma_index: None) -> None:
    """Aggregate baseline: at least 8/10 queries hit the expected source in top-3."""
    retriever = create_reranked_ensemble_retriever()
    hits = 0
    misses: list[str] = []

    for case in RETRIEVAL_BASELINE_CASES:
        top_sources = [
            doc.metadata.get("source")
            for doc in retriever.invoke(case.query)[:3]
        ]
        if case.expected_source in top_sources:
            hits += 1
        else:
            misses.append(f"{case.query!r} -> {top_sources}")

    assert hits >= RETRIEVAL_PRECISION_THRESHOLD, (
        f"Retrieval precision {hits}/{RETRIEVAL_CASE_COUNT} below threshold "
        f"{RETRIEVAL_PRECISION_THRESHOLD}. Misses: {misses}"
    )


def test_guideline_retriever_only_returns_guidelines(require_chroma_index: None) -> None:
    retriever = guideline_retriever()
    results = retriever.invoke("depression treatment recommendations")
    assert results
    assert all(doc.metadata.get("doc_type") == GUIDELINE_DOC_TYPE for doc in results)
    assert all(
        doc.metadata.get("source", "").startswith(("nice-", "who-"))
        for doc in results
    )


def test_research_retriever_only_returns_research(require_chroma_index: None) -> None:
    retriever = research_retriever()
    results = retriever.invoke("DBT mindfulness skills")
    assert results
    assert all(doc.metadata.get("doc_type") == RESEARCH_DOC_TYPE for doc in results)


def test_session_retriever_applies_filter(require_chroma_index: None) -> None:
    retriever = session_retriever()
    results = retriever.invoke("therapy session homework plan")
    assert all(doc.metadata.get("doc_type") == SESSION_DOC_TYPE for doc in results)
