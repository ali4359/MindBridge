"""Golden Q&A dataset for legal-domain RAGAS evaluation."""

from __future__ import annotations

EVAL_CASES: list[dict[str, str]] = [
    {
        "question": "What is the foreseeability rule for contract damages in Hadley v. Baxendale?",
        "ground_truth": (
            "Hadley v. Baxendale limits recoverable contract damages to those arising "
            "naturally from the breach or those in the reasonable contemplation of both "
            "parties at the time of contracting."
        ),
    },
    {
        "question": "What standard applies when a court decides summary judgment under Celotex?",
        "ground_truth": (
            "Under Celotex, summary judgment is appropriate when the moving party shows "
            "no genuine dispute of material fact and is entitled to judgment as a matter "
            "of law; the non-moving party must present specific facts creating a triable issue."
        ),
    },
    {
        "question": "What elements must a plaintiff prove in a breach of contract claim?",
        "ground_truth": (
            "A plaintiff must prove formation of a valid contract, performance or excuse, "
            "breach by the defendant, and resulting damages."
        ),
    },
    {
        "question": "How do courts evaluate whether a non-compete clause is reasonable?",
        "ground_truth": (
            "Courts assess whether a non-compete is reasonable in geographic scope, "
            "duration, and the legitimate business interest it protects, balancing "
            "employer interests against the employee's right to work."
        ),
    },
    {
        "question": "What does UCC Article 2 govern in commercial transactions?",
        "ground_truth": (
            "UCC Article 2 governs contracts for the sale of goods, including formation, "
            "performance, warranties, risk of loss, and remedies for breach between merchants "
            "and other parties."
        ),
    },
]
