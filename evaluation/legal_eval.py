"""Golden Q&A dataset for LexBridge legal-domain RAGAS evaluation.

Ten (question, ground_truth) pairs spanning case law, contract clauses, and statutory
interpretation — proves the evaluation suite is platform-generic, not mental-health-specific.
"""

from __future__ import annotations

EVAL_CASES: list[dict[str, str]] = [
    # --- Case law (doc_type: case_law) ---
    {
        "question": (
            "What is the foreseeability rule for contract damages in Hadley v. Baxendale?"
        ),
        "ground_truth": (
            "Hadley v. Baxendale limits recoverable contract damages to losses arising "
            "naturally from the breach in the ordinary course of events, or losses in the "
            "reasonable contemplation of both parties at the time of contracting."
        ),
    },
    {
        "question": (
            "What standard applies when a court decides summary judgment under Celotex?"
        ),
        "ground_truth": (
            "Under Celotex Corp. v. Catrett, summary judgment is proper when the moving "
            "party shows no genuine dispute of material fact and is entitled to judgment "
            "as a matter of law after adequate time for discovery."
        ),
    },
    {
        "question": (
            "What must the non-moving party show to survive a motion for summary judgment?"
        ),
        "ground_truth": (
            "The non-moving party must present specific facts, by affidavit or admissible "
            "evidence, that create a genuine dispute of material fact; mere allegations or "
            "speculation are insufficient to defeat summary judgment."
        ),
    },
    {
        "question": (
            "When are consequential damages recoverable for breach of contract after Hadley?"
        ),
        "ground_truth": (
            "Consequential damages are recoverable only if the breaching party knew or "
            "should have known of the special circumstances making such loss probable at "
            "the time the contract was made."
        ),
    },
    # --- Contract clauses (doc_type: contract) ---
    {
        "question": (
            "How do courts evaluate whether a non-compete clause is reasonable?"
        ),
        "ground_truth": (
            "Courts assess whether a non-compete is reasonable in duration, geographic "
            "scope, and the legitimate business interest it protects, balancing the "
            "employer's protectable interest against the employee's right to earn a "
            "livelihood."
        ),
    },
    {
        "question": (
            "What is the typical scope of an indemnification clause in a vendor contract?"
        ),
        "ground_truth": (
            "An indemnification clause generally requires one party to defend and hold the "
            "other harmless against third-party claims, losses, and expenses arising from "
            "specified conduct such as negligence, breach, or infringement tied to the "
            "indemnifying party's performance."
        ),
    },
    {
        "question": (
            "What elements must a plaintiff prove in a breach of contract claim?"
        ),
        "ground_truth": (
            "A plaintiff must prove a valid contract, performance or legal excuse for "
            "non-performance, the defendant's material breach, and resulting damages."
        ),
    },
    # --- Statutory interpretation (doc_type: statute) ---
    {
        "question": (
            "What does UCC Article 2 govern in commercial transactions?"
        ),
        "ground_truth": (
            "UCC Article 2 governs contracts for the sale of goods, including formation, "
            "performance obligations, warranties, risk of loss, and remedies for breach "
            "between merchants and other parties."
        ),
    },
    {
        "question": (
            "What is required for mutual assent under the Restatement of Contracts?"
        ),
        "ground_truth": (
            "Mutual assent requires an offer that manifests willingness to be bound on "
            "definite terms and an acceptance that mirrors the essential terms, creating "
            "a meeting of the minds on the contract's material provisions."
        ),
    },
    {
        "question": (
            "What limitations period applies to contract claims under the UCC?"
        ),
        "ground_truth": (
            "The UCC generally imposes a four-year statute of limitations for actions "
            "arising from a contract for the sale of goods, which the parties may reduce "
            "by agreement to not less than one year but may not extend."
        ),
    },
]
