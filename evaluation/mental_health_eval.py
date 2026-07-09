"""Golden Q&A dataset for MindBridge mental-health RAGAS evaluation."""

from __future__ import annotations

EVAL_CASES: list[dict[str, str]] = [
    {
        "question": (
            "What does the NICE guideline recommend as first-line pharmacological "
            "treatment for depression in adults?"
        ),
        "ground_truth": (
            "NICE recommends SSRIs as first-line pharmacological treatment for "
            "depression in adults, typically starting with a generic SSRI such as "
            "sertraline or citalopram, alongside psychological therapy."
        ),
    },
    {
        "question": (
            "What DBT distress tolerance skills can I teach a client for crisis survival?"
        ),
        "ground_truth": (
            "DBT distress tolerance crisis skills include TIPP (Temperature, Intense "
            "exercise, Paced breathing, Progressive muscle relaxation), distraction "
            "techniques, self-soothing, and improving the moment."
        ),
    },
    {
        "question": (
            "How can I use a thought record worksheet with a client who has "
            "generalized anxiety?"
        ),
        "ground_truth": (
            "Use a thought record to identify the triggering situation, automatic "
            "anxious thought, emotion and intensity, evidence for and against the "
            "thought, and a balanced alternative thought to reduce worry."
        ),
    },
    {
        "question": "What stepped-care approach does NICE recommend for generalized anxiety disorder?",
        "ground_truth": (
            "NICE recommends a stepped-care model for GAD: psychoeducation and "
            "active monitoring at step 2, low-intensity psychological interventions "
            "at step 3, and CBT or an SSRI/SNRI at step 4 for persistent symptoms."
        ),
    },
    {
        "question": "What does the WHO mhGAP guide recommend for depression in primary care?",
        "ground_truth": (
            "The WHO mhGAP intervention guide recommends psychoeducation, structured "
            "psychological interventions such as behavioral activation or problem "
            "management, and antidepressant medication when indicated, with follow-up "
            "in primary care."
        ),
    },
]
