"""Golden Q&A dataset for MindBridge mental-health RAGAS evaluation.

Twenty (question, ground_truth) pairs spanning guideline, workbook, and
session_note sources. Targets: faithfulness > 0.80, answer_relevancy > 0.75.
"""

from __future__ import annotations

EVAL_CASES: list[dict[str, str]] = [
    # --- NICE / WHO guidelines (doc_type: guideline) ---
    {
        "question": (
            "What does the NICE guideline recommend as first-line pharmacological "
            "treatment for depression in adults?"
        ),
        "ground_truth": (
            "NICE NG222 recommends an SSRI as first-line pharmacological treatment "
            "for depression in adults, typically sertraline or citalopram, prescribed "
            "at a licensed dose alongside information about depression and, when "
            "appropriate, a structured psychological intervention."
        ),
    },
    {
        "question": (
            "What stepped-care approach does NICE recommend for generalized anxiety disorder?"
        ),
        "ground_truth": (
            "NICE CG113 recommends a stepped-care model for GAD: education and active "
            "monitoring at step 2, low-intensity psychological interventions at step 3, "
            "and either CBT or an SSRI/SNRI at step 4 when symptoms persist."
        ),
    },
    {
        "question": (
            "What does the WHO mhGAP guide recommend for managing depression in primary care?"
        ),
        "ground_truth": (
            "The WHO mhGAP intervention guide recommends psychoeducation, structured "
            "psychological interventions such as behavioral activation or problem "
            "management, antidepressant medication when indicated, and regular follow-up "
            "in primary care."
        ),
    },
    {
        "question": (
            "What does NICE say about prescribing benzodiazepines for generalized anxiety disorder?"
        ),
        "ground_truth": (
            "NICE CG113 states benzodiazepines are not recommended for treating GAD "
            "because of the risk of dependence and withdrawal; if used in crisis they "
            "should be limited to short-term prescription at the lowest effective dose."
        ),
    },
    {
        "question": (
            "Does NICE recommend combining antidepressant medication with psychological "
            "therapy for moderate to severe depression?"
        ),
        "ground_truth": (
            "NICE NG222 recommends offering combination treatment — an antidepressant "
            "plus a high-intensity psychological intervention such as CBT — for adults "
            "with moderate or severe depression, particularly when an adequate trial of "
            "either treatment alone has been insufficient."
        ),
    },
    # --- DBT workbooks (doc_type: workbook / research) ---
    {
        "question": (
            "What DBT distress tolerance skills can I teach a client for crisis survival?"
        ),
        "ground_truth": (
            "DBT distress tolerance crisis skills include TIPP (Temperature change, "
            "Intense exercise, Paced breathing, Progressive muscle relaxation), "
            "distraction, self-soothing with the five senses, improving the moment, "
            "and pros and cons of acting on crisis urges."
        ),
    },
    {
        "question": (
            "What are the PLEASE skills in DBT emotion regulation?"
        ),
        "ground_truth": (
            "PLEASE skills reduce emotional vulnerability by treating PhysicaL illness, "
            "balancing Eating, avoiding mood-Altering substances, balancing Sleep, and "
            "getting Exercise."
        ),
    },
    {
        "question": (
            "How do I teach a client the DEAR MAN skill for interpersonal effectiveness?"
        ),
        "ground_truth": (
            "DEAR MAN structures an assertive request: Describe the situation, Express "
            "feelings, Assert wants clearly, Reinforce benefits; stay Mindful, Appear "
            "confident, and Negotiate when needed."
        ),
    },
    {
        "question": (
            "What is the difference between observe, describe, and participate in DBT mindfulness?"
        ),
        "ground_truth": (
            "In DBT mindfulness, Observe means noticing internal and external experience "
            "without judgment; Describe means putting words to observations without "
            "interpretation; Participate means fully engaging in the present activity "
            "with awareness."
        ),
    },
    # --- CBT workbooks (doc_type: workbook / research) ---
    {
        "question": (
            "How can I use a thought record worksheet with a client who has "
            "generalized anxiety?"
        ),
        "ground_truth": (
            "Guide the client through a thought record: identify the Situation, "
            "automatic Thought, Emotion and rating, Evidence for and against the "
            "thought, then generate a Balanced alternative thought and re-rate the "
            "emotion."
        ),
    },
    {
        "question": (
            "What is behavioral activation and how is it used for depression in CBT?"
        ),
        "ground_truth": (
            "Behavioral activation schedules rewarding or valued activities to reverse "
            "the depression cycle of withdrawal and low mood; the therapist helps the "
            "client monitor activity and mood, set graded tasks, and address avoidance."
        ),
    },
    {
        "question": (
            "What are the steps in a CBT problem-solving worksheet?"
        ),
        "ground_truth": (
            "A CBT problem-solving worksheet typically follows: define the Problem, "
            "brainstorm possible Solutions, list Pros and cons, choose the best Option, "
            "plan specific Actions, and Review the outcome."
        ),
    },
    {
        "question": (
            "What unhelpful thinking habits should a CBT therapist help clients identify?"
        ),
        "ground_truth": (
            "Common unhelpful thinking habits include all-or-nothing thinking, "
            "catastrophizing, mind reading, emotional reasoning, should statements, "
            "labelling, and discounting the positive."
        ),
    },
    {
        "question": (
            "How do behavioral experiments differ from thought records in CBT?"
        ),
        "ground_truth": (
            "A thought record examines evidence for beliefs in session; a behavioral "
            "experiment tests a specific prediction through planned real-world action, "
            "records what happened, and compares results to the original belief."
        ),
    },
    # --- Medication-focused (doc_type: guideline) ---
    {
        "question": (
            "What monitoring is recommended when starting an SSRI for depression?"
        ),
        "ground_truth": (
            "When starting an SSRI, NICE recommends reviewing the patient after about "
            "two weeks for side effects and emerging suicidal ideation, then regularly "
            "assessing response, tolerability, and adherence over the first months of "
            "treatment."
        ),
    },
    {
        "question": (
            "What does NICE recommend before stopping antidepressant medication?"
        ),
        "ground_truth": (
            "NICE advises tapering antidepressants gradually over at least four weeks "
            "and longer after higher doses or prolonged use, with monitoring for "
            "discontinuation symptoms and relapse, plus a clear follow-up plan."
        ),
    },
    # --- Patient-specific / session notes (doc_type: session_note) ---
    {
        "question": (
            "What homework was assigned in the depression session focused on behavioral "
            "activation scheduling?"
        ),
        "ground_truth": (
            "The session note documents homework to complete a daily activity schedule, "
            "rate mood before and after each activity, and schedule at least one "
            "pleasant and one mastery task for the coming week."
        ),
    },
    {
        "question": (
            "Which DBT crisis skills were practiced with the client who used TIPP and "
            "self-soothe during distress?"
        ),
        "ground_truth": (
            "The session note records practicing TIPP (paced breathing and cold "
            "temperature on the face) plus self-soothe using music and a scented "
            "lotion, with homework to use the skills card when urges peaked."
        ),
    },
    {
        "question": (
            "How was motivational interviewing used with the client ambivalent about "
            "antidepressant adherence?"
        ),
        "ground_truth": (
            "The session note describes exploring ambivalence with open questions, "
            "reflecting change and sustain talk, summarizing pros and cons of "
            "medication, and agreeing a small adherence experiment before the next "
            "psychiatry review."
        ),
    },
    {
        "question": (
            "What grounding and safety strategies were reviewed for the client with PTSD "
            "after a trauma trigger?"
        ),
        "ground_truth": (
            "The session note documents grounding with five-senses orientation, paced "
            "breathing, reviewing the written safety plan, confirming no current suicidal "
            "intent, and assigning practice of grounding when noticing early trigger "
            "signs."
        ),
    },
]
