"""Tests for config-driven safety review."""

from __future__ import annotations

import json

from langchain.chains import LLMChain
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from core import load_config
from core.config import UseCaseConfig
from core.safety import apply_safety, build_safety_chain, match_flag_patterns


def test_match_flag_patterns_crisis_text() -> None:
    config = load_config("configs/mental_health.yaml")
    matched = match_flag_patterns("I want to kill myself tonight.", config)
    assert matched


def test_match_flag_patterns_prescriptive_medical_advice() -> None:
    config = load_config("configs/mental_health.yaml")
    matched = match_flag_patterns(
        "You should start taking 20mg antidepressant medication daily.",
        config,
    )
    assert matched


def test_match_flag_patterns_safe_clinical_text() -> None:
    config = load_config("configs/mental_health.yaml")
    matched = match_flag_patterns(
        "NICE recommends CBT as a first-line psychological treatment.",
        config,
    )
    assert not matched


def test_apply_safety_returns_fallback_on_regex_match() -> None:
    config = load_config("configs/mental_health.yaml")
    result = apply_safety("I want to end my life.", config)

    assert result["safe"] is False
    assert result["text"] == config.safety.fallback_message
    assert result["matched_patterns"]


def test_build_safety_chain_returns_llm_chain() -> None:
    config = load_config("configs/mental_health.yaml")
    llm = FakeListChatModel(
        responses=[json.dumps({"safe": True, "violated_rules": [], "explanation": "ok"})]
    )
    chain = build_safety_chain(config, llm)

    assert isinstance(chain, LLMChain)
    raw = chain.invoke({"text": "CBT is a recommended psychotherapy."})
    verdict = raw["text"] if isinstance(raw.get("text"), dict) else raw
    assert verdict["safe"] is True


def test_legal_config_flags_guaranteed_outcome_claims() -> None:
    config = UseCaseConfig.model_validate(
        {
            "name": "legal",
            "display_name": "Legal Assistant",
            "safety": {
                "audience": "licensed attorneys",
                "fallback_message": "I cannot guarantee case outcomes or provide definitive legal predictions.",
                "flag_patterns": [
                    r"(?i)\b(guaranteed to win|will definitely win|certain victory|assured outcome)\b",
                    r"(?i)\b(100% chance of winning|no risk of losing)\b",
                ],
            },
        }
    )
    matched = match_flag_patterns("We are guaranteed to win this lawsuit.", config)
    assert matched

    result = apply_safety("We are guaranteed to win this lawsuit.", config)
    assert result["safe"] is False
    assert "cannot guarantee" in result["text"].lower()
