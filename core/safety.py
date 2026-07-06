"""Config-driven safety review — regex flags and LLM classification."""

from __future__ import annotations

import re
from typing import Any, Optional

from langchain.chains import LLMChain
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from core.config import UseCaseConfig

_SAFETY_REVIEW_TEMPLATE = """You are a safety reviewer for content shown to: {audience}.

Flag the content as unsafe if it matches ANY rule below (including semantic equivalents):
{rules}

Also flag if the content involves any prohibited capability:
{prohibited}

Content to review:
{text}

{format_instructions}"""


class SafetyVerdict(BaseModel):
    """Structured JSON verdict from the safety LLM chain."""

    safe: bool
    violated_rules: list[str] = Field(default_factory=list)
    explanation: str = ""


def match_flag_patterns(text: str, config: UseCaseConfig) -> list[str]:
    """Return regex patterns from ``config.safety.flag_patterns`` that match ``text``."""
    matched: list[str] = []
    for pattern in config.safety.flag_patterns:
        if re.search(pattern, text):
            matched.append(pattern)
    return matched


def build_safety_chain(config: UseCaseConfig, llm: BaseChatModel) -> LLMChain:
    """Build an ``LLMChain`` that classifies content using config-defined safety rules."""
    safety = config.safety
    parser = JsonOutputParser(pydantic_object=SafetyVerdict)

    rules = "\n".join(f"- {pattern}" for pattern in safety.flag_patterns) or "- (none configured)"
    prohibited = ", ".join(safety.prohibited_capabilities) or "(none configured)"

    prompt = PromptTemplate(
        input_variables=["text"],
        partial_variables={
            "audience": safety.audience,
            "rules": rules,
            "prohibited": prohibited,
            "format_instructions": parser.get_format_instructions(),
        },
        template=_SAFETY_REVIEW_TEMPLATE,
    )
    return LLMChain(llm=llm, prompt=prompt, output_parser=parser)


def apply_safety(
    text: str,
    config: UseCaseConfig,
    llm: Optional[BaseChatModel] = None,
) -> dict[str, Any]:
    """Run regex flags first, then optional LLM review; return safe text or fallback."""
    if not config.safety.enabled:
        return {"safe": True, "text": text, "matched_patterns": [], "verdict": None}

    matched = match_flag_patterns(text, config)
    if matched:
        return {
            "safe": False,
            "text": config.safety.fallback_message,
            "matched_patterns": matched,
            "verdict": None,
        }

    if llm is None:
        return {"safe": True, "text": text, "matched_patterns": [], "verdict": None}

    chain = build_safety_chain(config, llm)
    raw = chain.invoke({"text": text})
    verdict = raw.get("text", raw) if isinstance(raw, dict) else raw
    if isinstance(verdict, dict) and not verdict.get("safe", True):
        return {
            "safe": False,
            "text": config.safety.fallback_message,
            "matched_patterns": [],
            "verdict": verdict,
        }

    return {"safe": True, "text": text, "matched_patterns": [], "verdict": verdict}
