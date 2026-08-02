"""Keyword-based query router: decides whether a query goes to the RAG chain or the agent.

Zero LLM cost — routing is a config-driven keyword match, not a model call.
The signal lists live entirely in ``config.router`` (per-profile YAML); this
module stays domain-blind and never hard-codes vocabulary from any one
deployment.
"""

from __future__ import annotations

import logging
from typing import Optional

from core.config import UseCaseConfig

logger = logging.getLogger(__name__)


def route_query(question: str, entity_id: Optional[str], config: UseCaseConfig) -> str:
    """Return ``'rag'`` or ``'agent'`` for the given question.

    Rules, in priority order:
    1. No ``entity_id`` → always ``'rag'``.
    2. Any ``config.router.rag_signals`` phrase present in the question → ``'rag'``.
    3. Any ``config.router.agent_signals`` phrase present in the question → ``'agent'``.
    4. Otherwise (an entity_id is present but no signal matched) → ``'agent'``.
    """
    router_config = config.router
    rag_signals = router_config.rag_signals if router_config else []
    agent_signals = router_config.agent_signals if router_config else []

    lowered = question.lower()

    if not entity_id:
        route = "rag"
    elif any(signal.lower() in lowered for signal in rag_signals):
        route = "rag"
    elif any(signal.lower() in lowered for signal in agent_signals):
        route = "agent"
    else:
        route = "agent"

    logger.info(f"Route: {route} | Q: {question[:50]}")
    return route
