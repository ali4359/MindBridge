"""Tool-calling agent factory — prompt and tool wiring, no domain logic.

The system prompt comes entirely from ``config.agent.system_prompt``; this
module never hard-codes domain text. Tools are the fixed set from
``core.tools.entity_tools``, wired to the given source system and cache via
``init_tools`` before the agent is built.
"""

from __future__ import annotations

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable

from core.config import UseCaseConfig
from core.tools import entity_tools
from core.tools.cache import HybridCache
from core.tools.entity_tools import ENTITY_TOOLS
from core.tools.source_system import SourceSystemInterface


def build_agent(
    config: UseCaseConfig,
    llm: BaseChatModel,
    source_system: SourceSystemInterface,
    cache: HybridCache,
    *,
    retriever: BaseRetriever | None = None,
) -> AgentExecutor:
    """Build a tool-calling ``AgentExecutor`` whose prompt comes entirely from ``config.agent``."""
    if config.agent is None or not config.agent.system_prompt.strip():
        raise ValueError("agent.system_prompt is required in config to build an agent")

    entity_tools.init_tools(source_system, cache, retriever)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", config.agent.system_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm, ENTITY_TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=ENTITY_TOOLS,
        max_iterations=5,
        return_intermediate_steps=True,
        verbose=True,
    )


def build_direct_answer_chain(config: UseCaseConfig, llm: BaseChatModel) -> Runnable:
    """Build a single-call chain that answers from pre-loaded context, with no tool round trip.

    Used on a full cache hit: the entity context is already known, so there is
    nothing for a tool call to fetch — one generation call is enough.
    """
    if config.agent is None or not config.agent.system_prompt.strip():
        raise ValueError("agent.system_prompt is required in config to build an agent")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", config.agent.system_prompt),
            ("human", "## Pre-loaded entity context\n{context}\n\n## Question\n{question}"),
        ]
    )
    return prompt | llm | StrOutputParser()


async def answer_with_context(
    config: UseCaseConfig,
    llm: BaseChatModel,
    question: str,
    context: str,
) -> str:
    """Answer ``question`` from pre-loaded ``context`` in exactly one LLM call."""
    chain = build_direct_answer_chain(config, llm)
    return await chain.ainvoke({"context": context, "question": question})
