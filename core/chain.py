"""Config-driven LCEL RAG chain — no domain-specific prompt text."""

from __future__ import annotations

import re
from typing import Optional

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough
from neo4j import Driver

from core.config import UseCaseConfig


def format_retrieved_context(documents: list[Document], config: UseCaseConfig) -> str:
    """Format retrieved chunks using ``config.prompts.context_chunk_template``."""
    if not documents:
        return config.output_schema.empty_context_message

    template = config.prompts.context_chunk_template
    if not template:
        raise ValueError("prompts.context_chunk_template is required in config")

    blocks: list[str] = []
    for index, doc in enumerate(documents, start=1):
        meta = doc.metadata
        blocks.append(
            template.format(
                index=index,
                source=meta.get("source", "unknown"),
                page_number=meta.get("page_number", "?"),
                doc_type=meta.get("doc_type", "unknown"),
                content=doc.page_content.strip(),
            )
        )
    return "\n\n".join(blocks)


def format_entity_graph(
    graph: dict[str, list[str]],
    config: UseCaseConfig,
) -> str:
    """Format a Neo4j entity subgraph as structured prompt context."""
    if not graph:
        return config.prompts.empty_entity_context

    lines: list[str] = []
    for label, values in graph.items():
        if values:
            lines.append(f"- {label}: {', '.join(str(v) for v in values)}")
        else:
            lines.append(f"- {label}: (none)")

    return "\n".join(lines) if lines else config.prompts.empty_entity_context


def load_entity_context(
    *,
    entity_id: Optional[str],
    config: UseCaseConfig,
    driver: Optional[Driver] = None,
) -> str:
    """Load and format graph context for ``entity_id``, or the empty placeholder."""
    if not entity_id or config.graph_schema is None:
        return config.prompts.empty_entity_context

    from core.graph import get_graph_driver, read_entity_graph

    owns_driver = driver is None
    resolved = driver if driver is not None else get_graph_driver()
    try:
        graph = read_entity_graph(entity_id, config, resolved)
        return format_entity_graph(graph, config)
    finally:
        if owns_driver:
            resolved.close()


def _compose_human_template(config: UseCaseConfig) -> str:
    """Assemble the human message from config prompt headers only."""
    prompts = config.prompts
    sections: list[str] = []

    if prompts.entity_context_header.strip():
        sections.append(f"{prompts.entity_context_header.strip()}\n{{entity_context}}")

    if not prompts.knowledge_header.strip():
        raise ValueError("prompts.knowledge_header is required in config")
    sections.append(f"{prompts.knowledge_header.strip()}\n{{context}}")

    if prompts.question_header.strip():
        sections.append(f"{prompts.question_header.strip()}\n{{question}}")
    else:
        sections.append("{question}")

    if prompts.response_requirements.strip():
        sections.append(prompts.response_requirements.strip())

    return "\n\n".join(sections)


def build_prompt(config: UseCaseConfig) -> ChatPromptTemplate:
    """Build a ``ChatPromptTemplate`` entirely from ``config.prompts``."""
    if not config.prompts.system.strip():
        raise ValueError("prompts.system is required in config")

    return ChatPromptTemplate.from_messages(
        [
            ("system", config.prompts.system),
            ("human", _compose_human_template(config)),
        ]
    )


def response_appears_cited(text: str, config: UseCaseConfig) -> bool:
    """Heuristic citation check using ``config.output_schema`` patterns."""
    schema = config.output_schema
    if not schema.require_citations:
        return True

    pdf_re = re.compile(schema.pdf_citation_pattern, re.IGNORECASE)
    page_re = re.compile(schema.page_citation_pattern, re.IGNORECASE)
    return bool(pdf_re.search(text)) and bool(page_re.search(text))


def build_chain(
    config: UseCaseConfig,
    retriever: BaseRetriever,
    llm: BaseChatModel,
    *,
    entity_id: Optional[str] = None,
    driver: Optional[Driver] = None,
    entity_context: Optional[str] = None,
) -> Runnable[str, str]:
    """Assemble a config-driven RAG chain using LCEL pipe syntax.

    When ``entity_id`` is provided, the chain calls ``read_entity_graph()``,
    formats the subgraph, and injects it under ``config.prompts.entity_context_header``.
    The LLM then sees: entity graph summary + retrieved knowledge chunks + question.

    Pass ``entity_context`` to use pre-loaded text (e.g. a cache hit) instead of
    reading the graph.
    """
    prompt = build_prompt(config)

    def _format_context(documents: list[Document]) -> str:
        return format_retrieved_context(documents, config)

    def _entity_context(_question: str) -> str:
        if entity_context is not None:
            return entity_context
        return load_entity_context(entity_id=entity_id, config=config, driver=driver)

    chain: Runnable[str, str] = (
        RunnableParallel(
            {
                "context": retriever | RunnableLambda(_format_context),
                "entity_context": RunnableLambda(_entity_context),
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
