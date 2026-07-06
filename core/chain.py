"""Config-driven LCEL RAG chain — no domain-specific prompt text."""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough

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
) -> Runnable[str, str]:
    """Assemble a config-driven RAG chain using LCEL pipe syntax."""
    prompt = build_prompt(config)
    empty_entity_context = config.prompts.empty_entity_context

    def _format_context(documents: list[Document]) -> str:
        return format_retrieved_context(documents, config)

    chain: Runnable[str, str] = (
        RunnableParallel(
            {
                "context": retriever | RunnableLambda(_format_context),
                "entity_context": RunnableLambda(lambda _question: empty_entity_context),
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
