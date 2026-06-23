"""LCEL RAG chain: retrieve clinical context, generate grounded cited answers."""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough

from chains.llm import create_groq_llm
from chains.prompts import CLINICAL_RAG_PROMPT, format_retrieved_context
from retrieval.hybrid import create_reranked_ensemble_retriever

logger = logging.getLogger(__name__)

# Therapist-facing queries for smoke / integration tests over the demo corpus.
SAMPLE_THERAPIST_QUERIES: tuple[str, ...] = (
    "What does the NICE guideline recommend as first-line pharmacological treatment for depression in adults?",
    "What DBT distress tolerance skills can I teach a client for crisis survival?",
    "How can I use a thought record worksheet with a client who has generalized anxiety?",
)

_PDF_CITATION_RE = re.compile(r"\.pdf\b", re.IGNORECASE)
_PAGE_CITATION_RE = re.compile(r"\bp\.?\s*\d+", re.IGNORECASE)


def response_appears_cited(text: str) -> bool:
    """Heuristic: answer references a PDF source and a page number."""
    return bool(_PDF_CITATION_RE.search(text)) and bool(_PAGE_CITATION_RE.search(text))


def create_rag_chain(
    *,
    retriever: BaseRetriever | None = None,
    prompt: ChatPromptTemplate | None = None,
    llm: BaseChatModel | None = None,
) -> Runnable[str, str]:
    """Assemble the clinical RAG chain using LCEL pipe syntax."""
    resolved_retriever = retriever if retriever is not None else create_reranked_ensemble_retriever()
    resolved_prompt = prompt if prompt is not None else CLINICAL_RAG_PROMPT
    resolved_llm = llm if llm is not None else create_groq_llm()

    chain: Runnable[str, str] = (
        RunnableParallel(
            {
                "context": resolved_retriever | RunnableLambda(format_retrieved_context),
                "question": RunnablePassthrough(),
            }
        )
        | resolved_prompt
        | resolved_llm
        | StrOutputParser()
    )
    return chain


def _run_sample_queries(chain: Runnable[str, str] | None = None) -> list[dict[str, Any]]:
    """Run sample therapist queries and print grounded answers."""
    rag_chain = chain if chain is not None else create_rag_chain()
    results: list[dict[str, Any]] = []

    for query in SAMPLE_THERAPIST_QUERIES:
        logger.info("Query: %s", query)
        answer = rag_chain.invoke(query)
        cited = response_appears_cited(answer)
        results.append({"query": query, "answer": answer, "cited": cited})
        print(f"\n{'=' * 72}")
        print(f"Q: {query}")
        print(f"Cited: {cited}")
        print(f"A: {answer}")

    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = _run_sample_queries()
    cited_count = sum(1 for item in results if item["cited"])
    print(f"\n{'=' * 72}")
    print(f"Sample run complete: {cited_count}/{len(results)} responses include PDF + page citations")


if __name__ == "__main__":
    main()
