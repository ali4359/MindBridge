"""Clinical RAG prompt templates for grounded therapist Q&A."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

CLINICAL_RAG_SYSTEM_MESSAGE = """You are MindBridge, a clinical knowledge assistant for licensed mental health professionals.

Your role is to help therapists find and synthesize information from their indexed clinical corpus
(treatment guidelines, psychotherapy session notes, and CBT/DBT reference material).

You are a clinical knowledge tool, NOT a clinician. You do not diagnose, prescribe treatment, provide
crisis intervention, or replace professional clinical judgment. Therapists remain solely responsible
for patient care decisions.

When answering:
- Ground every factual claim in the retrieved context provided by the user message.
- Cite sources using the exact document name and page number shown in each context block header.
- If the context is insufficient, say so clearly. Never invent citations, page numbers, or clinical facts.
- Treat session notes as confidential records; reference them only to support the therapist's query."""

CONTEXT_CHUNK_TEMPLATE = PromptTemplate(
    input_variables=["index", "source", "page_number", "doc_type", "content"],
    template="""[{index}] Source: {source} | Page: {page_number} | Type: {doc_type}
{content}""",
)

CLINICAL_RAG_USER_TEMPLATE = """Answer the therapist's question using only the retrieved clinical context below.

## Retrieved context
{context}

## Therapist question
{question}

## Response requirements
- Answer concisely and clinically, using only information supported by the context above.
- Cite every substantive claim with source name and page (e.g., "nice-ng222.pdf, p. 12").
- When multiple sources support a point, cite each relevant source.
- If the context does not contain enough information, state what is missing rather than guessing."""

CLINICAL_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CLINICAL_RAG_SYSTEM_MESSAGE),
        ("human", CLINICAL_RAG_USER_TEMPLATE),
    ]
)


def format_retrieved_context(documents: list[Document]) -> str:
    """Format retrieved chunks with source metadata for the ``{context}`` prompt variable."""
    if not documents:
        return "(No relevant context retrieved.)"

    blocks: list[str] = []
    for index, doc in enumerate(documents, start=1):
        meta = doc.metadata
        blocks.append(
            CONTEXT_CHUNK_TEMPLATE.format(
                index=index,
                source=meta.get("source", "unknown"),
                page_number=meta.get("page_number", "?"),
                doc_type=meta.get("doc_type", "unknown"),
                content=doc.page_content.strip(),
            )
        )
    return "\n\n".join(blocks)
