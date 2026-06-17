"""Chroma metadata filters for doc-type–scoped retrieval."""

from __future__ import annotations

from ingestion.loader import DocType

GUIDELINE_DOC_TYPE: DocType = "guideline"
SESSION_DOC_TYPE: DocType = "session_note"
RESEARCH_DOC_TYPE: DocType = "research"


def filter_by_doc_type(doc_type: DocType) -> dict[str, str]:
    """Return a Chroma ``where`` clause restricting search to one document type."""
    return {"doc_type": doc_type}
