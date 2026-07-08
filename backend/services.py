"""Helpers used by platform API route handlers."""

from __future__ import annotations

import re
from typing import Any, Optional

from langchain_core.documents import Document

from backend.schemas import Citation
from core.config import UseCaseConfig


_CITATION_LINE_RE = re.compile(
    r"Source:\s*(?P<source>[^|]+?)\s*\|\s*Page:\s*(?P<page>[^|]+?)"
    r"(?:\s*\|\s*Type:\s*(?P<doc_type>\S+))?",
    re.IGNORECASE,
)


def extract_citations(answer: str, documents: list[Document]) -> list[Citation]:
    """Build citation list from retrieved docs, falling back to answer text."""
    seen: set[tuple[Any, ...]] = set()
    citations: list[Citation] = []

    for doc in documents:
        meta = doc.metadata or {}
        source = meta.get("source")
        if not source:
            continue
        key = (source, meta.get("page_number"), meta.get("doc_type"))
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            Citation(
                source=str(source),
                page_number=meta.get("page_number"),
                doc_type=meta.get("doc_type"),
            )
        )

    if citations:
        return citations

    for match in _CITATION_LINE_RE.finditer(answer):
        key = (
            match.group("source").strip(),
            match.group("page").strip(),
            (match.group("doc_type") or "").strip() or None,
        )
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            Citation(
                source=key[0],
                page_number=key[1],
                doc_type=key[2],
            )
        )
    return citations


def build_filtered_retriever(vectorstore: Any, config: UseCaseConfig, metadata_filter: dict[str, Any]):
    """Vector retriever restricted by a Chroma metadata ``where`` filter."""
    k = config.retrieval.top_k
    return vectorstore.as_retriever(
        search_kwargs={"k": k, "filter": metadata_filter},
    )


def resolve_retriever(
    *,
    request_app_state: Any,
    config: UseCaseConfig,
    metadata_filter: Optional[dict[str, Any]],
):
    """Return the shared retriever, or a filter-scoped vector retriever."""
    if metadata_filter:
        return build_filtered_retriever(
            request_app_state.vectorstore,
            config,
            metadata_filter,
        )
    return request_app_state.retriever


def profile_from_documents(
    documents: list[Document],
    config: UseCaseConfig,
) -> dict[str, Any]:
    """Derive a profile dict from entity chunk metadata using config fields."""
    profile: dict[str, Any] = {field: None for field in config.entities.profile_fields}
    for doc in documents:
        meta = doc.metadata or {}
        for field in config.entities.profile_fields:
            if profile[field] is None and field in meta and meta[field] is not None:
                profile[field] = meta[field]
    return profile


def session_count_from_documents(documents: list[Document]) -> int:
    """Count distinct sessions/notes from entity documents."""
    session_ids: set[str] = set()
    for doc in documents:
        meta = doc.metadata or {}
        session_id = meta.get("session_id")
        if session_id:
            session_ids.add(str(session_id))
            continue
        session_number = meta.get("session_number")
        if session_number is not None:
            session_ids.add(str(session_number))
    if session_ids:
        return len(session_ids)
    return 1 if documents else 0
