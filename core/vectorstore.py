"""Config-driven Chroma vector store build and reload."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from core.config import UseCaseConfig
from core.embeddings import get_embeddings

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_vectorstore(
    chunks: list[Document],
    config: UseCaseConfig,
    *,
    base_dir: Path = REPO_ROOT,
    reset: bool = True,
) -> Chroma:
    """Embed chunks and persist them to the configured Chroma path."""
    persist_directory = config.chroma_path(base=base_dir)
    collection_name = config.data.collection

    if reset and persist_directory.exists():
        shutil.rmtree(persist_directory)
        logger.info("Cleared existing index at %s", persist_directory)

    persist_directory.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Indexing %d chunk(s) into %s (collection=%s)",
        len(chunks),
        persist_directory,
        collection_name,
    )

    embeddings = get_embeddings(config)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_directory),
    )
    logger.info("Persisted %d vector(s)", vectorstore._collection.count())
    return vectorstore


def load_vectorstore(
    config: UseCaseConfig,
    *,
    base_dir: Path = REPO_ROOT,
) -> Chroma:
    """Reload a persisted Chroma store for the active use-case profile."""
    persist_directory = config.chroma_path(base=base_dir)
    if not persist_directory.is_dir():
        raise FileNotFoundError(
            f"No persisted index at {persist_directory}. "
            f"Run: USE_CASE_CONFIG=<config> python -m core ingest"
        )

    embeddings = get_embeddings(config)
    vectorstore = Chroma(
        collection_name=config.data.collection,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )
    logger.info(
        "Reloaded %d vector(s) from %s",
        vectorstore._collection.count(),
        persist_directory,
    )
    return vectorstore


def documents_from_vectorstore(vectorstore: Chroma) -> list[Document]:
    """Materialise all persisted chunks as LangChain Documents (for BM25/hybrid)."""
    result = vectorstore._collection.get(include=["documents", "metadatas"])
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    docs: list[Document] = []
    for content, metadata in zip(documents, metadatas):
        if not content:
            continue
        docs.append(
            Document(
                page_content=content,
                metadata=dict(metadata or {}),
            )
        )
    return docs


def lookup_entity_documents(
    vectorstore: Chroma,
    entity_id: str,
) -> list[Document]:
    """Return Chroma chunks whose metadata ``entity_id`` matches ``entity_id``."""
    result = vectorstore._collection.get(
        where={"entity_id": entity_id},
        include=["documents", "metadatas"],
    )
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    docs: list[Document] = []
    for content, metadata in zip(documents, metadatas):
        docs.append(
            Document(
                page_content=content or "",
                metadata=dict(metadata or {}),
            )
        )
    return docs


def append_documents(
    documents: list[Document],
    config: UseCaseConfig,
    *,
    base_dir: Path = REPO_ROOT,
) -> int:
    """Embed and append documents into the persisted Chroma collection."""
    if not documents:
        return 0

    persist_directory = config.chroma_path(base=base_dir)
    persist_directory.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings(config)
    vectorstore = Chroma(
        collection_name=config.data.collection,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )
    vectorstore.add_documents(documents)
    logger.info(
        "Appended %d document(s) → %s (collection=%s, total=%d)",
        len(documents),
        persist_directory,
        config.data.collection,
        vectorstore._collection.count(),
    )
    return len(documents)
