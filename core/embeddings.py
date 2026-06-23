"""Module-level singleton for HuggingFace embedding models."""

from __future__ import annotations

import logging
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings

from core.config import DataSourceConfig, UseCaseConfig

logger = logging.getLogger(__name__)

_embeddings: Optional[HuggingFaceEmbeddings] = None
_cached_model_name: Optional[str] = None


def _create_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """Build a CPU-backed embedder with L2-normalized vectors."""
    logger.info("Initialising embedding model %s", model_name)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _resolve_model_name(config: Optional[UseCaseConfig]) -> str:
    if config is None:
        return DataSourceConfig().embedding_model
    return config.data.embedding_model


def get_embeddings(config: Optional[UseCaseConfig] = None) -> HuggingFaceEmbeddings:
    """Return the shared ``HuggingFaceEmbeddings`` instance (loaded once per model name)."""
    global _embeddings, _cached_model_name

    model_name = _resolve_model_name(config)
    if _embeddings is None or _cached_model_name != model_name:
        _embeddings = _create_embeddings(model_name)
        _cached_model_name = model_name
    return _embeddings


def embed_query(
    text: str,
    *,
    config: Optional[UseCaseConfig] = None,
    embeddings: Optional[HuggingFaceEmbeddings] = None,
) -> list[float]:
    """Embed a single query string using the shared model."""
    model = embeddings if embeddings is not None else get_embeddings(config)
    return model.embed_query(text)
