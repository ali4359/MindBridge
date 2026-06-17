"""Local HuggingFace embedding model for MindBridge."""

from __future__ import annotations

import logging
import os

from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

SAMPLE_SENTENCE = "Cognitive behavioral therapy helps patients identify unhelpful thought patterns."


def create_embeddings() -> HuggingFaceEmbeddings:
    """Return a CPU-backed MiniLM embedder with L2-normalized vectors."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def embed_query(text: str, *, embeddings: HuggingFaceEmbeddings | None = None) -> list[float]:
    """Embed a single query string."""
    model = embeddings or create_embeddings()
    return model.embed_query(text)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    logger.info("Model cache directory (HF_HOME): %s", hf_home)
    logger.info("Loading %s (first run downloads ~80MB)", EMBEDDING_MODEL_NAME)

    embeddings = create_embeddings()
    vector = embed_query(SAMPLE_SENTENCE, embeddings=embeddings)

    if len(vector) != EMBEDDING_DIM:
        raise SystemExit(f"Expected {EMBEDDING_DIM}-dim vector, got {len(vector)}")

    print(f"Sample: {SAMPLE_SENTENCE!r}")
    print(f"Dimensions: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")


if __name__ == "__main__":
    main()
