"""
AI Knowledge Distillation Platform - Embedding Utilities.

Wrapper around sentence-transformers for generating text embeddings.
"""

import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load and cache the embedding model (singleton).

    Returns:
        Loaded SentenceTransformer model.
    """
    logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
    model = SentenceTransformer(settings.EMBEDDING_MODEL)
    logger.info(f"Embedding model loaded. Dimension: {model.get_sentence_embedding_dimension()}")
    return model


def generate_embedding(text: str) -> list[float]:
    """
    Generate embedding vector for a single text.

    Args:
        text: Input text to embed.

    Returns:
        Embedding vector as list of floats.
    """
    model = get_embedding_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def generate_embeddings_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """
    Generate embedding vectors for a batch of texts.

    Args:
        texts: List of input texts to embed.
        batch_size: Batch size for encoding.

    Returns:
        List of embedding vectors.
    """
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 100,
    )
    return embeddings.tolist()
