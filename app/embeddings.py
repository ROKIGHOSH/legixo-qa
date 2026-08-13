"""Free, local embeddings using sentence-transformers (no API key, no cost).

We use this instead of a paid embeddings API so the whole project can run
with zero paid keys (Gemini's free tier covers the LLM, this covers embeddings).
"""
from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

from app import config


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    # Cached so the (fairly large) model is only loaded into memory once per process.
    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts. Returns one vector (list[float]) per text."""
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> List[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]
