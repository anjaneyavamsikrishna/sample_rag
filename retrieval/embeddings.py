"""Task 4 - Embeddings: turn text into vectors (lists of numbers) that capture meaning.

Key ideas:
  * Texts with similar MEANING get vectors that point in similar directions, even with different words.
  * We normalise every vector to length 1, so  cosine similarity == a plain dot product.
  * Queries and documents are encoded differently (queries get an instruction prefix) - BGE's design.
  * The model is loaded lazily and only once: loading takes seconds, encoding takes milliseconds.
"""
from functools import lru_cache

import numpy as np

import config


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer   # imported here so other code can import
    return SentenceTransformer(config.EMBED_MODEL)           # this file without the heavy library


def embed_documents(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Encode chunks. Returns an array of shape (number_of_texts, dimension)."""
    if not texts:
        return np.zeros((0, get_model().get_sentence_embedding_dimension()))
    return np.asarray(get_model().encode(texts, batch_size=batch_size, normalize_embeddings=True))


def embed_query(query: str) -> np.ndarray:
    """Encode one question. Returns an array of shape (dimension,)."""
    return np.asarray(get_model().encode(config.QUERY_PREFIX + query, normalize_embeddings=True))


def rank(query: str, chunks: list[dict], vectors: np.ndarray, k: int = 3) -> list[dict]:
    """Brute-force search: score every chunk against the query and return the best k.
    This is exactly what a vector database does for us at scale (Task 5)."""
    scores = vectors @ embed_query(query)                    # dot product == cosine (both length 1)
    best = np.argsort(scores)[::-1][:k]
    return [{**chunks[i], "score": float(scores[i])} for i in best]
