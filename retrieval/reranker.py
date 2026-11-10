"""Task 6a - Reranking.

Vector search compares two PRE-COMPUTED vectors: fast, but it only sees each text on its own.
A cross-encoder reads the question and one chunk TOGETHER and scores how well the chunk answers it:
slower, much more accurate. So we do it in two stages: fast search for ~20 candidates, then rerank.
"""
import math
from functools import lru_cache

import config


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(config.RERANK_MODEL)


def _to_probability(scores) -> list[float]:
    """Guarantee scores are in 0..1. Current library versions already apply a sigmoid; if some version
    returned raw logits instead, apply it here. (Applying it twice would squash everything to ~0.5-0.7.)"""
    scores = [float(s) for s in scores]
    if any(s < 0.0 or s > 1.0 for s in scores):
        scores = [1.0 / (1.0 + math.exp(-s)) for s in scores]
    return scores


def _predict(pairs):
    model = get_model()
    try:
        import torch
        return model.predict(pairs, activation_fn=torch.nn.Sigmoid())   # be explicit: 0..1 scores
    except (TypeError, ImportError):
        return model.predict(pairs)                                     # older library versions


def rerank(question: str, chunks: list[dict], top_k: int | None = None) -> list[dict]:
    """Return copies of the chunks, best first, each with `rerank_score` and its old `vector_rank`."""
    if not chunks:
        return []
    scores = _to_probability(_predict([(question, c["text"]) for c in chunks]))
    ranked = sorted(
        ({**c, "rerank_score": s, "vector_rank": i} for i, (c, s) in enumerate(zip(chunks, scores), start=1)),
        key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k] if top_k else ranked
