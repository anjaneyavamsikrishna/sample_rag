"""Task 6b - The retriever: search -> rerank -> relevance gate.

The gate is what lets the assistant say "I couldn't find that" instead of guessing: if even the best
chunk scores below MIN_RERANK_SCORE, nothing is passed on as evidence.
"""
import config
from retrieval import vector_store
from retrieval.reranker import rerank


def retrieve(question: str, source: str | None = None) -> dict:
    candidates = vector_store.search(question, config.RETRIEVE_K, source)
    ranked = rerank(question, candidates)
    top = ranked[:config.TOP_K]
    kept = [c for c in top if c["rerank_score"] >= config.MIN_RERANK_SCORE]
    return {
        "relevant": bool(kept),                  # False => the caller must abstain
        "chunks": kept,                          # evidence that passed the gate
        "top": top,                              # before the gate, handy for debugging
        "best_score": ranked[0]["rerank_score"] if ranked else 0.0,
        "candidates": len(candidates),
    }
