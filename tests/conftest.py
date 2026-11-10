"""Shared test fixtures: an in-memory Qdrant and a tiny fake embedder (no model download)."""
import re
import zlib

import numpy as np
import pytest
from qdrant_client import QdrantClient

from retrieval import vector_store


def fake_vec(text: str, dim: int = 256) -> np.ndarray:
    v = np.zeros(dim)
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        v[zlib.crc32(w.encode()) % dim] += 1
    n = np.linalg.norm(v)
    return v / n if n else v


@pytest.fixture
def fake_embed(monkeypatch):
    """Replace the real embedding model with a tiny deterministic one."""
    monkeypatch.setattr(vector_store, "embed_documents", lambda texts: np.array([fake_vec(t) for t in texts]))
    monkeypatch.setattr(vector_store, "embed_query", lambda q: fake_vec(q))


@pytest.fixture
def store(monkeypatch, fake_embed):
    """Fresh empty in-memory vector store, with a fake embedder, for every test."""
    monkeypatch.setattr(vector_store, "_client", QdrantClient(":memory:"))
    return vector_store


STOP = {"what", "is", "the", "a", "an", "of", "do", "does", "how", "many", "much", "i", "can", "to", "for", "are", "my"}


class FakeCrossEncoder:
    """Scores a (question, chunk) pair by the share of question keywords found in the chunk."""

    def predict(self, pairs, **kwargs):
        out = []
        for q, text in pairs:
            words = [w for w in re.findall(r"[a-z0-9]+", q.lower()) if w not in STOP]
            have = set(re.findall(r"[a-z0-9]+", text.lower()))
            out.append(sum(w in have for w in words) / max(len(words), 1))
        return np.array(out)


@pytest.fixture
def fake_rerank(monkeypatch):
    from retrieval import reranker
    monkeypatch.setattr(reranker, "get_model", lambda: FakeCrossEncoder())


class FakeLLM:
    """Stands in for the real LLM: records every call and returns whatever `reply` is set to."""

    def __init__(self):
        self.calls, self.reply, self.error = [], "Employees receive 24 days of annual leave [1].", None
        self.replies = []                     # optional queue: one reply per call, in order

    def chat(self, system, user, temperature=0.0):
        self.calls.append({"system": system, "user": user})
        if self.error:
            raise self.error
        return self.replies.pop(0) if self.replies else self.reply


@pytest.fixture
def fake_llm(monkeypatch):
    from rag import llm
    fake = FakeLLM()
    monkeypatch.setattr(llm, "chat", fake.chat)
    monkeypatch.setattr(llm, "is_configured", lambda: True)
    return fake
