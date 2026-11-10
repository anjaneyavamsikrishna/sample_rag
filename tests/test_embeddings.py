"""Logic tests that run WITHOUT downloading the model (a fake model stands in)."""
import numpy as np

import config
from retrieval import embeddings


class FakeModel:
    def __init__(self):
        self.seen = []

    def encode(self, x, batch_size=32, normalize_embeddings=True):
        self.seen.append(x)
        return np.full((len(x), 4), 0.5) if isinstance(x, list) else np.full(4, 0.5)

    def get_sentence_embedding_dimension(self):
        return 4


def use_fake(monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(embeddings, "get_model", lambda: fake)
    return fake


def test_query_gets_prefix_documents_do_not(monkeypatch):
    fake = use_fake(monkeypatch)
    embeddings.embed_query("How many leave days?")
    embeddings.embed_documents(["Employees get 24 days."])
    assert fake.seen[0] == config.QUERY_PREFIX + "How many leave days?"
    assert fake.seen[1] == ["Employees get 24 days."]          # no prefix on documents


def test_shapes(monkeypatch):
    use_fake(monkeypatch)
    assert embeddings.embed_documents(["a", "b", "c"]).shape == (3, 4)
    assert embeddings.embed_query("q").shape == (4,)


def test_empty_input_does_not_crash(monkeypatch):
    use_fake(monkeypatch)
    assert embeddings.embed_documents([]).shape == (0, 4)


def test_rank_orders_by_score(monkeypatch):
    use_fake(monkeypatch)
    monkeypatch.setattr(embeddings, "embed_query", lambda q: np.array([1.0, 0.0]))
    vectors = np.array([[0.0, 1.0], [1.0, 0.0], [0.7, 0.7]])
    chunks = [{"text": "far"}, {"text": "exact"}, {"text": "close"}]
    result = embeddings.rank("q", chunks, vectors, k=2)
    assert [r["text"] for r in result] == ["exact", "close"]
    assert result[0]["score"] > result[1]["score"]
