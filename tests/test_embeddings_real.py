"""REAL-model tests. The first run downloads the BGE model (~130 MB). Needs internet once."""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sentence_transformers")

from ingestion.document_loader import load_document
from retrieval.chunker import chunk_sections
from retrieval import embeddings

DATA = Path(__file__).resolve().parent.parent / "data"
FILES = ["it_security_policy.pdf", "leave_policy.docx", "expense_policy.csv", "company_handbook.txt"]


@pytest.fixture(scope="module")
def corpus():
    chunks = []
    for name in FILES:
        data = (DATA / name).read_bytes()
        chunks += chunk_sections(load_document(name, data), name)
    return chunks, embeddings.embed_documents([c["embed_text"] for c in chunks])


def test_dimension_and_unit_length():
    v = embeddings.embed_documents(["hello world"])
    assert v.shape == (1, 384)
    assert abs(np.linalg.norm(v[0]) - 1.0) < 1e-3


def test_meaning_beats_keywords():
    q = embeddings.embed_query("How many vacation days do I get?")
    related, unrelated = embeddings.embed_documents([
        "Full-time employees are entitled to 24 days of paid annual leave per year.",
        "Passwords must be at least 14 characters long.",
    ])
    assert q @ related > q @ unrelated      # no shared keywords, yet the right one wins


@pytest.mark.parametrize("question, expected", [
    ("How many days of annual leave do employees get?", "24 days"),
    ("What is the minimum password length?", "at least 14 characters"),
    ("How many vacation days can I roll over to next year?", "carried forward"),
    ("How quickly must a security incident be reported?", "within 2 hours"),
    ("What is the daily hotel limit for international travel?", "Hotel - international,260"),
    ("How many weeks of parental leave for a primary caregiver?", "16 weeks paid"),
])
def test_right_chunk_in_top_3(corpus, question, expected):
    chunks, vectors = corpus
    top = embeddings.rank(question, chunks, vectors, k=3)
    assert any(expected in c["text"] for c in top), [c["location"] for c in top]


@pytest.mark.parametrize("question, expected", [
    ("How many days of annual leave do employees get?", "24 days"),
    ("What is the minimum password length?", "at least 14 characters"),
    ("How many vacation days can I roll over to next year?", "carried forward"),
    ("How quickly must a security incident be reported?", "within 2 hours"),
    ("What is the daily hotel limit for international travel?", "Hotel - international,260"),
    ("How many weeks of parental leave for a primary caregiver?", "16 weeks paid"),
])
def test_candidate_recall_at_10(corpus, question, expected):
    """The real pipeline retrieves ~20 candidates and reranks them (Task 6), so the answer
    MUST at least be inside the candidate set, even if it is not ranked first."""
    chunks, vectors = corpus
    assert any(expected in c["text"] for c in embeddings.rank(question, chunks, vectors, k=10))
