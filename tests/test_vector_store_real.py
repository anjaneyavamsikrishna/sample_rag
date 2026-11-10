"""REAL-model test: index the four sample files, then search by meaning."""
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

pytest.importorskip("sentence_transformers")

from ingestion.ingest import ingest_file
from retrieval import vector_store

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def loaded():
    old = vector_store._client
    vector_store._client = QdrantClient(":memory:")
    for name in ["it_security_policy.pdf", "leave_policy.docx", "expense_policy.csv", "company_handbook.txt"]:
        ingest_file(name, (DATA / name).read_bytes())
    yield vector_store
    vector_store._client = old


@pytest.mark.parametrize("question, expected", [
    ("How many days of annual leave do employees get?", "24 days"),
    ("What is the minimum password length?", "at least 14 characters"),
    ("How many vacation days can I roll over to next year?", "carried forward"),
    ("How quickly must a security incident be reported?", "within 2 hours"),
    ("What is the daily hotel limit for international travel?", "Hotel - international,260"),
    ("How many weeks of parental leave for a primary caregiver?", "16 weeks paid"),
])
def test_answer_is_in_top_10_candidates(loaded, question, expected):
    assert any(expected in c["text"] for c in loaded.search(question, limit=10))
