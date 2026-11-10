"""REAL-model tests (embedding model + reranker). First run downloads the reranker (~1 GB)."""
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

pytest.importorskip("sentence_transformers")

import config
from ingestion.ingest import ingest_file
from retrieval import vector_store
from retrieval.retriever import retrieve

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def loaded():
    old = vector_store._client
    vector_store._client = QdrantClient(":memory:")
    for name in ["it_security_policy.pdf", "leave_policy.docx", "expense_policy.csv", "company_handbook.txt"]:
        ingest_file(name, (DATA / name).read_bytes())
    yield
    vector_store._client = old


ANSWERABLE = [
    ("How many days of annual leave do employees get?", "24 days"),
    ("What is the minimum password length?", "at least 14 characters"),
    ("How many vacation days can I roll over to next year?", "carried forward"),
    ("How quickly must a security incident be reported?", "within 2 hours"),
    ("What is the daily hotel limit for international travel?", "Hotel - international,260"),
    ("How many weeks of parental leave for a primary caregiver?", "16 weeks paid"),
]


@pytest.mark.parametrize("question, expected", ANSWERABLE)
def test_best_chunk_after_reranking_has_the_answer(loaded, question, expected):
    top = retrieve(question)["top"]
    assert expected in top[0]["text"], [(c["source"], c["location"], round(c["rerank_score"], 3)) for c in top[:3]]


@pytest.mark.parametrize("question, expected", ANSWERABLE)
def test_answerable_questions_pass_the_gate(loaded, question, expected):
    r = retrieve(question)
    assert r["relevant"], f"best score {r['best_score']:.3f} < MIN_RERANK_SCORE {config.MIN_RERANK_SCORE}"


@pytest.mark.parametrize("question", [
    "What is the stock option vesting schedule?",
    "How much is the monthly remote-work internet stipend?",
])
def test_unanswerable_questions_are_blocked(loaded, question):
    r = retrieve(question)
    assert not r["relevant"], f"best score {r['best_score']:.3f} passed the gate; run calibrate_threshold.py"
