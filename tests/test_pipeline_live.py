"""LIVE test: real embedding model + real reranker + REAL LLM. Skipped unless .env configures an LLM.
Costs a few cents with a paid API. Run:  python -m pytest tests/test_pipeline_live.py -v -s"""
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

pytest.importorskip("sentence_transformers")

from ingestion.ingest import ingest_file
from rag import llm, pipeline
from retrieval import vector_store

pytestmark = pytest.mark.skipif(not llm.is_configured(), reason="no LLM configured in .env")
DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def loaded():
    old = vector_store._client
    vector_store._client = QdrantClient(":memory:")
    for name in ["it_security_policy.pdf", "leave_policy.docx", "expense_policy.csv", "company_handbook.txt"]:
        ingest_file(name, (DATA / name).read_bytes())
    yield
    vector_store._client = old


@pytest.mark.parametrize("question, must_contain, source", [
    ("How many days of annual leave does a full-time employee get?", "24", "leave_policy.docx"),
    ("What is the minimum password length?", "14", "it_security_policy.pdf"),
    ("How many vacation days can I roll over, and when do they expire?", "31 March", "leave_policy.docx"),
    ("How quickly must a security incident be reported?", "2 hours", "it_security_policy.pdf"),
    ("What is the daily hotel limit for international travel?", "260", "expense_policy.csv"),
    ("How many weeks of parental leave does a primary caregiver get?", "16", "leave_policy.docx"),
])
def test_answers_are_correct_and_cited(loaded, question, must_contain, source):
    r = pipeline.answer_question(question)
    assert not r["abstained"], r["reason"]
    assert must_contain in r["answer"], r["answer"]
    assert any(s["cited"] and s["source"] == source for s in r["sources"]), r["sources"]


@pytest.mark.parametrize("question", [
    "What is the stock option vesting schedule?",
    "How much is the monthly remote-work internet stipend?",
])
def test_unanswerable_questions_abstain(loaded, question):
    assert pipeline.answer_question(question)["abstained"]


def test_follow_up_question_uses_memory(loaded):
    first = pipeline.answer_question("What is the sick leave policy?")
    history = [{"role": "user", "content": "What is the sick leave policy?"},
               {"role": "assistant", "content": first["answer"]}]
    r = pipeline.answer_question("Do I need a doctor's note for it?", history=history)
    print("searched as:", r["standalone_question"])
    assert r["rewritten"] and "sick" in r["standalone_question"].lower()
    assert not r["abstained"] and any(s["cited"] and s["source"] == "leave_policy.docx" for s in r["sources"])
    assert "3" in r["answer"]
