from pathlib import Path

import pytest

import config
from ingestion.ingest import ingest_file
from retrieval import reranker, vector_store
from retrieval.retriever import retrieve

DATA = Path(__file__).resolve().parent.parent / "data"
FILES = ["it_security_policy.pdf", "leave_policy.docx", "expense_policy.csv", "company_handbook.txt"]


def c(text, cid, source="x.txt"):
    return {"text": text, "chunk_id": cid, "source": source, "location": "text"}


def test_rerank_moves_the_real_answer_to_the_top(fake_rerank):
    candidates = [c("The cafeteria serves lunch at noon.", "a"),
                  c("Employees receive 24 days of annual leave per year.", "b")]
    ranked = reranker.rerank("how many days of annual leave", candidates)
    assert [r["chunk_id"] for r in ranked] == ["b", "a"]
    assert ranked[0]["vector_rank"] == 2          # it was 2nd in vector search, now 1st


def test_scores_are_probabilities_and_input_is_untouched(fake_rerank):
    candidates = [c("annual leave days", "a"), c("lunch", "b")]
    ranked = reranker.rerank("annual leave", candidates)
    assert all(0.0 <= r["rerank_score"] <= 1.0 for r in ranked)
    assert "rerank_score" not in candidates[0]


def test_top_k_and_empty(fake_rerank):
    many = [c(f"leave {i}", str(i)) for i in range(6)]
    assert len(reranker.rerank("leave", many, top_k=2)) == 2
    assert reranker.rerank("leave", []) == []


def test_logits_are_converted_exactly_once():
    assert reranker._to_probability([0.2, 0.9]) == [0.2, 0.9]            # already 0..1: untouched
    out = reranker._to_probability([-4.0, 0.0, 4.0])                      # raw logits: sigmoid once
    assert 0 < out[0] < out[1] == 0.5 < out[2] < 1


@pytest.fixture
def loaded(store, fake_rerank, monkeypatch):
    monkeypatch.setattr(config, "MIN_RERANK_SCORE", 0.3)
    for name in FILES:
        ingest_file(name, (DATA / name).read_bytes())


def test_answerable_question_passes_the_gate(loaded):
    r = retrieve("How many days of annual leave do employees get?")
    assert r["relevant"] and r["chunks"][0]["source"] == "leave_policy.docx"
    assert r["candidates"] > 0 and all(x["rerank_score"] >= 0.3 for x in r["chunks"])


def test_unanswerable_question_is_blocked_by_the_gate(loaded):
    r = retrieve("What is the stock option vesting schedule?")
    assert r["relevant"] is False and r["chunks"] == []
    assert r["top"], "debug info should still show what was found"


def test_source_filter_is_respected(loaded):
    r = retrieve("annual leave days", source="company_handbook.txt")
    assert all(x["source"] == "company_handbook.txt" for x in r["top"])


def test_reranker_fixes_a_wrong_vector_order(fake_rerank, monkeypatch):
    """The exact bug you saw: vector search put the handbook above the leave policy."""
    wrong_first = [c("Holidays and vacation. Please read the Leave Policy for details.", "hb", "company_handbook.txt"),
                   c("Up to 5 unused annual leave days may be carried forward to next year.", "lp", "leave_policy.docx")]
    monkeypatch.setattr(vector_store, "search", lambda q, k, s=None: wrong_first)
    monkeypatch.setattr(config, "MIN_RERANK_SCORE", 0.1)
    r = retrieve("unused annual leave days carried forward next year")
    assert r["chunks"][0]["chunk_id"] == "lp"
