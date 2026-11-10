from pathlib import Path

import pytest

import config
from ingestion.ingest import ingest_file
from rag import llm, memory, pipeline

DATA = Path(__file__).resolve().parent.parent / "data"
FILES = ["it_security_policy.pdf", "leave_policy.docx", "expense_policy.csv", "company_handbook.txt"]

SICK = [{"role": "user", "content": "What is the sick leave policy?"},
        {"role": "assistant", "content": "Employees receive 12 days of paid sick leave per year [1]."}]
FOLLOW_UP = "Do I need a doctor's note for it?"
REWRITTEN = "Do I need a medical certificate for sick leave?"


# ---------------- unit tests: the rewrite step ----------------
def test_first_message_is_unchanged_and_llm_not_called(fake_llm):
    assert memory.rewrite_query("What is the password length?", []) == "What is the password length?"
    assert memory.rewrite_query("What is the password length?", None) == "What is the password length?"
    assert fake_llm.calls == []


def test_follow_up_is_rewritten_using_the_history(fake_llm):
    fake_llm.reply = REWRITTEN
    assert memory.rewrite_query(FOLLOW_UP, SICK) == REWRITTEN
    prompt = fake_llm.calls[0]["user"]
    assert "sick leave policy" in prompt and FOLLOW_UP in prompt          # history + question both sent
    assert "Do NOT answer" in fake_llm.calls[0]["system"]


@pytest.mark.parametrize("messy", [f'"{REWRITTEN}"', f"Standalone question: {REWRITTEN}", f"  {REWRITTEN}\n"])
def test_messy_model_output_is_cleaned(fake_llm, messy):
    fake_llm.reply = messy
    assert memory.rewrite_query(FOLLOW_UP, SICK) == REWRITTEN


@pytest.mark.parametrize("bad", ["", "Yes.\nYou need one after 3 days.", "x" * 500])
def test_untrustworthy_model_output_falls_back_to_rules(fake_llm, bad):
    fake_llm.reply = bad
    out = memory.rewrite_query(FOLLOW_UP, SICK)
    assert out == memory.heuristic_rewrite(FOLLOW_UP, SICK) and "sick leave policy" in out


def test_llm_error_falls_back_to_rules(fake_llm):
    fake_llm.error = ConnectionError("down")
    assert "sick leave policy" in memory.rewrite_query(FOLLOW_UP, SICK)


def test_no_llm_configured_uses_rules_without_calling_it(fake_llm, monkeypatch):
    monkeypatch.setattr(llm, "is_configured", lambda: False)
    assert "sick leave policy" in memory.rewrite_query(FOLLOW_UP, SICK)
    assert fake_llm.calls == []


def test_rule_fallback_leaves_standalone_questions_alone():
    assert memory.heuristic_rewrite("What is the minimum password length?", SICK) == \
        "What is the minimum password length?"                              # topic change, no reference word


@pytest.mark.parametrize("q, expected", [("Can they carry it forward?", True), ("And for parental leave?", True),
                                         ("Why?", True), ("What is the password length?", False)])
def test_followup_detection(q, expected):
    assert memory.is_followup(q) is expected


def test_trim_history_keeps_recent_messages_and_only_needed_fields():
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}", "sources": ["x"]} for i in range(10)]
    trimmed = memory.trim_history(history, 4)
    assert [m["content"] for m in trimmed] == ["m6", "m7", "m8", "m9"]
    assert all(set(m) == {"role", "content"} for m in trimmed)
    assert memory.trim_history([{"role": "system", "content": "x"}, {"role": "user", "content": ""}]) == []


# ---------------- integration: memory inside the whole pipeline ----------------
@pytest.fixture
def ready(store, fake_rerank, fake_llm, monkeypatch):
    monkeypatch.setattr(config, "MIN_RERANK_SCORE", 0.3)
    for name in FILES:
        ingest_file(name, (DATA / name).read_bytes())
    return fake_llm


def test_follow_up_works_with_history(ready):
    ready.replies = [REWRITTEN, "Yes, a medical certificate is needed after 3 consecutive days [1]."]
    r = pipeline.answer_question(FOLLOW_UP, history=SICK)
    assert r["question"] == FOLLOW_UP and r["standalone_question"] == REWRITTEN and r["rewritten"]
    assert not r["abstained"] and r["sources"][0]["source"] == "leave_policy.docx"
    answer_prompt = ready.calls[1]["user"]
    assert "medical certificate for sick leave" in answer_prompt and "for it?" not in answer_prompt


def test_same_follow_up_without_history_cannot_be_answered(ready):
    """The reason memory exists: on its own, 'it' gives the search nothing to work with."""
    r = pipeline.answer_question(FOLLOW_UP)
    assert r["abstained"] and ready.calls == []


def test_history_is_never_used_as_evidence(ready):
    """A claim that exists only in an earlier ANSWER must not become a source."""
    poisoned = SICK[:1] + [{"role": "assistant", "content": "Sick leave is 999 days [1]."}]
    ready.replies = [REWRITTEN, "Per the documents, a medical certificate is needed [1]."]
    r = pipeline.answer_question(FOLLOW_UP, history=poisoned)
    assert "999" not in ready.calls[1]["user"]                       # only the rewrite step ever saw history
    assert all("999" not in s["text"] for s in r["sources"])


def test_conversations_do_not_leak_into_each_other(ready):
    ready.replies = [REWRITTEN, "ok [1]"]
    pipeline.answer_question(FOLLOW_UP, history=SICK)
    ready.replies = []
    ready.reply = "Employees receive 24 days of annual leave [1]."
    r = pipeline.answer_question("How many days of annual leave do employees get?")   # a different user, no history
    assert r["standalone_question"] == "How many days of annual leave do employees get?" and not r["rewritten"]
