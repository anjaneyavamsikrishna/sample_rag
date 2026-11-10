import pytest

import app_logic as logic
import config
from rag import llm
from retrieval import vector_store

PDF_FAKE = b"%PDF-1.4 this is not really a pdf"


def test_load_sample_data_indexes_all_four_files(store):
    results = logic.load_sample_data()
    assert [r["status"] for r in results] == ["indexed"] * 4
    assert len(vector_store.list_documents()) == 4


def test_second_upload_is_reported_as_duplicate(store):
    logic.load_sample_data()
    assert {r["status"] for r in logic.load_sample_data()} == {"duplicate"}


def test_one_bad_file_does_not_stop_the_others(store):
    results = logic.process_uploads([("good.txt", b"Employees get 24 days of leave."),
                                     ("broken.pdf", PDF_FAKE),
                                     ("empty.txt", b""),
                                     ("virus.exe", b"MZ..."),
                                     ("also_good.txt", b"The cafeteria opens at noon.")])
    assert [r["status"] for r in results] == ["indexed", "error", "error", "error", "indexed"]
    assert "could not be read" in results[1]["detail"]
    assert "empty" in results[2]["detail"].lower() and "Unsupported" in results[3]["detail"]
    assert len(vector_store.list_documents()) == 2


def test_oversized_file_is_rejected(store, monkeypatch):
    monkeypatch.setattr(logic, "MAX_UPLOAD_MB", 0.0001)
    assert logic.process_uploads([("big.txt", b"x" * 1000)])[0]["status"] == "error"


def test_describe_upload_levels():
    assert logic.describe_upload({"status": "indexed", "source": "a", "chunks": 3})[0] == "success"
    assert logic.describe_upload({"status": "replaced", "source": "a", "chunks": 3})[0] == "success"
    assert logic.describe_upload({"status": "duplicate", "source": "a", "chunks": 0})[0] == "info"
    assert logic.describe_upload({"status": "error", "source": "a", "detail": "bad"}) == ("error", "a: bad")


def test_ask_never_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("database exploded")
    monkeypatch.setattr(logic, "answer_question", boom)
    r = logic.ask("hello?", [], None)
    assert r["mode"] == "error" and "RuntimeError" in r["error_detail"] and r["sources"] == []
    assert "Something went wrong" in r["answer"]


def test_ask_passes_only_role_and_content_as_history(monkeypatch):
    seen = {}
    monkeypatch.setattr(logic, "answer_question", lambda q, h, s: seen.update(q=q, h=h, s=s) or {})
    logic.ask("q?", [{"role": "assistant", "content": "a", "sources": [1, 2], "warnings": []}], "x.txt")
    assert seen == {"q": "q?", "h": [{"role": "assistant", "content": "a"}], "s": "x.txt"}


def test_source_title():
    assert logic.source_title({"source": "a.pdf", "location": "page 3"}) == "a.pdf · page 3"
    assert logic.source_title({"source": "a.txt", "location": None}) == "a.txt"


def test_llm_status_text(monkeypatch):
    monkeypatch.setattr(llm, "is_configured", lambda: False)
    assert "extractive" in logic.llm_status()
    monkeypatch.setattr(llm, "is_configured", lambda: True)
    assert config.LLM_MODEL in logic.llm_status()
