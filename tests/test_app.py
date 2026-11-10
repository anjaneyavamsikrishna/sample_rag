"""Headless smoke tests of the real Streamlit script (no browser, no real models)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

import app_logic as logic
import config
from retrieval import embeddings

APP = str(Path(__file__).resolve().parent.parent / "app.py")
Q = "How many days of annual leave do employees get?"


@pytest.fixture
def app(store, fake_rerank, fake_llm, monkeypatch):
    monkeypatch.setattr(embeddings, "get_model", lambda: None)      # app start-up must not load the real model
    monkeypatch.setattr(config, "MIN_RERANK_SCORE", 0.3)
    return AppTest.from_file(APP, default_timeout=60)


def texts(at):
    return " ".join(str(m.value) for m in at.markdown)


def test_empty_state_guides_the_user_and_blocks_chat(app):
    at = app.run()
    assert not at.exception
    assert any("Start by uploading" in i.value for i in at.info)
    assert at.chat_input[0].disabled


def test_full_conversation_flow(app, fake_llm):
    logic.load_sample_data()
    at = app.run()
    assert not at.exception and not at.chat_input[0].disabled

    at.chat_input[0].set_value(Q).run()
    assert not at.exception
    assert "24 days" in texts(at)                                   # the answer is displayed
    assert len(at.session_state["messages"]) == 2
    assert at.session_state["messages"][1]["sources"][0]["cited"] is True
    assert any("Sources" in e.label for e in at.expander)           # citations viewer is rendered

    fake_llm.replies = ["Do I need a medical certificate for sick leave?", "Yes, after 3 days [1]."]
    at.chat_input[0].set_value("Do I need a doctor's note for it?").run()
    assert not at.exception and len(at.session_state["messages"]) == 4


def test_unanswerable_question_shows_abstention_and_skips_llm(app, fake_llm):
    logic.load_sample_data()
    at = app.run()
    at.chat_input[0].set_value("What is the stock option vesting schedule?").run()
    assert not at.exception
    assert any("No supporting evidence" in c.value for c in at.caption)
    assert fake_llm.calls == []


def test_clear_conversation_button(app):
    logic.load_sample_data()
    at = app.run()
    at.chat_input[0].set_value(Q).run()
    next(b for b in at.sidebar.button if b.label == "Clear conversation").click().run()
    assert at.session_state["messages"] == []
