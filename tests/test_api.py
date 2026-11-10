from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api_client_demo
import app_logic
import config
from api import main
from retrieval import embeddings, reranker

DATA = Path(__file__).resolve().parent.parent / "data"
FILES = ["it_security_policy.pdf", "leave_policy.docx", "expense_policy.csv", "company_handbook.txt"]
Q = "How many days of annual leave do employees get?"


@pytest.fixture
def client(store, fake_rerank, fake_llm, monkeypatch):
    monkeypatch.setattr(config, "MIN_RERANK_SCORE", 0.3)
    monkeypatch.setattr(config, "API_KEY", "")
    return TestClient(main.app)                      # no `with`: the model-loading lifespan is skipped


def upload(client, names=FILES, extra=()):
    files = [("files", (n, (DATA / n).read_bytes())) for n in names] + list(extra)
    return client.post("/documents", files=files)


@pytest.fixture
def loaded(client):
    assert upload(client).status_code == 200
    return client


# ------------------------------ health & docs ------------------------------
def test_health_and_openapi(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["auth_required"] is False
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/documents", "/documents/{doc_id}", "/ask"} <= set(paths)


# ------------------------------ documents ------------------------------
def test_upload_list_duplicate_delete_cycle(client):
    r = upload(client)
    assert r.status_code == 200 and [x["status"] for x in r.json()["results"]] == ["indexed"] * 4
    docs = client.get("/documents").json()
    assert sorted(d["source"] for d in docs) == sorted(FILES)
    assert {x["status"] for x in upload(client).json()["results"]} == {"duplicate"}

    doc_id = docs[0]["doc_id"]
    assert client.delete(f"/documents/{doc_id}").json() == {"deleted": doc_id}
    assert len(client.get("/documents").json()) == 3
    assert client.delete(f"/documents/{doc_id}").status_code == 404


def test_mixed_batch_reports_each_file(client):
    r = upload(client, ["company_handbook.txt"], extra=[("files", ("broken.pdf", b"%PDF not real")),
                                                       ("files", ("tool.exe", b"MZ"))])
    assert r.status_code == 200
    assert [x["status"] for x in r.json()["results"]] == ["indexed", "error", "error"]


def test_batch_where_everything_fails_is_a_400(client):
    r = client.post("/documents", files=[("files", ("a.exe", b"x")), ("files", ("b.txt", b""))])
    assert r.status_code == 400 and len(r.json()["detail"]["results"]) == 2


def test_too_many_files_rejected(client, monkeypatch):
    monkeypatch.setattr(config, "API_MAX_FILES", 2)
    assert upload(client).status_code == 400


def test_oversized_file_is_reported_not_stored(client, monkeypatch):
    monkeypatch.setattr(app_logic, "MAX_UPLOAD_MB", 0.0001)
    r = client.post("/documents", files=[("files", ("big.txt", b"x" * 5000)),
                                         ("files", ("small.txt", b"hi"))])
    assert [x["status"] for x in r.json()["results"]] == ["error", "indexed"]


def test_folder_path_in_filename_is_stripped(client):
    client.post("/documents", files=[("files", ("../../etc/notes.txt", b"Hello there friends."))])
    assert [d["source"] for d in client.get("/documents").json()] == ["notes.txt"]


def test_malformed_doc_id_is_a_422(client):
    assert client.delete("/documents/not-a-valid-id").status_code == 422


def test_upload_without_files_is_a_422(client):
    assert client.post("/documents").status_code == 422


# ------------------------------ ask ------------------------------
def test_ask_returns_cited_answer_in_the_documented_shape(loaded):
    body = loaded.post("/ask", json={"question": Q}).json()
    assert body["answer"].startswith("Employees receive 24 days") and body["abstained"] is False
    assert body["mode"] == "llm" and body["rewritten"] is False
    assert body["sources"][0]["source"] == "leave_policy.docx" and body["sources"][0]["cited"] is True
    assert "error_detail" not in body                      # internal details are never exposed


def test_follow_up_with_history(loaded, fake_llm):
    fake_llm.replies = ["Do I need a medical certificate for sick leave?", "Yes, after 3 days [1]."]
    history = [{"role": "user", "content": "What is the sick leave policy?"},
               {"role": "assistant", "content": "12 days of paid sick leave [1]."}]
    body = loaded.post("/ask", json={"question": "Do I need a doctor's note for it?", "history": history}).json()
    assert body["rewritten"] and body["standalone_question"] == "Do I need a medical certificate for sick leave?"


def test_unanswerable_question_abstains_with_200(loaded, fake_llm):
    r = loaded.post("/ask", json={"question": "What is the stock option vesting schedule?"})
    assert r.status_code == 200 and r.json()["abstained"] is True and r.json()["sources"] == []
    assert fake_llm.calls == []


def test_source_filter_and_unknown_source(loaded):
    ok = loaded.post("/ask", json={"question": "annual leave days", "source": "company_handbook.txt"}).json()
    assert all(s["source"] == "company_handbook.txt" for s in ok["sources"])
    assert loaded.post("/ask", json={"question": Q, "source": "nope.pdf"}).status_code == 404


@pytest.mark.parametrize("payload", [
    {},                                                         # no question
    {"question": ""},
    {"question": "    "},                                       # blank
    {"question": "x" * 2001},                                   # too long
    {"question": "ok?", "history": [{"role": "system", "content": "evil"}]},      # invalid role
    {"question": "ok?", "history": [{"role": "user", "content": "m"}] * 21},      # history too long
    {"question": "ok?", "history": [{"role": "user", "content": ""}]},            # empty message
])
def test_invalid_ask_requests_are_422(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_unexpected_failure_is_a_generic_500(loaded, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret internal path /srv/db password=hunter2")
    monkeypatch.setattr(main, "answer_question", boom)
    r = loaded.post("/ask", json={"question": Q})
    assert r.status_code == 500 and "hunter2" not in r.text and "RuntimeError" not in r.text


def test_api_is_stateless_between_callers(loaded, fake_llm):
    fake_llm.replies = ["Do I need a medical certificate for sick leave?", "ok [1]"]
    loaded.post("/ask", json={"question": "Do I need a doctor's note for it?",
                              "history": [{"role": "user", "content": "What is the sick leave policy?"}]})
    body = loaded.post("/ask", json={"question": Q}).json()                  # a different caller, no history
    assert body["standalone_question"] == Q and body["rewritten"] is False


# ------------------------------ authentication ------------------------------
ENDPOINTS = [("get", "/documents", {}), ("post", "/ask", {"json": {"question": "hi there?"}}),
             ("delete", "/documents/0123456789abcdef", {}),
             ("post", "/documents", {"files": [("files", ("a.txt", b"hello world"))]})]


@pytest.mark.parametrize("method, url, kwargs", ENDPOINTS)
def test_protected_endpoints_require_the_key(client, monkeypatch, method, url, kwargs):
    monkeypatch.setattr(config, "API_KEY", "s3cret-key")
    call = getattr(client, method)
    assert call(url, **kwargs).status_code == 401
    assert call(url, headers={"X-API-Key": "wrong"}, **kwargs).status_code == 401
    assert call(url, headers={"X-API-Key": "s3cret-key"}, **kwargs).status_code != 401


def test_health_stays_open_when_auth_is_on(client, monkeypatch):
    monkeypatch.setattr(config, "API_KEY", "s3cret-key")
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["auth_required"] is True


# ------------------------------ start-up & demo script ------------------------------
def test_models_are_loaded_at_startup(store, monkeypatch):
    loaded = []
    monkeypatch.setattr(embeddings, "get_model", lambda: loaded.append("embed"))
    monkeypatch.setattr(reranker, "get_model", lambda: loaded.append("rerank"))
    with TestClient(main.app) as c:
        assert c.get("/health").status_code == 200
    assert loaded == ["embed", "rerank"]


def test_client_demo_script_runs_against_the_api(client, fake_llm):
    fake_llm.replies = ["Employees receive 24 days of annual leave [1].",
                        "How many days of sick leave do employees get?",
                        "Employees receive 12 days of sick leave [1]."]
    last = api_client_demo.run_demo(client, data_dir=DATA)
    assert last["rewritten"] and "12 days" in last["answer"]
