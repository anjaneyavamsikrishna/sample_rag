from pathlib import Path

from qdrant_client import QdrantClient

import config
from ingestion.ingest import ingest_file
from retrieval import vector_store as vs

DATA = Path(__file__).resolve().parent.parent / "data"


def chunk(text, n, source="a.txt", doc_id="d1", page=None):
    return {"text": text, "embed_text": text, "source": source, "page": page,
            "location": f"page {page}" if page else "text", "doc_id": doc_id,
            "chunk_id": f"{doc_id}_{n:03d}", "word_count": len(text.split())}


LEAVE = chunk("Employees receive 24 days of paid annual leave each year.", 0, "leave.txt", "L", 2)
FOOD = chunk("The cafeteria serves lunch from noon on weekdays.", 0, "food.txt", "F")


def test_search_on_empty_store_returns_nothing(store):
    assert store.search("anything") == []
    assert store.list_documents() == []


def test_search_finds_the_right_chunk_with_metadata(store):
    store.index_document("L", "leave.txt", [LEAVE])
    store.index_document("F", "food.txt", [FOOD])
    top = store.search("how many days of annual leave", limit=2)[0]
    assert top["source"] == "leave.txt" and top["page"] == 2 and "24 days" in top["text"]
    assert "vector_score" in top and "embed_text" not in top


def test_indexing_same_chunks_twice_does_not_duplicate(store):
    store.index_document("L", "leave.txt", [LEAVE])
    store.index_document("L", "leave.txt", [LEAVE])
    assert store.count() == 1


def test_same_name_replaces_old_version(store):
    store.index_document("L1", "leave.txt", [chunk("Old rule: 10 days.", 0, "leave.txt", "L1")])
    replaced = store.index_document("L2", "leave.txt", [chunk("New rule: 24 days.", 0, "leave.txt", "L2")])
    assert replaced is True and store.count() == 1
    assert "New rule" in store.search("rule")[0]["text"]


def test_delete_document(store):
    store.index_document("L", "leave.txt", [LEAVE])
    store.index_document("F", "food.txt", [FOOD])
    assert store.delete_document("L") is True and store.delete_document("L") is False
    assert [d["source"] for d in store.list_documents()] == ["food.txt"]


def test_source_filter(store):
    store.index_document("L", "leave.txt", [LEAVE])
    store.index_document("F", "food.txt", [FOOD])
    results = store.search("lunch leave", limit=5, source="food.txt")
    assert results and all(r["source"] == "food.txt" for r in results)


def test_list_documents_counts_chunks(store):
    store.index_document("D", "a.txt", [chunk("one", 0, "a.txt", "D"), chunk("two", 1, "a.txt", "D")])
    assert store.list_documents() == [{"doc_id": "D", "source": "a.txt", "chunks": 2}]


def test_data_survives_restart(tmp_path, monkeypatch, fake_embed):
    first = QdrantClient(path=str(tmp_path)); monkeypatch.setattr(vs, "_client", first)
    vs.index_document("L", "leave.txt", [LEAVE]); first.close()
    monkeypatch.setattr(vs, "_client", QdrantClient(path=str(tmp_path)))   # a "restart"
    assert vs.count() == 1 and vs.search("annual leave")[0]["source"] == "leave.txt"


def test_ingest_file_statuses(store):
    data = (DATA / "company_handbook.txt").read_bytes()
    first = ingest_file("company_handbook.txt", data)
    assert first["status"] == "indexed" and first["chunks"] > 1
    assert ingest_file("company_handbook.txt", data)["status"] == "duplicate"
    assert store.count() == first["chunks"]                       # nothing was added twice
    changed = ingest_file("company_handbook.txt", data + b"\n\nNew section\nSomething new here.")
    assert changed["status"] == "replaced"
    assert len(store.list_documents()) == 1


def test_ingest_all_four_sample_files(store):
    for name in ["it_security_policy.pdf", "leave_policy.docx", "expense_policy.csv", "company_handbook.txt"]:
        assert ingest_file(name, (DATA / name).read_bytes())["chunks"] > 0
    assert len(store.list_documents()) == 4
    sources = {r["source"] for r in store.search("annual leave days per year", limit=3)}
    assert "leave_policy.docx" in sources


def test_close_then_reopen_keeps_data(tmp_path, monkeypatch, fake_embed):
    """Lifecycle: use the real on-disk mode, close cleanly, reopen lazily, data still there."""
    monkeypatch.setattr(config, "QDRANT_PATH", str(tmp_path))
    monkeypatch.setattr(config, "QDRANT_URL", None)
    monkeypatch.setattr(vs, "_client", None)
    vs.index_document("L", "leave.txt", [LEAVE])
    vs.close_client()
    assert vs._client is None
    assert vs.count() == 1                       # get_client() reopened the same folder
    vs.close_client()
