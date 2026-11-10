"""Task 5 - Vector store (Qdrant).

Qdrant stores each chunk as a "point": a vector (for similarity search) plus a payload (the readable
text and citation metadata). It answers "which stored vectors are closest to this question vector?"
quickly, survives restarts, and supports filters, replacement and deletion.

Key design decisions:
  * Point IDs come from chunk_id (uuid5), so indexing the same chunk twice OVERWRITES, never duplicates.
  * Re-uploading a file with the same NAME replaces its old chunks (document replacement).
  * We embed `embed_text` but store the original `text` for display.
  * Local mode (a folder) is simplest; set QDRANT_URL to use a Docker server instead.
"""
import atexit
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

import config
from retrieval.embeddings import embed_documents, embed_query

_client = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = (QdrantClient(url=config.QDRANT_URL) if config.QDRANT_URL
                   else QdrantClient(path=config.QDRANT_PATH))
    return _client


def close_client() -> None:
    """Close the database cleanly while Python is still fully alive.
    Without this, the local client is closed by its __del__ during interpreter shutdown, which on
    newer Python versions prints 'ImportError: sys.meta_path is None' (harmless, but ugly)."""
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None          # drop the reference now, so cleanup happens now, not at shutdown


atexit.register(close_client)   # runs automatically when the program ends


def _exists() -> bool:
    return get_client().collection_exists(config.COLLECTION)


def _ensure_collection(dim: int) -> None:
    c = get_client()
    if c.collection_exists(config.COLLECTION):
        return
    c.create_collection(config.COLLECTION, vectors_config=models.VectorParams(
        size=dim, distance=models.Distance.COSINE))
    if config.QDRANT_URL:  # payload indexes speed up filters on a real server (no effect in local mode)
        for field in ("doc_id", "source"):
            c.create_payload_index(config.COLLECTION, field, models.PayloadSchemaType.KEYWORD)


def _equals(field: str, value: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key=field, match=models.MatchValue(value=value))])


def count(flt: models.Filter | None = None) -> int:
    if not _exists():
        return 0
    return get_client().count(config.COLLECTION, count_filter=flt, exact=True).count


def document_exists(doc_id: str) -> bool:
    return count(_equals("doc_id", doc_id)) > 0


def delete_by_source(source: str) -> bool:
    """Remove every chunk of a file name. Returns True if something was removed."""
    flt = _equals("source", source)
    if count(flt) == 0:
        return False
    get_client().delete(config.COLLECTION, points_selector=models.FilterSelector(filter=flt))
    return True


def delete_document(doc_id: str) -> bool:
    flt = _equals("doc_id", doc_id)
    if count(flt) == 0:
        return False
    get_client().delete(config.COLLECTION, points_selector=models.FilterSelector(filter=flt))
    return True


def index_document(doc_id: str, source: str, chunks: list[dict]) -> bool:
    """Store a document's chunks. Replaces an older version with the same file name.
    Returns True if it replaced one."""
    if not chunks:
        raise ValueError("No chunks to index")
    replaced = delete_by_source(source)
    vectors = embed_documents([c["embed_text"] for c in chunks])
    _ensure_collection(len(vectors[0]))
    points = [
        models.PointStruct(
            id=str(uuid5(NAMESPACE_URL, c["chunk_id"])),       # stable ID => no duplicates
            vector=v.tolist(),
            payload={k: val for k, val in c.items() if k != "embed_text"})
        for c, v in zip(chunks, vectors)
    ]
    for i in range(0, len(points), 64):
        get_client().upsert(config.COLLECTION, points=points[i:i + 64])
    return replaced


def search(query: str, limit: int = 20, source: str | None = None) -> list[dict]:
    """Return the `limit` chunks closest in meaning to the question (optionally one file only)."""
    if count() == 0:
        return []
    result = get_client().query_points(
        config.COLLECTION, query=embed_query(query).tolist(), limit=limit, with_payload=True,
        query_filter=_equals("source", source) if source else None)
    return [{**p.payload, "vector_score": p.score} for p in result.points]


def list_documents() -> list[dict]:
    if not _exists():
        return []
    docs, offset = {}, None
    while True:
        points, offset = get_client().scroll(config.COLLECTION, limit=500, offset=offset,
                                             with_payload=["doc_id", "source"], with_vectors=False)
        for p in points:
            d = docs.setdefault(p.payload["doc_id"], {"doc_id": p.payload["doc_id"],
                                                      "source": p.payload["source"], "chunks": 0})
            d["chunks"] += 1
        if offset is None:
            break
    return sorted(docs.values(), key=lambda d: d["source"])
