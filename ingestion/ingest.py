"""One function that does the whole upload pipeline: parse -> chunk -> embed -> store."""
import hashlib

from ingestion.document_loader import load_document
from retrieval import vector_store
from retrieval.chunker import chunk_sections


def ingest_file(filename: str, data: bytes) -> dict:
    # doc_id is a fingerprint of the CONTENT: same bytes -> same id.
    doc_id = hashlib.sha256(data).hexdigest()[:16]

    if vector_store.document_exists(doc_id):
        return {"status": "duplicate", "doc_id": doc_id, "source": filename, "chunks": 0}

    chunks = chunk_sections(load_document(filename, data), doc_id)
    if not chunks:
        raise ValueError(f"No text could be extracted from {filename}.")
    replaced = vector_store.index_document(doc_id, filename, chunks)
    return {"status": "replaced" if replaced else "indexed",
            "doc_id": doc_id, "source": filename, "chunks": len(chunks)}
