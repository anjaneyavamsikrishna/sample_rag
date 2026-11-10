"""Task 10 - FastAPI layer.   Run:  uvicorn api.main:app --reload     then open http://127.0.0.1:8000/docs

NOTE: the embedded (folder) Qdrant database can be opened by only ONE program. To run this API and the
Streamlit app at the same time, start Qdrant in Docker and set QDRANT_URL=http://localhost:6333 in .env.
"""
import logging
import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import PurePath

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi import Path as PathParam
from fastapi.concurrency import run_in_threadpool

import app_logic as logic
import config
from api.schemas import AskRequest, AskResponse, DocumentOut, UploadResponse
from rag import llm
from rag.pipeline import answer_question
from retrieval import embeddings, reranker, vector_store

log = logging.getLogger("api")
write_lock = threading.Lock()          # the local database should not be written by two requests at once


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(embeddings.get_model)      # load models at start-up, not on the first request
    await run_in_threadpool(reranker.get_model)
    yield
    vector_store.close_client()


app = FastAPI(title="Enterprise Knowledge Assistant API", version="1.0.0", lifespan=lifespan,
              description="Upload documents, then ask questions and receive cited answers.")


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """If API_KEY is set in .env, every protected request must carry the same value in X-API-Key."""
    expected = config.API_KEY
    if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
        raise HTTPException(401, "Invalid or missing API key.", headers={"WWW-Authenticate": "ApiKey"})


protected = [Depends(require_api_key)]


@app.get("/health")
def health():
    return {"status": "ok", "auth_required": bool(config.API_KEY), "llm_configured": llm.is_configured()}


def _ingest_locked(items):
    with write_lock:
        return logic.process_uploads(items)


@app.post("/documents", response_model=UploadResponse, dependencies=protected)
async def upload_documents(files: list[UploadFile] = File(...)):
    """Index one or more files. Each file gets its own result, so one bad file never blocks the rest."""
    if len(files) > config.API_MAX_FILES:
        raise HTTPException(400, f"At most {config.API_MAX_FILES} files per request.")
    limit = int(logic.MAX_UPLOAD_MB * 1024 * 1024)
    items = []
    for f in files:
        name = PurePath((f.filename or "unnamed").replace("\\", "/")).name[:200] or "unnamed"   # no folder paths
        items.append((name, await f.read(limit + 1)))        # read one byte past the limit to detect oversize
    results = await run_in_threadpool(_ingest_locked, items)
    if all(r["status"] == "error" for r in results):
        raise HTTPException(400, detail={"message": "No file could be processed.", "results": results})
    return {"results": results}


@app.get("/documents", response_model=list[DocumentOut], dependencies=protected)
def list_documents():
    return vector_store.list_documents()


@app.delete("/documents/{doc_id}", dependencies=protected)
def delete_document(doc_id: str = PathParam(pattern=r"^[0-9a-f]{16}$")):
    with write_lock:
        deleted = vector_store.delete_document(doc_id)
    if not deleted:
        raise HTTPException(404, "No document with that id.")
    return {"deleted": doc_id}


@app.post("/ask", response_model=AskResponse, dependencies=protected)
def ask(req: AskRequest):
    """Ask a question. `abstained: true` means the documents do not contain a supported answer."""
    if req.source and req.source not in {d["source"] for d in vector_store.list_documents()}:
        raise HTTPException(404, f"No indexed document named '{req.source}'.")
    try:
        return answer_question(req.question, [m.model_dump() for m in req.history], req.source)
    except Exception:
        log.exception("ask failed")                          # details go to the server log only
        raise HTTPException(500, "Unable to process the question.")
