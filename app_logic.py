"""Everything the web app DOES, kept separate from how it LOOKS so it can be tested without a browser."""
import logging
from pathlib import Path

import config
from ingestion.ingest import ingest_file
from rag import llm
from rag.pipeline import answer_question
from retrieval import vector_store

log = logging.getLogger(__name__)

MAX_UPLOAD_MB = 25
SUPPORTED_TYPES = ["pdf", "docx", "csv", "txt"]
DATA_DIR = Path(__file__).resolve().parent / "data"


def process_uploads(files: list[tuple[str, bytes]]) -> list[dict]:
    """Index each (filename, bytes). One bad file never stops the others: every file gets a status dict."""
    results = []
    for name, data in files:
        try:
            if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
                raise ValueError(f"File is larger than {MAX_UPLOAD_MB} MB.")
            if not data:
                raise ValueError("File is empty.")
            results.append(ingest_file(name, data))
        except ValueError as exc:                       # our own, user-readable problems
            results.append({"status": "error", "source": name, "chunks": 0, "detail": str(exc)})
        except Exception as exc:                        # corrupt PDF, password-protected file, ...
            log.warning("Could not process %s: %s: %s", name, type(exc).__name__, exc)
            results.append({"status": "error", "source": name, "chunks": 0,
                            "detail": "The file could not be read. It may be corrupt or password-protected."})
    return results


def load_sample_data() -> list[dict]:
    files = [(p.name, p.read_bytes()) for p in sorted(DATA_DIR.glob("*")) if p.suffix[1:].lower() in SUPPORTED_TYPES]
    return process_uploads(files)


def describe_upload(result: dict) -> tuple[str, str]:
    """(level, message) for the UI: level is 'success', 'info' or 'error'."""
    name, status = result["source"], result["status"]
    if status == "indexed":
        return "success", f"Indexed {name} ({result['chunks']} chunks)"
    if status == "replaced":
        return "success", f"Updated {name}: the old version was replaced ({result['chunks']} chunks)"
    if status == "duplicate":
        return "info", f"{name} is already indexed (identical content), skipped"
    return "error", f"{name}: {result.get('detail', 'failed')}"


def history_for_pipeline(messages: list[dict]) -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def ask(question: str, messages: list[dict], source: str | None) -> dict:
    """Never raises: a failure becomes a friendly result the UI can show."""
    try:
        return answer_question(question, history_for_pipeline(messages), source)
    except Exception as exc:
        log.exception("answer_question failed")
        return {"answer": "Something went wrong while answering. Please try again; details are in the terminal.",
                "sources": [], "warnings": [], "mode": "error", "abstained": False, "reason": type(exc).__name__,
                "error_detail": f"{type(exc).__name__}: {exc}", "standalone_question": question,
                "rewritten": False, "best_score": 0.0}


def source_title(src: dict) -> str:
    return f"{src['source']} · {src['location']}" if src.get("location") else src["source"]


def llm_status() -> str:
    return (f"Answers by `{config.LLM_MODEL}`" if llm.is_configured()
            else "No LLM configured: extractive mode (shows the best passage)")
