"""Central settings for the whole project.

How it works:
  .env  (plain NAME=value lines, never committed)  ->  os.environ  ->  this file  ->  rest of the code
Every setting has a safe default, so the project still runs if .env is missing a line.
Other files do:  import config   and use  config.EMBED_MODEL, config.LLM_MODEL, ...
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root, no matter which folder you run Python from.
# (Real environment variables win over .env, which is handy for servers and tests.)
load_dotenv(Path(__file__).resolve().parent / ".env")

# --- Embeddings (Task 4) ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# Tied to the BGE model's training, so it lives in code, not in .env.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# --- Vector database (Task 5) ---
COLLECTION = "enterprise_documents"
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_data")
QDRANT_URL = os.getenv("QDRANT_URL") or None          # empty or missing -> use the local folder

# --- Retrieval + reranking (Task 6) ---
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "20"))        # candidates pulled from the vector database
TOP_K = int(os.getenv("TOP_K", "5"))                   # chunks kept after reranking
# Answers need at least this reranker score (0..1). CALIBRATE it: python calibrate_threshold.py
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "0.05"))

# --- Conversation memory (Task 8) ---
HISTORY_MESSAGES = int(os.getenv("HISTORY_MESSAGES", "6"))   # last N messages used to understand a follow-up

# --- API (Task 10) ---
API_KEY = os.getenv("API_KEY", "")        # empty = open API (development only). Set it to require X-API-Key.
API_MAX_FILES = int(os.getenv("API_MAX_FILES", "10"))

# --- LLM (Task 7) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")


def describe() -> dict:
    """Current settings with secrets masked, safe to print."""
    key = OPENAI_API_KEY
    masked = "(not set)" if not key or key == "your_api_key_here" else key[:3] + "..." + key[-2:]
    return {"EMBED_MODEL": EMBED_MODEL, "RERANK_MODEL": RERANK_MODEL, "RETRIEVE_K": RETRIEVE_K,
            "TOP_K": TOP_K, "MIN_RERANK_SCORE": MIN_RERANK_SCORE, "QDRANT_PATH": QDRANT_PATH, "QDRANT_URL": QDRANT_URL,
            "LLM_MODEL": LLM_MODEL, "OPENAI_BASE_URL": OPENAI_BASE_URL, "OPENAI_API_KEY": masked}


if __name__ == "__main__":          # run:  python config.py   to check your .env is being read
    for name, value in describe().items():
        print(f"{name:16} = {value}")
