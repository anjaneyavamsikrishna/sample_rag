"""Task 8 - Conversation memory.

Retrieval needs a COMPLETE question. "Do I need a doctor's note for it?" matches nothing useful because
the search engine cannot know what "it" is. So before searching we rewrite the follow-up into a standalone
question using the recent conversation:

    history:   user: What is the sick leave policy?   assistant: 12 days of paid sick leave ...
    follow-up: "Do I need a doctor's note for it?"
    rewritten: "Do I need a doctor's note for sick leave?"

Design rules:
  * History is passed in explicitly on every call. No global state, so one user's conversation can never
    leak into another's (the Streamlit app and the API keep one history per session).
  * History is used ONLY to rewrite the question. Earlier answers are never fed to the answering step as
    facts: every answer must be backed by freshly retrieved evidence.
  * If the LLM is unavailable or returns something odd, a simple rule-based fallback is used.
"""
import logging
import re

import config
from rag import llm
from rag.prompts import REWRITE_SYSTEM, build_rewrite_prompt

log = logging.getLogger(__name__)

REFERENCE_WORDS = {"it", "its", "they", "them", "their", "this", "that", "those", "these",
                   "he", "she", "his", "her", "there", "one", "ones"}
FOLLOWUP_STARTS = ("and ", "also ", "but ", "what about", "how about", "why ", "so ")
LABEL = re.compile(r"^(standalone question|rewritten question|rewritten|question|query)\s*:\s*", re.I)


def trim_history(history: list[dict] | None, max_messages: int | None = None) -> list[dict]:
    """Keep only the most recent messages, and only the role/content fields."""
    limit = max_messages if max_messages is not None else config.HISTORY_MESSAGES
    clean = [{"role": m["role"], "content": str(m["content"])} for m in (history or [])
             if m.get("role") in ("user", "assistant") and m.get("content")]
    return clean[-limit:] if limit > 0 else []


def _last_user_message(history: list[dict]) -> str | None:
    return next((m["content"] for m in reversed(history) if m["role"] == "user"), None)


def is_followup(question: str) -> bool:
    words = re.findall(r"[a-z']+", question.lower())
    return (len(words) <= 3 or bool(REFERENCE_WORDS & set(words))
            or question.lower().startswith(FOLLOWUP_STARTS))


def heuristic_rewrite(question: str, history: list[dict]) -> str:
    """No-LLM fallback: if it looks like a follow-up, prepend the previous user question."""
    previous = _last_user_message(history)
    if previous and is_followup(question):
        return f"{previous.strip().rstrip('?')}? {question.strip()}"
    return question


def _clean(reply: str, original: str) -> str | None:
    """Return a safe one-line question, or None if the model's reply cannot be trusted."""
    text = LABEL.sub("", reply.strip()).strip().strip('"\'`').strip()
    if not text or "\n" in text or len(text) > 4 * len(original) + 150:
        return None                       # empty, multi-line, or rambling: probably an answer, not a question
    return text


def rewrite_query(question: str, history: list[dict] | None) -> str:
    history = trim_history(history)
    if not history:
        return question                   # first message: nothing to resolve
    if not llm.is_configured():
        return heuristic_rewrite(question, history)
    try:
        cleaned = _clean(llm.chat(REWRITE_SYSTEM, build_rewrite_prompt(question, history)), question)
    except Exception as exc:              # a rewrite failure must never block the answer
        log.warning("Query rewrite failed (%s: %s); using rule-based fallback", type(exc).__name__, exc)
        return heuristic_rewrite(question, history)
    return cleaned or heuristic_rewrite(question, history)
