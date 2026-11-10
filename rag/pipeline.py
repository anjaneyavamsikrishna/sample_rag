"""Task 7 - Answer generation: retrieve -> (gate) -> LLM -> validate citations -> result.

Three layers stop made-up answers:
  1. the relevance gate (no good evidence  -> never call the LLM),
  2. the prompt (the LLM must reply NOT_ENOUGH_EVIDENCE if the answer isn't there),
  3. citation validation (citations pointing at evidence that doesn't exist are removed and flagged).
"""
import logging
import re

import config
from rag import llm, memory
from rag.prompts import ANSWER_SYSTEM, NO_ANSWER_MARKER, build_user_prompt
from retrieval.retriever import retrieve

NOT_FOUND = "I could not find sufficient evidence in the uploaded documents to answer that."
LLM_DOWN = "The language model is unavailable right now, so I could not write an answer. The most relevant passages are listed below."
NO_LLM = "No language model is configured, so here is the most relevant passage instead (extractive mode)."

log = logging.getLogger(__name__)
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def clean_citations(answer: str, n_evidence: int) -> tuple[str, set[int], set[int]]:
    """Return (answer without invalid citations, valid cited ids, invalid ids that were removed)."""
    valid, invalid = set(), set()

    def fix(match):
        ids = [int(x) for x in re.findall(r"\d+", match.group(1))]
        good = [i for i in ids if 1 <= i <= n_evidence]
        valid.update(good)
        invalid.update(i for i in ids if i not in good)
        return "".join(f"[{i}]" for i in good)

    return _CITATION.sub(fix, answer), valid, invalid


def _scrub(text: str) -> str:
    """Never let the API key leak into messages, even partially."""
    key = config.OPENAI_API_KEY
    return text.replace(key, "***") if key else text


def _sources(chunks: list[dict], cited: set[int]) -> list[dict]:
    return [{"id": i, "source": c["source"], "page": c.get("page"), "location": c.get("location"),
             "score": round(c["rerank_score"], 3), "text": c["text"], "cited": i in cited}
            for i, c in enumerate(chunks, start=1)]


def _result(answer, chunks=(), cited=(), *, question, standalone=None, mode="llm", abstained=False,
            reason=None, warnings=(), best_score=0.0, error_detail=None):
    standalone = standalone or question
    return {"answer": answer, "sources": _sources(list(chunks), set(cited)), "question": question,
            "standalone_question": standalone, "rewritten": standalone != question,
            "mode": mode, "abstained": abstained, "reason": reason, "warnings": list(warnings),
            "best_score": round(best_score, 3), "error_detail": error_detail}


def answer_question(question: str, history: list[dict] | None = None, source: str | None = None) -> dict:
    """history = earlier messages of THIS conversation: [{"role": "user"|"assistant", "content": "..."}]."""
    question = question.strip()
    if not question:
        raise ValueError("Question is empty.")

    standalone = memory.rewrite_query(question, history)           # "...for it?" -> "...for sick leave?"

    def finish(answer, chunks=(), cited=(), **kw):
        return _result(answer, chunks, cited, question=question, standalone=standalone, **kw)

    found = retrieve(standalone, source)
    if not found["relevant"]:                                       # layer 1: the gate
        return finish(NOT_FOUND, abstained=True, reason="no_relevant_evidence", best_score=found["best_score"])
    chunks = found["chunks"]

    if not llm.is_configured():                                     # no-key mode: show the evidence
        return finish(f"{NO_LLM}\n\n{chunks[0]['text']} [1]", chunks, {1}, mode="extractive",
                      best_score=found["best_score"])

    try:
        raw = llm.chat(ANSWER_SYSTEM, build_user_prompt(standalone, chunks))   # answered from NEW evidence only
    except Exception as exc:                                        # network, key, quota ... never crash
        detail = _scrub(f"{type(exc).__name__}: {exc}")
        log.warning("LLM call failed: %s", detail)                  # the REAL reason is never hidden
        return finish(LLM_DOWN, chunks, mode="error", reason=type(exc).__name__,
                      best_score=found["best_score"], error_detail=detail)

    if NO_ANSWER_MARKER in raw:                                     # layer 2: the model itself abstains
        return finish(NOT_FOUND, abstained=True, reason="model_found_insufficient_evidence",
                      best_score=found["best_score"])

    answer, cited, invalid = clean_citations(raw, len(chunks))      # layer 3: citation validation
    warnings = []
    if invalid:
        warnings.append(f"Removed invalid citation(s): {sorted(invalid)}")
    if not cited:
        warnings.append("The answer contains no citations; treat it with caution.")
    return finish(answer, chunks, cited, warnings=warnings, best_score=found["best_score"])
