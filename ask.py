"""Ask the assistant from the terminal.

    python ask.py "How many vacation days can I roll over?"     # one question
    python ask.py                                               # interactive; empty line to quit
"""
import sys

from _shared_scripts import ensure_indexed
from rag.pipeline import answer_question
from retrieval import vector_store


def show(question: str, history: list[dict] | None = None) -> dict:
    r = answer_question(question, history)
    if r["rewritten"]:
        print(f"\n(searched as: {r['standalone_question']})")
    print(f"\n{r['answer']}\n")
    if r["abstained"]:
        print(f"(abstained: {r['reason']}, best evidence score {r['best_score']})")
    if r["mode"] == "error":
        print(f"LLM ERROR: {r['error_detail']}")
        print("-> run  python check_llm.py  for a step-by-step diagnosis")
    for w in r["warnings"]:
        print("WARNING:", w)
    for s in r["sources"]:
        mark = "cited" if s["cited"] else "retrieved"
        print(f"  [{s['id']}] {s['source']} - {s['location']}  ({mark}, score {s['score']})")
    print(f"  mode: {r['mode']}")
    return r


ensure_indexed()
if len(sys.argv) > 1:
    show(" ".join(sys.argv[1:]))
else:
    history = []                                     # this conversation's memory
    while (q := input("\nQuestion (empty to quit): ").strip()):
        r = show(q, history)
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": r["answer"]}]
vector_store.close_client()
