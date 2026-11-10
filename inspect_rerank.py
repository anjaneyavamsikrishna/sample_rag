"""See vector search vs reranking side by side.

    python inspect_rerank.py "How many vacation days can I roll over to next year?"
"""
import sys

import config
from _shared_scripts import ensure_indexed
from retrieval import vector_store
from retrieval.reranker import rerank

ensure_indexed()
question = " ".join(sys.argv[1:]) or "How many vacation days can I roll over to next year?"
candidates = vector_store.search(question, config.RETRIEVE_K)
ranked = rerank(question, candidates)

print(f"Q: {question}\n")
print("VECTOR SEARCH order (top 5):")
for i, c in enumerate(candidates[:5], 1):
    print(f"  {i}. cos={c['vector_score']:.3f}  {c['source']} [{c['location']}]")
print(f"\nAFTER RERANKING (threshold = {config.MIN_RERANK_SCORE}):")
for i, c in enumerate(ranked[:5], 1):
    gate = "PASS" if c["rerank_score"] >= config.MIN_RERANK_SCORE else "below threshold"
    print(f"  {i}. score={c['rerank_score']:.3f} (was #{c['vector_rank']})  {c['source']} [{c['location']}]  {gate}")
    print(f"     {c['text'][:85].replace(chr(10), ' ')}...")
vector_store.close_client()
