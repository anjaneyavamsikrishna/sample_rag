"""Pick MIN_RERANK_SCORE from DATA instead of guessing.

For questions you KNOW are answerable and ones you KNOW are not, it records the best reranker score
each gets. A good threshold sits between the two groups.   Run: python calibrate_threshold.py
Add your own questions to the lists below for a more trustworthy result."""
import math

import config
from _shared_scripts import ensure_indexed
from retrieval import vector_store
from retrieval.reranker import rerank

ANSWERABLE = [
    "How many days of annual leave do employees get?",
    "How much time off do I get per year?",
    "What is the minimum password length?",
    "How many vacation days can I roll over to next year?",
    "How quickly must a security incident be reported?",
    "What is the daily hotel limit for international travel?",
    "How many weeks of parental leave for a primary caregiver?",
    "What is the sick leave policy?",
    "Do I need a doctor's note for sick leave?",
    "My laptop was lost. What must I do?",
    "Which days do we work from the office?",
]
UNANSWERABLE = [
    "What is the stock option vesting schedule?",
    "How much is the monthly remote-work internet stipend?",
    "Who is the CEO of the company?",
    "What was the company's revenue last year?",
    "What is the best pizza recipe?",
    "How do I reset my Netflix password?",
]

ensure_indexed()


def best_score(q):
    ranked = rerank(q, vector_store.search(q, config.RETRIEVE_K))
    return ranked[0]["rerank_score"] if ranked else 0.0


rows = [(best_score(q), "answerable", q) for q in ANSWERABLE] + \
       [(best_score(q), "NOT in docs", q) for q in UNANSWERABLE]
print(f"{'score':>7}  {'type':11}  question")
for score, kind, q in sorted(rows, reverse=True):
    print(f"{score:7.3f}  {kind:11}  {q}")

lowest_ok = min(s for s, k, _ in rows if k == "answerable")
highest_bad = max(s for s, k, _ in rows if k != "answerable")
print(f"\nlowest  answerable score : {lowest_ok:.3f}")
print(f"highest unanswerable score: {highest_bad:.3f}")
if lowest_ok > highest_bad:
    suggestion = math.sqrt(max(lowest_ok, 1e-4) * max(highest_bad, 1e-4))
    print("The two groups are cleanly separated.")
    print(f"\nPut this in your .env file:\nMIN_RERANK_SCORE={suggestion:.3f}")
else:
    print("WARNING: the groups overlap, so no single threshold is perfect.")
    print(f"Safer against made-up answers: MIN_RERANK_SCORE={highest_bad + 0.01:.3f} (a few real questions may be refused)")
    print(f"Safer against refusing real questions: MIN_RERANK_SCORE={max(lowest_ok - 0.01, 0.01):.3f}")
print(f"(current setting: MIN_RERANK_SCORE={config.MIN_RERANK_SCORE})")
vector_store.close_client()
