"""Find out WHY a question is retrieved badly. Run:  python diagnose_retrieval.py

For each setup it prints the RANK of the chunk that really contains the answer (1 = best).
Setups:  A = your original failing setup   D = the new default.   Compare the columns."""
from ingestion.document_loader import load_document
from retrieval import embeddings
from retrieval.chunker import chunk_sections

FILES = ["it_security_policy.pdf", "leave_policy.docx", "expense_policy.csv", "company_handbook.txt"]
SECTIONS = {n: load_document(n, open(f"data/{n}", "rb").read()) for n in FILES}

QUESTIONS = [
    ("How many days of annual leave do employees get?", "24 days"),
    ("How much time off do I get per year?", "24 days"),
    ("What is the minimum password length?", "at least 14 characters"),
    ("How many vacation days can I roll over to next year?", "carried forward"),
    ("How quickly must a security incident be reported?", "within 2 hours"),
    ("What is the daily hotel limit for international travel?", "Hotel - international,260"),
    ("How many weeks of parental leave for a primary caregiver?", "16 weeks paid"),
]
SETUPS = {  # name: (max_words, overlap, by_heading, text field that gets embedded)
    "A: big chunks, raw":      (150, 40, False, "text"),
    "B: + title context":      (150, 40, False, "embed_text"),
    "C: + split by heading":   (100, 25, True,  "text"),
    "D: heading + title (new)": (100, 25, True,  "embed_text"),
}


def build(setup):
    mw, ov, by_heading, field = setup
    chunks = [c for n in FILES for c in chunk_sections(SECTIONS[n], n, mw, ov, by_heading)]
    return chunks, embeddings.embed_documents([c[field] for c in chunks])


def answer_rank(question, expected, chunks, vectors):
    ranked = embeddings.rank(question, chunks, vectors, k=len(chunks))
    for i, c in enumerate(ranked, 1):
        if expected in c["text"]:
            return i, ranked
    return None, ranked


built = {name: build(s) for name, s in SETUPS.items()}
print(f"{'question':58}" + "".join(f"{n[:2]:>5}" for n in SETUPS))
wins = dict.fromkeys(SETUPS, 0)
for q, exp in QUESTIONS:
    row = f"{q[:56]:58}"
    for name, (chunks, vectors) in built.items():
        r, _ = answer_rank(q, exp, chunks, vectors)
        wins[name] += (r or 99) <= 3
        row += f"{(r or '-'):>5}"
    print(row)
print("\nanswered in top 3:", {n[:2]: f"{w}/{len(QUESTIONS)}" for n, w in wins.items()})
print("chunks per setup :", {n[:2]: len(built[n][0]) for n in SETUPS})

q, exp = QUESTIONS[3]
print(f"\nFull ranking for setup A on: {q}")
r, ranked = answer_rank(q, exp, *built["A: big chunks, raw"])
for i, c in enumerate(ranked[:6], 1):
    print(f"  {i}. {c['score']:.3f} {c['source']} [{c['location']}]" + ("  <-- answer" if exp in c["text"] else ""))
print(f"  ... answer chunk is at rank {r}")
