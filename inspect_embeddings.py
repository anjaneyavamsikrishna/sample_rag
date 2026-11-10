"""Look inside the embeddings. Run: python inspect_embeddings.py   (downloads the model on first run)"""
import numpy as np

from ingestion.document_loader import load_document
from retrieval.chunker import chunk_sections
from retrieval import embeddings

FILES = ["it_security_policy.pdf", "leave_policy.docx", "expense_policy.csv", "company_handbook.txt"]
chunks = []
for name in FILES:
    chunks += chunk_sections(load_document(name, open(f"data/{name}", "rb").read()), name)

vectors = embeddings.embed_documents([c["embed_text"] for c in chunks])
print(f"{len(chunks)} chunks -> vectors of shape {vectors.shape}")
print("first 6 numbers of chunk 0's vector:", np.round(vectors[0][:6], 4))
print("vector length (should be 1.0):", round(float(np.linalg.norm(vectors[0])), 4))

QUESTIONS = [
    ("How many days of annual leave does a full-time employee get?", "24 days"),
    ("What is the minimum password length?", "at least 14 characters"),
    ("How many vacation days can I roll over, and when do they expire?", "carried forward"),
    ("How quickly must a security incident be reported?", "within 2 hours"),
    ("What is the daily hotel limit for international travel?", "Hotel - international,260"),
    ("How many weeks of parental leave does a primary caregiver get?", "16 weeks paid"),
    ("What is the stock option vesting schedule?", None),            # NOT in the documents
    ("How much is the monthly remote-work internet stipend?", None),  # NOT in the documents
]
for q, expected in QUESTIONS:
    top = embeddings.rank(q, chunks, vectors, k=3)
    print("\nQ:", q)
    for i, c in enumerate(top, 1):
        mark = "  <-- has the answer" if expected and expected in c["text"] else ""
        print(f"  {i}. score={c['score']:.3f}  {c['source']} [{c['location']}]{mark}")
