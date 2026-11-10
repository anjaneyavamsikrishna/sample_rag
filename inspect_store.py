"""Index the sample files into the PERSISTENT local database, then search it.

    python inspect_store.py                      # index data/ and ask a default question
    python inspect_store.py "your question"      # ask your own
Run it twice: the second run should report every file as 'duplicate' (nothing is stored twice)."""
import sys
from pathlib import Path

from ingestion.ingest import ingest_file
from retrieval import vector_store

for path in sorted(Path("data").iterdir()):
    if path.suffix.lower() in {".pdf", ".docx", ".csv", ".txt"}:
        r = ingest_file(path.name, path.read_bytes())
        print(f"{r['status']:10} {r['source']:26} chunks={r['chunks']}")

print("\nIn the database now:")
for d in vector_store.list_documents():
    print(f"  {d['source']:26} {d['chunks']} chunks   id={d['doc_id']}")
print("  total chunks:", vector_store.count())

question = " ".join(sys.argv[1:]) or "How many vacation days can I roll over to next year?"
print(f"\nQ: {question}")
for i, c in enumerate(vector_store.search(question, limit=5), 1):
    print(f"  {i}. score={c['vector_score']:.3f}  {c['source']} [{c['location']}]")
    print(f"     {c['text'][:90].replace(chr(10), ' ')}...")

vector_store.close_client()
