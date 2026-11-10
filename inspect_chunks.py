import hashlib

from ingestion.document_loader import load_document
from retrieval.chunker import chunk_sections

total = 0
for name in ["it_security_policy.pdf", "leave_policy.docx",
             "expense_policy.csv", "company_handbook.txt"]:
    data = open(f"data/{name}", "rb").read()
    doc_id = hashlib.sha256(data).hexdigest()[:8]          # fingerprint of the file content
    chunks = chunk_sections(load_document(name, data), doc_id)
    total += len(chunks)
    print("=" * 70)
    print(f"{name}: {len(chunks)} chunk(s)")
    for c in chunks:
        print(f"  {c['chunk_id']} | {c['location']:<14} | {c['word_count']:>3} words | "
              f"{c['text'][:70].replace(chr(10), ' ')}...")
print("\nTOTAL chunks:", total)

# Show the overlap with your own eyes (handbook):
data = open("data/company_handbook.txt", "rb").read()
cs = chunk_sections(load_document("company_handbook.txt", data), "x")
print("\n--- END of chunk 0 ---\n", cs[0]["text"][-200:])
print("\n--- START of chunk 1 ---\n", cs[1]["text"][:200])
