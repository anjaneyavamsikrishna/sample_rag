from ingestion.document_loader import load_document

for name in ["it_security_policy.pdf", "leave_policy.docx",
             "expense_policy.csv", "company_handbook.txt"]:
    with open(f"data/{name}", "rb") as f:
        sections = load_document(name, f.read())
    print("=" * 60)
    print(name, "->", len(sections), "section(s)")
    for s in sections:
        print(f"  [{s['location']}] page={s['page']}")
        print("  ", s["text"][:150].replace("\n", " "), "...")
