"""Task 2 - Document parsing.

Every supported file type is converted into the SAME shape:
    {"text": str, "source": filename, "page": int | None, "location": str}
so later stages (chunking, vector store, UI) never care about the original format.
"""
import io
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

CSV_ROWS_PER_SECTION = 10


def _section(text: str, source: str, page: int | None, location: str) -> dict:
    return {"text": text, "source": source, "page": page, "location": location}


def _unwrap_lines(text: str) -> str:
    """PDFs hard-wrap lines for layout ("within 2\\nhours"). Re-join those lines, but keep a
    line break after sentence-ending punctuation so headings and clauses stay separated."""
    out = []
    for line in (ln.strip() for ln in text.splitlines()):
        if not line:
            continue
        if out and not out[-1].endswith((".", ":", ";", "?", "!")):
            out[-1] += " " + line
        else:
            out.append(line)
    return "\n".join(out)


def _load_pdf(filename: str, data: bytes) -> list[dict]:
    sections = []
    with fitz.open(stream=data, filetype="pdf") as pdf:        # open from memory, not disk
        for page_number, page in enumerate(pdf, start=1):      # humans count pages from 1
            text = _unwrap_lines(page.get_text("text"))
            if text:                                           # skip blank pages
                sections.append(_section(text, filename, page_number, f"page {page_number}"))
    if not sections:
        raise ValueError(f"No extractable text in {filename}. It may be a scanned PDF that needs OCR.")
    return sections


def _load_docx(filename: str, data: bytes) -> list[dict]:
    doc = Document(io.BytesIO(data))
    lines = []
    # doc.paragraphs ignores tables, so walk the body in order and handle both element types.
    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            text = Paragraph(child, doc).text.strip()
            if text:
                lines.append(text)
        elif child.tag.endswith("}tbl"):
            for row in Table(child, doc).rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    lines.append(" | ".join(cells))
    text = "\n".join(lines)
    return [_section(text, filename, None, "document body")] if text.strip() else []


def _load_csv(filename: str, data: bytes) -> list[dict]:
    # dtype=str keeps "007" as "007"; fillna("") avoids the literal text "nan"
    df = pd.read_csv(io.BytesIO(data), dtype=str).fillna("")
    sections = []
    for start in range(0, len(df), CSV_ROWS_PER_SECTION):
        part = df.iloc[start:start + CSV_ROWS_PER_SECTION]
        # to_csv(index=False) writes the header again, so every section keeps its column names
        sections.append(_section(part.to_csv(index=False), filename, None,
                                 f"rows {start + 1}-{start + len(part)}"))
    return sections


def _load_txt(filename: str, data: bytes) -> list[dict]:
    text = data.decode("utf-8-sig", errors="replace")          # tolerate BOMs and bad bytes
    return [_section(text, filename, None, "text")] if text.strip() else []


_LOADERS = {".pdf": _load_pdf, ".docx": _load_docx, ".csv": _load_csv, ".txt": _load_txt}


def load_document(filename: str, file_bytes: bytes) -> list[dict]:
    ext = Path(filename).suffix.lower()
    if ext not in _LOADERS:
        raise ValueError(f"Unsupported file type: {ext}")
    return _LOADERS[ext](filename, file_bytes)


if __name__ == "__main__":
    # Lets you run this file directly:  python -m ingestion.document_loader data/leave_policy.docx
    import sys

    if len(sys.argv) < 2:
        sys.exit("Usage: python -m ingestion.document_loader <path-to-file>")
    path = Path(sys.argv[1])
    for sec in load_document(path.name, path.read_bytes()):
        print(f"--- {sec['source']} | {sec['location']} | page={sec['page']}")
        print(sec["text"][:300], "\n")
