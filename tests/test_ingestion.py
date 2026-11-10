from pathlib import Path

import fitz
import pytest

from ingestion.document_loader import load_document

DATA = Path(__file__).resolve().parent.parent / "data"


def load(name):
    return load_document(name, (DATA / name).read_bytes())


def test_pdf_has_three_pages_with_numbers():
    sections = load("it_security_policy.pdf")
    assert [s["page"] for s in sections] == [1, 2, 3]
    assert "14 characters" in sections[0]["text"]
    assert "2 hours" in sections[2]["text"]


def test_pdf_metadata():
    s = load("it_security_policy.pdf")[0]
    assert s["source"] == "it_security_policy.pdf"
    assert s["location"] == "page 1"


def test_docx_includes_table_content():
    sections = load("leave_policy.docx")
    text = " ".join(s["text"] for s in sections)
    assert "24 days" in text
    assert "16 weeks" in text          # this lives inside the table
    assert sections[0]["page"] is None


def test_csv_split_keeps_header():
    sections = load("expense_policy.csv")
    assert len(sections) == 2          # 16 rows -> 10 + 6
    assert all(s["text"].startswith("category,") for s in sections)
    assert sections[1]["location"] == "rows 11-16"


def test_txt_loads():
    assert "Hyderabad" in load("company_handbook.txt")[0]["text"]


def test_unsupported_type():
    with pytest.raises(ValueError):
        load_document("virus.exe", b"x")


def test_empty_pdf_raises():
    blank = fitz.open(); blank.new_page()
    with pytest.raises(ValueError):
        load_document("scan.pdf", blank.tobytes())
