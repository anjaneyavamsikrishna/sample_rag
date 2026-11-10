from pathlib import Path

import pytest

from ingestion.document_loader import load_document
from retrieval.chunker import chunk_sections, chunk_text

DATA = Path(__file__).resolve().parent.parent / "data"

# 40 numbered sentences of exactly 6 words each = 240 words
LONG = " ".join(f"Sentence number {i} is here now." for i in range(1, 41))


def sections(name):
    return load_document(name, (DATA / name).read_bytes())


def test_short_text_is_one_chunk():
    assert len(chunk_text("Employees get 24 days of leave.")) == 1


def test_long_text_is_split_and_sized():
    chunks = chunk_text(LONG, max_words=60, overlap_words=12)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 60 for c in chunks)


def test_consecutive_chunks_overlap():
    chunks = chunk_text(LONG, max_words=60, overlap_words=12)
    for a, b in zip(chunks, chunks[1:]):
        last_sentence = a.split(". ")[-1].rstrip(".")
        assert last_sentence in b, "the end of one chunk must reappear in the next"


def test_no_text_is_lost():
    chunks = " ".join(chunk_text(LONG, max_words=60, overlap_words=12))
    for i in range(1, 41):
        assert f"Sentence number {i} is here now." in chunks


def test_giant_sentence_still_terminates():
    giant = "word " * 500
    chunks = chunk_text(giant, max_words=100, overlap_words=10)
    assert len(chunks) >= 5 and all(len(c.split()) <= 100 for c in chunks)


def test_invalid_settings_rejected():
    with pytest.raises(ValueError):
        chunk_text("x", max_words=50, overlap_words=50)
    with pytest.raises(ValueError):
        chunk_text("x", max_words=0, overlap_words=0)


def test_metadata_and_stable_ids():
    secs = sections("it_security_policy.pdf")
    a, b = chunk_sections(secs, "doc1"), chunk_sections(secs, "doc1")
    assert [c["chunk_id"] for c in a] == [c["chunk_id"] for c in b]      # same input -> same ids
    assert len({c["chunk_id"] for c in a}) == len(a)                     # ids are unique
    assert all(c["source"] == "it_security_policy.pdf" and c["page"] in (1, 2, 3) for c in a)


def test_key_facts_survive_intact_with_correct_page():
    chunks = chunk_sections(sections("it_security_policy.pdf"), "d")
    hit = [c for c in chunks if "within 2 hours" in c["text"]]
    assert hit and hit[0]["page"] == 3
    assert any("at least 14 characters" in c["text"] for c in chunks)


def test_csv_header_in_every_chunk():
    chunks = chunk_sections(sections("expense_policy.csv"), "d")
    assert all(c["text"].startswith("category,daily_limit_usd") for c in chunks)
    assert any("Hotel - international,260,Yes" in c["text"] for c in chunks)


def test_docx_table_row_is_in_a_chunk():
    chunks = chunk_sections(sections("leave_policy.docx"), "d")
    assert any("16 weeks paid" in c["text"] for c in chunks)


def test_handbook_splits_into_several_chunks():
    chunks = chunk_sections(sections("company_handbook.txt"), "d")
    assert len(chunks) >= 2 and all(c["word_count"] <= 150 for c in chunks)


def test_chunks_do_not_mix_sections():
    """Regression for the 'vacation roll-over' retrieval failure: one topic per chunk."""
    chunks = chunk_sections(sections("leave_policy.docx"), "d")
    carry = [c for c in chunks if "carried forward" in c["text"]]
    assert carry and all("medical certificate" not in c["text"] for c in carry)
    assert carry[0]["text"].startswith("1. Annual Leave")        # heading travels with its text


def test_handbook_chunks_start_with_their_heading():
    chunks = chunk_sections(sections("company_handbook.txt"), "d")
    hybrid = [c for c in chunks if "three days a week" in c["text"]]
    assert hybrid and hybrid[0]["text"].startswith("Hybrid working")


def test_embed_text_adds_title_but_text_stays_original():
    c = chunk_sections(sections("leave_policy.docx"), "d")[1]
    assert c["embed_text"].startswith("leave policy - ")
    assert "leave policy" not in c["text"].lower().split("\n")[0]    # shown evidence is unmodified
