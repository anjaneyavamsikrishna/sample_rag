"""Task 3 - Chunking.

Why: a whole document is too big to embed or send to an LLM for every question, and one vector for a
whole page is too blurry to match a specific fact. We cut sections into small, overlapping chunks.

Design choices:
  * Cut at SENTENCE boundaries, never in the middle of a fact.
  * Size is measured in WORDS (simple, no model download). 1 word ~ 1.3 tokens, so 150 words ~ 200 tokens.
    A later task swaps this for the embedding model's real tokenizer.
  * OVERLAP: the last sentences of one chunk are repeated at the start of the next, so a fact that sits
    on a boundary is still fully visible in at least one chunk.
  * Every chunk carries the metadata of its section (source, page, location) plus a stable chunk_id.
"""
import re

MAX_WORDS = 100
OVERLAP_WORDS = 25


def _word_count(text: str) -> int:
    return len(text.split())


def _split_units(text: str) -> list[str]:
    """Break text into sentence-sized units. A very long sentence is hard-split by words."""
    units = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        units.extend(p for p in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", line) if p)
    return units


def _is_heading(line: str) -> bool:
    """A short line with no sentence punctuation and no table separator looks like a heading."""
    words = line.split()
    return 1 <= len(words) <= 8 and "|" not in line and not line.endswith((".", ";", ",", "?", "!"))


def _split_blocks(text: str) -> list[tuple[str, list[str]]]:
    """Group lines into (heading, body_lines) blocks so a chunk never mixes two sections."""
    blocks: list[tuple[str, list[str]]] = []
    heading, body = "", []
    for line in (ln.strip() for ln in text.splitlines()):
        if not line:
            continue
        if _is_heading(line):
            if body:                                   # a body already started -> new section
                blocks.append((heading, body))
                heading, body = line, []
            else:                                      # consecutive headings -> merge them
                heading = f"{heading} / {line}" if heading else line
        else:
            body.append(line)
    if heading or body:
        blocks.append((heading, body))
    return blocks


def _hard_split(unit: str, max_words: int) -> list[str]:
    words = unit.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def _pack(units: list[str], max_words: int, overlap_words: int) -> list[list[str]]:
    """Greedily fill chunks with units; start each new chunk with the previous chunk's tail."""
    prepared = []
    for u in units:
        prepared.extend(_hard_split(u, max_words) if _word_count(u) > max_words else [u])

    chunks, cur, cur_words = [], [], 0
    for unit in prepared:
        w = _word_count(unit)
        if cur and cur_words + w > max_words:
            chunks.append(cur)
            tail, tail_words = [], 0
            for prev in reversed(cur):                       # walk backwards to build the overlap
                if tail_words + _word_count(prev) > overlap_words:
                    break
                tail.insert(0, prev)
                tail_words += _word_count(prev)
            if tail_words + w > max_words:                   # overlap must never crowd out new text
                tail, tail_words = [], 0
            cur, cur_words = tail, tail_words
        cur.append(unit)
        cur_words += w
    if cur:
        chunks.append(cur)
    return chunks


def chunk_text(text: str, max_words: int = MAX_WORDS, overlap_words: int = OVERLAP_WORDS,
               sep: str = " ") -> list[str]:
    if max_words <= 0 or overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("Need max_words > 0 and 0 <= overlap_words < max_words")
    return [sep.join(group) for group in _pack(_split_units(text), max_words, overlap_words)]


def _title(source: str) -> str:
    """'leave_policy.docx' -> 'leave policy' (a little context for the embedding model)."""
    return re.sub(r"[_\-]+", " ", source.rsplit(".", 1)[0])


def chunk_sections(sections: list[dict], doc_id: str, max_words: int = MAX_WORDS,
                   overlap_words: int = OVERLAP_WORDS, by_heading: bool = True) -> list[dict]:
    """Turn parsed sections (Task 2 output) into chunks that keep their citation metadata.

    text        - exactly what was in the document (shown to users as evidence)
    embed_text  - document title + text; this is what we turn into a vector (better matching)
    """
    chunks, n = [], 0
    for sec in sections:
        is_csv = sec["source"].lower().endswith(".csv")
        if is_csv:
            # Rows are the units, and the header line is repeated in every chunk.
            header, *rows = [ln for ln in sec["text"].splitlines() if ln.strip()]
            budget = max(max_words - _word_count(header), 1)
            groups = _pack(rows, budget, min(overlap_words, budget - 1)) if rows else []
            pieces = [header + "\n" + "\n".join(g) for g in groups] or [header]
        elif by_heading:
            pieces = []
            for heading, body_lines in _split_blocks(sec["text"]):
                budget = max(max_words - _word_count(heading), 1)
                parts = chunk_text("\n".join(body_lines), budget, min(overlap_words, budget - 1)) or [""]
                pieces += [f"{heading}\n{p}".strip() for p in parts]   # heading repeats in every piece
        else:
            pieces = chunk_text(sec["text"], max_words, overlap_words)

        for piece in pieces:
            chunks.append({
                "text": piece, "source": sec["source"], "page": sec["page"],
                "location": sec["location"], "doc_id": doc_id,
                "chunk_id": f"{doc_id}_{n:03d}", "word_count": _word_count(piece),
                "embed_text": f"{_title(sec['source'])} - {piece}",
            })
            n += 1
    return chunks
