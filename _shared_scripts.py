"""Helper for the inspect/calibrate scripts: make sure the sample data is indexed (safe to repeat)."""
from pathlib import Path

from ingestion.ingest import ingest_file


def ensure_indexed(folder="data"):
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() in {".pdf", ".docx", ".csv", ".txt"}:
            ingest_file(path.name, path.read_bytes())
