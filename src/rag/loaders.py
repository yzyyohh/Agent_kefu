from __future__ import annotations

import csv
from pathlib import Path

try:
    from pypdf import PdfReader  # type: ignore
except ImportError:
    PdfReader = None  # type: ignore


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".csv"}


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    if PdfReader is None:
        return ""
    reader = PdfReader(str(path))
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages)


def _read_csv(path: Path) -> str:
    rows: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(" | ".join(row))
    return "\n".join(rows)


def load_documents(data_dir: Path) -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    for file_path in sorted(data_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        content = ""
        if file_path.suffix.lower() in {".txt", ".md"}:
            content = _read_txt(file_path)
        elif file_path.suffix.lower() == ".pdf":
            content = _read_pdf(file_path)
        elif file_path.suffix.lower() == ".csv":
            content = _read_csv(file_path)

        if content.strip():
            docs.append((str(file_path), content.strip()))
    return docs
