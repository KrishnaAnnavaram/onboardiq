"""Load onboarding documents from a folder.

Markdown and plain-text files may start with a small front-matter block:

    ---
    role: data_engineer, data_analyst
    level: junior
    topic: sql
    ---

Values are plain ``key: value`` lines; comma-separated values become lists.
PDF files are supported when the optional ``pdf`` extra (pypdf) is installed;
PDFs carry no front-matter, so they apply to every role and level.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

from onboardiq.types import Document

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
PDF_SUFFIXES = {".pdf"}


def parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Split ``text`` into (metadata, body). Text without front-matter returns ({}, text)."""
    lines = text.lstrip("﻿").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    meta: dict[str, Any] = {}
    for end, line in enumerate(lines[1:], start=1):
        stripped = line.strip()
        if stripped == "---":
            body = "\n".join(lines[end + 1:]).lstrip("\n")
            return meta, body
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        value = value.strip().strip("[]")
        items = [v.strip().strip('"').strip("'") for v in value.split(",") if v.strip()]
        meta[key.strip().lower()] = items if len(items) > 1 else (items[0] if items else "")
    return {}, text  # unterminated block: treat everything as body


def _doc_id(relative_path: str) -> str:
    return hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:12]


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Reading PDFs needs the optional extra: pip install 'onboardiq[pdf]'") from exc
    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def iter_document_paths(folder: Path) -> Iterable[Path]:
    suffixes = TEXT_SUFFIXES | PDF_SUFFIXES
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes and path.name.lower() != "readme.md":
            yield path


def load_document(path: Path, root: Path) -> Document:
    relative = path.relative_to(root).as_posix()
    if path.suffix.lower() in PDF_SUFFIXES:
        meta, body = {}, _read_pdf(path)
    else:
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
    return Document(doc_id=_doc_id(relative), source=relative, text=body, metadata=meta)


def load_folder(folder: str | Path) -> list[Document]:
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Document folder not found: {root}")
    docs = [load_document(p, root) for p in iter_document_paths(root)]
    return [d for d in docs if d.text.strip()]
