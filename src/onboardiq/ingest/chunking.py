"""Structure-aware chunking.

Documents are first cut at Markdown headings so a chunk never straddles two
sections, and each chunk remembers its heading path ("Guide > Setup"). Long
sections are packed paragraph by paragraph up to ``chunk_size`` characters;
paragraphs that are still too long are split at sentence boundaries, and only as
a last resort at word boundaries. Consecutive chunks of the same section share
``overlap`` characters of trailing context.
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable

from onboardiq.types import ANY, Chunk, Document, normalize_tag

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sections(text: str) -> list[tuple[str, str]]:
    """Return (heading_path, body) pairs in document order."""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((" > ".join(title for _, title in stack), body))
        buffer.clear()

    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        match = None if in_code else _HEADING.match(line)
        if match:
            flush()
            depth = len(match.group(1))
            while stack and stack[-1][0] >= depth:
                stack.pop()
            stack.append((depth, match.group(2).strip()))
        else:
            buffer.append(line)
    flush()
    return sections


def _split_long(paragraph: str, size: int) -> list[str]:
    if len(paragraph) <= size:
        return [paragraph]
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(paragraph):
        if len(sentence) > size:  # a single enormous sentence: fall back to words
            words, sentence_parts, part = sentence.split(), [], ""
            for word in words:
                if part and len(part) + 1 + len(word) > size:
                    sentence_parts.append(part)
                    part = word
                else:
                    part = f"{part} {word}".strip()
            if part:
                sentence_parts.append(part)
        else:
            sentence_parts = [sentence]
        for part in sentence_parts:
            if current and len(current) + 1 + len(part) > size:
                pieces.append(current)
                current = part
            else:
                current = f"{current} {part}".strip()
    if current:
        pieces.append(current)
    return pieces


def _tail(text: str, overlap: int) -> str:
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 else ""
    cut = text[-overlap:]
    space = cut.find(" ")
    return cut[space + 1:] if 0 <= space < len(cut) - 1 else cut


def pack_section(body: str, chunk_size: int, overlap: int) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("chunk overlap must be smaller than chunk size")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        # leave room for the carried-over overlap and the paragraph separator
        tidy = "\n".join(" ".join(line.split()) for line in paragraph.splitlines() if line.strip())
        units.extend(_split_long(tidy, max(chunk_size - overlap - 2, 1)))
    chunks: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + 2 + len(unit) > chunk_size:
            chunks.append(current)
            carry = _tail(current, overlap)
            current = f"{carry}\n\n{unit}" if carry else unit
        else:
            current = f"{current}\n\n{unit}" if current else unit
    if current:
        chunks.append(current)
    return chunks


def _tags(value: object) -> tuple[str, ...]:
    if value is None or value == "":
        return (ANY,)
    values = value if isinstance(value, list) else [value]
    tags = tuple(sorted({normalize_tag(str(v)) for v in values}))
    return (ANY,) if ANY in tags else tags


def chunk_document(doc: Document, chunk_size: int = 700, overlap: int = 100) -> list[Chunk]:
    roles = _tags(doc.metadata.get("role", doc.metadata.get("roles")))
    levels = _tags(doc.metadata.get("level", doc.metadata.get("levels")))
    topic = str(doc.metadata.get("topic", ""))
    chunks: list[Chunk] = []
    for heading, body in split_sections(doc.text):
        for text in pack_section(body, chunk_size, overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}-{len(chunks):03d}",
                    doc_id=doc.doc_id,
                    source=doc.source,
                    text=text,
                    heading=heading,
                    roles=roles,
                    levels=levels,
                    topic=topic,
                )
            )
    return chunks


def content_key(text: str) -> str:
    """Fingerprint used to drop duplicate chunks (case, punctuation and spacing ignored)."""
    normalized = " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def chunk_corpus(docs: Iterable[Document], chunk_size: int = 700, overlap: int = 100) -> list[Chunk]:
    """Chunk every document and drop exact/near-exact duplicate chunks across documents."""
    seen: set[str] = set()
    out: list[Chunk] = []
    for doc in docs:
        for chunk in chunk_document(doc, chunk_size, overlap):
            key = content_key(chunk.text)
            if key in seen:
                continue
            seen.add(key)
            out.append(chunk)
    return out
