"""Turn ``[n]`` markers in model output into citations of concrete chunks."""
from __future__ import annotations

import re
from typing import Sequence

from onboardiq.types import Citation, Hit

_GROUP = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def extract_markers(text: str) -> list[int]:
    """Citation numbers in order of first appearance; handles [1], [1][2] and [1, 2]."""
    seen: list[int] = []
    for group in _GROUP.findall(text):
        for part in group.split(","):
            number = int(part)
            if number not in seen:
                seen.append(number)
    return seen


def _snippet(text: str, limit: int = 160) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3].rstrip() + "..."


def resolve_citations(answer: str, hits: Sequence[Hit]) -> tuple[str, list[Citation]]:
    """Validate markers against the passages actually shown to the model.

    Markers that point at no passage are removed from the text. If the model cited
    nothing valid, every passage it was given is attached as a source, so an
    answer is never returned without provenance.
    """
    valid = range(1, len(hits) + 1)

    def keep_valid(match: re.Match) -> str:
        numbers = [n.strip() for n in match.group(1).split(",") if int(n) in valid]
        return f"[{', '.join(numbers)}]" if numbers else ""

    cleaned = _GROUP.sub(keep_valid, answer)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned).strip()
    cited = [n for n in extract_markers(cleaned) if n in valid] or list(valid)
    citations = [
        Citation(n, hits[n - 1].chunk.chunk_id, hits[n - 1].chunk.source, hits[n - 1].chunk.heading,
                 _snippet(hits[n - 1].chunk.text))
        for n in cited
    ]
    return cleaned, citations


def format_citations(citations: Sequence[Citation]) -> str:
    if not citations:
        return ""
    lines = ["Sources:"]
    for c in citations:
        where = f"{c.source} > {c.heading}" if c.heading else c.source
        lines.append(f"[{c.marker}] {where} ({c.chunk_id})")
    return "\n".join(lines)
