"""Metadata filtering by role and seniority level.

Documents declare who they are for in front-matter. A chunk tagged ``all`` (or
untagged) is visible to everyone. The filter widens step by step instead of ever
returning an empty candidate set: exact role+level, then role only, then the
whole corpus.
"""
from __future__ import annotations

from typing import Sequence

from onboardiq.types import ANY, Chunk, normalize_tag


def _matches(tags: Sequence[str], wanted: str) -> bool:
    return wanted == ANY or ANY in tags or wanted in tags


def filter_positions(chunks: Sequence[Chunk], role: str | None, level: str | None) -> tuple[list[int], str]:
    """Return (allowed chunk positions, which filter stage produced them)."""
    role_tag, level_tag = normalize_tag(role), normalize_tag(level)
    exact = [i for i, c in enumerate(chunks) if _matches(c.roles, role_tag) and _matches(c.levels, level_tag)]
    if exact:
        return exact, "role+level"
    by_role = [i for i, c in enumerate(chunks) if _matches(c.roles, role_tag)]
    if by_role:
        return by_role, "role"
    return list(range(len(chunks))), "none"
