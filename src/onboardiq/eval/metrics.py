"""Retrieval metrics over ranked lists of relevance judgements."""
from __future__ import annotations

from typing import Sequence


def recall_at_k(matched: Sequence[set[int]], n_relevant: int, k: int) -> float:
    """Share of the relevant targets found in the top ``k`` results.

    ``matched[i]`` is the set of relevant-target ids satisfied by the i-th result.
    """
    if n_relevant <= 0:
        raise ValueError("a query needs at least one relevant target")
    found: set[int] = set()
    for targets in matched[:k]:
        found |= targets
    return len(found) / n_relevant


def reciprocal_rank(matched: Sequence[set[int]]) -> float:
    for rank, targets in enumerate(matched, start=1):
        if targets:
            return 1.0 / rank
    return 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
