"""Reciprocal-rank fusion (Cormack, Clarke & Buettcher, 2009).

Each ranked list contributes ``weight / (k + rank)`` to every item it contains
(rank starts at 1). Items found by several retrievers accumulate score, and an
item found by only one retriever still competes on equal footing, so neither
list can silently shadow the other.
"""
from __future__ import annotations

from typing import Hashable, Mapping, Sequence, TypeVar

T = TypeVar("T", bound=Hashable)


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[T]],
    k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> list[tuple[T, float, dict[str, int]]]:
    """Fuse named ranked lists. Returns (item, fused_score, {list_name: rank}) best-first."""
    if k <= 0:
        raise ValueError("RRF constant k must be positive")
    scores: dict[T, float] = {}
    ranks: dict[T, dict[str, int]] = {}
    first_seen: dict[T, int] = {}
    for name, ranking in rankings.items():
        weight = 1.0 if weights is None else weights.get(name, 1.0)
        for rank, item in enumerate(ranking, start=1):
            if name in ranks.get(item, {}):
                continue  # ignore duplicates inside one list
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
            ranks.setdefault(item, {})[name] = rank
            first_seen.setdefault(item, len(first_seen))
    ordered = sorted(scores, key=lambda item: (-scores[item], first_seen[item]))
    return [(item, scores[item], ranks[item]) for item in ordered]
