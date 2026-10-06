"""Exact cosine-similarity search over a normalised embedding matrix."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def cosine_search(matrix: np.ndarray, query_vector: np.ndarray, k: int,
                  allowed: Sequence[int] | None = None) -> list[tuple[int, float]]:
    if matrix.size == 0:
        return []
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    positions = np.arange(matrix.shape[0]) if allowed is None else np.asarray(list(allowed), dtype=int)
    if positions.size == 0:
        return []
    sims = matrix[positions] @ query
    order = np.argsort(-sims, kind="stable")[:k]
    return [(int(positions[i]), float(sims[i])) for i in order]
