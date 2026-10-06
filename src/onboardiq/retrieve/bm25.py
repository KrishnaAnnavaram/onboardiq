"""Okapi BM25 over pre-tokenised chunks, with JSON-serialisable state."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Sequence

from onboardiq.retrieve.text import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.term_freqs: list[dict[str, int]] = []
        self.doc_lengths: list[int] = []
        self.doc_freq: dict[str, int] = {}
        self.avg_length = 0.0

    @classmethod
    def build(cls, texts: Sequence[str], k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        index = cls(k1, b)
        df: Counter[str] = Counter()
        for text in texts:
            counts = Counter(tokenize(text))
            index.term_freqs.append(dict(counts))
            index.doc_lengths.append(sum(counts.values()))
            df.update(counts.keys())
        index.doc_freq = dict(df)
        index.avg_length = (sum(index.doc_lengths) / len(index.doc_lengths)) if index.doc_lengths else 0.0
        return index

    def __len__(self) -> int:
        return len(self.term_freqs)

    def idf(self, term: str) -> float:
        n, df = len(self), self.doc_freq.get(term, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def scores(self, query: str, allowed: Sequence[int] | None = None) -> dict[int, float]:
        """Score every allowed chunk position that shares at least one term with ``query``."""
        terms = set(tokenize(query))
        positions = range(len(self)) if allowed is None else allowed
        out: dict[int, float] = {}
        avg = self.avg_length or 1.0
        for pos in positions:
            tf = self.term_freqs[pos]
            score = 0.0
            for term in terms:
                f = tf.get(term)
                if not f:
                    continue
                norm = f + self.k1 * (1 - self.b + self.b * self.doc_lengths[pos] / avg)
                score += self.idf(term) * f * (self.k1 + 1) / norm
            if score > 0:
                out[pos] = score
        return out

    def search(self, query: str, k: int, allowed: Sequence[int] | None = None) -> list[tuple[int, float]]:
        ranked = sorted(self.scores(query, allowed).items(), key=lambda item: (-item[1], item[0]))
        return ranked[:k]

    def to_dict(self) -> dict[str, Any]:
        return {"k1": self.k1, "b": self.b, "term_freqs": self.term_freqs, "doc_lengths": self.doc_lengths}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BM25Index":
        index = cls(data["k1"], data["b"])
        index.term_freqs = [dict(tf) for tf in data["term_freqs"]]
        index.doc_lengths = list(data["doc_lengths"])
        df: Counter[str] = Counter()
        for tf in index.term_freqs:
            df.update(tf.keys())
        index.doc_freq = dict(df)
        index.avg_length = (sum(index.doc_lengths) / len(index.doc_lengths)) if index.doc_lengths else 0.0
        return index
