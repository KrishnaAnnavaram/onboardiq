"""Hybrid retriever: metadata filter -> BM25 + vector candidates -> RRF -> top k."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from onboardiq.index.store import IndexStore
from onboardiq.providers.embeddings import Embedder
from onboardiq.retrieve.filters import filter_positions
from onboardiq.retrieve.fusion import reciprocal_rank_fusion
from onboardiq.retrieve.vector import cosine_search
from onboardiq.types import Hit

Mode = Literal["hybrid", "bm25", "vector"]


@dataclass
class RetrievalResult:
    hits: list[Hit]
    filter_stage: str


class HybridRetriever:
    def __init__(self, store: IndexStore, embedder: Embedder, candidate_k: int = 20, rrf_k: int = 60):
        if embedder.name != store.embedder_name:
            raise ValueError(
                f"index was built with {store.embedder_name!r} but the query embedder is {embedder.name!r}; "
                "rebuild the index"
            )
        self.store, self.embedder = store, embedder
        self.candidate_k, self.rrf_k = candidate_k, rrf_k

    def retrieve(self, query: str, k: int = 4, role: str | None = None, level: str | None = None,
                 mode: Mode = "hybrid") -> RetrievalResult:
        allowed, stage = filter_positions(self.store.chunks, role, level)
        if not query.strip() or not allowed:
            return RetrievalResult([], stage)

        lexical = self.store.bm25.search(query, self.candidate_k, allowed)
        query_vector = self.embedder.embed([query])[0]
        semantic = cosine_search(self.store.vectors, query_vector, self.candidate_k, allowed)

        bm25_scores = dict(lexical)
        vector_scores = dict(semantic)
        if mode == "bm25":
            ranked = [(pos, score, {"bm25": r}) for r, (pos, score) in enumerate(lexical, start=1)]
        elif mode == "vector":
            ranked = [(pos, score, {"vector": r}) for r, (pos, score) in enumerate(semantic, start=1)]
        elif mode == "hybrid":
            ranked = reciprocal_rank_fusion(
                {"bm25": [pos for pos, _ in lexical], "vector": [pos for pos, _ in semantic]}, k=self.rrf_k
            )
        else:
            raise ValueError(f"unknown retrieval mode: {mode!r}")

        hits: list[Hit] = []
        for pos, score, ranks in ranked[:k]:
            if pos not in vector_scores:
                vector_scores[pos] = float(self.store.vectors[pos] @ query_vector)
            hits.append(Hit(self.store.chunks[pos], float(score), bm25_scores.get(pos, 0.0),
                            vector_scores[pos], dict(ranks)))
        return RetrievalResult(hits, stage)


def has_evidence(hits: list[Hit], min_similarity: float) -> bool:
    """True if at least one hit shares query terms (BM25 > 0) or is semantically close enough."""
    return any(h.bm25_score > 0 or h.vector_score >= min_similarity for h in hits)
