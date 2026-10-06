"""Service layer: wires settings, providers, index, retriever and answerer together.

The CLI and the Streamlit UI both talk only to ``Assistant``.
"""
from __future__ import annotations

from typing import Sequence

from onboardiq.config import Settings
from onboardiq.generate.answer import Answerer
from onboardiq.index.store import IndexStore, build_or_load
from onboardiq.providers import make_embedder, make_llm
from onboardiq.providers.embeddings import Embedder
from onboardiq.providers.llm import ChatLLM
from onboardiq.retrieve.hybrid import HybridRetriever
from onboardiq.types import Answer


class Assistant:
    def __init__(self, settings: Settings, embedder: Embedder | None = None, llm: ChatLLM | None = None,
                 force_rebuild: bool = False):
        self.settings = settings
        self.embedder = embedder or make_embedder(settings)
        self.llm = llm or make_llm(settings)
        self.store, self.rebuilt = build_or_load(
            settings.docs_dir, settings.index_dir, self.embedder,
            settings.chunk_size, settings.chunk_overlap, force=force_rebuild,
        )
        self.retriever = HybridRetriever(self.store, self.embedder, candidate_k=settings.candidate_k)
        self.answerer = Answerer(self.retriever, self.llm, top_k=settings.top_k,
                                 min_similarity=settings.min_similarity, history_turns=settings.history_turns)

    @property
    def index(self) -> IndexStore:
        return self.store

    def ask(self, question: str, role: str | None = None, level: str | None = None,
            history: Sequence[tuple[str, str]] = ()) -> Answer:
        return self.answerer.ask(question, role, level, history)
