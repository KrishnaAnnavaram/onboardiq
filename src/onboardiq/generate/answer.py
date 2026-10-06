"""Grounded answer generation: one retrieval, one LLM call, validated citations.

The passages the retriever returns are exactly the passages placed in the prompt
and exactly the passages citations can point to. When retrieval finds no
supporting evidence the assistant refuses without calling the model.
"""
from __future__ import annotations

import uuid
from typing import Sequence

from onboardiq.generate.citations import resolve_citations
from onboardiq.generate.prompts import build_messages
from onboardiq.providers.llm import ChatLLM
from onboardiq.retrieve.hybrid import HybridRetriever, has_evidence
from onboardiq.types import LEVEL_LABELS, ROLE_LABELS, Answer, normalize_tag

REFUSAL = (
    "I couldn't find this in the onboarding documents available for a {level} {role}. "
    "Try rephrasing, or ask your team lead or onboarding buddy."
)


class Answerer:
    def __init__(self, retriever: HybridRetriever, llm: ChatLLM, top_k: int = 4, min_similarity: float = 0.15,
                 history_turns: int = 3):
        self.retriever, self.llm = retriever, llm
        self.top_k, self.min_similarity, self.history_turns = top_k, min_similarity, history_turns

    def ask(self, question: str, role: str | None = None, level: str | None = None,
            history: Sequence[tuple[str, str]] = ()) -> Answer:
        role_tag, level_tag = normalize_tag(role), normalize_tag(level)
        result = self.retriever.retrieve(question, k=self.top_k, role=role_tag, level=level_tag)
        answer_id = uuid.uuid4().hex[:12]
        if not result.hits or not has_evidence(result.hits, self.min_similarity):
            text = REFUSAL.format(level=LEVEL_LABELS.get(level_tag, "team").lower(),
                                  role=ROLE_LABELS.get(role_tag, "member"))
            return Answer(question, text, [], result.hits, role_tag, level_tag, refused=True, answer_id=answer_id)

        messages = build_messages(question, result.hits, role_tag, level_tag, history, self.history_turns)
        raw = self.llm.complete(messages)
        text, citations = resolve_citations(raw, result.hits)
        return Answer(question, text, citations, result.hits, role_tag, level_tag, answer_id=answer_id)
