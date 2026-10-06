from __future__ import annotations

from pathlib import Path

import pytest

from onboardiq.config import Settings
from onboardiq.providers.embeddings import HashingEmbedder
from onboardiq.providers.llm import EchoLLM

REPO = Path(__file__).resolve().parents[1]
SAMPLE_DOCS = REPO / "sample_docs"
GOLDEN = REPO / "eval" / "golden_set.jsonl"


class CountingEmbedder(HashingEmbedder):
    """Hashing embedder that records how many texts it was asked to embed."""

    def __init__(self):
        super().__init__()
        self.calls: list[int] = []

    def embed(self, texts):
        self.calls.append(len(texts))
        return super().embed(texts)


class SpyLLM:
    name = "spy"

    def __init__(self, reply: str = "The answer is in the docs [1]."):
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages):
        self.calls.append(messages)
        return self.reply


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings.from_env(env={
        "ONBOARDIQ_DOCS_DIR": str(SAMPLE_DOCS),
        "ONBOARDIQ_INDEX_DIR": str(tmp_path / "index"),
        "ONBOARDIQ_FEEDBACK_DB": str(tmp_path / "feedback.sqlite"),
    })


@pytest.fixture
def embedder() -> CountingEmbedder:
    return CountingEmbedder()


@pytest.fixture
def echo() -> EchoLLM:
    return EchoLLM()
