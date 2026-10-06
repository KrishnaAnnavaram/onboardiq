"""Embedding providers behind one small interface.

``HashingEmbedder`` is deterministic, dependency-free and offline: it hashes word
unigrams, bigrams and character trigrams into a fixed-size signed vector. It is
what tests and the offline demo use. The API-backed embedders talk to any
OpenAI-compatible ``/embeddings`` endpoint or to a local Ollama server.
"""
from __future__ import annotations

import hashlib
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from onboardiq.providers.http import ProviderError, post_json
from onboardiq.retrieve.text import tokenize


@runtime_checkable
class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, d) float32 matrix of L2-normalised vectors."""
        ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self.dim, (1.0 if (value >> 63) & 1 else -1.0)

    def _features(self, text: str) -> list[tuple[str, float]]:
        tokens = tokenize(text)
        feats: list[tuple[str, float]] = [(f"w:{t}", 1.0) for t in tokens]
        feats += [(f"b:{a}_{b}", 0.5) for a, b in zip(tokens, tokens[1:])]
        for token in tokens:
            padded = f"<{token}>"
            feats += [(f"c:{padded[i:i + 3]}", 0.2) for i in range(len(padded) - 2)]
        return feats

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, weight in self._features(text):
                index, sign = self._bucket(feature)
                out[row, index] += sign * weight
        return l2_normalize(out)


class OpenAIEmbedder:
    """Any OpenAI-compatible embeddings endpoint (OpenAI, Azure-style proxies, vLLM, LM Studio...)."""

    def __init__(self, model: str, api_key: str, base_url: str, batch_size: int = 64):
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is not set")
        self.model, self.api_key, self.base_url, self.batch_size = model, api_key, base_url, batch_size
        self.name = f"openai:{model}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            data = post_json(f"{self.base_url}/embeddings", {"model": self.model, "input": batch},
                             headers={"Authorization": f"Bearer {self.api_key}"})
            items = sorted(data["data"], key=lambda item: item["index"])
            rows.extend(item["embedding"] for item in items)
        return l2_normalize(np.array(rows, dtype=np.float32))


class OllamaEmbedder:
    def __init__(self, model: str, host: str):
        self.model, self.host = model, host
        self.name = f"ollama:{model}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        data = post_json(f"{self.host}/api/embed", {"model": self.model, "input": list(texts)})
        return l2_normalize(np.array(data["embeddings"], dtype=np.float32))
