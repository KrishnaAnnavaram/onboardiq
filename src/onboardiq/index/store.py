"""Persisted index: chunks + BM25 statistics + embedding matrix.

Documents are embedded exactly once, at build time. The manifest stores a
fingerprint of the source files, the chunking parameters and the embedder name;
``build_or_load`` reuses the saved index whenever that fingerprint still
matches, so answering a question only ever embeds the question itself.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from onboardiq.ingest.chunking import chunk_corpus
from onboardiq.ingest.loaders import iter_document_paths, load_folder
from onboardiq.providers.embeddings import Embedder
from onboardiq.retrieve.bm25 import BM25Index
from onboardiq.retrieve.text import TOKENIZER_VERSION
from onboardiq.types import Chunk

FORMAT_VERSION = 1


def corpus_fingerprint(docs_dir: Path, chunk_size: int, overlap: int, embedder_name: str) -> str:
    h = hashlib.sha256(f"v{FORMAT_VERSION}|t{TOKENIZER_VERSION}|{chunk_size}|{overlap}|{embedder_name}".encode())
    for path in iter_document_paths(docs_dir):
        h.update(path.relative_to(docs_dir).as_posix().encode("utf-8"))
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


@dataclass
class IndexStore:
    chunks: list[Chunk]
    bm25: BM25Index
    vectors: np.ndarray
    embedder_name: str
    fingerprint: str = ""

    @classmethod
    def from_chunks(cls, chunks: Sequence[Chunk], embedder: Embedder, fingerprint: str = "") -> "IndexStore":
        chunks = list(chunks)
        texts = [f"{c.heading}\n{c.text}" if c.heading else c.text for c in chunks]
        vectors = embedder.embed(texts) if texts else np.zeros((0, 1), dtype=np.float32)
        return cls(chunks, BM25Index.build(texts), vectors, embedder.name, fingerprint)

    def save(self, directory: str | Path) -> None:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        (out / "bm25.json").write_text(json.dumps(self.bm25.to_dict()), encoding="utf-8")
        np.save(out / "vectors.npy", self.vectors)
        manifest = {
            "format": FORMAT_VERSION,
            "fingerprint": self.fingerprint,
            "embedder": self.embedder_name,
            "chunks": len(self.chunks),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "IndexStore":
        src = Path(directory)
        manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
        with (src / "chunks.jsonl").open(encoding="utf-8") as fh:
            chunks = [Chunk.from_dict(json.loads(line)) for line in fh if line.strip()]
        bm25 = BM25Index.from_dict(json.loads((src / "bm25.json").read_text(encoding="utf-8")))
        vectors = np.load(src / "vectors.npy")
        return cls(chunks, bm25, vectors, manifest["embedder"], manifest.get("fingerprint", ""))

    @staticmethod
    def read_manifest(directory: str | Path) -> dict | None:
        path = Path(directory) / "manifest.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None


def build_index(docs_dir: str | Path, embedder: Embedder, chunk_size: int = 700, overlap: int = 100,
                index_dir: str | Path | None = None) -> IndexStore:
    docs_path = Path(docs_dir)
    fingerprint = corpus_fingerprint(docs_path, chunk_size, overlap, embedder.name)
    chunks = chunk_corpus(load_folder(docs_path), chunk_size, overlap)
    store = IndexStore.from_chunks(chunks, embedder, fingerprint)
    if index_dir is not None:
        store.save(index_dir)
    return store


def build_or_load(docs_dir: str | Path, index_dir: str | Path, embedder: Embedder, chunk_size: int = 700,
                  overlap: int = 100, force: bool = False) -> tuple[IndexStore, bool]:
    """Return (store, rebuilt). Rebuilds only if forced or the corpus/config changed."""
    docs_path = Path(docs_dir)
    manifest = IndexStore.read_manifest(index_dir)
    if not force and manifest is not None:
        expected = corpus_fingerprint(docs_path, chunk_size, overlap, embedder.name)
        if manifest.get("fingerprint") == expected and manifest.get("format") == FORMAT_VERSION:
            return IndexStore.load(index_dir), False
    return build_index(docs_path, embedder, chunk_size, overlap, index_dir), True
