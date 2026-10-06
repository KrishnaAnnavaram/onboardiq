"""Problem 3: documents were re-read and re-embedded on every message."""
import shutil

import pytest
from conftest import SAMPLE_DOCS, CountingEmbedder, SpyLLM

from onboardiq.index.store import IndexStore, build_or_load
from onboardiq.providers.embeddings import HashingEmbedder
from onboardiq.retrieve.hybrid import HybridRetriever
from onboardiq.service import Assistant


def test_index_is_built_once_then_loaded(tmp_path, embedder):
    index_dir = tmp_path / "index"
    store, rebuilt = build_or_load(SAMPLE_DOCS, index_dir, embedder)
    assert rebuilt and len(embedder.calls) == 1 and embedder.calls[0] == len(store.chunks)

    again, rebuilt = build_or_load(SAMPLE_DOCS, index_dir, embedder)
    assert not rebuilt
    assert len(embedder.calls) == 1, "loading a cached index must not embed documents again"
    assert [c.chunk_id for c in again.chunks] == [c.chunk_id for c in store.chunks]
    assert again.vectors.shape == store.vectors.shape


def test_changed_documents_or_settings_trigger_rebuild(tmp_path, embedder):
    docs = tmp_path / "docs"
    shutil.copytree(SAMPLE_DOCS, docs)
    index_dir = tmp_path / "index"
    build_or_load(docs, index_dir, embedder)
    (docs / "new_note.md").write_text("---\nrole: all\nlevel: all\n---\n# New\n\nA brand new note.", encoding="utf-8")
    _, rebuilt = build_or_load(docs, index_dir, embedder)
    assert rebuilt
    _, rebuilt = build_or_load(docs, index_dir, embedder, chunk_size=400)
    assert rebuilt


def test_answering_embeds_only_the_question(settings, embedder):
    assistant = Assistant(settings, embedder=embedder, llm=SpyLLM())
    before = len(embedder.calls)
    for question in ("How do backfills work?", "Who approves production write access?"):
        assistant.ask(question, "data_engineer", "junior")
    assert embedder.calls[before:] == [1, 1]


def test_query_embedder_must_match_index(settings):
    store, _ = build_or_load(settings.docs_dir, settings.index_dir, HashingEmbedder(dim=512))
    with pytest.raises(ValueError, match="rebuild the index"):
        HybridRetriever(store, HashingEmbedder(dim=256))


def test_saved_index_roundtrip(tmp_path):
    store, _ = build_or_load(SAMPLE_DOCS, tmp_path / "i", CountingEmbedder())
    loaded = IndexStore.load(tmp_path / "i")
    assert loaded.embedder_name == store.embedder_name
    assert loaded.chunks[0] == store.chunks[0]
