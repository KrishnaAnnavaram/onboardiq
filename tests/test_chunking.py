from conftest import SAMPLE_DOCS

from onboardiq.ingest.chunking import chunk_corpus, chunk_document, pack_section, split_sections
from onboardiq.ingest.loaders import load_folder, parse_front_matter
from onboardiq.types import ANY, Document

GUIDE = """# Guide

Intro paragraph.

## Setup

Install the tools.

### Access

Request warehouse access.

```
# not a heading inside a code fence
```

## Usage

Run the job.
"""


def test_sections_follow_heading_hierarchy():
    sections = split_sections(GUIDE)
    headings = [h for h, _ in sections]
    assert headings == ["Guide", "Guide > Setup", "Guide > Setup > Access", "Guide > Usage"]
    access_body = dict(sections)["Guide > Setup > Access"]
    assert "# not a heading" in access_body  # a comment inside a code fence stays in the body


def test_chunks_respect_size_and_carry_overlap():
    paragraphs = "\n\n".join(f"Sentence number {i} talks about pipelines and retries." for i in range(40))
    chunks = pack_section(paragraphs, chunk_size=300, overlap=60)
    assert len(chunks) > 3
    assert all(len(c) <= 300 for c in chunks)
    for previous, current in zip(chunks, chunks[1:]):
        carried = current.split("\n\n")[0]
        assert carried and carried in previous  # the overlap comes from the end of the previous chunk


def test_very_long_sentence_is_split_on_words():
    text = " ".join(["word"] * 500)
    chunks = pack_section(text, chunk_size=120, overlap=20)
    assert all(len(c) <= 120 for c in chunks)
    assert sum(c.count("word") for c in chunks) >= 500


def test_overlap_must_be_smaller_than_chunk():
    import pytest

    with pytest.raises(ValueError):
        pack_section("text", chunk_size=50, overlap=50)


def test_chunk_never_straddles_sections():
    doc = Document("d1", "guide.md", GUIDE, {})
    for chunk in chunk_document(doc, chunk_size=200, overlap=20):
        assert not ("Install the tools" in chunk.text and "Run the job" in chunk.text)


def test_front_matter_parsed_into_tags():
    text = "---\nrole: Data Engineer, data_analyst\nlevel: Junior\ntopic: sql\n---\n# T\n\nx"
    meta, body = parse_front_matter(text)
    assert meta == {"role": ["Data Engineer", "data_analyst"], "level": "Junior", "topic": "sql"}
    assert body.startswith("# T")
    chunk = chunk_document(Document("d", "s.md", body, meta))[0]
    assert chunk.roles == ("data_analyst", "data_engineer")
    assert chunk.levels == ("junior",)
    assert chunk.topic == "sql"


def test_missing_or_unterminated_front_matter_applies_to_everyone():
    meta, body = parse_front_matter("---\nrole: x\nno closing fence")
    assert meta == {} and body.startswith("---")
    chunk = chunk_document(Document("d", "s.md", "# A\n\ntext", {}))[0]
    assert chunk.roles == (ANY,) and chunk.levels == (ANY,)


def test_duplicate_chunks_across_documents_are_dropped():
    # problem 9: near-duplicate documents produced redundant hits
    text = "# Roadmap\n\nLearn SQL first, then dashboards."
    copy = "# Roadmap\n\nlearn  SQL first,  then dashboards!"
    chunks = chunk_corpus([Document("a", "a.md", text, {}), Document("b", "b.md", copy, {})])
    assert len(chunks) == 1


def test_sample_docs_load_with_metadata():
    docs = load_folder(SAMPLE_DOCS)
    assert len(docs) >= 8
    assert all(d.metadata.get("role") for d in docs), "every sample doc declares its audience"
    assert not any(d.source.lower() == "readme.md" for d in docs)
