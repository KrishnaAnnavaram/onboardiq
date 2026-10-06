from onboardiq.retrieve.bm25 import BM25Index
from onboardiq.retrieve.filters import filter_positions
from onboardiq.retrieve.text import tokenize
from onboardiq.types import Chunk, normalize_tag


def test_tokens_are_whole_words():
    # problem 4: substring matching made the keyword "EDA" match "needed"
    assert "eda" not in tokenize("Whatever is needed for the job")
    assert "eda" in tokenize("Start with EDA.")
    assert tokenize("C++ and C# in SQL") == ["c++", "c#", "sql"]


def test_plural_folding_is_conservative():
    assert tokenize("retries dashboards access status analysis") == [
        "retry", "dashboard", "access", "status", "analysis"]


def test_bm25_ranks_matching_chunk_first_and_roundtrips():
    texts = ["retries use exponential backoff", "dashboards read the metric layer", "backfill one month at a time"]
    index = BM25Index.build(texts)
    assert index.search("how do retries work", k=3)[0][0] == 0
    assert index.search("nothing relevant here", k=3) == []
    restored = BM25Index.from_dict(index.to_dict())
    assert restored.scores("metric layer") == index.scores("metric layer")


def test_bm25_respects_allowed_positions():
    index = BM25Index.build(["alpha beta", "alpha gamma"])
    assert [pos for pos, _ in index.search("alpha", k=5, allowed=[1])] == [1]


def _chunk(cid, roles, levels):
    return Chunk(cid, "d", "s.md", "text", roles=roles, levels=levels)


CHUNKS = [
    _chunk("everyone", ("all",), ("all",)),
    _chunk("de-junior", ("data_engineer",), ("junior",)),
    _chunk("de-senior", ("data_engineer",), ("senior",)),
    _chunk("ds-any", ("data_scientist",), ("all",)),
]


def test_metadata_filter_exact_match_includes_shared_docs():
    allowed, stage = filter_positions(CHUNKS, "Data Engineer", "Junior")
    assert stage == "role+level"
    assert [CHUNKS[i].chunk_id for i in allowed] == ["everyone", "de-junior"]


def test_filter_widens_instead_of_returning_nothing():
    # problem 4: an empty filter result used to crash BM25/FAISS construction
    only_role = [CHUNKS[1], CHUNKS[2]]
    allowed, stage = filter_positions(only_role, "data_engineer", "mid")
    assert stage == "role" and len(allowed) == 2
    allowed, stage = filter_positions(only_role, "data_analyst", "mid")
    assert stage == "none" and len(allowed) == 2
    assert filter_positions([], "data_analyst", "mid") == ([], "none")


def test_normalize_tag():
    assert normalize_tag("Data Engineer") == "data_engineer"
    assert normalize_tag("data-engineer") == "data_engineer"
    assert normalize_tag(None) == normalize_tag("") == normalize_tag("any") == "all"
