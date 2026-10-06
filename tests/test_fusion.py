import pytest

from onboardiq.retrieve.fusion import reciprocal_rank_fusion


def test_rrf_scores_match_formula():
    fused = reciprocal_rank_fusion({"bm25": ["a", "b"], "vector": ["b", "c"]}, k=60)
    scores = {item: score for item, score, _ in fused}
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["c"] == pytest.approx(1 / 62)
    assert fused[0][0] == "b"  # found by both retrievers


def test_rrf_records_ranks_per_list():
    fused = reciprocal_rank_fusion({"bm25": ["a", "b"], "vector": ["b"]})
    ranks = {item: r for item, _, r in fused}
    assert ranks["b"] == {"bm25": 2, "vector": 1}
    assert ranks["a"] == {"bm25": 1}


def test_vector_results_are_not_shadowed_by_bm25():
    # problem 2: the old "hybrid" put BM25 first and cut to 3, so vector hits never surfaced
    fused = reciprocal_rank_fusion({"bm25": ["a", "b", "c"], "vector": ["d", "e", "f"]})
    top3 = [item for item, _, _ in fused[:3]]
    assert "d" in top3


def test_duplicates_within_one_list_count_once():
    fused = reciprocal_rank_fusion({"bm25": ["a", "a", "b"]})
    scores = {item: score for item, score, _ in fused}
    assert scores["a"] == pytest.approx(1 / 61)


def test_weights_and_invalid_k():
    fused = reciprocal_rank_fusion({"bm25": ["a"], "vector": ["b"]}, weights={"vector": 2.0})
    assert fused[0][0] == "b"
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"x": ["a"]}, k=0)
