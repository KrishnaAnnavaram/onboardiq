"""Problem 7: quality claims had no evaluation behind them."""
import pytest
from conftest import GOLDEN

from onboardiq.eval.metrics import recall_at_k, reciprocal_rank
from onboardiq.eval.runner import evaluate, load_golden, matched_targets
from onboardiq.service import Assistant


def test_recall_and_mrr():
    matched = [set(), {0}, set(), {1}]
    assert recall_at_k(matched, 2, 1) == 0.0
    assert recall_at_k(matched, 2, 2) == 0.5
    assert recall_at_k(matched, 2, 4) == 1.0
    assert reciprocal_rank(matched) == 0.5
    assert reciprocal_rank([set(), set()]) == 0.0
    with pytest.raises(ValueError):
        recall_at_k(matched, 0, 1)


def test_every_golden_target_exists_in_the_corpus(settings, embedder, echo):
    assistant = Assistant(settings, embedder=embedder, llm=echo)
    golden = load_golden(GOLDEN)
    assert len(golden) >= 25
    for item in golden:
        assert any(matched_targets(c, item.relevant) for c in assistant.store.chunks), item.qid


def test_hybrid_retrieval_meets_a_floor_on_the_golden_set(settings, embedder, echo):
    assistant = Assistant(settings, embedder=embedder, llm=echo)
    golden = load_golden(GOLDEN)
    for use_filter in (True, False):
        report = evaluate(assistant.retriever, golden, use_filter=use_filter)
        assert set(report) == {"bm25", "vector", "hybrid"}
        assert report["hybrid"]["recall@5"] >= 0.9
        assert report["hybrid"]["mrr"] >= 0.8
