"""Problem 5: feedback was recorded as "Yes" on first render and never updated."""
import csv

import pytest

from onboardiq.feedback.store import FeedbackStore
from onboardiq.types import Answer, Citation


def _answer(answer_id="a1"):
    return Answer("How do I get access?", "Use the access portal [1].",
                  [Citation(1, "c1", "welcome.md", "Access", "...")], [], "data_analyst", "junior",
                  answer_id=answer_id)


def test_nothing_is_stored_until_the_user_submits(tmp_path):
    store = FeedbackStore(tmp_path / "fb.sqlite")
    assert store.rows() == []


def test_resubmitting_updates_the_same_row(tmp_path):
    store = FeedbackStore(tmp_path / "fb.sqlite")
    store.submit(_answer(), "helpful")
    store.submit(_answer(), "not_helpful", "  missing the approval step  ")
    rows = store.rows()
    assert len(rows) == 1
    assert rows[0]["rating"] == "not_helpful"
    assert rows[0]["comment"] == "missing the approval step"
    assert rows[0]["sources"] == '["c1"]'
    assert store.summary() == {"helpful": 0, "not_helpful": 1}


def test_invalid_rating_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        FeedbackStore(tmp_path / "fb.sqlite").submit(_answer(), "Yes")


def test_feedback_persists_and_exports(tmp_path):
    path = tmp_path / "fb.sqlite"
    FeedbackStore(path).submit(_answer("a1"), "helpful")
    FeedbackStore(path).submit(_answer("a2"), "not_helpful")
    out = tmp_path / "export.csv"
    assert FeedbackStore(path).export_csv(out) == 2
    with out.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["answer_id"]: r["rating"] for r in rows} == {"a1": "helpful", "a2": "not_helpful"}


def test_in_memory_store():
    store = FeedbackStore(":memory:")
    store.submit(_answer(), "helpful")
    assert len(store.rows()) == 1
    store.close()
