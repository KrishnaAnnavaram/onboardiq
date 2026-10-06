"""Headless smoke test of the Streamlit UI (skipped when the 'ui' extra is not installed)."""
from pathlib import Path

import pytest
from conftest import SAMPLE_DOCS

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "src" / "onboardiq" / "app.py"


def test_app_answers_and_stores_explicit_feedback(tmp_path, monkeypatch):
    monkeypatch.setenv("ONBOARDIQ_DOCS_DIR", str(SAMPLE_DOCS))
    monkeypatch.setenv("ONBOARDIQ_INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("ONBOARDIQ_FEEDBACK_DB", str(tmp_path / "fb.sqlite"))
    monkeypatch.setenv("ONBOARDIQ_LLM_PROVIDER", "echo")
    monkeypatch.setenv("ONBOARDIQ_EMBED_PROVIDER", "hashing")

    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    assert not at.exception
    # default sidebar selection is a mid-level data analyst
    at.chat_input[0].set_value("Which warehouse <b>layer</b> should dashboards read from?").run()
    assert not at.exception
    rendered = " ".join(m.value for m in at.markdown)
    assert "<b>" not in rendered and "&lt;b&gt;" in rendered
    assert "Sources:" in rendered

    from onboardiq.feedback.store import FeedbackStore

    assert FeedbackStore(tmp_path / "fb.sqlite").rows() == []  # nothing stored before submit
    at.radio[0].set_value("not_helpful")
    submit = next(b for b in at.button if b.label == "Submit feedback")
    submit.click().run()
    assert not at.exception
    assert at.success and "saved" in at.success[0].value
    rows = FeedbackStore(tmp_path / "fb.sqlite").rows()
    assert len(rows) == 1 and rows[0]["rating"] == "not_helpful"
