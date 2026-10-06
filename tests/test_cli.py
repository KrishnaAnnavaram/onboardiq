"""The ``onboardiq`` command line, driven in-process with the offline providers."""
import io
import json

import pytest
from conftest import SAMPLE_DOCS

from onboardiq import cli
from onboardiq.feedback.store import FeedbackStore


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no stray .env from the repository is read
    monkeypatch.setenv("ONBOARDIQ_DOCS_DIR", str(SAMPLE_DOCS))
    monkeypatch.setenv("ONBOARDIQ_INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("ONBOARDIQ_FEEDBACK_DB", str(tmp_path / "fb.sqlite"))
    monkeypatch.setenv("ONBOARDIQ_LLM_PROVIDER", "echo")
    monkeypatch.setenv("ONBOARDIQ_EMBED_PROVIDER", "hashing")
    return tmp_path


def test_index_then_ask_json(env, capsys):
    assert cli.main(["index"]) == 0
    assert "Index built" in capsys.readouterr().out
    assert cli.main(["index"]) == 0
    assert "up to date" in capsys.readouterr().out
    assert cli.main(["ask", "Who approves production write access?", "--role", "data_engineer",
                     "--level", "junior", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["refused"] is False and payload["citations"]


def test_chat_rating_before_any_answer_is_not_sent_as_a_question(env, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("+\nWhat is the retry policy?\n-\n\n"))
    assert cli.main(["chat", "--role", "data_engineer", "--level", "junior"]) == 0
    out = capsys.readouterr().out
    assert "nothing to rate yet" in out
    assert out.count("Sources:") == 1  # only the real question produced an answer
    rows = FeedbackStore(env / "fb.sqlite").rows()
    assert len(rows) == 1 and rows[0]["rating"] == "not_helpful"
    assert rows[0]["question"] == "What is the retry policy?"
