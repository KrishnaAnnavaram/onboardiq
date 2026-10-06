"""Problem 1: retrieved passages must be what the generator sees and cites."""
from conftest import SpyLLM

from onboardiq.generate.prompts import build_messages, history_window, system_prompt
from onboardiq.service import Assistant


def test_retrieved_chunks_reach_the_prompt_verbatim(settings, embedder):
    llm = SpyLLM()
    assistant = Assistant(settings, embedder=embedder, llm=llm)
    answer = assistant.ask("How many times does a failing task retry?", "data_engineer", "junior")

    assert len(llm.calls) == 1, "exactly one LLM call per question (no agent hop)"
    prompt = llm.calls[0][-1]["content"]
    assert answer.hits, "retrieval returned passages"
    for number, hit in enumerate(answer.hits, start=1):
        assert f"[{number}] {hit.chunk.location}" in prompt
        assert hit.chunk.text in prompt
    assert any("three times" in hit.chunk.text for hit in answer.hits)


def test_citations_point_at_the_passages_that_were_shown(settings, embedder):
    llm = SpyLLM("Tasks retry three times [1], then page on-call [2]. Also see [9].")
    answer = Assistant(settings, embedder=embedder, llm=llm).ask("How do retries work?", "data_engineer", "junior")
    assert "[9]" not in answer.text  # invalid marker removed
    assert [c.marker for c in answer.citations] == [1, 2]
    assert [c.chunk_id for c in answer.citations] == [h.chunk.chunk_id for h in answer.hits[:2]]


def test_no_evidence_means_refusal_without_calling_the_model(settings, embedder):
    llm = SpyLLM()
    answer = Assistant(settings, embedder=embedder, llm=llm).ask("zzqx vvbn plorf?", "data_analyst", "mid")
    assert answer.refused and answer.citations == []
    assert llm.calls == []


def test_echo_llm_end_to_end_is_grounded_and_cited(settings, embedder, echo):
    answer = Assistant(settings, embedder=embedder, llm=echo).ask(
        "What threshold of population stability index fires a drift alert?", "data_scientist", "senior")
    assert not answer.refused
    assert "0.2" in answer.text and "[1]" in answer.text
    assert answer.citations and answer.citations[0].source == "ml_deployment.md"


def test_role_and_level_shape_the_system_prompt():
    junior = system_prompt("data_analyst", "junior")
    senior = system_prompt("data_engineer", "senior")
    assert "Junior Data Analyst" in junior and "step by step" in junior
    assert "Senior Data Engineer" in senior and "skip basics" in senior
    assert "pipelines" in senior and "SQL" in junior


def test_role_filter_keeps_other_roles_out(settings, embedder):
    answer = Assistant(settings, embedder=embedder, llm=SpyLLM()).ask(
        "How do I roll back a deployed model?", "data_analyst", "junior")
    assert all(h.chunk.source != "ml_deployment.md" for h in answer.hits)


def test_history_window_is_bounded_and_drops_old_markers():
    # problem 8: chat history grew without bound inside the prompt
    history = [(f"question {i}", f"answer {i} [1]") for i in range(10)]
    window = history_window(history, turns=3)
    assert len(window) == 6
    assert window[0]["content"] == "question 7"
    assert all("[1]" not in m["content"] for m in window if m["role"] == "assistant")
    assert history_window(history, turns=0) == []
    messages = build_messages("q", [], "data_analyst", "mid", history, history_turns=2)
    assert len(messages) == 1 + 4 + 1
