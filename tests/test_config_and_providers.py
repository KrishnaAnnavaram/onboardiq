import numpy as np
import pytest

from onboardiq.config import Settings, load_dotenv
from onboardiq.providers import EchoLLM, HashingEmbedder, ProviderError, make_embedder, make_llm


def test_defaults_are_offline():
    s = Settings.from_env(env={})
    assert s.llm_provider == "echo" and s.embed_provider == "hashing"
    assert isinstance(make_llm(s), EchoLLM)
    assert isinstance(make_embedder(s), HashingEmbedder)


def test_env_overrides():
    s = Settings.from_env(env={"ONBOARDIQ_TOP_K": "7", "ONBOARDIQ_LLM_PROVIDER": "Ollama",
                               "OLLAMA_HOST": "http://box:11434/"})
    assert s.top_k == 7 and s.llm_provider == "ollama" and s.ollama_host == "http://box:11434"


def test_api_key_is_not_shown_in_repr():
    s = Settings.from_env(env={"OPENAI_API_KEY": "not-a-real-key"})
    assert "not-a-real-key" not in repr(s)


def test_openai_provider_requires_a_key():
    s = Settings.from_env(env={"ONBOARDIQ_LLM_PROVIDER": "openai", "ONBOARDIQ_EMBED_PROVIDER": "openai"})
    with pytest.raises(ProviderError):
        make_llm(s)
    with pytest.raises(ProviderError):
        make_embedder(s)
    with pytest.raises(ProviderError):
        make_llm(Settings.from_env(env={"ONBOARDIQ_LLM_PROVIDER": "mystery"}))


def test_dotenv_never_overrides_existing_variables(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("ONBOARDIQ_TEST_A=from_file\nONBOARDIQ_TEST_B=from_file\n# comment\n", encoding="utf-8")
    monkeypatch.setenv("ONBOARDIQ_TEST_A", "from_shell")
    monkeypatch.delenv("ONBOARDIQ_TEST_B", raising=False)
    loaded = load_dotenv(env_file)
    assert loaded == {"ONBOARDIQ_TEST_B": "from_file"}
    import os

    assert os.environ["ONBOARDIQ_TEST_A"] == "from_shell"
    monkeypatch.delenv("ONBOARDIQ_TEST_B")


def test_hashing_embedder_is_deterministic_and_normalised():
    a = HashingEmbedder().embed(["airflow retries", "dashboard metrics"])
    b = HashingEmbedder().embed(["airflow retries", "dashboard metrics"])
    assert np.array_equal(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    q = HashingEmbedder().embed(["retries in airflow"])[0]
    assert q @ a[0] > q @ a[1]


def test_echo_llm_quotes_and_cites_context():
    prompt = "Question: q\n\n<context>\n[1] a.md (c1)\nFirst fact. More.\n\n[2] b.md (c2)\nSecond fact.\n</context>"
    reply = EchoLLM().complete([{"role": "user", "content": prompt}])
    assert reply.endswith("First fact. [1] Second fact. [2]")
