"""Settings, read from environment variables (and an optional .env file).

Nothing here holds a secret by default: with no variables set, onboardiq uses the
offline fake providers so it can be demoed and tested without any API key.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

PREFIX = "ONBOARDIQ_"


def load_dotenv(path: str | os.PathLike = ".env") -> dict[str, str]:
    """Minimal .env reader. Existing environment variables always win."""
    loaded: dict[str, str] = {}
    env_path = Path(path)
    if not env_path.is_file():
        return loaded
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name, "").strip()
    return int(raw) if raw else default


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name, "").strip()
    return float(raw) if raw else default


def _str(env: Mapping[str, str], name: str, default: str) -> str:
    raw = env.get(name, "").strip()
    return raw or default


@dataclass
class Settings:
    docs_dir: Path = Path("sample_docs")
    index_dir: Path = Path(".index")
    feedback_db: Path = Path(".index/feedback.sqlite")

    llm_provider: str = "echo"          # echo | openai | ollama
    llm_model: str = "gpt-4o-mini"
    embed_provider: str = "hashing"     # hashing | openai | ollama
    embed_model: str = "text-embedding-3-small"
    temperature: float = 0.1

    top_k: int = 4
    candidate_k: int = 20
    chunk_size: int = 700
    chunk_overlap: int = 100
    history_turns: int = 3
    min_similarity: float = 0.15

    openai_api_key: str = field(default="", repr=False)
    openai_base_url: str = "https://api.openai.com/v1"
    ollama_host: str = "http://localhost:11434"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, dotenv: bool = True) -> "Settings":
        if env is None:
            if dotenv:
                load_dotenv()
            env = os.environ
        p = PREFIX
        defaults = cls()
        return cls(
            docs_dir=Path(_str(env, p + "DOCS_DIR", str(defaults.docs_dir))),
            index_dir=Path(_str(env, p + "INDEX_DIR", str(defaults.index_dir))),
            feedback_db=Path(_str(env, p + "FEEDBACK_DB", str(defaults.feedback_db))),
            llm_provider=_str(env, p + "LLM_PROVIDER", defaults.llm_provider).lower(),
            llm_model=_str(env, p + "LLM_MODEL", defaults.llm_model),
            embed_provider=_str(env, p + "EMBED_PROVIDER", defaults.embed_provider).lower(),
            embed_model=_str(env, p + "EMBED_MODEL", defaults.embed_model),
            temperature=_float(env, p + "TEMPERATURE", defaults.temperature),
            top_k=_int(env, p + "TOP_K", defaults.top_k),
            candidate_k=_int(env, p + "CANDIDATE_K", defaults.candidate_k),
            chunk_size=_int(env, p + "CHUNK_SIZE", defaults.chunk_size),
            chunk_overlap=_int(env, p + "CHUNK_OVERLAP", defaults.chunk_overlap),
            history_turns=_int(env, p + "HISTORY_TURNS", defaults.history_turns),
            min_similarity=_float(env, p + "MIN_SIMILARITY", defaults.min_similarity),
            openai_api_key=env.get("OPENAI_API_KEY", ""),
            openai_base_url=_str(env, "OPENAI_BASE_URL", defaults.openai_base_url).rstrip("/"),
            ollama_host=_str(env, "OLLAMA_HOST", defaults.ollama_host).rstrip("/"),
        )
