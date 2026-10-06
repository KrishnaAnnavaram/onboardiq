"""Provider factory: pick the embedder and chat model from Settings."""
from __future__ import annotations

from onboardiq.config import Settings
from onboardiq.providers.embeddings import Embedder, HashingEmbedder, OllamaEmbedder, OpenAIEmbedder
from onboardiq.providers.http import ProviderError
from onboardiq.providers.llm import ChatLLM, EchoLLM, OllamaChatLLM, OpenAIChatLLM


def make_embedder(settings: Settings) -> Embedder:
    kind = settings.embed_provider
    if kind in ("hashing", "fake", "offline"):
        return HashingEmbedder()
    if kind == "openai":
        return OpenAIEmbedder(settings.embed_model, settings.openai_api_key, settings.openai_base_url)
    if kind == "ollama":
        return OllamaEmbedder(settings.embed_model, settings.ollama_host)
    raise ProviderError(f"unknown embedding provider: {kind!r}")


def make_llm(settings: Settings) -> ChatLLM:
    kind = settings.llm_provider
    if kind in ("echo", "fake", "offline"):
        return EchoLLM()
    if kind == "openai":
        return OpenAIChatLLM(settings.llm_model, settings.openai_api_key, settings.openai_base_url,
                             settings.temperature)
    if kind == "ollama":
        return OllamaChatLLM(settings.llm_model, settings.ollama_host, settings.temperature)
    raise ProviderError(f"unknown LLM provider: {kind!r}")


__all__ = ["Embedder", "ChatLLM", "EchoLLM", "HashingEmbedder", "ProviderError", "make_embedder", "make_llm"]
