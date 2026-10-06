"""Chat LLM providers behind one small interface.

``EchoLLM`` is an offline, deterministic stand-in: it answers extractively by
quoting, from each of the top numbered context passages, the sentence that best
overlaps the question, and citing it, so the whole pipeline (retrieval -> prompt -> citations) can be run
and tested without a model.
"""
from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from onboardiq.providers.http import ProviderError, post_json
from onboardiq.retrieve.text import tokenize

Message = dict[str, str]

_CONTEXT_BLOCK = re.compile(r"^\[(\d+)\][^\n]*\n(.*?)(?=^\[\d+\]|\Z|^</context>)", re.M | re.S)


@runtime_checkable
class ChatLLM(Protocol):
    name: str

    def complete(self, messages: list[Message]) -> str:
        ...


class EchoLLM:
    name = "echo"

    def __init__(self, max_passages: int = 2):
        self.max_passages = max_passages
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message]) -> str:
        self.calls.append(messages)
        prompt = messages[-1]["content"]
        question_line = prompt.split("\n", 1)[0].removeprefix("Question:")
        question_terms = set(tokenize(question_line))
        context = prompt.split("<context>", 1)[-1].split("</context>", 1)[0]
        sentences = []
        for marker, body in _CONTEXT_BLOCK.findall(context)[: self.max_passages]:
            candidates = re.split(r"(?<=[.!?])\s+", " ".join(body.split()))
            # quote the sentence that shares the most words with the question (first one on ties)
            best = max(candidates, key=lambda s: len(question_terms & set(tokenize(s))))
            sentences.append(f"{best} [{marker}]")
        if not sentences:
            return "I could not find this in the onboarding documents."
        return "Based on the onboarding documents: " + " ".join(sentences)


class OpenAIChatLLM:
    """Any OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(self, model: str, api_key: str, base_url: str, temperature: float = 0.1):
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is not set")
        self.model, self.api_key, self.base_url, self.temperature = model, api_key, base_url, temperature
        self.name = f"openai:{model}"

    def complete(self, messages: list[Message]) -> str:
        data = post_json(
            f"{self.base_url}/chat/completions",
            {"model": self.model, "messages": messages, "temperature": self.temperature},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        return (data["choices"][0]["message"]["content"] or "").strip()


class OllamaChatLLM:
    def __init__(self, model: str, host: str, temperature: float = 0.1):
        self.model, self.host, self.temperature = model, host, temperature
        self.name = f"ollama:{model}"

    def complete(self, messages: list[Message]) -> str:
        data = post_json(
            f"{self.host}/api/chat",
            {"model": self.model, "messages": messages, "stream": False,
             "options": {"temperature": self.temperature}},
        )
        return (data.get("message", {}).get("content") or "").strip()
