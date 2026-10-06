"""Safe rendering of user and model text for the UI.

All text is HTML-escaped before it reaches Streamlit, and the UI never enables
``unsafe_allow_html``, so neither a question nor a model answer can inject markup.
"""
from __future__ import annotations

import html

from onboardiq.generate.citations import format_citations
from onboardiq.types import Answer


def escape(text: str) -> str:
    return html.escape(text, quote=True)


def answer_markdown(answer: Answer) -> str:
    body = escape(answer.text)
    sources = format_citations(answer.citations)
    if sources:
        lines = escape(sources).splitlines()
        body += "\n\n**" + lines[0] + "**\n" + "\n".join(f"- {line}" for line in lines[1:])
    return body


def answer_plain(answer: Answer) -> str:
    sources = format_citations(answer.citations)
    return f"{answer.text}\n\n{sources}" if sources else answer.text
