"""Tokenisation shared by BM25 and the hashing embedder.

Tokens are whole words (regex word boundaries, lower-cased), so a term such as
"eda" can never match inside "needed" the way raw substring checks do.
"""
from __future__ import annotations

import re

TOKENIZER_VERSION = 2  # bump whenever tokenisation changes; persisted indexes are rebuilt

_TOKEN = re.compile(r"[a-z0-9]+(?:[+#][a-z0-9+#]*)?")

STOPWORDS = frozenset(
    """a an and are as at be but by can do does for from has have how i if in into is it its
    me my of on or our should so than that the their then there these they this to up us was
    we what when where which who why will with you your""".split()
)


def light_stem(token: str) -> str:
    """Conservative plural folding: "retries" -> "retry", "dashboards" -> "dashboard"."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return [light_stem(t) for t in tokens]
