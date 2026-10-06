"""Evaluate retrieval against a hand-written golden set.

Golden-set lines are JSON objects::

    {"id": "de-01", "question": "...", "role": "data_engineer", "level": "junior",
     "relevant": [{"source": "pipelines.md", "heading": "Retries"}]}

A retrieved chunk satisfies a target when its source file matches and, if a
heading is given, the chunk's heading path contains it (case-insensitive).
Targets are matched on file + section rather than chunk ids, so the golden set
survives changes to the chunk size.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from onboardiq.eval.metrics import mean, recall_at_k, reciprocal_rank
from onboardiq.retrieve.hybrid import HybridRetriever
from onboardiq.types import Chunk

MODES = ("bm25", "vector", "hybrid")
KS = (1, 3, 5)


@dataclass
class GoldenItem:
    qid: str
    question: str
    role: str
    level: str
    relevant: list[dict[str, str]]


def load_golden(path: str | Path) -> list[GoldenItem]:
    items = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        data: dict[str, Any] = json.loads(line)
        if not data.get("relevant"):
            raise ValueError(f"golden set line {line_no} has no relevant targets")
        items.append(GoldenItem(data["id"], data["question"], data.get("role", "all"),
                                data.get("level", "all"), list(data["relevant"])))
    return items


def matched_targets(chunk: Chunk, targets: Sequence[dict[str, str]]) -> set[int]:
    out = set()
    for i, target in enumerate(targets):
        if chunk.source != target["source"]:
            continue
        heading = target.get("heading", "")
        if heading and heading.lower() not in chunk.heading.lower():
            continue
        out.add(i)
    return out


def evaluate(retriever: HybridRetriever, golden: Iterable[GoldenItem], modes: Sequence[str] = MODES,
             ks: Sequence[int] = KS, use_filter: bool = True) -> dict[str, dict[str, float]]:
    """Mean recall@k and MRR per retrieval mode.

    With ``use_filter=False`` the role/level metadata filter is switched off, which
    measures the retrievers alone over the whole corpus (a harder setting).
    """
    golden = list(golden)
    depth = max(ks)
    report: dict[str, dict[str, float]] = {}
    for mode in modes:
        recalls: dict[int, list[float]] = {k: [] for k in ks}
        rrs: list[float] = []
        for item in golden:
            role, level = (item.role, item.level) if use_filter else (None, None)
            hits = retriever.retrieve(item.question, k=depth, role=role, level=level, mode=mode).hits
            matched = [matched_targets(h.chunk, item.relevant) for h in hits]
            for k in ks:
                recalls[k].append(recall_at_k(matched, len(item.relevant), k))
            rrs.append(reciprocal_rank(matched))
        row = {f"recall@{k}": mean(recalls[k]) for k in ks}
        row["mrr"] = mean(rrs)
        report[mode] = row
    return report


def format_report(report: dict[str, dict[str, float]], n_queries: int, title: str = "") -> str:
    columns = list(next(iter(report.values())).keys()) if report else []
    lines = [title or f"Retrieval evaluation on {n_queries} questions", "",
             "| retriever | " + " | ".join(columns) + " |",
             "|---|" + "---:|" * len(columns)]
    for mode, row in report.items():
        lines.append(f"| {mode} | " + " | ".join(f"{row[c]:.3f}" for c in columns) + " |")
    return "\n".join(lines)
