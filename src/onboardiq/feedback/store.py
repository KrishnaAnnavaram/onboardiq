"""Feedback stored in SQLite.

Feedback is only written when the user explicitly submits a rating, and a later
submission for the same answer updates the existing row instead of being lost.
There is no default rating: an answer nobody rated has no feedback row.
"""
from __future__ import annotations

import csv
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from onboardiq.types import Answer

RATINGS = ("helpful", "not_helpful")

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    answer_id  TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    role       TEXT NOT NULL,
    level      TEXT NOT NULL,
    question   TEXT NOT NULL,
    answer     TEXT NOT NULL,
    sources    TEXT NOT NULL,
    refused    INTEGER NOT NULL,
    rating     TEXT NOT NULL CHECK (rating IN ('helpful', 'not_helpful')),
    comment    TEXT NOT NULL DEFAULT ''
)
"""

COLUMNS = ("answer_id", "created_at", "updated_at", "role", "level", "question", "answer", "sources",
           "refused", "rating", "comment")


class FeedbackStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._memory = sqlite3.connect(":memory:") if str(self.path) == ":memory:" else None
        with self._connect() as conn:
            conn.execute(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection inside a transaction; file connections are always closed."""
        if self._memory is not None:
            with self._memory:
                yield self._memory
            return
        conn = sqlite3.connect(self.path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def submit(self, answer: Answer, rating: str, comment: str = "") -> None:
        if rating not in RATINGS:
            raise ValueError(f"rating must be one of {RATINGS}, got {rating!r}")
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        sources = json.dumps([c.chunk_id for c in answer.citations])
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback (answer_id, created_at, updated_at, role, level, question, answer,
                                      sources, refused, rating, comment)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(answer_id) DO UPDATE SET
                    rating = excluded.rating, comment = excluded.comment, updated_at = excluded.updated_at
                """,
                (answer.answer_id, now, now, answer.role, answer.level, answer.question, answer.text,
                 sources, int(answer.refused), rating, comment.strip()),
            )

    def rows(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            cursor = conn.execute(f"SELECT {', '.join(COLUMNS)} FROM feedback ORDER BY created_at, answer_id")
            return [dict(zip(COLUMNS, row)) for row in cursor.fetchall()]

    def summary(self) -> dict[str, int]:
        counts = {"helpful": 0, "not_helpful": 0}
        for row in self.rows():
            counts[row["rating"]] += 1
        return counts

    def export_csv(self, path: str | Path) -> int:
        rows = self.rows()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        return len(rows)

    def close(self) -> None:
        if self._memory is not None:
            self._memory.close()
