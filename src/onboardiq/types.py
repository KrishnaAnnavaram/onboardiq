"""Plain data types shared by every layer of the pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

ROLES = ("data_scientist", "data_engineer", "data_analyst")
LEVELS = ("junior", "mid", "senior")
ANY = "all"

ROLE_LABELS = {
    "data_scientist": "Data Scientist",
    "data_engineer": "Data Engineer",
    "data_analyst": "Data Analyst",
}
LEVEL_LABELS = {"junior": "Junior", "mid": "Mid-level", "senior": "Senior"}


def normalize_tag(value: str | None) -> str:
    """Map free-form role/level spellings ("Data Engineer", "data-engineer", "MID") to canonical tags."""
    if value is None:
        return ANY
    tag = "_".join(str(value).strip().lower().replace("-", " ").split())
    if tag in ("", "any", "*", "everyone"):
        return ANY
    return tag


@dataclass(frozen=True)
class Document:
    """One source file after loading, before chunking."""

    doc_id: str
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text with the metadata it inherited from its document."""

    chunk_id: str
    doc_id: str
    source: str
    text: str
    heading: str = ""
    roles: tuple[str, ...] = (ANY,)
    levels: tuple[str, ...] = (ANY,)
    topic: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["roles"] = list(self.roles)
        data["levels"] = list(self.levels)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            source=data["source"],
            text=data["text"],
            heading=data.get("heading", ""),
            roles=tuple(data.get("roles", [ANY])),
            levels=tuple(data.get("levels", [ANY])),
            topic=data.get("topic", ""),
        )

    @property
    def location(self) -> str:
        return f"{self.source} > {self.heading}" if self.heading else self.source


@dataclass
class Hit:
    """A chunk returned by a retriever, with the evidence that put it there."""

    chunk: Chunk
    score: float
    bm25_score: float = 0.0
    vector_score: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Citation:
    marker: int
    chunk_id: str
    source: str
    heading: str
    snippet: str


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Citation]
    hits: list[Hit]
    role: str
    level: str
    refused: bool = False
    answer_id: str = ""
