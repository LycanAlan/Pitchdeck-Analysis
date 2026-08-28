"""Domain types shared by every layer.

These are the only shapes that cross module boundaries. Ingestion produces a
DeckDocument; indexing and retrieval consume Chunks; generation produces an
Answer. Keeping them here is what lets the retrieval, evaluation and API layers
stay free of duplicated dict-shuffling.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path


class ExtractionMode(str, Enum):
    """How a page's text was recovered."""

    TEXT_LAYER = "text_layer"  # PDF had an embedded text layer
    VISION = "vision"  # page was rendered and transcribed by a vision model


class ChunkKind(str, Enum):
    """What a retrievable chunk represents."""

    TRANSCRIPT = "transcript"  # verbatim slide text
    SUMMARY = "summary"  # model-written structured summary


@dataclass
class Chunk:
    """One retrievable unit."""

    deck: str
    slide: int
    kind: ChunkKind
    text: str
    section: str = ""

    @property
    def id(self) -> str:
        return f"{self.deck}::{self.slide}::{self.kind.value}"

    def to_metadata(self) -> dict:
        return {
            "deck": self.deck,
            "slide": self.slide,
            "kind": self.kind.value,
            "section": self.section,
            "chunk_id": self.id,
        }


@dataclass
class ScoredChunk:
    """A chunk with the score assigned by whichever retriever returned it."""

    chunk: Chunk
    score: float

    @property
    def slide(self) -> int:
        return self.chunk.slide

    @property
    def deck(self) -> str:
        return self.chunk.deck


@dataclass
class Slide:
    """A single analysed slide of a deck."""

    slide: int
    mode: ExtractionMode
    transcript: str = ""
    section: str = ""
    summary: str = ""
    key_points: list[str] = field(default_factory=list)

    def to_chunks(self, deck: str) -> list[Chunk]:
        """Explode into retrievable units. Empty fields yield nothing."""
        chunks = []
        if self.transcript.strip():
            chunks.append(
                Chunk(deck, self.slide, ChunkKind.TRANSCRIPT, self.transcript, self.section)
            )
        if self.summary.strip():
            body = self.summary
            if self.key_points:
                body += "\n" + "\n".join(f"- {p}" for p in self.key_points)
            chunks.append(Chunk(deck, self.slide, ChunkKind.SUMMARY, body, self.section))
        return chunks


@dataclass
class DeckDocument:
    """A fully ingested deck. This is the on-disk unit of work."""

    name: str
    slides: list[Slide] = field(default_factory=list)

    @property
    def vision_slides(self) -> int:
        return sum(1 for s in self.slides if s.mode is ExtractionMode.VISION)

    def chunks(self) -> list[Chunk]:
        return [c for s in self.slides for c in s.to_chunks(self.name)]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"name": self.name, "slides": [asdict(s) for s in self.slides]}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DeckDocument":
        raw = json.loads(path.read_text(encoding="utf-8"))
        slides = [
            Slide(
                slide=s["slide"],
                mode=ExtractionMode(s["mode"]),
                transcript=s.get("transcript", ""),
                section=s.get("section", ""),
                summary=s.get("summary", ""),
                key_points=s.get("key_points", []),
            )
            for s in raw["slides"]
        ]
        return cls(name=raw["name"], slides=slides)


@dataclass
class Citation:
    deck: str
    slide: int
    kind: str


@dataclass
class Answer:
    """Generator output plus the evidence it was grounded on."""

    text: str
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[ScoredChunk] = field(default_factory=list)
    rewritten_query: str = ""
    retrieval_rounds: int = 1
    grounded: bool = True

    @classmethod
    def from_chunks(cls, text: str, scored: list[ScoredChunk], **kw) -> "Answer":
        seen, cites = set(), []
        for sc in scored:
            key = (sc.chunk.deck, sc.chunk.slide)
            if key not in seen:
                seen.add(key)
                cites.append(Citation(sc.chunk.deck, sc.chunk.slide, sc.chunk.kind.value))
        return cls(text=text, citations=cites, retrieved=scored, **kw)


@dataclass
class EvalQuestion:
    """One ground-truth item."""

    id: str
    deck: str
    question: str
    expected_answer: str
    gold_slides: list[int]
    category: str = "factual"  # factual | numeric | multi_hop | unanswerable
