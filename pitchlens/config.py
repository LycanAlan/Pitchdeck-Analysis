"""Central configuration. Every tunable in the system lives here exactly once."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

ROOT = Path(__file__).resolve().parent.parent

# Free-tier quota is granted per model per day (20 requests), so a model chain is
# not merely failover for 503s — it is how the pipeline gets a workable budget at
# all. GeminiClient drops a model permanently once it 429s, so listing many
# same-tier models multiplies the throughput available in a single run.
FLASH_CHAIN: tuple[str, ...] = (
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-flash-latest",
    "gemini-3.7-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash-lite-preview",
    "gemini-flash-lite-latest",
)


@dataclass(frozen=True)
class ModelConfig:
    """Model routing.

    Chains exist because the Gemini endpoint returns 503/404 unpredictably per
    model; the client walks the chain until one answers. The judge chain is
    deliberately a *different, stronger* family than the generator chain so the
    evaluator is not grading its own output.
    """

    vision: tuple[str, ...] = FLASH_CHAIN
    analysis: tuple[str, ...] = FLASH_CHAIN
    generation: tuple[str, ...] = FLASH_CHAIN
    # A different, stronger family than the generator so answers are not graded
    # by the model that wrote them; falls back into the flash tier only if the
    # pro models are exhausted, which is recorded in the run's LLM usage table.
    judge: tuple[str, ...] = ("gemini-3.1-pro-preview", "gemini-pro-latest") + FLASH_CHAIN

    embedding: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class RetrievalConfig:
    candidates: int = 12  # per-retriever pool before fusion
    final_k: int = 5  # chunks handed to the generator
    rrf_k: int = 60  # reciprocal-rank-fusion damping constant
    eval_k_values: tuple[int, ...] = (1, 3, 5, 10)


@dataclass(frozen=True)
class IngestConfig:
    render_dpi: int = 140
    # A page with fewer than this many characters in its PDF text layer is
    # treated as image-only and routed to the vision extractor.
    text_layer_min_chars: int = 60
    max_workers: int = 8


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    decks: Path = ROOT / "data" / "decks"
    documents: Path = ROOT / "data" / "documents"
    indices: Path = ROOT / "data" / "indices"
    eval_sets: Path = ROOT / "data" / "eval"
    results: Path = ROOT / "results"

    def ensure(self) -> None:
        for p in (self.decks, self.documents, self.indices, self.eval_sets, self.results):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    models: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    paths: Paths = field(default_factory=Paths)

    @property
    def api_key(self) -> str:
        return os.environ["GEMINI_API_KEY"]


settings = Settings()
