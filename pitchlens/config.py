"""Central configuration. Every tunable in the system lives here exactly once."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

ROOT = Path(__file__).resolve().parent.parent

# Ordered by free-tier daily quota, not by capability. Measured from the 429
# payloads: every full flash model allows 20 requests/day, while the flash-lite
# family allows far more. Transcribing a slide is close to OCR, so lite is nearly
# as good at it, and a chain that opens with the 20/day models exhausts itself on
# a single medium deck. The chain is failover *and* budget.
FLASH_CHAIN: tuple[str, ...] = (
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash-lite-preview",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-flash-latest",
    "gemini-3.7-flash",
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

    # Both run through ONNX Runtime rather than PyTorch. Same weights, same
    # outputs (embeddings match torch to float32 precision), roughly a third of
    # the memory -- which is what lets the service run on a 512 MB free tier.
    embedding: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder: str = "Xenova/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class RetrievalConfig:
    candidates: int = 12  # per-retriever pool before fusion
    final_k: int = 5  # chunks handed to the generator
    rrf_k: int = 60  # reciprocal-rank-fusion damping constant
    eval_k_values: tuple[int, ...] = (1, 3, 5, 10)

    @property
    def rerank_enabled(self) -> bool:
        """Whether the service may offer cross-encoder modes.

        The cross-encoder needs roughly 250 MB on top of the ~290 MB the rest of
        the service occupies, so it does not fit a 512 MB container -- measured,
        after one was OOM-killed. Deployments set PITCHLENS_RERANK=0 and serve
        `hybrid` (recall@3 0.892 at 14 ms); everywhere else it stays on, which is
        why the ablation still reports all five strategies.
        """
        return os.environ.get("PITCHLENS_RERANK", "1") != "0"


@dataclass(frozen=True)
class IngestConfig:
    render_dpi: int = 140
    # A page with fewer than this many characters in its PDF text layer is
    # treated as image-only and routed to the vision extractor.
    text_layer_min_chars: int = 60
    # Tuned to the free tier's per-minute limit rather than to the CPU: more
    # concurrency just converts throughput into 429s and backoff.
    max_workers: int = 4


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
        """Empty when unset rather than raising.

        Retrieval, indexing and every ranking measurement are entirely local, so
        the absence of a key must not stop the process from starting — it only
        rules out generation. Callers that genuinely need a key check
        `has_api_key` and fail with something more useful than a KeyError.
        """
        return os.environ.get("GEMINI_API_KEY", "")

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)


settings = Settings()
