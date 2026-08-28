"""Shared plumbing for the measurement scripts.

Only things genuinely used by more than one script live here. Table rendering is
re-exported from pitchlens.evaluation.reporting rather than reimplemented, so
there is exactly one table formatter in the project.
"""

from __future__ import annotations

import csv
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pitchlens.config import settings  # noqa: E402
from pitchlens.domain import Chunk, DeckDocument, EvalQuestion  # noqa: E402
from pitchlens.evaluation.reporting import ResultsBundle, ResultsTable  # noqa: E402
from pitchlens.index.embedder import get_embedder  # noqa: E402
from pitchlens.index.store import VectorIndex  # noqa: E402
from pitchlens.llm import GeminiClient  # noqa: E402
from pitchlens.retrieval.factory import RetrieverFactory  # noqa: E402

EVAL_SET_PATH = settings.paths.eval_sets / "eval_set.json"
INDEX_DIR = settings.paths.indices / "corpus"

Bundle = ResultsBundle


class Table(ResultsTable):
    """Incremental builder over the one table formatter.

    Scripts accumulate rows as they iterate decks or modes, which reads better
    than assembling a list of dicts up front. It subclasses ResultsTable rather
    than reimplementing rendering, so there is still exactly one formatter.
    """

    def __init__(self, title: str, columns: list[str] | None = None, rows: list[dict] | None = None):
        super().__init__(title, list(rows or []))
        self._columns_hint = columns

    def add(self, *values, **fields) -> None:
        """Add a row positionally against the declared columns, or by keyword."""
        self.rows.append(dict(zip(self._columns_hint, values)) if values else fields)

    def note(self, text: str) -> None:
        self.note_text = text

__all__ = [
    "EVAL_SET_PATH",
    "INDEX_DIR",
    "Bundle",
    "Chunk",
    "DeckDocument",
    "EvalQuestion",
    "Failures",
    "Stopwatch",
    "Table",
    "banner",
    "chunks_of",
    "csv_list",
    "deck_pdfs",
    "gemini_client",
    "load_documents",
    "load_eval_set",
    "open_index",
    "percentile",
    "retriever_factory",
    "settings",
    "stats_table",
    "step",
    "write_csv",
]


# ── console ───────────────────────────────────────────────────────────────────

def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}")


def step(text: str) -> None:
    print(f"  {text}")


# ── timing ────────────────────────────────────────────────────────────────────

class Stopwatch:
    """Elapsed-time context manager.

    `seconds` reads live inside the block and freezes on exit, so callers can
    time a step and report it without caring which side of the block they are on.
    """

    def __enter__(self) -> "Stopwatch":
        self._start = time.perf_counter()
        self._stopped: float | None = None
        return self

    def __exit__(self, *_exc) -> None:
        self._stopped = time.perf_counter() - self._start

    @property
    def seconds(self) -> float:
        return self._stopped if self._stopped is not None else time.perf_counter() - self._start


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile. Small samples make interpolation false precision."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round(pct / 100 * len(ordered) + 0.5)) - 1, len(ordered) - 1)
    return ordered[max(index, 0)]


# ── failure collection ────────────────────────────────────────────────────────

class Failures:
    """Collects per-item errors so one bad deck does not abort a whole run.

    The user's requirement is that nothing fails silently, so failures are
    surfaced as their own table at the end of every script rather than swallowed.
    """

    def __init__(self, label: str):
        self.label = label
        self.rows: list[dict] = []

    @contextmanager
    def guard(self, item: str):
        try:
            yield
        except Exception as exc:  # noqa: BLE001 - reported, not hidden
            self.rows.append({"item": item, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  !! {item}: {type(exc).__name__}: {exc}")

    def table(self) -> ResultsTable | None:
        return ResultsTable(f"Failures ({self.label})", self.rows) if self.rows else None

    def report(self) -> None:
        if table := self.table():
            table.render()


# ── corpus access ─────────────────────────────────────────────────────────────

def csv_list(raw: str | None) -> list[str] | None:
    return [p.strip() for p in raw.split(",") if p.strip()] if raw else None


def _select(paths: list[Path], names: list[str] | None, limit: int | None) -> list[Path]:
    if names:
        wanted = {n.lower() for n in names}
        paths = [p for p in paths if p.stem.lower() in wanted]
    return paths[:limit] if limit else paths


def deck_pdfs(deck_dir: Path | None = None, names=None, limit=None) -> list[Path]:
    return _select(sorted((deck_dir or settings.paths.decks).glob("*.pdf")), names, limit)


def load_documents(doc_dir: Path | None = None, names=None, limit=None) -> list[DeckDocument]:
    paths = _select(sorted((doc_dir or settings.paths.documents).glob("*.json")), names, limit)
    return [DeckDocument.load(p) for p in paths]


def chunks_of(documents: Iterable[DeckDocument]) -> list[Chunk]:
    return [c for d in documents for c in d.chunks()]


def load_eval_set(path: Path | None = None, decks=None, limit=None) -> list[EvalQuestion]:
    import json

    raw = json.loads((path or EVAL_SET_PATH).read_text(encoding="utf-8"))
    questions = [EvalQuestion(**item) for item in raw]
    if decks:
        wanted = {d.lower() for d in decks}
        questions = [q for q in questions if q.deck.lower() in wanted]
    return questions[:limit] if limit else questions


# ── index / retrieval ─────────────────────────────────────────────────────────

def open_index(documents: list[DeckDocument] | None = None, index_dir: Path | None = None) -> VectorIndex:
    """Load the persisted index, rebuilding only if the corpus fingerprint moved."""
    docs = documents if documents is not None else load_documents()
    return VectorIndex.load_or_build(index_dir or INDEX_DIR, chunks_of(docs), get_embedder())


def retriever_factory(index: VectorIndex) -> RetrieverFactory:
    return RetrieverFactory(index)


# ── llm ───────────────────────────────────────────────────────────────────────

def gemini_client() -> GeminiClient:
    return GeminiClient()


def stats_table(client: GeminiClient, title: str = "LLM usage") -> ResultsTable:
    s = client.stats
    return ResultsTable(
        title,
        [
            {
                "calls": s.calls,
                "failures": s.failures,
                "fallbacks": s.fallbacks,
                "wall_seconds": round(s.seconds, 1),
                "models_used": ", ".join(f"{m}x{n}" for m, n in s.by_model.items()),
            }
        ],
    )


# ── output ────────────────────────────────────────────────────────────────────

def write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path
