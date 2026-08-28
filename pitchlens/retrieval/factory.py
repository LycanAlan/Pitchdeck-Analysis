"""Construction of every retrieval strategy from a single mode string."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Retriever
from .dense import DenseRetriever
from .hybrid import HybridRetriever
from .rerank import RerankDecorator
from .sparse import BM25Retriever

if TYPE_CHECKING:
    from ..index.store import VectorIndex

_RERANK = "+rerank"


class RetrieverFactory:
    """Turns a mode name into a wired-up Retriever.

    This is what reduces the ablation to a loop over `MODES`: every strategy the
    study compares is addressable by string, so the evaluation harness, the API
    and the UI all select a strategy the same way and none of them import a
    concrete retriever class.
    """

    MODES: tuple[str, ...] = ("dense", "bm25", "hybrid", "dense+rerank", "hybrid+rerank")

    def __init__(self, index: VectorIndex):
        self.index = index
        self._built: dict[str, Retriever] = {}

    def build(self, mode: str) -> Retriever:
        base, _, suffix = mode.partition("+")
        retriever = self._base(base)
        return RerankDecorator(retriever) if suffix else retriever

    def build_all(self) -> list[Retriever]:
        return [self.build(mode) for mode in self.MODES]

    def _base(self, mode: str) -> Retriever:
        # Memoised because "bm25", "hybrid" and "hybrid+rerank" all want the same
        # BM25 index, and rebuilding it per mode would be the ablation's largest
        # avoidable cost.
        if mode not in self._built:
            self._built[mode] = self._construct(mode)
        return self._built[mode]

    def _construct(self, mode: str) -> Retriever:
        match mode:
            case "dense":
                return DenseRetriever(self.index)
            case "bm25":
                return BM25Retriever(self.index.chunks)
            case "hybrid":
                return HybridRetriever([self._base("dense"), self._base("bm25")])
        raise ValueError(f"unknown retrieval mode {mode!r}; expected one of {self.MODES}")
