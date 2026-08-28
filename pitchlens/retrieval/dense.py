"""Dense (embedding similarity) retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..domain import ScoredChunk
from .base import Retriever

if TYPE_CHECKING:
    from ..index.store import VectorIndex


class DenseRetriever(Retriever):
    """Nearest neighbours in embedding space.

    The vector index already owns the embedder and the similarity metric, so
    this is a thin adapter: its only job is to present that index through the
    `Retriever` interface the ablation harness speaks.
    """

    name = "dense"

    def __init__(self, index: VectorIndex):
        self.index = index

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        return self.index.search(query, k)
