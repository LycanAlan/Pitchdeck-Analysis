"""Rank fusion over any number of retrievers."""

from __future__ import annotations

from ..config import settings
from ..domain import Chunk, ScoredChunk
from .base import Retriever


class HybridRetriever(Retriever):
    """Reciprocal Rank Fusion of N retrievers.

    Fusion is rank-based, not score-based, because the inputs are not on a
    comparable scale: BM25 is unbounded and corpus-dependent while cosine
    similarity sits in [-1, 1]. Averaging them would let whichever retriever
    happens to emit larger magnitudes dominate, and any fix for that (min-max or
    z-score normalisation) is sensitive to the shape of each candidate pool.
    RRF only reads positions, so it needs no normalisation and no per-retriever
    weight to tune.
    """

    name = "hybrid"

    def __init__(
        self,
        retrievers: list[Retriever],
        rrf_k: int = settings.retrieval.rrf_k,
        candidates: int = settings.retrieval.candidates,
    ):
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.candidates = candidates

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        fused: dict[str, float] = {}
        seen: dict[str, Chunk] = {}

        for retriever in self.retrievers:
            for rank, scored in enumerate(retriever.retrieve(query, self.candidates), start=1):
                cid = scored.chunk.id
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
                seen.setdefault(cid, scored.chunk)

        top = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:k]
        return [ScoredChunk(seen[cid], score) for cid, score in top]
