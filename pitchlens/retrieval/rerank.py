"""Cross-encoder re-ranking, applied as a decorator over any base retriever."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from ..config import settings
from ..domain import ScoredChunk
from .base import Retriever, RetrieverDecorator

if TYPE_CHECKING:
    from fastembed.rerank.cross_encoder import TextCrossEncoder

# ONNX Runtime sizes its allocation arena to the largest batch it is handed, and
# a batch is padded to its longest member. Scoring twelve full slide transcripts
# in one go peaked near 760 MB, which is fatal in a 512 MB container. Small
# batches of truncated text cost nothing in quality -- this cross-encoder
# truncates at 512 tokens regardless, so the tail was never being read.
_BATCH = 4
_MAX_CHARS = 2000


@lru_cache(maxsize=None)
def get_cross_encoder(model_name: str = settings.models.cross_encoder) -> TextCrossEncoder:
    """One model instance per name, shared by every decorator in the process.

    Loading the weights costs seconds and the ablation builds several re-ranked
    retrievers over the same model, so a per-instance load would dominate the
    measured latency. The import stays inside the function so merely importing
    this package does not pay for the ONNX runtime.
    """
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    # One ONNX thread: extra threads each carry their own allocation arena, which
    # buys nothing on a 0.1-CPU instance and costs memory the container does not
    # have.
    return TextCrossEncoder(model_name, threads=1)


class RerankDecorator(RetrieverDecorator):
    """Re-scores a wider candidate pool with a cross-encoder, then cuts to k.

    The base retriever compares query and chunk through independently computed
    representations; the cross-encoder reads the pair jointly, which is far more
    accurate but too slow to run over the whole corpus. Hence the two stages:
    cheap recall first, expensive precision over the survivors.
    """

    def __init__(
        self,
        inner: Retriever,
        candidates: int = settings.retrieval.candidates,
        model_name: str = settings.models.cross_encoder,
    ):
        super().__init__(inner)
        self.candidates = candidates
        self.model_name = model_name

    @property
    def name(self) -> str:
        return f"{self.inner.name}+rerank"

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        pool = self.inner.retrieve(query, self.candidates)
        if not pool:
            return []

        scores = get_cross_encoder(self.model_name).rerank(
            query,
            [scored.chunk.text[:_MAX_CHARS] for scored in pool],
            batch_size=_BATCH,
        )
        ranked = sorted(zip(pool, scores), key=lambda pair: pair[1], reverse=True)[:k]
        return [ScoredChunk(scored.chunk, float(score)) for scored, score in ranked]
