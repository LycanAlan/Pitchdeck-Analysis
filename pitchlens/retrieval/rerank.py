"""Cross-encoder re-ranking, applied as a decorator over any base retriever."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from ..config import settings
from ..domain import ScoredChunk
from .base import Retriever, RetrieverDecorator

if TYPE_CHECKING:
    from fastembed.rerank.cross_encoder import TextCrossEncoder


@lru_cache(maxsize=None)
def get_cross_encoder(model_name: str = settings.models.cross_encoder) -> TextCrossEncoder:
    """One model instance per name, shared by every decorator in the process.

    Loading the weights costs seconds and the ablation builds several re-ranked
    retrievers over the same model, so a per-instance load would dominate the
    measured latency. The import stays inside the function so merely importing
    this package does not pay for the ONNX runtime.
    """
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name)


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
            query, [scored.chunk.text for scored in pool]
        )
        ranked = sorted(zip(pool, scores), key=lambda pair: pair[1], reverse=True)[:k]
        return [ScoredChunk(scored.chunk, float(score)) for scored, score in ranked]
