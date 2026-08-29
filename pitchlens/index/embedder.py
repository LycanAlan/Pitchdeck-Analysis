"""Sentence embeddings.

Runs the MiniLM weights through ONNX Runtime rather than PyTorch. The model is
identical -- embeddings match the torch implementation to float32 precision
(cosine 1.000000) so a persisted index stays valid across the switch -- but the
resident footprint drops from ~474 MB to ~199 MB. That is the difference between
fitting a 512 MB free-tier container and not.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding

from ..config import settings


class Embedder:
    """Encodes text to L2-normalised float32 vectors."""

    # Batch size trades throughput for peak memory. ONNX Runtime sizes its arena
    # to the largest batch it sees, and 64 pushed a full-corpus build past a
    # 512 MB container. 16 keeps the peak survivable at negligible cost, since
    # serving embeds one query at a time anyway.
    def __init__(self, model_name: str = settings.models.embedding, batch_size: int = 16):
        self.name = model_name
        self._batch_size = batch_size
        self._model = TextEmbedding(model_name)

    @property
    def dimension(self) -> int:
        return len(self.encode_one("dimension probe"))

    def encode(self, texts: list[str]) -> np.ndarray:
        """(n, dim) float32, unit norm, so FAISS inner product means cosine."""
        vectors = np.asarray(
            list(self._model.embed(texts, batch_size=self._batch_size)), dtype=np.float32
        )
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.ascontiguousarray(vectors)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Process-wide singleton. Loading the model costs seconds; every retriever
    and every script shares this one instance."""
    return Embedder()
