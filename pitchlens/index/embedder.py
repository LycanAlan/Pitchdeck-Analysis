"""Text -> vector, once per process.

The embedding model is the one genuinely expensive object in the retrieval
stack: constructing it costs seconds and hundreds of megabytes. Every retriever,
the index builder and the evaluation harness therefore share one instance via
`get_embedder()` instead of each constructing their own.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import settings


class Embedder:
    """Encodes text into unit-norm float32 vectors.

    Normalisation happens here rather than at the index because it is a property
    of the *vectors*, not of one consumer: with unit vectors a FAISS inner
    product is exactly cosine similarity, and any other consumer gets the same
    guarantee for free.
    """

    def __init__(self, model_name: str = settings.models.embedding, batch_size: int = 64):
        self.name = model_name
        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # FAISS requires contiguous float32; sentence-transformers may hand back
        # float64 or a view depending on the backend.
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Process-wide singleton. Cached because model load dominates cold start."""
    return Embedder()
