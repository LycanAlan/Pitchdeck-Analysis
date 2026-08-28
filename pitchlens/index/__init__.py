"""Embedding and the persistent dense index."""

from __future__ import annotations

from .embedder import Embedder, get_embedder
from .store import VectorIndex

__all__ = ["Embedder", "get_embedder", "VectorIndex"]
