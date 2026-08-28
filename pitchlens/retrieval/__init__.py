"""Retrieval strategies.

Import the interface and the factory from here; the concrete strategy classes
are exported for type annotations and direct construction, but callers that
select a strategy should go through `RetrieverFactory`.
"""

from __future__ import annotations

from .base import Retriever, RetrieverDecorator
from .dense import DenseRetriever
from .factory import RetrieverFactory
from .hybrid import HybridRetriever
from .rerank import RerankDecorator
from .sparse import BM25Retriever

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "RerankDecorator",
    "Retriever",
    "RetrieverDecorator",
    "RetrieverFactory",
]
