"""The retrieval contract.

Every retrieval strategy in the system — dense, sparse, fused, re-ranked — is a
`Retriever`. That single interface is what makes the ablation study free: the
evaluation harness takes any Retriever and never learns which one it holds, so
adding a strategy adds zero branches anywhere else in the codebase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..domain import ScoredChunk


class Retriever(ABC):
    """Maps a query to ranked chunks."""

    name: str = "retriever"

    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        """Return the top-k chunks for a query, best first."""


class RetrieverDecorator(Retriever):
    """Base for retrievers that wrap another retriever (e.g. re-ranking).

    Composition rather than inheritance keeps strategies orthogonal: any
    decorator works over any base retriever without a combinatorial explosion of
    subclasses.
    """

    def __init__(self, inner: Retriever):
        self.inner = inner

    @property
    def base_name(self) -> str:
        return self.inner.name
