"""Sparse lexical retrieval (BM25).

Kept as a first-class strategy because pitch decks are dense with exact tokens
an embedding model blurs together — round names, ticker-like product codes and
money figures. BM25 matches those literally; the dense retriever does not.
"""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from ..domain import Chunk, ScoredChunk
from .base import Retriever

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Split on non-alphanumerics, lowercased.

    Deck copy is full of mixed letter-digit tokens ("$8.7B", "M2-M20", "Q3").
    Splitting on \\W keeps each alphanumeric run whole — "8", "7b", "m2", "m20" —
    where a letters-only tokenizer would strip the digits that carry the meaning.
    """
    return _TOKEN.findall(text.lower())


class BM25Retriever(Retriever):
    """Okapi BM25 over the same chunk set the vector index holds.

    Built directly on rank_bm25 rather than through LangChain's wrapper so the
    raw, un-normalised BM25 score reaches the caller — the fusion layer ranks on
    position, and the evaluation layer reports the real number.
    """

    name = "bm25"

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = np.argsort(scores)[::-1][:k]
        # BM25 scores a chunk 0 when it shares no query term; emitting those
        # would pad the result with arbitrary non-matches and inflate recall.
        return [ScoredChunk(self.chunks[i], float(scores[i])) for i in ranked if scores[i] > 0]
