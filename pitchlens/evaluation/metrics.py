"""Ranking metrics.

Deliberately dependency-free: these are the numbers the whole project is judged
on, so they are plain arithmetic over plain lists and can be unit-tested without
a model, an index or a config. Every function takes `retrieved` as an ordered
list of slide identifiers, best first.
"""

from __future__ import annotations

from math import log2

__all__ = [
    "dedupe",
    "hit_at_k",
    "recall_at_k",
    "precision_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
]


def dedupe(items: list[int]) -> list[int]:
    """Distinct identifiers, first occurrence wins.

    One slide is indexed as both a transcript chunk and a summary chunk, so a
    retriever legitimately returns the same slide twice. Counting it twice would
    inflate precision and let a single slide occupy several top-k slots, so the
    collapse happens once, here, before any metric reads a rank position.
    """
    seen: set[int] = set()
    unique: list[int] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _prepare(retrieved: list[int], gold: list[int], k: int) -> tuple[list[int], set[int]]:
    """The deduped top-k ranking and the distinct gold set."""
    return dedupe(retrieved)[:k], set(gold)


def hit_at_k(retrieved: list[int], gold: list[int], k: int) -> float:
    """1.0 if any gold identifier appears in the top-k, else 0.0."""
    top, relevant = _prepare(retrieved, gold, k)
    return float(any(item in relevant for item in top))


def recall_at_k(retrieved: list[int], gold: list[int], k: int) -> float:
    """Fraction of the gold set found in the top-k."""
    top, relevant = _prepare(retrieved, gold, k)
    if not relevant:
        return 0.0
    return len(relevant.intersection(top)) / len(relevant)


def precision_at_k(retrieved: list[int], gold: list[int], k: int) -> float:
    """Fraction of the top-k that is gold.

    The denominator is the number of distinct slides actually shown, not k: after
    deduping, a system asked for 10 chunks may only have 6 distinct slides to
    offer, and charging it for four slots it never filled measures the index size
    rather than the ranking.
    """
    top, relevant = _prepare(retrieved, gold, k)
    if not top:
        return 0.0
    return len(relevant.intersection(top)) / len(top)


def reciprocal_rank(retrieved: list[int], gold: list[int]) -> float:
    """1 / rank of the first gold hit over the whole ranking, 0.0 if there is none."""
    relevant = set(gold)
    for position, item in enumerate(dedupe(retrieved), start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved: list[int], gold: list[int], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance.

    DCG = sum(rel_i / log2(i + 2)); the ideal ranking front-loads min(|gold|, k)
    hits, so IDCG uses the same discount over that many positions.
    """
    top, relevant = _prepare(retrieved, gold, k)
    if not relevant:
        return 0.0
    dcg = sum(1.0 / log2(i + 2) for i, item in enumerate(top) if item in relevant)
    idcg = sum(1.0 / log2(i + 2) for i in range(min(len(relevant), k)))
    if not idcg:
        return 0.0
    return dcg / idcg
