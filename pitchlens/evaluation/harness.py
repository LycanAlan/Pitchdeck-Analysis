"""Evaluation harnesses.

Two things get measured and they share one shape: walk the eval set, time each
unit of work, fold per-question records into a single flat summary row that the
reporting layer can print without knowing what was measured. Only the middle
step differs between retrieval and generation, so only the middle step is
subclassed.
"""

from __future__ import annotations

import statistics
from abc import ABC
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Protocol

from ..config import settings
from ..domain import Answer, EvalQuestion, ScoredChunk
from ..retrieval.base import Retriever
from .judge import AnswerJudge, format_context
from .metrics import hit_at_k, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank

__all__ = ["AnswerSystem", "BaseEvaluator", "RetrievalEvaluator", "AnswerEvaluator"]

UNANSWERABLE = "unanswerable"

_RANKING_METRICS: dict[str, Callable[[list[int], list[int], int], float]] = {
    "hit": hit_at_k,
    "recall": recall_at_k,
    "precision": precision_at_k,
    "ndcg": ndcg_at_k,
}

# A system that declines has to be recognised without a second model call, so
# this is a lexical pass over the refusal phrasings the generator prompt asks
# for. Routing abstention through the judge would make the rate cost money and
# drift between runs.
_ABSTENTION_MARKERS = (
    "not stated",
    "does not state",
    "not mentioned",
    "does not mention",
    "not specified",
    "not provided",
    "no information",
    "not contain",
    "cannot be determined",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "i don't know",
    "insufficient",
)


class AnswerSystem(Protocol):
    """What both the plain Answerer and the corrective agent expose."""

    name: str

    def answer(self, question: str) -> Answer: ...


def ranked_slide_ids(chunks: Sequence[ScoredChunk], deck: str) -> list[int]:
    """Rank positions as slide numbers, with foreign-deck results kept distinct.

    The index spans every deck, so a result can be slide 7 of the wrong deck. It
    consumed a top-k slot and must keep its rank position, but it must never
    match gold slide 7 -- hence a distinct negative id per foreign result, which
    cannot collide with a gold slide and cannot be deduped against another
    foreign result.
    """
    return [sc.slide if sc.deck == deck else -(i + 1) for i, sc in enumerate(chunks)]


def _abstained(answer: Answer) -> bool:
    if not answer.grounded:
        return True
    lowered = answer.text.lower()
    return any(marker in lowered for marker in _ABSTENTION_MARKERS)


class BaseEvaluator(ABC):
    """Shared aggregation for every evaluator."""

    system_key = "system"

    def __init__(self, questions: list[EvalQuestion]):
        self.questions = questions
        self.records: list[dict] = []

    @staticmethod
    def _mean(values: Iterable[float]) -> float:
        values = list(values)
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _median(values: Iterable[float]) -> float:
        values = list(values)
        return statistics.median(values) if values else 0.0

    @classmethod
    def _pct(cls, flags: Iterable[bool]) -> float:
        """Percentage in [0, 100] of true flags."""
        return 100.0 * cls._mean(float(bool(flag)) for flag in flags)

    def _summarize(self, name: str, metrics: dict[str, float], records: list[dict]) -> dict:
        """The one flat row every report consumes.

        `queries` is the number of questions the metrics were actually computed
        over and `errors` the number excluded, so a headline number can never be
        read without the denominator that produced it.
        """
        self.records = records
        return {
            self.system_key: name,
            **metrics,
            "queries": len(records),
            "errors": sum(1 for r in records if r.get("error")),
        }


class RetrievalEvaluator(BaseEvaluator):
    """Ranking quality of any Retriever, with no knowledge of which one it holds."""

    system_key = "retriever"

    def evaluate(
        self,
        retriever: Retriever,
        k_values: tuple[int, ...] = settings.retrieval.eval_k_values,
    ) -> dict:
        """Ranking metrics averaged over the answerable questions.

        Unanswerable questions carry no gold slides, so scoring them would add a
        run of guaranteed zeros to every mean and understate the retriever; they
        are skipped and counted in `skipped_unanswerable` instead.
        """
        scored = [q for q in self.questions if q.category != UNANSWERABLE]
        depth = max(k_values)
        records = [self._score(retriever, q, k_values, depth) for q in scored]

        metrics: dict[str, float] = {
            f"{metric}@{k}": self._mean(r[f"{metric}@{k}"] for r in records)
            for metric in _RANKING_METRICS
            for k in k_values
        }
        metrics["mrr"] = self._mean(r["rr"] for r in records)
        metrics["median_latency_ms"] = self._median(r["latency_ms"] for r in records)

        summary = self._summarize(retriever.name, metrics, records)
        summary["skipped_unanswerable"] = len(self.questions) - len(scored)
        return summary

    def _score(
        self, retriever: Retriever, question: EvalQuestion, k_values: tuple[int, ...], depth: int
    ) -> dict:
        # Retrieval is measured serially: under a thread pool the wall clock
        # would report lock and BLAS contention rather than retriever latency.
        started = perf_counter()
        chunks = retriever.retrieve(question.question, depth)
        latency_ms = (perf_counter() - started) * 1000.0

        ranked = ranked_slide_ids(chunks, question.deck)
        record = {
            "id": question.id,
            "deck": question.deck,
            "category": question.category,
            "question": question.question,
            "gold_slides": ", ".join(str(s) for s in question.gold_slides),
            "retrieved_slides": ", ".join(str(s) for s in ranked if s > 0),
            "rr": reciprocal_rank(ranked, question.gold_slides),
            "latency_ms": latency_ms,
            "error": "",
        }
        for name, fn in _RANKING_METRICS.items():
            for k in k_values:
                record[f"{name}@{k}"] = fn(ranked, question.gold_slides, k)
        return record


class AnswerEvaluator(BaseEvaluator):
    """End-to-end answer quality of any system exposing `.answer` and `.name`."""

    system_key = "system"

    def __init__(self, questions: list[EvalQuestion], judge: AnswerJudge, max_workers: int = 4):
        super().__init__(questions)
        self.judge = judge
        self.max_workers = max_workers

    def evaluate(self, answerer: AnswerSystem) -> dict:
        """Judged answer metrics.

        `*_pct` keys are percentages in [0, 100]; `abstention_rate` is a fraction
        in [0, 1] measured over every question, answerable or not, since
        declining on an unanswerable question is the desired behaviour and
        declining on an answerable one is a failure.
        """
        # Generation is network-bound, so a small pool turns a serial run of
        # minutes into one of seconds. It is bounded to stay under the API's
        # concurrency limits; mean latency is therefore per-call wall time under
        # that much load, not an isolated single-request measurement.
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            records = list(pool.map(lambda q: self._score(answerer, q), self.questions))

        judged = [r for r in records if not r["error"]]
        metrics = {
            "correct_pct": self._pct(r["label"] == "correct" for r in judged),
            "partially_correct_pct": self._pct(r["label"] == "partially_correct" for r in judged),
            "hallucination_pct": self._pct(r["label"] == "hallucinated" for r in judged),
            "faithfulness_pct": self._pct(r["faithful"] for r in judged),
            "abstention_rate": self._mean(float(r["abstained"]) for r in records),
            "mean_latency_s": self._mean(r["latency_s"] for r in judged),
        }

        summary = self._summarize(answerer.name, metrics, records)
        summary["records"] = records
        return summary

    def _score(self, answerer: AnswerSystem, question: EvalQuestion) -> dict:
        record = {
            "id": question.id,
            "deck": question.deck,
            "category": question.category,
            "question": question.question,
            "expected": question.expected_answer,
            "gold_slides": ", ".join(str(s) for s in question.gold_slides),
            "generated": "",
            "label": "",
            "faithful": False,
            "reason": "",
            "abstained": False,
            "context_hit": 0.0,
            "citations": 0,
            "retrieval_rounds": 0,
            "latency_s": 0.0,
            "error": "",
        }
        started = perf_counter()
        try:
            answer = answerer.answer(question.question)
            record["latency_s"] = perf_counter() - started
            verdict = self.judge.judge(
                question.question,
                question.expected_answer,
                answer.text,
                format_context(answer.retrieved),
            )
        except Exception as exc:  # noqa: BLE001
            # A flaky model call must not void a whole run, but it must not be
            # silently graded either: the record carries the error and is
            # excluded from every quality denominator above.
            record["latency_s"] = perf_counter() - started
            record["error"] = f"{type(exc).__name__}: {exc}"
            return record

        ranked = ranked_slide_ids(answer.retrieved, question.deck)
        record |= {
            "generated": answer.text,
            "label": verdict.label,
            "faithful": verdict.faithful,
            "reason": verdict.reason,
            "abstained": _abstained(answer),
            "context_hit": hit_at_k(ranked, question.gold_slides, settings.retrieval.final_k),
            "citations": len(answer.citations),
            "retrieval_rounds": answer.retrieval_rounds,
        }
        return record
