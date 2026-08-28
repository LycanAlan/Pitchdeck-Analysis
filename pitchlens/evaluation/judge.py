"""LLM-as-judge grading of generated answers.

Correctness and faithfulness are graded as two independent axes because they
fail independently: an answer can restate the reference perfectly while citing
context that never mentioned it, and a faithful answer can still be useless.
Collapsing them into one score hides exactly the failure mode a RAG system is
built to prevent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..config import settings
from ..domain import ScoredChunk
from ..llm import LLMClient

__all__ = ["Verdict", "AnswerJudge", "format_context", "LABELS"]

LABELS = ("correct", "partially_correct", "incorrect", "hallucinated")

# Judges reach for these spellings often enough to be worth mapping; anything
# outside the enum is a parse failure, not a silent default.
_ALIASES = {
    "partial": "partially_correct",
    "partially": "partially_correct",
    "hallucination": "hallucinated",
    "hallucinating": "hallucinated",
    "wrong": "incorrect",
}

_TRUTHY = {"true", "yes", "1", "y"}

_RUBRIC = """You grade one answer produced by a retrieval-augmented question answering \
system over a startup pitch deck. Grade strictly and return JSON only.

Decide two things independently.

1. "label" - how the SYSTEM ANSWER compares to the REFERENCE ANSWER:
   "correct"            every fact the reference asserts is present and nothing is
                        contradicted. Numbers must match. Different wording with the
                        same meaning is correct. If the reference says the deck does
                        not answer this, and the system declined to answer, that is
                        correct.
   "partially_correct"  part of the reference is captured, something material is
                        missing, but nothing is contradicted.
   "incorrect"          contradicts the reference, answers a different question, or
                        declines when the reference gives a real answer.
   "hallucinated"       asserts a specific fact - a number, name, date or claim - that
                        appears in neither the reference nor the retrieved context.
                        Prefer this over "incorrect" whenever an invented specific is
                        present.

2. "faithful" - true only if EVERY claim in the SYSTEM ANSWER is supported by the
   RETRIEVED CONTEXT. Judge this against the context alone and never against the
   reference: an answer can be factually correct yet unfaithful because the context
   it was given does not support it. An answer that only declines to answer is
   faithful.

Respond with exactly this JSON and nothing else:
{"label": "<one of correct|partially_correct|incorrect|hallucinated>", "faithful": true, "reason": "<at most 25 words>"}"""


@dataclass
class Verdict:
    label: str
    faithful: bool
    reason: str

    @property
    def correct(self) -> bool:
        return self.label == "correct"

    @property
    def hallucinated(self) -> bool:
        return self.label == "hallucinated"


def format_context(chunks: Sequence[ScoredChunk]) -> str:
    """Render retrieved chunks the one way the judge ever sees them."""
    return "\n\n".join(
        f"[{sc.chunk.deck} | slide {sc.chunk.slide} | {sc.chunk.kind.value}]\n{sc.chunk.text}"
        for sc in chunks
    )


def _normalise_label(raw: object) -> str:
    token = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    token = _ALIASES.get(token, token)
    if token not in LABELS:
        # Coercing an unrecognised label to a default would quietly move the
        # headline accuracy number. The caller records this as an error instead
        # and drops the question from every denominator.
        raise ValueError(f"judge returned an unknown label: {raw!r}")
    return token


def _as_bool(raw: object) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in _TRUTHY


class AnswerJudge:
    """Grades generated answers with a model from the judge chain.

    The judge chain in config is a different and stronger family than the
    generation chain on purpose: a model asked to grade its own output rates it
    higher than a neutral grader does, so reusing the generator here would buy a
    few points of self-preference bias rather than measure anything.
    """

    name = "llm_judge"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def judge(self, question: str, expected: str, generated: str, context: str) -> Verdict:
        prompt = (
            f"{_RUBRIC}\n\n"
            f"QUESTION\n{question}\n\n"
            f"REFERENCE ANSWER\n{expected}\n\n"
            f"RETRIEVED CONTEXT\n{context or '(nothing was retrieved)'}\n\n"
            f"SYSTEM ANSWER\n{generated}\n"
        )
        payload = self.llm.complete_json(prompt, chain=settings.models.judge)
        return Verdict(
            label=_normalise_label(payload["label"]),
            faithful=_as_bool(payload.get("faithful")),
            reason=str(payload.get("reason", "")).strip(),
        )
