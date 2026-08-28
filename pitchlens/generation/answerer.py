"""One-shot RAG: retrieve once, generate once.

This is the control arm of the agentic ablation. It shares its retrieve and
generate steps with the corrective agent, so any accuracy gap between them
comes from the grade/rewrite loop alone.
"""

from __future__ import annotations

from ..domain import Answer
from .base import RAGPipeline


class Answerer(RAGPipeline):
    """Baseline: no grading, no rewriting, no second look."""

    @property
    def name(self) -> str:
        return f"baseline[{self.retriever.name}]"

    def answer(self, question: str) -> Answer:
        scored = self.retrieve(question)
        return Answer.from_chunks(self.generate(question, scored), scored)
