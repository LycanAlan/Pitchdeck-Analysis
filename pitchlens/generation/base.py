"""The retrieve-and-generate machinery shared by every answering strategy.

The baseline answerer and the corrective agent differ only in what happens
*between* retrieval and generation. Both steps therefore live here once, so the
two arms of the agentic ablation cannot silently drift apart — a difference in
measured accuracy is then attributable to the loop and nothing else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import settings
from ..domain import Answer, ScoredChunk
from ..llm import LLMClient
from ..retrieval.base import Retriever
from .prompts import ANSWER_PROMPT, format_context


class RAGPipeline(ABC):
    """Common interface and shared steps for question answering.

    Subclasses implement `answer`; the evaluation harness only ever sees this
    type, which is what lets it swap baseline for agent with zero branching.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        k: int = settings.retrieval.final_k,
    ):
        self.retriever = retriever
        self.llm = llm
        self.k = k

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier for this arm, used as the results-table row label."""

    @abstractmethod
    def answer(self, question: str) -> Answer:
        """Answer a question against the indexed decks."""

    def retrieve(self, query: str) -> list[ScoredChunk]:
        return self.retriever.retrieve(query, self.k)

    def generate(self, question: str, scored: list[ScoredChunk]) -> str:
        prompt = ANSWER_PROMPT.format(context=format_context(scored), question=question)
        return self.llm.complete(prompt, chain=settings.models.generation)
