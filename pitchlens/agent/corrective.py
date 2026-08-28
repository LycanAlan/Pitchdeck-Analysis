"""Corrective RAG: retrieve, grade the evidence, rewrite the query, retry.

The failure this fixes is retrieval, not generation. When a question is phrased
unlike the deck ("how much are they raising?" vs a slide titled "Use of
Funds"), a single retrieval pass returns plausible-but-wrong chunks and the
generator dutifully answers from them. Grading the evidence *before* generating
turns that silent error into a second, better-phrased query.

`CorrectiveRAGAgent` deliberately exposes the same `answer`/`name` surface as
`Answerer`, so the evaluation harness swaps the arms without a single branch.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from ..config import settings
from ..domain import Answer, ScoredChunk
from ..generation.base import RAGPipeline
from ..generation.prompts import GRADE_PROMPT, REWRITE_PROMPT, format_context
from ..llm import LLMClient
from ..retrieval.base import Retriever


class AgentState(TypedDict):
    """Everything that flows between graph nodes."""

    question: str  # the user's original wording, never mutated
    query: str  # what retrieval actually ran with this round
    scored: list[ScoredChunk]
    rounds: int
    grounded: bool
    reason: str
    rewritten: str  # last rewrite, empty when the first pass sufficed
    answer: str


class CorrectiveRAGAgent(RAGPipeline):
    """Grade-and-rewrite loop over the shared retrieve/generate steps."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        k: int = settings.retrieval.final_k,
        max_rounds: int = 3,
    ):
        super().__init__(retriever, llm, k)
        self.max_rounds = max_rounds
        self._graph = self._build()

    @property
    def name(self) -> str:
        return f"corrective[{self.retriever.name}]"

    def answer(self, question: str) -> Answer:
        state: AgentState = {
            "question": question,
            "query": question,
            "scored": [],
            "rounds": 0,
            "grounded": False,
            "reason": "",
            "rewritten": "",
            "answer": "",
        }
        # Each round costs three node visits, so the recursion budget has to
        # track max_rounds rather than sit at langgraph's default.
        final = self._graph.invoke(state, {"recursion_limit": 3 * self.max_rounds + 4})
        return Answer.from_chunks(
            final["answer"],
            final["scored"],
            rewritten_query=final["rewritten"],
            retrieval_rounds=final["rounds"],
            grounded=final["grounded"],
        )

    def _build(self):
        graph = StateGraph(AgentState)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("grade", self._grade_node)
        graph.add_node("rewrite", self._rewrite_node)
        graph.add_node("generate", self._generate_node)

        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "grade")
        graph.add_conditional_edges("grade", self._route, ["rewrite", "generate"])
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("generate", END)
        return graph.compile()

    def _retrieve_node(self, state: AgentState) -> dict:
        return {
            "scored": self.retrieve(state["query"]),
            "rounds": state["rounds"] + 1,
        }

    def _grade_node(self, state: AgentState) -> dict:
        prompt = GRADE_PROMPT.format(
            context=format_context(state["scored"]),
            question=state["question"],
        )
        verdict = self.llm.complete_json(prompt, chain=settings.models.judge)
        # The grader is asked for "yes"/"no" strings rather than JSON booleans
        # because models emit `true`, `"true"` and `"yes"` interchangeably.
        return {
            "grounded": str(verdict["sufficient"]).strip().lower().startswith("y"),
            "reason": verdict.get("reason", ""),
        }

    def _rewrite_node(self, state: AgentState) -> dict:
        prompt = REWRITE_PROMPT.format(
            question=state["question"],
            query=state["query"],
            reason=state["reason"],
        )
        rewritten = self.llm.complete(prompt, chain=settings.models.analysis).strip().strip('"')
        return {"query": rewritten, "rewritten": rewritten}

    def _generate_node(self, state: AgentState) -> dict:
        return {"answer": self.generate(state["question"], state["scored"])}

    def _route(self, state: AgentState) -> str:
        """Stop on good evidence, or when the retry budget is spent."""
        if state["grounded"] or state["rounds"] >= self.max_rounds:
            return "generate"
        return "rewrite"
