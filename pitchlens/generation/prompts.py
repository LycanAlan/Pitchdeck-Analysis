"""Every prompt in the generation path, written once.

The baseline answerer and the corrective agent must be compared on identical
wording — an ablation that changes the prompt between arms measures the prompt,
not the agent. So both import from here and neither owns prompt text.
"""

from __future__ import annotations

from ..domain import ScoredChunk

# The refusal is a fixed string so the evaluation harness can detect abstention
# by exact match instead of asking a judge whether the model hedged.
CANNOT_ANSWER = "I cannot find the answer in the provided context."

NO_CONTEXT = "(no context was retrieved)"


def format_context(scored: list[ScoredChunk]) -> str:
    """Render chunks as labelled evidence blocks.

    Deck and slide labels are part of the rendering rather than the caller's job
    because the answerer, the grader and the citation check must all see the
    same view of the evidence.
    """
    if not scored:
        return NO_CONTEXT

    blocks = []
    for rank, sc in enumerate(scored, start=1):
        chunk = sc.chunk
        header = f"[{rank}] deck={chunk.deck} | slide={chunk.slide} | kind={chunk.kind.value}"
        if chunk.section:
            header += f" | section={chunk.section}"
        blocks.append(f"{header}\n{chunk.text.strip()}")
    return "\n\n".join(blocks)


ANSWER_PROMPT = f"""You are a venture analyst answering questions about startup pitch decks.

Answer the question using ONLY the context below.

Rules:
- Use only facts stated in the context. Never use outside knowledge, never infer
  numbers that are not written, and never guess.
- If the context does not contain the answer, reply with exactly this sentence
  and nothing else: {CANNOT_ANSWER}
- Quote figures, dates and names exactly as they appear.
- Reference the evidence inline as (deck, slide N).
- Be concise: a few sentences, no preamble.

Context:
{{context}}

Question: {{question}}

Answer:"""


GRADE_PROMPT = """You are grading retrieved evidence, not answering the question.

Decide whether the context below contains enough information to answer the
question completely and correctly. Judge only sufficiency of the evidence — do
not use outside knowledge, and do not reward context that is merely on-topic.

Context:
{context}

Question: {question}

Reply with JSON only:
{{"sufficient": "yes" or "no", "reason": "one short sentence"}}"""


REWRITE_PROMPT = """A retrieval system failed to find evidence for a question about a
startup pitch deck. Rewrite the query so it retrieves better.

Original question: {question}
Query that was tried: {query}
Why it failed: {reason}

Guidance:
- Keep the original intent exactly; do not answer the question.
- Prefer the vocabulary a pitch deck slide would actually use (e.g. "ARR",
  "TAM", "go-to-market", "use of funds", "burn rate") over conversational
  phrasing.
- Expand abbreviations and add the obvious synonyms a slide title might carry.

Output only the rewritten query, on one line, with no quotes or explanation."""
