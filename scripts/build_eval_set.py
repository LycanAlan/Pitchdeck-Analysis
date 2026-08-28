"""Generate a verified ground-truth eval set from the ingested decks.

    python scripts/build_eval_set.py --per-deck 6

Needs GEMINI_API_KEY. Every generated item is checked against the deck it came
from; items citing slides that do not exist are dropped and reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import (  # noqa: E402
    EVAL_SET_PATH,
    DeckDocument,
    EvalQuestion,
    Failures,
    Table,
    banner,
    csv_list,
    gemini_client,
    load_documents,
    settings,
    stats_table,
    step,
)

CATEGORIES = ("factual", "numeric", "multi_hop", "unanswerable")

PROMPT = """You are building a retrieval-evaluation set for the pitch deck "{deck}".

Below are its slides. Write exactly {n} questions an analyst would ask about THIS deck.

Distribute them across these categories:
- factual: answered by one slide's stated content.
- numeric: the answer is a specific figure, percentage or amount on a slide.
- multi_hop: answering requires combining two or more DIFFERENT slides.
- unanswerable: a plausible question about this company that these slides do NOT answer.

Rules:
- gold_slides must list the slide numbers that contain the evidence, exactly as
  numbered below. multi_hop items must list two or more.
- unanswerable items must have gold_slides: [] and expected_answer stating the
  deck does not cover it.
- Quote figures exactly as they appear on the slide.
- Never ask about slide numbering or the deck as a document ("what is on slide 4").

Return ONLY a JSON array of objects with keys:
question, expected_answer, gold_slides, category

SLIDES
{slides}
"""


class EvalSetBuilder:
    """Generates and verifies eval questions, one deck at a time."""

    def __init__(self, client, per_deck: int = 6, max_slides: int = 40, slide_chars: int = 700):
        self.client = client
        self.per_deck = per_deck
        self.max_slides = max_slides
        self.slide_chars = slide_chars
        self.dropped: list[tuple[str, str]] = []

    def build(self, document: DeckDocument) -> list[EvalQuestion]:
        raw = self.client.complete_json(
            PROMPT.format(
                deck=document.name, n=self.per_deck, slides=self._render_slides(document)
            ),
            chain=settings.models.analysis,
        )
        items = raw if isinstance(raw, list) else raw.get("questions", [])
        return self._verify(document, items)

    def _render_slides(self, document: DeckDocument) -> str:
        blocks = []
        for slide in self._sample(document.slides):
            body = (slide.summary or slide.transcript).strip()[: self.slide_chars]
            header = f"[slide {slide.slide}]"
            if slide.section:
                header += f" section: {slide.section}"
            blocks.append(f"{header}\n{body}")
        return "\n\n".join(blocks)

    def _sample(self, slides: list) -> list:
        """Evenly spaced rather than random: the same deck yields the same prompt."""
        if len(slides) <= self.max_slides:
            return slides
        stride = len(slides) / self.max_slides
        return [slides[int(i * stride)] for i in range(self.max_slides)]

    def _verify(self, document: DeckDocument, items: list[dict]) -> list[EvalQuestion]:
        real = {s.slide for s in document.slides}
        verified: list[EvalQuestion] = []

        for position, item in enumerate(items):
            question = str(item.get("question", "")).strip()
            category = str(item.get("category", "factual")).strip().lower()
            gold = [int(s) for s in item.get("gold_slides", []) if str(s).lstrip("-").isdigit()]
            label = f"{document.name}#{position}"

            reason = self._reject(question, category, gold, real)
            if reason:
                self.dropped.append((label, reason))
                continue

            verified.append(
                EvalQuestion(
                    id=f"{document.name}-{len(verified):02d}",
                    deck=document.name,
                    question=question,
                    expected_answer=str(item.get("expected_answer", "")).strip(),
                    gold_slides=sorted(set(gold)),
                    category=category,
                )
            )
        return verified

    def _reject(self, question: str, category: str, gold: list[int], real: set[int]) -> str | None:
        if not question:
            return "empty question"
        if category not in CATEGORIES:
            return f"unknown category {category!r}"
        if category == "unanswerable":
            return f"unanswerable item cites slides {gold}" if gold else None
        missing = [s for s in gold if s not in real]
        if missing:
            return f"gold_slides {missing} are not slides of this deck"
        if not gold:
            return "no gold slides"
        if category == "multi_hop" and len(set(gold)) < 2:
            return "multi_hop item cites a single slide"
        return None


def collect_tables(
    per_deck: int = 6,
    names: list[str] | None = None,
    limit: int | None = None,
    out: Path | None = None,
    max_slides: int = 40,
) -> list[Table]:
    out = out or EVAL_SET_PATH
    documents = load_documents(names=names, limit=limit)

    client = gemini_client()
    builder = EvalSetBuilder(client, per_deck, max_slides)
    failures = Failures("eval-set generation")
    questions: list[EvalQuestion] = []

    banner(f"Generating up to {per_deck} questions for each of {len(documents)} decks")
    for position, document in enumerate(documents, start=1):
        with failures.guard(document.name):
            produced = builder.build(document)
            questions.extend(produced)
            step(f"[{position}/{len(documents)}] {document.name}: kept {len(produced)}/{per_deck}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([asdict(q) for q in questions], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    by_deck = Table("Eval set by deck", ["deck", "kept", *CATEGORIES])
    for document in documents:
        mine = [q for q in questions if q.deck == document.name]
        by_deck.add(document.name, len(mine), *[sum(1 for q in mine if q.category == c) for c in CATEGORIES])

    by_deck.note(f"wrote {len(questions)} verified questions to {out}")
    by_deck.note(f"requested {per_deck * len(documents)}, dropped {len(builder.dropped)} in verification")
    by_deck.note(f"gold slides verified against the real slide numbers of each DeckDocument")

    tables = [by_deck]
    if builder.dropped:
        dropped = Table("Dropped in verification", ["item", "reason"])
        for label, reason in builder.dropped:
            dropped.add(label, reason)
        tables.append(dropped)

    failures.report()
    return [t for t in (*tables, stats_table(client, "LLM usage (eval set)"), failures.table()) if t]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ground-truth eval questions from the ingested DeckDocuments and write "
        "them to data/eval/eval_set.json. Each item is verified against its deck before being kept. "
        "Requires GEMINI_API_KEY.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--per-deck", type=int, default=6, help="questions requested per deck")
    parser.add_argument("--decks", type=str, default=None, help="comma-separated subset of deck names")
    parser.add_argument("--limit", type=int, default=None, help="only use the first N decks")
    parser.add_argument("--out", type=Path, default=EVAL_SET_PATH, help="output JSON path")
    parser.add_argument("--max-slides", type=int, default=40, help="slides sampled into each prompt")
    args = parser.parse_args()

    tables = collect_tables(
        args.per_deck, csv_list(args.decks), args.limit, args.out, args.max_slides
    )
    for table in tables:
        table.show()


if __name__ == "__main__":
    main()
