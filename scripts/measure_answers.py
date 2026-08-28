"""Answer quality: one-shot RAG vs the corrective agent, scored by an LLM judge.

Needs GEMINI_API_KEY. The judge runs on a deliberately stronger and different
model family than the generator (see settings.models.judge) so the system is not
grading its own output.

    python scripts/measure_answers.py
    python scripts/measure_answers.py --systems baseline --limit 20
"""

from __future__ import annotations

import argparse

from _common import (
    Bundle,
    Failures,
    Table,
    banner,
    csv_list,
    gemini_client,
    load_documents,
    load_eval_set,
    open_index,
    retriever_factory,
    settings,
    stats_table,
    step,
    write_csv,
)

SYSTEMS = ("baseline", "corrective")


def _build_system(kind: str, retriever, client):
    """Both arms satisfy the same answer()/name interface, so the harness never branches."""
    from pitchlens.agent.corrective import CorrectiveRAGAgent
    from pitchlens.generation.answerer import Answerer

    return Answerer(retriever, client) if kind == "baseline" else CorrectiveRAGAgent(retriever, client)


def collect_tables(systems=None, mode=None, decks=None, limit=None, workers=4) -> list[Table]:
    from pitchlens.evaluation.harness import AnswerEvaluator
    from pitchlens.evaluation.judge import AnswerJudge

    questions = load_eval_set(decks=decks, limit=limit)
    banner("Answer quality")
    step(f"{len(questions)} questions")

    index = open_index(load_documents())
    factory = retriever_factory(index)
    retrieval_mode = mode or "hybrid+rerank"
    retriever = factory.build(retrieval_mode)
    step(f"retriever: {retrieval_mode}")

    client = gemini_client()
    evaluator = AnswerEvaluator(questions, AnswerJudge(client), max_workers=workers)

    failures = Failures("answer evaluation")
    rows, records = [], []
    for kind in systems or list(SYSTEMS):
        with failures.guard(kind):
            step(f"evaluating {kind} ...")
            row = evaluator.evaluate(_build_system(kind, retriever, client))
            rows.append(row)
            for record in evaluator.records:
                records.append({"system": kind, **record})

    if records:
        path = write_csv(settings.paths.results / "answer_details.csv", records)
        step(f"per-question detail: {path}")

    tables = [Table(f"Answer quality (n={len(questions)}, retriever={retrieval_mode})", rows=rows)]
    tables.append(stats_table(client, "LLM usage (answer eval)"))
    if failed := failures.table():
        tables.append(failed)
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Answer quality: baseline vs corrective agent, LLM-judged. Requires GEMINI_API_KEY.",
    )
    parser.add_argument("--systems", type=str, help=f"comma-separated subset of {','.join(SYSTEMS)}")
    parser.add_argument("--retriever", type=str, help="retrieval mode (default hybrid+rerank)")
    parser.add_argument("--decks", type=str, help="comma-separated deck names")
    parser.add_argument("--limit", type=int, help="cap number of questions")
    parser.add_argument("--workers", type=int, default=4, help="concurrent questions")
    args = parser.parse_args()

    bundle = Bundle("PitchLens — answer quality")
    for table in collect_tables(
        csv_list(args.systems), args.retriever, csv_list(args.decks), args.limit, args.workers
    ):
        bundle.add_table(table)
    bundle.render()
    print(f"\nwritten: {bundle.write()}")


if __name__ == "__main__":
    main()
