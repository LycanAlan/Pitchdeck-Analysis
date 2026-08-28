"""The retrieval ablation: every strategy, same questions, same metrics.

Runs entirely OFFLINE. Embeddings, BM25 and the cross-encoder are all local
models, so this needs no GEMINI_API_KEY and costs nothing to re-run.

    python scripts/measure_retrieval.py
    python scripts/measure_retrieval.py --modes dense,hybrid+rerank
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import (
    Bundle,
    Failures,
    Table,
    banner,
    csv_list,
    load_documents,
    load_eval_set,
    open_index,
    retriever_factory,
    settings,
    step,
)


def collect_tables(modes=None, decks=None, limit=None, k_values=None) -> list[Table]:
    questions = load_eval_set(decks=decks, limit=limit)
    banner("Retrieval ablation")
    step(f"{len(questions)} questions over {len({q.deck for q in questions})} decks")

    documents = load_documents()
    index = open_index(documents)
    step(f"index: {len(index)} chunks from {len(documents)} decks")

    factory = retriever_factory(index)
    selected = modes or list(factory.MODES)
    ks = tuple(k_values or settings.retrieval.eval_k_values)

    from pitchlens.evaluation.harness import RetrievalEvaluator

    evaluator = RetrievalEvaluator(questions)
    failures = Failures("retrieval")
    rows = []
    for mode in selected:
        with failures.guard(mode):
            step(f"evaluating {mode} ...")
            rows.append(evaluator.evaluate(factory.build(mode), k_values=ks))

    tables = [Table(f"Retrieval ablation (n={len(questions)} questions)", rows=rows)]
    if failed := failures.table():
        tables.append(failed)
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieval ablation across strategies. Runs fully offline; no API key needed.",
    )
    parser.add_argument("--modes", type=str, help="comma-separated subset of retrieval modes")
    parser.add_argument("--decks", type=str, help="comma-separated deck names to restrict to")
    parser.add_argument("--limit", type=int, help="cap number of questions")
    parser.add_argument("--k", type=str, help="comma-separated k values, e.g. 1,3,5,10")
    args = parser.parse_args()

    ks = [int(v) for v in csv_list(args.k)] if args.k else None
    tables = collect_tables(csv_list(args.modes), csv_list(args.decks), args.limit, ks)

    bundle = Bundle("PitchLens — retrieval ablation")
    for table in tables:
        bundle.add_table(table)
    bundle.render()
    print(f"\nwritten: {bundle.write()}")


if __name__ == "__main__":
    main()
