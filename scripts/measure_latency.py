"""Index persistence and query latency.

Produces the cold-build vs warm-load number (the payoff for persisting the FAISS
index instead of rebuilding it every process start) and per-mode query latency.

Runs entirely OFFLINE.

    python scripts/measure_latency.py
"""

from __future__ import annotations

import argparse
import shutil

from _common import (
    INDEX_DIR,
    Bundle,
    Stopwatch,
    Table,
    banner,
    chunks_of,
    csv_list,
    load_documents,
    load_eval_set,
    open_index,
    percentile,
    retriever_factory,
    settings,
    step,
)


def _persistence_row(documents) -> list[dict]:
    """Time a from-scratch build against a load of the persisted index."""
    from pitchlens.index.embedder import get_embedder
    from pitchlens.index.store import VectorIndex

    chunks = chunks_of(documents)
    embedder = get_embedder()  # warm the model first so we time indexing, not model load
    embedder.encode_one("warmup")

    scratch = INDEX_DIR.parent / "_latency_probe"
    shutil.rmtree(scratch, ignore_errors=True)

    with Stopwatch() as cold:
        index = VectorIndex.build(chunks, embedder)
        index.save(scratch)

    with Stopwatch() as warm:
        VectorIndex.load(scratch, embedder)

    shutil.rmtree(scratch, ignore_errors=True)
    return [
        {
            "chunks": len(chunks),
            "cold_build_s": round(cold.seconds, 2),
            "warm_load_s": round(warm.seconds, 3),
            "speedup": f"{cold.seconds / max(warm.seconds, 1e-6):.0f}x",
        }
    ]


def _latency_rows(documents, questions, modes) -> list[dict]:
    factory = retriever_factory(open_index(documents))
    k = settings.retrieval.final_k
    rows = []
    for mode in modes or list(factory.MODES):
        retriever = factory.build(mode)
        retriever.retrieve(questions[0].question, k)  # exclude lazy model load from the sample
        samples = []
        for q in questions:
            with Stopwatch() as watch:
                retriever.retrieve(q.question, k)
            samples.append(watch.seconds * 1000)
        rows.append(
            {
                "retriever": mode,
                "p50_ms": round(percentile(samples, 50), 1),
                "p95_ms": round(percentile(samples, 95), 1),
                "queries": len(samples),
            }
        )
        step(f"{mode}: p50 {rows[-1]['p50_ms']}ms  p95 {rows[-1]['p95_ms']}ms")
    return rows


def collect_tables(modes=None, limit=None) -> list[Table]:
    banner("Latency and index persistence")
    documents = load_documents()
    questions = load_eval_set(limit=limit)
    step(f"{len(documents)} decks, {len(questions)} queries")

    return [
        Table("Index persistence (cold build vs warm load)", rows=_persistence_row(documents)),
        Table("Query latency by retrieval mode", rows=_latency_rows(documents, questions, modes)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index persistence and query latency. Runs fully offline; no API key needed.",
    )
    parser.add_argument("--modes", type=str, help="comma-separated subset of retrieval modes")
    parser.add_argument("--limit", type=int, help="cap number of queries")
    args = parser.parse_args()

    bundle = Bundle("PitchLens — latency")
    for table in collect_tables(csv_list(args.modes), args.limit):
        bundle.add_table(table)
    bundle.render()
    print(f"\nwritten: {bundle.write()}")


if __name__ == "__main__":
    main()
