"""Regenerate every number in the README in one pass.

Offline by default (ingestion coverage, retrieval ablation, latency). Add
--with-llm to also run the judged answer-quality comparison, which costs API
calls.

    python scripts/run_all_measurements.py
    python scripts/run_all_measurements.py --with-llm
"""

from __future__ import annotations

import argparse

from _common import Bundle, Failures, banner

import measure_ingestion
import measure_latency
import measure_retrieval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full measurement suite into results/RESULTS.md")
    parser.add_argument("--with-llm", action="store_true", help="also run judged answer quality (uses API)")
    parser.add_argument("--limit", type=int, help="cap questions per measurement")
    args = parser.parse_args()

    bundle = Bundle("PitchLens — measurement suite")
    failures = Failures("suite")

    stages = [
        ("ingestion coverage", lambda: measure_ingestion.collect_tables()),
        ("retrieval ablation", lambda: measure_retrieval.collect_tables(limit=args.limit)),
        ("latency", lambda: measure_latency.collect_tables(limit=args.limit)),
    ]
    if args.with_llm:
        import measure_answers

        stages.append(("answer quality", lambda: measure_answers.collect_tables(limit=args.limit)))

    for label, run in stages:
        with failures.guard(label):
            for table in run():
                bundle.add_table(table)

    if failed := failures.table():
        bundle.add_table(failed)

    banner("Suite complete")
    bundle.render()
    print(f"\nwritten: {bundle.write()}")


if __name__ == "__main__":
    main()
