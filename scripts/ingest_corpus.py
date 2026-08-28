"""Ingest every deck PDF into a persisted DeckDocument.

    python scripts/ingest_corpus.py --limit 3

Needs GEMINI_API_KEY: image-only pages are transcribed by the vision model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import (  # noqa: E402
    DeckDocument,
    Failures,
    Stopwatch,
    Table,
    banner,
    csv_list,
    deck_pdfs,
    gemini_client,
    settings,
    stats_table,
    step,
)


def _pipeline(client):
    from pitchlens.ingest.pipeline import IngestionPipeline

    return IngestionPipeline(client)


def collect_tables(
    deck_dir: Path | None = None,
    out_dir: Path | None = None,
    names: list[str] | None = None,
    limit: int | None = None,
    analyze: bool = True,
    skip_existing: bool = False,
    vision: bool = True,
) -> list[Table]:
    out_dir = out_dir or settings.paths.documents
    out_dir.mkdir(parents=True, exist_ok=True)
    pdfs = deck_pdfs(deck_dir, names, limit)

    client = gemini_client()
    pipeline = _pipeline(client)
    failures = Failures("ingestion")
    rows: list[tuple[DeckDocument, float, bool]] = []

    banner(f"Ingesting {len(pdfs)} decks -> {out_dir} (analyze={analyze})")
    for position, pdf in enumerate(pdfs, start=1):
        target = out_dir / f"{pdf.stem}.json"
        if skip_existing and target.exists():
            step(f"[{position}/{len(pdfs)}] {pdf.stem}: cached")
            rows.append((DeckDocument.load(target), 0.0, True))
            continue

        # One deck at a time rather than pipeline.ingest_all so a single
        # malformed PDF is reported and the remaining decks still finish --
        # these runs cost real API calls and are slow to repeat.
        step(f"[{position}/{len(pdfs)}] {pdf.stem}: ingesting")
        with failures.guard(pdf.stem), Stopwatch() as watch:
            document = pipeline.ingest(pdf, analyze=analyze, vision=vision)
            document.save(target)
            rows.append((document, watch.seconds, False))
            step(
                f"    {len(document.slides)} slides, {document.vision_slides} via vision, "
                f"{len(document.chunks())} chunks in {watch.seconds:.1f}s"
            )

    per_deck = Table(
        "Ingestion routes", ["deck", "slides", "text_layer", "vision", "chunks", "seconds", "cached"]
    )
    for document, seconds, cached in rows:
        vision = document.vision_slides
        per_deck.add(
            document.name,
            len(document.slides),
            len(document.slides) - vision,
            vision,
            len(document.chunks()),
            round(seconds, 1),
            cached,
        )

    slides = sum(len(d.slides) for d, _, _ in rows)
    vision = sum(d.vision_slides for d, _, _ in rows)
    per_deck.note(f"decks ingested: {len(rows)}/{len(pdfs)}")
    per_deck.note(f"slides: {slides:,} total, {vision:,} routed to vision ({vision / max(slides, 1):.1%})")
    per_deck.note(f"chunks written: {sum(len(d.chunks()) for d, _, _ in rows):,}")

    failures.report()
    return [t for t in (per_deck, stats_table(client, "LLM usage (ingestion)"), failures.table()) if t]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the adaptive ingestion pipeline over every deck PDF and persist one "
        "DeckDocument JSON per deck. Requires GEMINI_API_KEY.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--deck-dir", type=Path, default=settings.paths.decks, help="directory of PDFs")
    parser.add_argument("--out", type=Path, default=settings.paths.documents, help="output directory")
    parser.add_argument("--decks", type=str, default=None, help="comma-separated subset of deck names")
    parser.add_argument("--limit", type=int, default=None, help="only ingest the first N decks")
    parser.add_argument(
        "--no-analyze", action="store_true", help="transcribe only; skip the slide analysis pass"
    )
    parser.add_argument(
        "--skip-existing", action="store_true", help="reuse decks already ingested into --out"
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="disable vision transcription; image pages yield nothing (the baseline extractor)",
    )
    args = parser.parse_args()

    tables = collect_tables(
        args.deck_dir,
        args.out,
        csv_list(args.decks),
        args.limit,
        analyze=not args.no_analyze,
        skip_existing=args.skip_existing,
        vision=not args.text_only,
    )
    for table in tables:
        table.show()


if __name__ == "__main__":
    main()
