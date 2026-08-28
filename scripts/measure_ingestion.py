"""Measure how much of the corpus a text-only extractor can actually read.

This is the headline number: PDF text layers are missing or near-empty on most
real pitch decks, so a text-only pipeline silently loses whole decks. Adaptive
extraction (text layer where it exists, vision transcription where it does not)
covers every deck by construction; the ratio between the two is the coverage
multiple reported here.

    python scripts/measure_ingestion.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz  # noqa: E402  (pymupdf)

from _common import Failures, Table, banner, csv_list, deck_pdfs, settings, step  # noqa: E402


@dataclass
class DeckProfile:
    """Text-layer profile of one PDF, measured without any model call."""

    name: str
    pages: int
    chars: int
    page_chars: list[int] = field(default_factory=list)

    @property
    def chars_per_page(self) -> float:
        return self.chars / self.pages if self.pages else 0.0

    def text_pages(self, min_chars: int) -> int:
        return sum(1 for c in self.page_chars if c >= min_chars)

    def page_coverage(self, min_chars: int) -> float:
        return self.text_pages(min_chars) / self.pages if self.pages else 0.0

    def is_text_layer(self, min_chars: int, min_coverage: float) -> bool:
        """Usable by a text-only extractor: enough pages clear the threshold."""
        return self.page_coverage(min_chars) >= min_coverage

    @classmethod
    def read(cls, pdf: Path) -> "DeckProfile":
        with fitz.open(pdf) as doc:
            page_chars = [len(page.get_text("text").strip()) for page in doc]
        return cls(pdf.stem, len(page_chars), sum(page_chars), page_chars)


def collect_tables(
    deck_dir: Path | None = None,
    names: list[str] | None = None,
    limit: int | None = None,
    min_coverage: float = 0.8,
) -> list[Table]:
    min_chars = settings.ingest.text_layer_min_chars
    pdfs = deck_pdfs(deck_dir, names, limit)

    banner(f"Text-layer audit of {len(pdfs)} decks (threshold {min_chars} chars/page)")
    failures = Failures("ingestion audit")
    profiles: list[DeckProfile] = []

    for pdf in pdfs:
        with failures.guard(pdf.stem):
            profile = DeckProfile.read(pdf)
            profiles.append(profile)
            verdict = "text_layer" if profile.is_text_layer(min_chars, min_coverage) else "image_only"
            step(
                f"{profile.name:<20} {profile.pages:>3} pages  "
                f"{profile.chars_per_page:>8.0f} chars/page  {verdict}"
            )

    per_deck = Table(
        "Per-deck text layer",
        ["deck", "pages", "text_chars", "chars_per_page", "text_pages", "page_coverage", "class"],
    )
    for p in sorted(profiles, key=lambda x: x.chars_per_page):
        per_deck.add(
            p.name,
            p.pages,
            p.chars,
            round(p.chars_per_page, 1),
            p.text_pages(min_chars),
            round(p.page_coverage(min_chars), 3),
            "text_layer" if p.is_text_layer(min_chars, min_coverage) else "image_only",
        )

    total_decks = len(profiles)
    text_decks = sum(1 for p in profiles if p.is_text_layer(min_chars, min_coverage))
    total_pages = sum(p.pages for p in profiles)
    text_pages = sum(p.text_pages(min_chars) for p in profiles)

    summary = Table("Extraction coverage", ["extractor", "decks_usable", "pct_decks", "pages_readable"])
    summary.add("text-only", f"{text_decks}/{total_decks}", round(100 * text_decks / total_decks, 1), text_pages)
    summary.add("adaptive (text + vision)", f"{total_decks}/{total_decks}", 100.0, total_pages)

    deck_multiple = total_decks / text_decks if text_decks else float("inf")
    page_multiple = total_pages / text_pages if text_pages else float("inf")
    summary.note(f"deck coverage multiple: {deck_multiple:.2f}x ({text_decks} -> {total_decks} decks)")
    summary.note(f"page coverage multiple: {page_multiple:.2f}x ({text_pages:,} -> {total_pages:,} pages)")
    summary.note(
        f"a deck counts as text-only usable when >={min_coverage:.0%} of its pages carry "
        f">={min_chars} characters (settings.ingest.text_layer_min_chars)"
    )
    summary.note("measured with pymupdf only -- no API key, no model calls")

    failures.report()
    return [t for t in (summary, per_deck, failures.table()) if t is not None]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the PDF text layer of every deck and report how many are usable by a "
        "TEXT-ONLY extractor versus by ADAPTIVE (text + vision) extraction, plus the coverage "
        "multiple between them. Runs entirely offline with pymupdf: NO API KEY IS REQUIRED and no "
        "model is ever called.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--deck-dir", type=Path, default=settings.paths.decks, help="directory of PDFs")
    parser.add_argument("--decks", type=str, default=None, help="comma-separated subset of deck names")
    parser.add_argument("--limit", type=int, default=None, help="only audit the first N decks")
    parser.add_argument(
        "--min-page-coverage",
        type=float,
        default=0.8,
        help="fraction of pages that must clear the char threshold for a deck to count as text_layer",
    )
    args = parser.parse_args()

    for table in collect_tables(args.deck_dir, csv_list(args.decks), args.limit, args.min_page_coverage):
        table.show()


if __name__ == "__main__":
    main()
