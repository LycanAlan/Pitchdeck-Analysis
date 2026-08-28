"""Download the 18-deck evaluation corpus from HuggingFace.

    python scripts/fetch_corpus.py
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import Failures, Stopwatch, Table, banner, csv_list, settings, step  # noqa: E402

DATASET = "skyforclouds/pitch-deckz"
BASE_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/main/files"

DECKS = (
    "Brex",
    "Dwolla",
    "Monzo",
    "Nium_replica",
    "Oscar_Health",
    "Revolut",
    "Transferwise",
    "airbnb",
    "coinbase",
    "dropbox",
    "facebook",
    "frontapp",
    "linkedin_series_b",
    "mixpanel",
    "square",
    "uber",
    "wework_2021",
    "shopify_2025",
)

# The plain urllib default User-Agent gets 403'd by the HuggingFace CDN.
HEADERS = {"User-Agent": "pitchlens-corpus-fetcher/1.0"}


class DeckFetcher:
    """Fetches one deck, skipping work already on disk."""

    def __init__(self, out_dir: Path, force: bool = False):
        self.out_dir = out_dir
        self.force = force

    def fetch(self, name: str) -> tuple[str, str, int]:
        """Return (name, status, bytes). Status is 'downloaded' or 'cached'."""
        target = self.out_dir / f"{name}.pdf"
        if target.exists() and not self.force:
            return name, "cached", target.stat().st_size

        # Download to a sidecar first: a half-written .pdf would look "cached"
        # on the next run and silently poison every downstream measurement.
        partial = target.with_suffix(".pdf.part")
        request = urllib.request.Request(f"{BASE_URL}/{name}.pdf", headers=HEADERS)
        with urllib.request.urlopen(request, timeout=120) as response:
            partial.write_bytes(response.read())
        partial.replace(target)
        return name, "downloaded", target.stat().st_size


def collect_tables(
    out_dir: Path | None = None,
    names: list[str] | None = None,
    workers: int = 6,
    force: bool = False,
) -> list[Table]:
    out_dir = out_dir or settings.paths.decks
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = names or list(DECKS)

    banner(f"Fetching {len(wanted)} decks -> {out_dir}")
    fetcher = DeckFetcher(out_dir, force)
    failures = Failures("fetch")
    results: list[tuple[str, str, int]] = []

    with Stopwatch() as watch, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetcher.fetch, name): name for name in wanted}
        for future in futures:
            name = futures[future]
            with failures.guard(name):
                result = future.result()
                results.append(result)
                step(f"[{result[1]:>10}] {name}.pdf  {result[2] / 1_000_000:.2f} MB")

    results.sort()
    table = Table("Corpus", ["deck", "status", "size_mb"])
    for name, status, size in results:
        table.add(name, status, round(size / 1_000_000, 2))

    downloaded = sum(1 for _, s, _ in results if s == "downloaded")
    total_mb = sum(size for _, _, size in results) / 1_000_000
    table.note(f"dataset: {DATASET}")
    table.note(f"decks on disk: {len(results)}/{len(wanted)} ({downloaded} newly downloaded)")
    table.note(f"total size: {total_mb:.1f} MB in {watch.seconds:.1f}s")

    failures.report()
    return [t for t in (table, failures.table()) if t is not None]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Download the {len(DECKS)}-deck pitch corpus from the HuggingFace "
        f"dataset {DATASET}. Files already present are skipped.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", type=Path, default=settings.paths.decks, help="destination directory")
    parser.add_argument("--decks", type=str, default=None, help="comma-separated subset of deck names")
    parser.add_argument("--workers", type=int, default=6, help="parallel downloads")
    parser.add_argument("--force", action="store_true", help="re-download decks already on disk")
    args = parser.parse_args()

    for table in collect_tables(args.out, csv_list(args.decks), args.workers, args.force):
        table.show()


if __name__ == "__main__":
    main()
