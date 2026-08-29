"""PDF -> DeckDocument.

The one entry point the rest of the system uses to get decks onto disk.
Everything downstream (indexing, retrieval, evaluation) reads the saved JSON and
never touches a PDF again.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

from ..config import settings
from ..domain import DeckDocument
from ..llm import LLMClient
from .analyzer import SlideAnalyzer
from .extractor import AdaptiveExtractor, RouteStats


class IngestionPipeline:
    """Extracts, analyses and persists decks, recording how each page was read."""

    def __init__(self, llm: LLMClient):
        self._llm = llm
        self._analyzer = SlideAnalyzer(llm)
        self.route_stats: dict[str, RouteStats] = {}

    def ingest(self, pdf_path: Path, analyze: bool = True, vision: bool = True) -> DeckDocument:
        # A fresh extractor per deck keeps its counter scoped to this deck.
        extractor = AdaptiveExtractor(self._llm, vision=vision)
        with pymupdf.open(pdf_path) as pdf:
            slides = extractor.extract_document(pdf)

        if analyze:
            slides = self._analyzer.analyze_many(slides)

        deck = DeckDocument(name=pdf_path.stem, slides=slides)
        self.route_stats[deck.name] = extractor.stats
        return deck

    def ingest_all(
        self,
        pdf_dir: Path | None = None,
        out_dir: Path | None = None,
        analyze: bool = True,
        vision: bool = True,
    ) -> list[DeckDocument]:
        """Ingest a directory of decks, reusing anything already on disk.

        Vision transcription is the dominant cost of the whole project — one
        billed API call per slide — so an already-ingested deck is loaded rather
        than re-read. That makes re-running the pipeline after a crash, a new
        deck drop, or a retrieval change effectively free.
        """
        pdf_dir = pdf_dir or settings.paths.decks
        out_dir = out_dir or settings.paths.documents
        out_dir.mkdir(parents=True, exist_ok=True)

        decks: list[DeckDocument] = []
        for pdf_path in sorted(pdf_dir.glob("*.pdf")):
            out_path = out_dir / f"{pdf_path.stem}.json"
            if out_path.exists():
                deck = DeckDocument.load(out_path)
                self.route_stats[deck.name] = RouteStats.from_slides(deck.slides)
            else:
                deck = self.ingest(pdf_path, analyze=analyze, vision=vision)
                # A deck that yielded no chunks read nothing at all — every page
                # failed, almost always because the vision quota ran out. Saving
                # it would be worse than failing: the existence check above would
                # skip it on every future run, so the deck could never recover.
                if not deck.chunks():
                    raise RuntimeError(
                        f"{pdf_path.stem}: every page failed to extract; not saved so a "
                        f"later run can retry it"
                    )
                deck.save(out_path)
            decks.append(deck)
        return decks

    @property
    def corpus_stats(self) -> RouteStats:
        """Text-layer vs vision coverage across every deck seen this run."""
        return RouteStats.merge(self.route_stats.values())
