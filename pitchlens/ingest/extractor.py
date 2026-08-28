"""Per-page text recovery.

This is the module the whole project exists to justify. Of the 18 pitch decks we
measured, 15 (83%) carry no PDF text layer whatsoever — every slide is a flat
image and `get_text()` returns "". A pipeline that assumes either strategy is
wrong on most of the corpus, so the *page* picks the extractor, not the author.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import pymupdf

from ..config import settings
from ..domain import ExtractionMode, Slide
from ..llm import LLMClient

VISION_PROMPT = """You are transcribing one slide from a startup pitch deck.

Transcribe ALL visible text VERBATIM, exactly as written. This includes:
- headings, body copy, captions, footnotes and logos rendered as text
- every number: revenue figures, percentages, dates, multiples, currency symbols
- chart content: axis labels, axis tick values, series names, legends, data labels
- table content: keep rows on their own lines and separate cells with " | "

Rules:
- Do not paraphrase, summarise, translate, correct or reorder anything.
- Do not invent values. If a number is unreadable, write [unreadable].
- Read top-to-bottom, left-to-right, so the reading order matches the layout.
- For a chart with no printed data labels, add one short line describing the
  trend after the transcribed labels, prefixed with "CHART:".
- Output the transcription only. No preamble, no commentary, no markdown fences.
"""


@dataclass
class RouteStats:
    """How many pages took each extraction route.

    Kept as a counter over the enum rather than two named integers so adding an
    extraction mode never touches the accounting or reporting code.
    """

    counts: dict[ExtractionMode, int] = field(
        default_factory=lambda: dict.fromkeys(ExtractionMode, 0)
    )
    unreadable: int = 0

    def record(self, mode: ExtractionMode) -> None:
        self.counts[mode] += 1

    @classmethod
    def from_slides(cls, slides: Iterable[Slide]) -> RouteStats:
        """Recover stats from an already-ingested deck loaded off disk."""
        stats = cls()
        for slide in slides:
            stats.record(slide.mode)
        return stats

    @classmethod
    def merge(cls, parts: Iterable[RouteStats]) -> RouteStats:
        total = cls()
        for part in parts:
            for mode, n in part.counts.items():
                total.counts[mode] += n
            total.unreadable += part.unreadable
        return total

    @property
    def text_layer(self) -> int:
        return self.counts[ExtractionMode.TEXT_LAYER]

    @property
    def vision(self) -> int:
        return self.counts[ExtractionMode.VISION]

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def vision_share(self) -> float:
        return self.vision / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "pages": self.total,
            "text_layer": self.text_layer,
            "vision": self.vision,
            "unreadable": self.unreadable,
            "vision_share": round(self.vision_share, 4),
        }


class PageExtractor(ABC):
    """Turns one PDF page into a Slide."""

    mode: ExtractionMode

    @abstractmethod
    def extract(self, page: pymupdf.Page, page_number: int) -> Slide:
        """Return the slide for `page`, numbered from 1."""


class TextLayerExtractor(PageExtractor):
    """Reads the embedded text layer. Free, instant, and exact when it exists."""

    mode = ExtractionMode.TEXT_LAYER

    def extract(self, page: pymupdf.Page, page_number: int) -> Slide:
        return Slide(
            slide=page_number,
            mode=self.mode,
            transcript=page.get_text().strip(),
        )


class VisionExtractor(PageExtractor):
    """Renders the page and has a vision model transcribe it.

    The fallback for the 83% of decks exported as images. Costs an API call per
    page, which is why AdaptiveExtractor only reaches for it when it must.
    """

    mode = ExtractionMode.VISION

    def __init__(self, llm: LLMClient):
        self._llm = llm

    @staticmethod
    def render(page: pymupdf.Page) -> bytes:
        return page.get_pixmap(dpi=settings.ingest.render_dpi).tobytes("png")

    def transcribe(self, png: bytes, page_number: int) -> Slide:
        """Transcribe already-rendered bytes.

        Split out from `extract` so callers can render on one thread — MuPDF page
        objects are not thread-safe — and then fan the API calls out across a
        pool, which is the difference between a 30-minute and a 5-minute corpus.
        """
        transcript = self._llm.complete(VISION_PROMPT, chain=settings.models.vision, image=png)
        return Slide(slide=page_number, mode=self.mode, transcript=transcript.strip())

    def extract(self, page: pymupdf.Page, page_number: int) -> Slide:
        return self.transcribe(self.render(page), page_number)


class AdaptiveExtractor(PageExtractor):
    """Routes each page to the cheapest extractor that can actually read it.

    The decision is per page, not per deck: decks are routinely mixed, with a
    text-native title slide followed by twenty exported-image slides.
    """

    def __init__(self, llm: LLMClient, vision: bool = True):
        self._text = TextLayerExtractor()
        self._vision = VisionExtractor(llm)
        self._vision_enabled = vision
        self.stats = RouteStats()

    def _uses_text_layer(self, page: pymupdf.Page) -> bool:
        return len(page.get_text().strip()) >= settings.ingest.text_layer_min_chars

    def extract(self, page: pymupdf.Page, page_number: int) -> Slide:
        chosen = self._text if self._uses_text_layer(page) else self._vision
        slide = chosen.extract(page, page_number)
        self.stats.record(slide.mode)
        return slide

    def extract_document(
        self, document: pymupdf.Document, max_workers: int | None = None
    ) -> list[Slide]:
        """Extract a whole deck, transcribing image pages concurrently.

        Rendering stays on the calling thread because MuPDF pages are not
        thread-safe; only the network-bound transcription is parallelised.
        """
        slides: dict[int, Slide] = {}
        pending: list[tuple[int, bytes]] = []

        for number, page in enumerate(document, start=1):
            if self._uses_text_layer(page):
                slides[number] = self._text.extract(page, number)
                self.stats.record(ExtractionMode.TEXT_LAYER)
            elif self._vision_enabled:
                pending.append((number, self._vision.render(page)))
            else:
                # Text-only mode: this is precisely the baseline the coverage
                # comparison measures, so it is a real runnable pipeline rather
                # than a hypothetical — image pages simply yield nothing.
                slides[number] = Slide(slide=number, mode=ExtractionMode.VISION)
                self.stats.unreadable += 1

        if pending:
            workers = max_workers or settings.ingest.max_workers
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(self._vision.transcribe, png, number): number
                    for number, png in pending
                }
                for future in as_completed(futures):
                    number = futures[future]
                    try:
                        slides[number] = future.result()
                        self.stats.record(ExtractionMode.VISION)
                    except Exception:  # noqa: BLE001 - counted, then reported by the caller
                        # One unreadable page must not cost the other fifty. The
                        # slide is kept with an empty transcript so numbering stays
                        # aligned with the PDF; it simply contributes no chunks.
                        slides[number] = Slide(slide=number, mode=ExtractionMode.VISION)
                        self.stats.unreadable += 1

        return [slides[n] for n in sorted(slides)]
