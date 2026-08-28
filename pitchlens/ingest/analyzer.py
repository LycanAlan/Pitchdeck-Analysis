"""Slide-level semantic enrichment.

A verbatim transcript retrieves badly on its own: a slide headed "0 -> 1.2M ARR
in 14 months" contains none of the words a user types ("traction", "growth",
"revenue"). Attaching a section label and a written summary gives the indexer a
second, vocabulary-rich chunk per slide to match against.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from ..config import settings
from ..domain import Slide
from ..llm import LLMClient

SECTIONS: tuple[str, ...] = (
    "Title",
    "Problem",
    "Solution",
    "Product",
    "Market",
    "Business Model",
    "Traction",
    "Competition",
    "Team",
    "Financials",
    "Funding Ask",
    "Other",
)

_ANALYSIS_PROMPT = """You are analysing one slide from a startup pitch deck.

Slide {number} transcript:
---
{transcript}
---

Classify and summarise it. Reply with JSON only, no markdown fence:
{{
  "section": one of {sections},
  "summary": "1-3 sentences describing what this slide claims, in the vocabulary
              an investor would use to search for it",
  "key_points": ["up to 5 short factual points"]
}}

Rules:
- "section" must be copied exactly from the list. Use "Other" if none fit.
- Reproduce every figure exactly as it appears; never round or estimate.
- Base everything on the transcript alone. Do not add outside knowledge.
"""


class SlideAnalyzer:
    """Fills in section, summary and key_points on extracted slides."""

    def __init__(self, llm: LLMClient):
        self._llm = llm

    def analyze(self, slide: Slide) -> Slide:
        if not slide.transcript.strip():
            return slide  # a blank slide has nothing to classify; skip the paid call

        data = self._llm.complete_json(
            _ANALYSIS_PROMPT.format(
                number=slide.slide,
                transcript=slide.transcript,
                sections=list(SECTIONS),
            ),
            chain=settings.models.analysis,
        )

        section = data.get("section", "")
        # Section labels drive section-filtered retrieval downstream, so an
        # off-taxonomy invention from the model has to collapse to "Other"
        # rather than silently widening the vocabulary.
        slide.section = section if section in SECTIONS else "Other"
        slide.summary = data.get("summary", "")
        slide.key_points = list(data.get("key_points", []))
        return slide

    def analyze_many(self, slides: list[Slide], max_workers: int | None = None) -> list[Slide]:
        """Analyse slides concurrently, preserving input order.

        Each slide is an independent network round-trip, so the wall-clock cost
        of a deck is otherwise the sum of ~20 sequential API calls.
        """
        workers = max_workers or settings.ingest.max_workers
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self.analyze, slides))
