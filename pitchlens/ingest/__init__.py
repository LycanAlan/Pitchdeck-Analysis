"""Ingestion: pitch-deck PDFs into DeckDocuments."""

from __future__ import annotations

from .analyzer import SECTIONS, SlideAnalyzer
from .extractor import (
    VISION_PROMPT,
    AdaptiveExtractor,
    PageExtractor,
    RouteStats,
    TextLayerExtractor,
    VisionExtractor,
)
from .pipeline import IngestionPipeline

__all__ = [
    "SECTIONS",
    "VISION_PROMPT",
    "AdaptiveExtractor",
    "IngestionPipeline",
    "PageExtractor",
    "RouteStats",
    "SlideAnalyzer",
    "TextLayerExtractor",
    "VisionExtractor",
]
