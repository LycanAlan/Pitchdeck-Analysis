"""Answer generation: prompts, the shared pipeline, and the baseline answerer."""

from __future__ import annotations

from .answerer import Answerer
from .base import RAGPipeline
from .prompts import (
    ANSWER_PROMPT,
    CANNOT_ANSWER,
    GRADE_PROMPT,
    REWRITE_PROMPT,
    format_context,
)

__all__ = [
    "ANSWER_PROMPT",
    "CANNOT_ANSWER",
    "GRADE_PROMPT",
    "REWRITE_PROMPT",
    "Answerer",
    "RAGPipeline",
    "format_context",
]
