"""The single gateway for every model call in the system.

Before this existed the project instantiated Gemini in three places with three
different model names and no shared retry policy. Everything now goes through
GeminiClient, so model routing, fallback, JSON coercion and call accounting are
written once.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from google import genai
from google.genai import types

from .config import settings

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)

# A retired model never comes back; neither does a rejected key.
_TERMINAL = ("NOT_FOUND", "404", "PERMISSION_DENIED", "API_KEY_INVALID")

_RETRY_DELAY = re.compile(r"retry in ([\d.]+)s")


def _is_terminal(exc: Exception) -> bool:
    """Whether this model should be dropped from the chain for the rest of the run.

    A 429 is not automatically terminal, and treating it as such was costing far
    more throughput than the quota itself: the free tier limits requests *per
    minute* as well as per day, and a burst of concurrent calls trips the minute
    limit immediately. Reading every 429 as "this model is finished" burned
    through the whole chain in seconds over a limit that clears in under one.
    Only a per-day exhaustion is genuinely terminal for the run.
    """
    text = str(exc)
    if any(marker in text for marker in _TERMINAL):
        return True
    return "RESOURCE_EXHAUSTED" in text and "PerDay" in text


def _retry_after(exc: Exception) -> float | None:
    """Seconds the API asked us to wait, when it said."""
    match = _RETRY_DELAY.search(str(exc))
    return float(match.group(1)) if match else None


@dataclass
class CallStats:
    """Call accounting, used by the measurement scripts to report cost."""

    calls: int = 0
    failures: int = 0
    fallbacks: int = 0
    seconds: float = 0.0
    by_model: dict[str, int] = field(default_factory=dict)

    def record(self, model: str, elapsed: float) -> None:
        self.calls += 1
        self.seconds += elapsed
        self.by_model[model] = self.by_model.get(model, 0) + 1


class LLMClient(ABC):
    """Interface the rest of the codebase depends on."""

    @abstractmethod
    def complete(self, prompt: str, *, chain: tuple[str, ...], image: bytes | None = None) -> str:
        """Return raw text for a prompt, optionally grounded on a page image."""

    def complete_json(self, prompt: str, *, chain: tuple[str, ...], image: bytes | None = None):
        """Return parsed JSON. Models fence their JSON often enough to strip it here."""
        raw = self.complete(prompt, chain=chain, image=image)
        match = _JSON_FENCE.search(raw)
        if match:
            raw = match.group(1)
        start = min((i for i in (raw.find("{"), raw.find("[")) if i != -1), default=0)
        return json.loads(raw[start:])


class GeminiClient(LLMClient):
    """Gemini-backed client with per-model fallback.

    The public endpoint returns 503 (overloaded) and 404 (model retired) without
    warning, so a single hardcoded model name makes the whole pipeline brittle.
    `complete` walks the supplied chain and only raises if every model fails.
    """

    def __init__(self, api_key: str | None = None, attempts_per_model: int = 3):
        self._client = genai.Client(api_key=api_key or settings.api_key)
        self._attempts = attempts_per_model
        self._exhausted: set[str] = set()
        self.stats = CallStats()

    @property
    def exhausted(self) -> set[str]:
        """Models that hit their daily quota or were retired during this run."""
        return set(self._exhausted)

    def complete(self, prompt: str, *, chain: tuple[str, ...], image: bytes | None = None) -> str:
        parts = [prompt]
        if image is not None:
            parts.insert(0, types.Part.from_bytes(data=image, mime_type="image/png"))

        last: Exception | None = None
        for position, model in enumerate(chain):
            if model in self._exhausted:
                continue
            for attempt in range(self._attempts):
                started = time.perf_counter()
                try:
                    response = self._client.models.generate_content(model=model, contents=parts)
                    self.stats.record(model, time.perf_counter() - started)
                    if position:
                        self.stats.fallbacks += 1
                    return (response.text or "").strip()
                except Exception as exc:  # noqa: BLE001 - any failure moves us along the chain
                    last = exc
                    self.stats.failures += 1
                    if _is_terminal(exc):
                        self._exhausted.add(model)
                        break
                    if attempt + 1 < self._attempts:
                        # Honour the server's own backoff when it gives one: a
                        # per-minute limit clears on a schedule the API knows and
                        # we do not.
                        wait = _retry_after(exc)
                        time.sleep(min(wait, 30) if wait else 2**attempt)
        raise RuntimeError(f"all models failed: {chain}") from last


class NullClient(LLMClient):
    """Stand-in for offline runs. Retrieval measurement needs no API at all."""

    def complete(self, prompt: str, *, chain: tuple[str, ...], image: bytes | None = None) -> str:
        raise RuntimeError("NullClient cannot generate; this path requires an API key")
