"""Helpers for per-call LLM usage accounting and run-level cost tracking."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("skydiscover.llm")

# ---------------------------------------------------------------------------
# Default model pricing table  (USD per 1M tokens)
# Format: model_name -> (input_price, output_price)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    # OpenAI GPT-5.x family
    "gpt-5.4": (2.5, 15.0),
    "gpt-5.4-mini": (0.75, 4.5),
    "gpt-5.4-nano": (0.2, 1.25),
    "gpt-5.4-pro": (30.0, 180.0),
    "gpt-5.2": (1.75, 14.0),
    "gpt-5.2-pro": (21.0, 168.0),
    "gpt-5.1": (1.25, 10.0),
    "gpt-5": (1.25, 10.0),
    "gpt-5-mini": (0.25, 2.0),
    "gpt-5-nano": (0.05, 0.4),
    "gpt-5-pro": (15.0, 120.0),
    # OpenAI GPT-4.x family
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1-nano": (0.1, 0.4),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-2024-05-13": (5.0, 15.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4-turbo-2024-04-09": (10.0, 30.0),
    "gpt-4-0125-preview": (10.0, 30.0),
    "gpt-4-1106-preview": (10.0, 30.0),
    "gpt-4-1106-vision-preview": (10.0, 30.0),
    "gpt-4-0613": (30.0, 60.0),
    "gpt-4-0314": (30.0, 60.0),
    "gpt-4-32k": (60.0, 120.0),
    # OpenAI GPT-3.5 family
    "gpt-3.5-turbo": (0.5, 1.5),
    "gpt-3.5-turbo-0125": (0.5, 1.5),
    "gpt-3.5-turbo-1106": (1.0, 2.0),
    "gpt-3.5-turbo-0613": (1.5, 2.0),
    "gpt-3.5-turbo-16k-0613": (3.0, 4.0),
    # OpenAI o-series reasoning models
    "o1": (15.0, 60.0),
    "o1-pro": (150.0, 600.0),
    "o1-mini": (1.1, 4.4),
    "o3": (2.0, 8.0),
    "o3-pro": (20.0, 80.0),
    "o3-mini": (1.1, 4.4),
    "o4-mini": (1.1, 4.4),
    # OpenAI specialized
    "o3-deep-research": (10.0, 40.0),
    "o4-mini-deep-research": (2.0, 8.0),
    "computer-use-preview": (3.0, 12.0),
    # Google Gemini models (also accessible via OpenAI-compatible APIs)
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.15, 0.6),
    "gemini-3-pro": (2.0, 12.0),
    "gemini-3.1-pro": (2.0, 12.0),
    "gemini-3.1-flash": (0.25, 1.5),
    "gemini/gemini-2.5-pro": (1.25, 10.0),
    "gemini/gemini-2.5-flash": (0.15, 0.6),
    "gemini/gemini-3-pro": (2.0, 12.0),
    "gemini/gemini-3.1-pro": (2.0, 12.0),
    "gemini/gemini-3.1-flash": (0.25, 1.5),
    "gemini/gemini-3-pro-preview": (2.0, 12.0),
    # Anthropic Claude models (accessible via OpenAI-compatible APIs)
    "claude-opus-4-6": (5.0, 25.0),
    "claude-opus-4-5": (5.0, 25.0),
    "claude-opus-4-1": (15.0, 75.0),
    "claude-opus-4": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def lookup_default_pricing(
    model_name: str,
) -> Tuple[Optional[float], Optional[float]]:
    """Return (input_price, output_price) per 1M tokens for a known model.

    Tries exact match first, then prefix match (longest prefix wins) so that
    dated snapshot names like ``gpt-5-20260301`` resolve to the base model.
    Returns (None, None) if no match is found.
    """
    if not model_name:
        return None, None

    name = model_name.strip().lower()

    # Exact match
    if name in DEFAULT_MODEL_PRICING:
        return DEFAULT_MODEL_PRICING[name]

    # Prefix match – longest prefix wins
    best_key: Optional[str] = None
    for key in DEFAULT_MODEL_PRICING:
        if name.startswith(key) and (best_key is None or len(key) > len(best_key)):
            best_key = key
    if best_key is not None:
        return DEFAULT_MODEL_PRICING[best_key]

    return None, None


def _safe_int(value: Any) -> int:
    """Convert usage counters to non-negative ints."""
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(parsed, 0)


def extract_usage_counts(usage: Any) -> tuple[int, int, int]:
    """Extract input/output/total token counts from provider usage objects."""
    if usage is None:
        return 0, 0, 0

    input_tokens = _safe_int(
        getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
    )
    output_tokens = _safe_int(
        getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
    )
    total_tokens = _safe_int(getattr(usage, "total_tokens", None))
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens

    # Gemini thinking models report thinking tokens only in total_tokens,
    # not in completion_tokens.  Attribute the gap to output (billed at
    # output rate) so cost calculations are correct.
    thinking_gap = total_tokens - (input_tokens + output_tokens)
    if thinking_gap > 0:
        output_tokens += thinking_gap

    return input_tokens, output_tokens, total_tokens


def compute_token_costs(
    input_tokens: int,
    output_tokens: int,
    input_price_per_million_tokens: Optional[float],
    output_price_per_million_tokens: Optional[float],
) -> tuple[float, float, float]:
    """Compute USD cost for input/output token usage."""
    input_cost = 0.0
    output_cost = 0.0

    if input_price_per_million_tokens is not None:
        input_cost = float(input_tokens) * float(input_price_per_million_tokens) / 1_000_000.0
    if output_price_per_million_tokens is not None:
        output_cost = float(output_tokens) * float(output_price_per_million_tokens) / 1_000_000.0

    return input_cost, output_cost, input_cost + output_cost


@dataclass
class _UsageBucket:
    """Aggregate counters for one usage category."""

    call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    total_cost_usd: float = 0.0

    def add(self, response: Any) -> None:
        self.call_count += 1
        self.input_tokens += _safe_int(getattr(response, "input_tokens", 0))
        self.output_tokens += _safe_int(getattr(response, "output_tokens", 0))
        self.total_tokens += _safe_int(getattr(response, "total_tokens", 0))
        self.input_cost_usd += float(getattr(response, "input_cost_usd", 0.0) or 0.0)
        self.output_cost_usd += float(getattr(response, "output_cost_usd", 0.0) or 0.0)
        self.total_cost_usd += float(getattr(response, "total_cost_usd", 0.0) or 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_count": self.call_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "input_cost_usd": self.input_cost_usd,
            "output_cost_usd": self.output_cost_usd,
            "total_cost_usd": self.total_cost_usd,
        }


@dataclass
class CostTracker:
    """Thread-safe run-level tracker for LLM usage across multiple pools."""

    buckets: Dict[str, _UsageBucket] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def record(self, response: Any, usage_category: str) -> None:
        """Record one LLM response under the requested category."""
        if response is None:
            return
        category = usage_category or "unknown"
        with self._lock:
            bucket = self.buckets.setdefault(category, _UsageBucket())
            bucket.add(response)

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serializable usage summary."""
        with self._lock:
            by_category = {name: bucket.to_dict() for name, bucket in self.buckets.items()}

        total = _UsageBucket()
        for bucket_dict in by_category.values():
            total.call_count += bucket_dict["call_count"]
            total.input_tokens += bucket_dict["input_tokens"]
            total.output_tokens += bucket_dict["output_tokens"]
            total.total_tokens += bucket_dict["total_tokens"]
            total.input_cost_usd += bucket_dict["input_cost_usd"]
            total.output_cost_usd += bucket_dict["output_cost_usd"]
            total.total_cost_usd += bucket_dict["total_cost_usd"]

        return {
            "total": total.to_dict(),
            "by_category": by_category,
        }
