"""
Utilities for metric scoring and formatting.
"""

import math
from typing import Any, Dict, List, Optional


def is_numeric_metric(value: Any) -> bool:
    """Return True for real numeric values, excluding bools.

    Python's ``bool`` is a subclass of ``int``, so ``isinstance(True, int)``
    is ``True``.  Flag metrics like ``timeout: True`` must not be treated as
    numeric fitness contributions.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def get_score(metrics: Dict[str, Any]) -> float:
    """Return combined_score if available, otherwise average of all numeric metric values."""
    if not metrics:
        return 0.0
    if "combined_score" in metrics:
        try:
            return float(metrics["combined_score"])
        except (ValueError, TypeError):
            pass
    numeric_values = [v for v in metrics.values() if is_numeric_metric(v)]
    return sum(numeric_values) / len(numeric_values) if numeric_values else 0.0


def format_metrics(metrics: Dict[str, Any]) -> str:
    """Format a metrics dict for logging, handling both numeric and string values."""
    if not metrics:
        return ""

    parts = []
    for name, value in metrics.items():
        if is_numeric_metric(value):
            try:
                parts.append(f"{name}={value:.4f}")
            except (ValueError, TypeError):
                parts.append(f"{name}={value}")
        else:
            parts.append(f"{name}={value}")

    return ", ".join(parts)


def normalize_metric_value(
    key: str,
    value: Any,
    higher_is_better: Dict[str, bool],
) -> Optional[float]:
    """Convert a metric to an internal score where larger is always better.

    Args:
        key: Metric name used to look up direction in *higher_is_better*.
        value: Raw metric value (must be numeric, else returns ``None``).
        higher_is_better: Mapping of metric names to direction.  Missing keys
            default to ``True`` (i.e. higher is better).

    Returns:
        Normalised float (negated when the metric should be minimised), or
        ``None`` when *value* is not numeric.
    """
    if not is_numeric_metric(value):
        return None
    normalized = float(value)
    if math.isnan(normalized):
        return None
    if not higher_is_better.get(key, True):
        normalized = -normalized
    return normalized


def compute_proxy_score(
    metrics: Dict[str, Any],
    *,
    fitness_key: Optional[str] = None,
    pareto_objectives: Optional[List[str]] = None,
    higher_is_better: Optional[Dict[str, bool]] = None,
) -> float:
    """Compute a scalar proxy score from a metrics dict.

    Implements a single fallback chain used by both the database layer and
    the prompt/context builder:

    1. ``fitness_key`` (normalised via *higher_is_better*)
    2. ``combined_score`` (taken as-is)
    3. Average of normalised *pareto_objectives* values
    4. :func:`get_score` (generic numeric average)

    Returns ``-inf`` when *metrics* is empty so callers can distinguish
    "no data" from "score is zero".
    """
    if not metrics:
        return float("-inf")

    hib = higher_is_better or {}

    if fitness_key is not None:
        normalized = normalize_metric_value(fitness_key, metrics.get(fitness_key), hib)
        if normalized is not None:
            return normalized

    combined_score = metrics.get("combined_score")
    if is_numeric_metric(combined_score):
        return float(combined_score)

    if pareto_objectives:
        objective_values = []
        for objective in pareto_objectives:
            normalized = normalize_metric_value(objective, metrics.get(objective), hib)
            if normalized is not None:
                objective_values.append(normalized)
        if objective_values:
            return sum(objective_values) / len(objective_values)

    return get_score(metrics)


def format_improvement(parent_metrics: Dict[str, Any], child_metrics: Dict[str, Any]) -> str:
    """Format the per-metric delta between parent and child for logging."""
    if not parent_metrics or not child_metrics:
        return ""

    parts = []
    for metric, child_value in child_metrics.items():
        if metric in parent_metrics:
            parent_value = parent_metrics[metric]
            if is_numeric_metric(child_value) and is_numeric_metric(parent_value):
                try:
                    parts.append(f"{metric}={child_value - parent_value:+.4f}")
                except (ValueError, TypeError):
                    continue

    return ", ".join(parts)
