"""
Utilities for metric scoring and formatting.
"""

from typing import Any, Dict, Optional


def get_score(metrics: Dict[str, Any]) -> float:
    """Return combined_score if available, otherwise average of all numeric metric values."""
    if not metrics:
        return 0.0
    if "combined_score" in metrics:
        try:
            return float(metrics["combined_score"])
        except (ValueError, TypeError):
            pass
    numeric_values = [v for v in metrics.values() if isinstance(v, (int, float))]
    return sum(numeric_values) / len(numeric_values) if numeric_values else 0.0


def format_metrics(metrics: Dict[str, Any]) -> str:
    """Format a metrics dict for logging, handling both numeric and string values."""
    if not metrics:
        return ""

    parts = []
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
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
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    normalized = float(value)
    if not higher_is_better.get(key, True):
        normalized = -normalized
    return normalized


def format_improvement(parent_metrics: Dict[str, Any], child_metrics: Dict[str, Any]) -> str:
    """Format the per-metric delta between parent and child for logging."""
    if not parent_metrics or not child_metrics:
        return ""

    parts = []
    for metric, child_value in child_metrics.items():
        if metric in parent_metrics:
            parent_value = parent_metrics[metric]
            if isinstance(child_value, (int, float)) and isinstance(parent_value, (int, float)):
                try:
                    parts.append(f"{metric}={child_value - parent_value:+.4f}")
                except (ValueError, TypeError):
                    continue

    return ", ".join(parts)
