"""
Search-space analyzer: characterises which regions of the hyperparameter space
the top-k solutions occupy and how effectively the space was explored.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np

from skydiscover.extras.evolve_analyzer.quantitative.bundle import SearchSpaceMetrics


def _is_numeric(value: Any) -> bool:
    """Return True if *value* can be treated as a real number."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def analyze_search_space(
    records: List[dict],
    top_k: int = 10,
) -> SearchSpaceMetrics:
    """Compute search-space metrics from iteration records."""
    has_params = any(r.get("parameters") for r in records)
    if not records or not has_params:
        return SearchSpaceMetrics(
            param_distributions={},
            bound_hit_params=[],
            frozen_params=[],
            effective_dimensionality=0,
            trial_to_param_ratio=None,
        )

    total_iterations = sum(
        1 for r in records if r.get("evaluation_status") in ("success", None)
    )

    records_with_params = [r for r in records if r.get("parameters")]

    # ── Top-k selection (for bound-hit detection only) ────────────────────────
    sorted_by_score = sorted(
        records_with_params,
        key=lambda r: r.get("child_score", float("-inf")),
        reverse=True,
    )
    top_records = sorted_by_score[:top_k]

    # ── Collect all parameter keys across ALL records ─────────────────────────
    all_param_keys: set = set()
    for rec in records_with_params:
        all_param_keys.update(rec["parameters"].keys())

    # ── param_distributions computed over ALL records ─────────────────────────
    # Using all records (not just top-k) gives accurate ranges and avoids
    # selection bias where top-k may not cover the full observed range.
    param_distributions: Dict[str, dict] = {}

    for key in all_param_keys:
        values = [
            rec["parameters"][key]
            for rec in records_with_params
            if key in rec["parameters"]
        ]
        if not values:
            continue

        numeric_values = [v for v in values if _is_numeric(v)]

        if len(numeric_values) == len(values):
            arr = np.array(numeric_values, dtype=float)
            param_distributions[key] = {
                "type": "numeric",
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=0)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "count": len(arr),
            }
        else:
            counts: Dict[str, int] = {}
            for v in values:
                label = str(v)
                counts[label] = counts.get(label, 0) + 1
            param_distributions[key] = {
                "type": "categorical",
                "counts": counts,
                "count": len(values),
            }

    # ── bound_hit_params ──────────────────────────────────────────────────────
    # A parameter "hits a bound" when the top-k values all sit at the
    # same extreme (min or max) of the full distribution.
    bound_hit_params: List[str] = []

    for key in all_param_keys:
        dist = param_distributions.get(key, {})
        if dist.get("type") != "numeric":
            continue

        all_values = [
            r["parameters"][key]
            for r in records_with_params
            if key in r["parameters"] and _is_numeric(r["parameters"][key])
        ]
        if len(all_values) < 3:
            continue

        global_min = min(all_values)
        global_max = max(all_values)

        if global_min == global_max:
            continue  # constant — not a bound hit

        top_values = [
            rec["parameters"][key]
            for rec in top_records
            if key in rec["parameters"] and _is_numeric(rec["parameters"][key])
        ]
        if not top_values:
            continue

        all_at_min = all(v == global_min for v in top_values)
        all_at_max = all(v == global_max for v in top_values)
        if all_at_min or all_at_max:
            bound_hit_params.append(key)

    # ── frozen_params ─────────────────────────────────────────────────────────
    # Parameters that were observed across multiple records but never varied.
    # Low variance here signals the search collapsed rather than deliberately
    # converging: e.g. island_id always=0 means multi-island diversity was
    # never used; change_type always="Full rewrite" means no mutation variety.
    frozen_params: List[str] = []

    for key in all_param_keys:
        dist = param_distributions.get(key, {})
        if dist.get("type") == "numeric":
            if dist.get("std", 0.0) == 0.0 and dist.get("count", 0) > 1:
                frozen_params.append(key)
        elif dist.get("type") == "categorical":
            distinct = len(dist.get("counts", {}))
            if distinct == 1 and dist.get("count", 0) > 1:
                frozen_params.append(key)

    # ── effective_dimensionality ──────────────────────────────────────────────
    effective_dimensionality = 0

    for key in all_param_keys:
        dist = param_distributions.get(key, {})
        if dist.get("type") == "numeric":
            if dist.get("std", 0.0) > 0.0:
                effective_dimensionality += 1
        elif dist.get("type") == "categorical":
            distinct = len(dist.get("counts", {}))
            if distinct > 1:
                effective_dimensionality += 1

    # ── trial_to_param_ratio ──────────────────────────────────────────────────
    trial_to_param_ratio: Optional[float] = None
    if effective_dimensionality > 0:
        trial_to_param_ratio = total_iterations / effective_dimensionality

    # ── dominant_island_id ────────────────────────────────────────────────────
    # When island_id is a bound-hit param, record which specific island the
    # top-k solutions clustered on (the min or max island ID).
    dominant_island_id: Optional[int] = None
    if "island_id" in bound_hit_params:
        top_island_values = [
            rec["parameters"]["island_id"]
            for rec in top_records
            if "island_id" in rec["parameters"]
            and _is_numeric(rec["parameters"]["island_id"])
        ]
        if top_island_values:
            dominant_island_id = int(top_island_values[0])

    return SearchSpaceMetrics(
        param_distributions=param_distributions,
        bound_hit_params=bound_hit_params,
        frozen_params=sorted(frozen_params),
        effective_dimensionality=effective_dimensionality,
        trial_to_param_ratio=trial_to_param_ratio,
        dominant_island_id=dominant_island_id,
    )
