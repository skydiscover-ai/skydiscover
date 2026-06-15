"""
Sub-metric trajectory analyzer.

Extracts per-component metrics from evaluator_metrics, tracks their
trajectories across iterations, and identifies which metric drove the
overall score improvement from seed to best.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from skydiscover.extras.evolve_analyzer.quantitative.bundle import SubMetricAnalysis, SubMetricStats


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Metric names (substring match) where a lower value is better.
_LOWER_IS_BETTER_KEYWORDS = (
    "_ms",
    "latency",
    "eviction_rate",
    "evictions",
    "error_rate",
    "misses",
    "p99_",
    "p95_",
    "tpot",
)

# Raw count fields that add noise without signal when shown alongside rates.
_COUNT_METRICS = {"lookup_total", "lookup_hits", "lookup_misses"}

# The primary composite score — tracked separately via child_score.
_EXCLUDE_METRICS = {"combined_score"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lower_is_better(name: str) -> bool:
    nl = name.lower()
    return any(kw in nl for kw in _LOWER_IS_BETTER_KEYWORDS)


def _flatten_metrics(em: dict) -> Dict[str, float]:
    """
    Flatten one level of nesting in an evaluator_metrics dict.

    Top-level numeric values are kept as-is.
    Values inside nested dicts (e.g. em["metrics"]["mean_ttft_ms"]) are
    unwrapped to the inner key name.
    Non-numeric values are ignored.
    """
    flat: Dict[str, float] = {}
    for k, v in em.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                    flat[kk] = float(vv)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            flat[k] = float(v)
    return flat


def _seed_metrics(records: List[dict]) -> Dict[str, float]:
    """Return the metrics of the seed program.

    Primary: parent_metrics of the earliest iteration that has them (set by
    adapters that can recover the seed program's evaluation results).
    Fallback: evaluator_metrics of an explicit iteration-0 record, which some
    frameworks emit when the seed itself is logged as an iteration.
    """
    sorted_recs = sorted(records, key=lambda r: r.get("iteration", 0))
    for rec in sorted_recs:
        pm = rec.get("parent_metrics")
        if isinstance(pm, dict) and pm:
            return _flatten_metrics(pm)
    # Fallback: iteration 0 logged as an ordinary record
    for rec in sorted_recs:
        if rec.get("iteration") == 0:
            em = rec.get("evaluator_metrics")
            if isinstance(em, dict) and em:
                return _flatten_metrics(em)
    return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_sub_metrics(records: List[dict]) -> Optional[SubMetricAnalysis]:
    """
    Compute per-sub-metric statistics from evaluator_metrics fields.

    Parameters
    ----------
    records:
        Normalised iteration records (dicts) from load_evolve_records.

    Returns
    -------
    SubMetricAnalysis, or None if no records contain evaluator_metrics.
    """
    records_with_em = [
        r for r in records
        if isinstance(r.get("evaluator_metrics"), dict)
    ]
    if not records_with_em:
        return None

    seed = _seed_metrics(records)

    # Collect (iteration, value) pairs per metric, only for valid evaluated iterations.
    # Exclude any record with a negative combined score (sentinel, correctness
    # failure, or any other penalty) since its sub-metrics are not meaningful.
    series: Dict[str, List[Tuple[int, float]]] = {}
    for rec in records_with_em:
        child_score = rec.get("child_score")
        if child_score is None:
            continue
        if float(child_score) < 0:
            continue
        if rec.get("evaluation_status") == "crash":
            continue
        iteration = rec.get("iteration", 0)
        for name, val in _flatten_metrics(rec["evaluator_metrics"]).items():
            if name in _EXCLUDE_METRICS or name in _COUNT_METRICS:
                continue
            series.setdefault(name, []).append((iteration, val))

    if not series:
        return None

    stats: Dict[str, SubMetricStats] = {}
    for name, iter_vals in series.items():
        lib = _lower_is_better(name)
        values = [v for _, v in iter_vals]
        iterations = [i for i, _ in iter_vals]

        best_val = min(values) if lib else max(values)
        worst_val = max(values) if lib else min(values)
        best_idx = values.index(best_val)
        best_iter = iterations[best_idx]

        # Final value by last iteration number
        final_val = sorted(iter_vals, key=lambda t: t[0])[-1][1]

        seed_val = seed.get(name)
        if seed_val is not None and abs(seed_val) > 1e-9:
            # Positive = improvement in the metric's direction
            if lib:
                improvement = (seed_val - best_val) / abs(seed_val)
            else:
                improvement = (best_val - seed_val) / abs(seed_val)
        else:
            improvement = None

        stats[name] = SubMetricStats(
            name=name,
            best=best_val,
            worst=worst_val,
            final=final_val,
            best_iteration=best_iter,
            seed_value=seed_val,
            improvement_vs_seed=improvement,
            lower_is_better=lib,
        )

    # Primary driver: metric with the largest absolute improvement vs seed
    candidates = [
        (name, s.improvement_vs_seed)
        for name, s in stats.items()
        if s.improvement_vs_seed is not None
    ]
    primary_driver: Optional[str] = None
    if candidates:
        primary_driver = max(candidates, key=lambda t: abs(t[1]))[0]

    return SubMetricAnalysis(metrics=stats, primary_driver=primary_driver)
