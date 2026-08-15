"""
Ceiling analyzer: estimates whether the optimization run is approaching its
performance ceiling and whether continued iteration is likely to be productive.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from skydiscover.extras.evolve_analyzer.quantitative.bundle import CeilingMetrics


def _linregress_slope(x: List[float], y: List[float]) -> float:
    """Return the slope of the OLS fit, using scipy when available."""
    try:
        from scipy.stats import linregress as _sp_linregress  # type: ignore

        result = _sp_linregress(x, y)
        return float(result.slope)
    except ImportError:
        coeffs = np.polyfit(x, y, 1)
        return float(coeffs[0])


def _ttest_ind_pvalue(a: List[float], b: List[float]) -> Optional[float]:
    """Return the two-tailed p-value from an independent t-test."""
    try:
        from scipy.stats import ttest_ind as _sp_ttest  # type: ignore

        result = _sp_ttest(a, b, equal_var=False)
        return float(result.pvalue)
    except ImportError:
        # Welch's t-test implemented with numpy only.
        arr_a = np.array(a, dtype=float)
        arr_b = np.array(b, dtype=float)
        mean_a, mean_b = np.mean(arr_a), np.mean(arr_b)
        var_a = np.var(arr_a, ddof=1) if len(arr_a) > 1 else 0.0
        var_b = np.var(arr_b, ddof=1) if len(arr_b) > 1 else 0.0
        n_a, n_b = len(arr_a), len(arr_b)
        denom = np.sqrt(var_a / n_a + var_b / n_b)
        if denom == 0:
            return 1.0
        t_stat = (mean_a - mean_b) / denom
        # Welch–Satterthwaite degrees of freedom.
        num_df = (var_a / n_a + var_b / n_b) ** 2
        den_df = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1) if n_a > 1 and n_b > 1 else 1.0
        df = num_df / den_df if den_df != 0 else 1.0
        # Approximate p-value using the regularized incomplete beta function.
        # Fall back to a rough estimate when scipy is absent.
        try:
            from scipy.special import btdtr  # type: ignore

            p = float(btdtr(df / 2, 0.5, df / (df + t_stat ** 2)))
            return 2.0 * min(p, 1.0 - p)
        except ImportError:
            # Very rough approximation for large df.
            z = abs(t_stat)
            p_approx = 2.0 * (1.0 / (1.0 + 0.33267 * z)) * (
                0.4361836 - 0.1201676 * (1.0 / (1.0 + 0.33267 * z))
                + 0.9372980 * (1.0 / (1.0 + 0.33267 * z)) ** 2
            )
            return float(np.clip(p_approx, 0.0, 1.0))


def analyze_ceiling(records: List[dict]) -> CeilingMetrics:
    """Compute ceiling / diminishing-returns metrics from iteration records."""
    if not records:
        return CeilingMetrics(
            marginal_improvement_trend="flat",
            plateau_p_value=None,
            estimated_gain_probability=None,
            flat_trend_start_iteration=None,
            early_stop_suggested_at=None,
            early_stop_actual_at=None,
            wasted_iterations_after_suggestion=None,
        )

    sorted_records = sorted(records, key=lambda r: r.get("iteration", 0))
    n = len(sorted_records)

    # ── Build best_so_far curve ───────────────────────────────────────────────
    scores = [r.get("child_score", 0.0) for r in sorted_records]
    best_so_far: List[float] = []
    running_best = scores[0]
    for s in scores:
        if s > running_best:
            running_best = s
        best_so_far.append(running_best)

    # Deltas between consecutive best_so_far values.
    bsf_deltas = [
        best_so_far[i] - best_so_far[i - 1] for i in range(1, n)
    ]

    # ── flat_trend_start_iteration ────────────────────────────────────────────
    # The iteration after the last *meaningful* improvement to best_so_far.
    # An improvement is meaningful if it exceeds 0.5% of the total score range,
    # so micro-gains (e.g. Δ=0.037 on a 137-point range) don't delay detection.
    flat_trend_start_iteration: Optional[int] = None
    total_range = best_so_far[-1] - best_so_far[0]
    meaningful_threshold = total_range * 0.005 if total_range > 0 else 0.0
    last_meaningful_idx = 0
    for i in range(1, n):
        if best_so_far[i] - best_so_far[i - 1] >= meaningful_threshold:
            last_meaningful_idx = i
    if last_meaningful_idx < n - 1:
        flat_trend_start_iteration = int(
            sorted_records[last_meaningful_idx + 1].get("iteration", last_meaningful_idx + 1)
        )

    # ── marginal_improvement_trend ────────────────────────────────────────────
    marginal_improvement_trend: str
    if len(bsf_deltas) >= 2:
        x = list(range(len(bsf_deltas)))
        slope = _linregress_slope(x, bsf_deltas)
        if slope < -0.0001:
            marginal_improvement_trend = "declining"
        elif abs(slope) <= 0.0001:
            marginal_improvement_trend = "flat"
        else:
            marginal_improvement_trend = "improving"
    else:
        # Not enough data to fit a line — default to flat.
        marginal_improvement_trend = "flat"

    # ── plateau detection ─────────────────────────────────────────────────────
    # Plateau = last 20% of iterations where best_so_far did not change.
    plateau_start_idx = max(0, int(n * 0.80))
    plateau_bsf = best_so_far[plateau_start_idx:]
    in_plateau = len(set(plateau_bsf)) == 1  # all identical

    # ── plateau_p_value ───────────────────────────────────────────────────────
    plateau_p_value: Optional[float] = None
    if in_plateau:
        plateau_deltas = [
            r.get("score_delta", 0.0) for r in sorted_records[plateau_start_idx:]
        ]
        non_plateau_deltas = [
            r.get("score_delta", 0.0) for r in sorted_records[:plateau_start_idx]
        ]
        if len(plateau_deltas) >= 10 and len(non_plateau_deltas) >= 10:
            plateau_p_value = _ttest_ind_pvalue(plateau_deltas, non_plateau_deltas)

    # ── estimated_gain_probability ────────────────────────────────────────────
    # Probability that a future iteration will advance the *global best*
    # (frontier), not merely beat the parent.
    estimated_gain_probability: Optional[float] = None
    if n >= 8:
        tail_start_idx = max(0, int(n * 0.75))
        frontier_advances = sum(
            1
            for i in range(tail_start_idx, n)
            if best_so_far[i] > best_so_far[i - 1]
        )
        tail_length = n - tail_start_idx
        estimated_gain_probability = frontier_advances / tail_length

    # ── early_stop_suggested_at ───────────────────────────────────────────────
    early_stop_suggested_at: Optional[int] = None
    for rec in sorted_records:
        if rec.get("early_stop_suggested") is True:
            early_stop_suggested_at = int(rec.get("iteration", 0))
            break

    # ── early_stop_actual_at ──────────────────────────────────────────────────
    early_stop_actual_at: Optional[int] = int(
        sorted_records[-1].get("iteration", n - 1)
    )

    # ── wasted_iterations_after_suggestion ───────────────────────────────────
    wasted_iterations_after_suggestion: Optional[int] = None
    if early_stop_suggested_at is not None and early_stop_actual_at is not None:
        wasted_iterations_after_suggestion = (
            early_stop_actual_at - early_stop_suggested_at
        )

    return CeilingMetrics(
        marginal_improvement_trend=marginal_improvement_trend,
        plateau_p_value=plateau_p_value,
        estimated_gain_probability=estimated_gain_probability,
        flat_trend_start_iteration=flat_trend_start_iteration,
        early_stop_suggested_at=early_stop_suggested_at,
        early_stop_actual_at=early_stop_actual_at,
        wasted_iterations_after_suggestion=wasted_iterations_after_suggestion,
    )
