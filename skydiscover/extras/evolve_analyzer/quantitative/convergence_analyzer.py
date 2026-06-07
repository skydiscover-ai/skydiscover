"""
Convergence analysis for evolutionary optimization runs.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from skydiscover.extras.evolve_analyzer.quantitative.bundle import ConvergenceMetrics


def analyze_convergence(records: List[dict], window: int = 10) -> ConvergenceMetrics:
    """
    Analyze the convergence behaviour of an evolutionary optimization run.

    Parameters
    ----------
    records:
        Raw JSONL records (each a dict) from the run.
    window:
        Rolling-window size used for mean and variance calculations.

    Returns
    -------
    ConvergenceMetrics
    """
    if not records:
        return ConvergenceMetrics(
            best_so_far_curve=[],
            rolling_mean=[],
            rolling_variance=[],
            change_points=[],
            convergence_rate=0.0,
            improvement_per_eval=0.0,
            time_to_best_fraction=0.0,
            plateau_onset_iteration=None,
        )

    sorted_records = sorted(records, key=lambda r: r.get("iteration", 0))
    total_iterations = len(sorted_records)

    # ── best-so-far curve (exclude crash evaluations) ─────────────────────────
    best_so_far_curve: List[float] = []
    running_best: Optional[float] = None

    for rec in sorted_records:
        if rec.get("evaluation_status") != "crash":
            score = rec.get("child_score", float("-inf"))
            if running_best is None or score > running_best:
                running_best = score
        # If every record so far has been a crash, carry forward None → use NaN
        best_so_far_curve.append(running_best if running_best is not None else float("nan"))

    scores = np.array(
        [r.get("child_score", float("nan")) for r in sorted_records], dtype=float
    )

    # ── rolling mean and variance ──────────────────────────────────────────────
    rolling_mean: List[float] = []
    rolling_variance: List[float] = []

    for i in range(total_iterations):
        start = max(0, i - window + 1)
        window_scores = scores[start : i + 1]
        valid = window_scores[~np.isnan(window_scores)]
        if len(valid) == 0:
            rolling_mean.append(float("nan"))
            rolling_variance.append(float("nan"))
        else:
            rolling_mean.append(float(np.mean(valid)))
            rolling_variance.append(float(np.var(valid, ddof=0)))

    # ── change points: indices where best_so_far strictly increases ───────────
    change_points: List[int] = []
    for i in range(1, total_iterations):
        prev = best_so_far_curve[i - 1]
        curr = best_so_far_curve[i]
        if not np.isnan(curr) and not np.isnan(prev) and curr > prev:
            change_points.append(sorted_records[i].get("iteration", i))

    # ── scalar summary metrics ─────────────────────────────────────────────────
    valid_best = [v for v in best_so_far_curve if not np.isnan(v)]
    initial_best = valid_best[0] if valid_best else 0.0
    final_best = valid_best[-1] if valid_best else 0.0

    convergence_rate = (
        (final_best - initial_best) / total_iterations
        if total_iterations > 0
        else 0.0
    )

    non_crash_count = sum(
        1 for r in sorted_records if r.get("evaluation_status") != "crash"
    )
    improvement_per_eval = (
        (final_best - initial_best) / non_crash_count
        if non_crash_count > 0
        else 0.0
    )

    # Index (position) where the best score was first achieved
    best_index: int = 0
    if valid_best:
        for i in range(len(sorted_records)):
            if not np.isnan(best_so_far_curve[i]) and best_so_far_curve[i] >= final_best:
                best_index = i
                break

    time_to_best_fraction = (
        best_index / (total_iterations - 1) if total_iterations > 1 else 0.0
    )

    # ── plateau onset: last iteration index where best_so_far improves ────────
    plateau_onset_iteration: Optional[int] = None
    last_improvement_idx = -1
    for i in range(1, total_iterations):
        prev = best_so_far_curve[i - 1]
        curr = best_so_far_curve[i]
        if not np.isnan(curr) and not np.isnan(prev) and curr > prev:
            last_improvement_idx = i

    if last_improvement_idx >= 0 and last_improvement_idx < total_iterations - 1:
        # There was an improvement, but not at the very last step → plateau exists
        plateau_onset_iteration = sorted_records[last_improvement_idx + 1].get(
            "iteration", last_improvement_idx + 1
        )

    return ConvergenceMetrics(
        best_so_far_curve=best_so_far_curve,
        rolling_mean=rolling_mean,
        rolling_variance=rolling_variance,
        change_points=change_points,
        convergence_rate=convergence_rate,
        improvement_per_eval=improvement_per_eval,
        time_to_best_fraction=time_to_best_fraction,
        plateau_onset_iteration=plateau_onset_iteration,
    )
