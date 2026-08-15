"""
Regression analysis for evolutionary optimization runs.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from skydiscover.extras.evolve_analyzer.quantitative.bundle import RegressionMetrics


def analyze_regressions(records: List[dict]) -> RegressionMetrics:
    """
    Analyse regression events (iterations where the child score is worse than the
    parent score) within an evolutionary optimization run.

    Parameters
    ----------
    records:
        Raw JSONL records (each a dict) from the run.

    Returns
    -------
    RegressionMetrics
    """
    if not records:
        return RegressionMetrics(
            regression_frequency=0.0,
            severity_distribution={"mild": 0, "moderate": 0, "severe": 0},
            mean_recovery_time=0.0,
            death_spiral_periods=[],
        )

    sorted_records = sorted(records, key=lambda r: r.get("iteration", 0))

    # Filter out failed iterations (negative scores, crashes) — their extreme
    # scores create artificial regressions and hide real ones.
    valid_records = [
        r for r in sorted_records
        if r.get("evaluation_status") != "crash"
        and not (
            r.get("child_score") is not None
            and float(r.get("child_score", 0)) < 0
        )
    ]
    total = len(valid_records)

    # Recompute score_delta between adjacent *valid* records since the
    # pre-computed deltas reference potentially-removed sentinel neighbours.
    deltas: list = []
    for i, r in enumerate(valid_records):
        if i == 0:
            deltas.append(0.0)
        else:
            cur = r.get("child_score")
            prev = valid_records[i - 1].get("child_score")
            if cur is not None and prev is not None:
                deltas.append(float(cur) - float(prev))
            else:
                deltas.append(0.0)

    # ── regression mask ───────────────────────────────────────────────────────
    is_regression = [d < 0.0 for d in deltas]
    regression_count = sum(is_regression)
    regression_frequency = regression_count / total if total > 0 else 0.0

    # ── severity distribution ─────────────────────────────────────────────────
    severity_distribution = {"mild": 0, "moderate": 0, "severe": 0}
    for i, reg in enumerate(is_regression):
        if reg:
            d = deltas[i]
            if d >= -0.01:
                severity_distribution["mild"] += 1
            elif d >= -0.05:
                severity_distribution["moderate"] += 1
            else:
                severity_distribution["severe"] += 1

    # ── best-so-far curve (valid evaluations only) ─────────────────────────────
    best_so_far: List[float] = []
    running_best: Optional[float] = None
    for rec in valid_records:
        score = rec.get("child_score", float("-inf"))
        if score is None:
            score = float("-inf")
        if running_best is None or score > running_best:
            running_best = score
        best_so_far.append(running_best if running_best is not None else float("nan"))

    # ── mean recovery time ────────────────────────────────────────────────────
    recovery_times: List[int] = []

    for i, reg in enumerate(is_regression):
        if not reg:
            continue
        # best score the run had achieved just before this regression
        pre_regression_best = best_so_far[i - 1] if i > 0 else best_so_far[i]

        # Search subsequent iterations for recovery
        recovered = False
        for j in range(i + 1, total):
            if best_so_far[j] > pre_regression_best:
                recovery_times.append(j - i)
                recovered = True
                break

        if not recovered:
            # Use remaining iterations as the recovery time
            recovery_times.append(total - 1 - i)

    mean_recovery_time = float(np.mean(recovery_times)) if recovery_times else 0.0

    # ── death spiral periods: runs of 3+ consecutive regressions ─────────────
    death_spiral_periods: List[Tuple[int, int]] = []

    run_start: Optional[int] = None
    run_length = 0

    for i, reg in enumerate(is_regression):
        if reg:
            if run_start is None:
                run_start = i
                run_length = 1
            else:
                run_length += 1
        else:
            if run_start is not None and run_length >= 3:
                start_iter = valid_records[run_start].get("iteration", run_start)
                end_iter = valid_records[run_start + run_length - 1].get(
                    "iteration", run_start + run_length - 1
                )
                death_spiral_periods.append((start_iter, end_iter))
            run_start = None
            run_length = 0

    # Handle a spiral that extends to the very last record
    if run_start is not None and run_length >= 3:
        start_iter = valid_records[run_start].get("iteration", run_start)
        end_iter = valid_records[run_start + run_length - 1].get(
            "iteration", run_start + run_length - 1
        )
        death_spiral_periods.append((start_iter, end_iter))

    return RegressionMetrics(
        regression_frequency=regression_frequency,
        severity_distribution=severity_distribution,
        mean_recovery_time=mean_recovery_time,
        death_spiral_periods=death_spiral_periods,
    )
