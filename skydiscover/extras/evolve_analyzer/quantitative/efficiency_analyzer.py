"""
Efficiency analyzer: measures how effectively each LLM call, dollar, and hour
of compute translated into score improvement.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from skydiscover.extras.evolve_analyzer.quantitative.bundle import EfficiencyMetrics


def analyze_efficiency(records: List[dict]) -> EfficiencyMetrics:
    """Compute efficiency metrics from a list of JSONL iteration records."""
    if not records:
        return EfficiencyMetrics(
            improvement_per_llm_call=0.0,
            improvement_per_dollar=None,
            improvement_per_hour=None,
            improvement_per_eval_call=0.0,
            productive_phase_fraction=1.0,
            wasted_phase_fraction=0.0,
            pareto_frontier=[],
        )

    sorted_records = sorted(records, key=lambda r: r.get("iteration", 0))
    total_iterations = len(sorted_records)

    # ── Score bookkeeping ─────────────────────────────────────────────────────
    # Build best_so_far curve using child_score at each iteration.
    # Sentinel/crash scores (≤ -9999) and missing scores are treated as NaN
    # so they don't corrupt initial_score or best_so_far.
    scores: List[float] = []
    for r in sorted_records:
        s = r.get("child_score")
        if s is None or float(s) < 0:
            scores.append(float("nan"))
        else:
            scores.append(float(s))

    best_so_far: List[float] = []
    running_best = float("nan")
    for s in scores:
        if not math.isnan(s) and (math.isnan(running_best) or s > running_best):
            running_best = s
        best_so_far.append(running_best)

    initial_score: float = next((s for s in scores if not math.isnan(s)), 0.0)
    final_best: float = best_so_far[-1] if not math.isnan(best_so_far[-1]) else initial_score
    total_gain: float = final_best - initial_score

    # ── improvement_per_llm_call ──────────────────────────────────────────────
    improvement_per_llm_call: float = (
        total_gain / total_iterations if total_iterations > 0 else 0.0
    )

    # ── improvement_per_dollar ────────────────────────────────────────────────
    improvement_per_dollar: Optional[float] = None
    cost_values = [
        r["llm_cost_usd"]
        for r in sorted_records
        if r.get("llm_cost_usd") is not None
    ]
    if cost_values:
        total_cost = sum(cost_values)
        if total_cost > 0:
            improvement_per_dollar = total_gain / total_cost

    # ── improvement_per_hour ──────────────────────────────────────────────────
    improvement_per_hour: Optional[float] = None
    timestamps = [
        r["timestamp"]
        for r in sorted_records
        if r.get("timestamp") is not None
    ]
    if len(timestamps) >= 2:
        total_seconds = max(timestamps) - min(timestamps)
        total_hours = total_seconds / 3600.0
        if total_hours > 0:
            improvement_per_hour = total_gain / total_hours

    # ── productive_phase_fraction ─────────────────────────────────────────────
    # Plateau onset = last iteration index where best_so_far improved.
    # After that index the curve is flat → "wasted" phase.
    plateau_onset_idx: int = 0
    for idx in range(1, total_iterations):
        if (
            not math.isnan(best_so_far[idx])
            and not math.isnan(best_so_far[idx - 1])
            and best_so_far[idx] > best_so_far[idx - 1]
        ):
            plateau_onset_idx = idx

    # plateau_onset_idx is the last index where an improvement occurred.
    # Iterations 0..plateau_onset_idx are "productive" (inclusive).
    productive_count = plateau_onset_idx + 1  # inclusive upper bound

    if total_gain == 0.0:
        # Score never improved; treat everything as productive (no wasted tail).
        productive_phase_fraction = 1.0
    else:
        productive_phase_fraction = productive_count / total_iterations

    wasted_phase_fraction = 1.0 - productive_phase_fraction

    # ── Pareto frontier: (cumulative_cost, best_score) ────────────────────────
    # Use actual cost if available, otherwise use iteration number as proxy.
    use_cost = bool(cost_values)

    pareto_frontier: List[Tuple[float, float]] = []
    cumulative_cost = 0.0
    frontier_best = -float("inf")

    for idx, rec in enumerate(sorted_records):
        if use_cost:
            cumulative_cost += rec.get("llm_cost_usd") or 0.0
        else:
            cumulative_cost = float(idx + 1)  # 1-indexed iteration proxy

        current_score = best_so_far[idx]
        if not math.isnan(current_score) and current_score > frontier_best:
            frontier_best = current_score
            pareto_frontier.append((cumulative_cost, current_score))

    return EfficiencyMetrics(
        improvement_per_llm_call=improvement_per_llm_call,
        improvement_per_dollar=improvement_per_dollar,
        improvement_per_hour=improvement_per_hour,
        improvement_per_eval_call=improvement_per_llm_call,  # alias per spec
        productive_phase_fraction=productive_phase_fraction,
        wasted_phase_fraction=wasted_phase_fraction,
        pareto_frontier=pareto_frontier,
    )
