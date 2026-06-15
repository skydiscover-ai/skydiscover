"""
Infrastructure failure analyzer: detects server crashes and eval-time degradation
patterns that corrupt iteration scores with sentinel values.

Sentinel values (-10999 score, 0.0 eval_time) indicate the evaluation server was
unreachable, not a code failure. This analyzer classifies the failure cause so
the report synthesizer can surface accurate diagnostics.
"""
from __future__ import annotations

import statistics
from typing import List, Optional, Tuple

from skydiscover.extras.evolve_analyzer.quantitative.bundle import InfrastructureMetrics

_SENTINEL_SCORE_THRESHOLD = -9999.0
_SENTINEL_P90_THRESHOLD = 9999.0
_DEGRADATION_SPIKE_RATIO = 1.5     # eval_time > 1.5× baseline → degradation
_CRASH_COUNT_THRESHOLD = 5         # > 5 contiguous sentinels → crash, not noise
_BASELINE_WINDOW = 10              # last N non-sentinel iters before first sentinel


def _is_sentinel(rec: dict) -> bool:
    score = rec.get("child_score")
    if score is not None and float(score) <= _SENTINEL_SCORE_THRESHOLD:
        return True
    em = rec.get("evaluator_metrics") or {}
    p90 = em.get("p90_trajectory_time")
    if p90 is not None and float(p90) >= _SENTINEL_P90_THRESHOLD:
        return True
    return False


def _contiguous_block_length(flags: List[bool], start: int) -> int:
    """Return the length of the contiguous True block starting at index start."""
    n = 0
    for i in range(start, len(flags)):
        if flags[i]:
            n += 1
        else:
            break
    return n


def analyze_infrastructure(records: List[dict]) -> Optional[InfrastructureMetrics]:
    """Detect infrastructure failure patterns in iteration records.

    Reads `_infra_log_signals` injected by ingestion (SkyDiscover JSONL path only)
    and propagates it to `InfrastructureMetrics.log_evidence`. The field is then
    stripped from records so it doesn't pollute downstream DataFrames.
    """
    if not records:
        return None

    # Collect log signals injected by ingestion, then strip to keep DataFrames clean.
    log_signals = next(
        (r["_infra_log_signals"] for r in records if "_infra_log_signals" in r),
        None,
    )
    for r in records:
        r.pop("_infra_log_signals", None)

    sorted_records = sorted(records, key=lambda r: r.get("iteration", 0))
    n = len(sorted_records)

    sentinel_flags = [_is_sentinel(r) for r in sorted_records]
    sentinel_count = sum(sentinel_flags)

    if sentinel_count == 0:
        return InfrastructureMetrics(
            sentinel_count=0,
            sentinel_fraction=0.0,
            first_sentinel_iteration=None,
            crash_onset_iteration=None,
            degradation_window=None,
            failure_cause="NONE",
            affected_iterations=[],
            eval_time_spike_ratio=None,
            log_evidence=log_signals,
        )

    sentinel_fraction = sentinel_count / n
    affected_iterations = [
        int(sorted_records[i].get("iteration", i))
        for i in range(n)
        if sentinel_flags[i]
    ]
    first_sentinel_idx = next(i for i, f in enumerate(sentinel_flags) if f)
    first_sentinel_iteration = int(sorted_records[first_sentinel_idx].get("iteration", first_sentinel_idx))

    # ── Crash onset: first sentinel with near-zero eval_time ─────────────────
    # < 1.0s distinguishes a ConnectionRefusedError (instant) from a slow evaluator
    crash_onset_iteration: Optional[int] = None
    for i in range(n):
        if sentinel_flags[i]:
            et = sorted_records[i].get("eval_time_seconds")
            if et is not None and float(et) < 1.0:
                crash_onset_iteration = int(sorted_records[i].get("iteration", i))
                break

    # ── Eval-time baseline from last N non-sentinel iters before first sentinel
    pre_sentinel_non_sentinel = [
        sorted_records[i]
        for i in range(first_sentinel_idx)
        if not sentinel_flags[i] and sorted_records[i].get("eval_time_seconds") is not None
    ]
    baseline_window = pre_sentinel_non_sentinel[:_BASELINE_WINDOW]
    baseline_eval_time: Optional[float] = None
    if baseline_window:
        times = [float(r["eval_time_seconds"]) for r in baseline_window]
        baseline_eval_time = statistics.median(times)

    # ── Degradation window: high eval_time + real score before first sentinel ─
    degradation_window: Optional[Tuple[int, int]] = None
    eval_time_spike_ratio: Optional[float] = None

    if baseline_eval_time is not None and baseline_eval_time > 0:
        degraded_iters = []
        peak_eval_time = 0.0
        for i in range(first_sentinel_idx):
            if sentinel_flags[i]:
                continue
            et = sorted_records[i].get("eval_time_seconds")
            if et is not None and float(et) > _DEGRADATION_SPIKE_RATIO * baseline_eval_time:
                degraded_iters.append(int(sorted_records[i].get("iteration", i)))
                if float(et) > peak_eval_time:
                    peak_eval_time = float(et)
        if degraded_iters:
            degradation_window = (min(degraded_iters), max(degraded_iters))
            eval_time_spike_ratio = peak_eval_time / baseline_eval_time

    # ── Failure cause classification ──────────────────────────────────────────
    # Is the sentinel block contiguous?
    max_contiguous = max(
        _contiguous_block_length(sentinel_flags, i)
        for i in range(n)
        if sentinel_flags[i]
    )

    if sentinel_count == 0:
        failure_cause = "NONE"
    elif max_contiguous >= _CRASH_COUNT_THRESHOLD:
        # Large contiguous block of sentinels → server crash
        failure_cause = "INFRA_CRASH"
    elif degradation_window is not None:
        # Degradation present but no large crash block
        failure_cause = "INFRA_DEGRADATION"
    else:
        # Isolated sentinels, no degradation pattern
        failure_cause = "EVALUATOR_NOISE"

    return InfrastructureMetrics(
        sentinel_count=sentinel_count,
        sentinel_fraction=sentinel_fraction,
        first_sentinel_iteration=first_sentinel_iteration,
        crash_onset_iteration=crash_onset_iteration,
        degradation_window=degradation_window,
        failure_cause=failure_cause,
        affected_iterations=affected_iterations,
        eval_time_spike_ratio=eval_time_spike_ratio,
        log_evidence=log_signals,
    )
