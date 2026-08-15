"""
evaluator_analyzer.py
---------------------
Functions for classifying per-iteration evaluation outcomes and surfacing
evaluator-level signals: failure modes, sub-metric divergence, cascade stage
breakdowns, and high-variance flags.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Literal, Optional


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

FailureMode = Literal[
    "success",
    "partial",
    "worse",
    "wrong_output",
    "timeout",
    "crash",
    "format_invalid",
]

# Names of sub-metrics that represent correctness / accuracy.
# A value of 0 (or 0.0) in any of these signals a wrong-output failure.
_CORRECTNESS_METRIC_NAMES = frozenset(
    {
        "correctness",
        "accuracy",
        "correct",
        "is_correct",
        "pass_rate",
        "pass",
        "exact_match",
    }
)


# ---------------------------------------------------------------------------
# classify_failure
# ---------------------------------------------------------------------------

def classify_failure(record: dict) -> str:
    """
    Classify a single evaluation record into one of the :data:`FailureMode`
    categories.

    Priority order (first matching rule wins):
    1. "crash"          – evaluation_status == "crash"
    2. "timeout"        – evaluation_status == "timeout"
    3. "format_invalid" – format_valid == False
    4. "wrong_output"   – any correctness/accuracy sub-metric == 0
    5. "success"        – score_delta >= 0.001
    6. "partial"        – at least one sub-metric improved despite combined
                          score not improving
    7. "worse"          – everything else

    Parameters
    ----------
    record:
        A single JSONL record dict.

    Returns
    -------
    str
        One of the :data:`FailureMode` literal values.
    """
    status = record.get("evaluation_status", "success")

    # --- hard failures -------------------------------------------------------
    if status == "crash":
        return "crash"
    if status == "timeout":
        return "timeout"
    if not record.get("format_valid", True):
        return "format_invalid"

    # --- score delta ---------------------------------------------------------
    score_delta = record.get("score_delta")
    if score_delta is None:
        child = record.get("child_score")
        parent = record.get("parent_score")
        if child is not None and parent is not None:
            score_delta = child - parent
        else:
            score_delta = 0.0

    # --- wrong output via sub-metrics ----------------------------------------
    metrics: Optional[dict] = record.get("evaluator_metrics")
    if metrics:
        for metric_name, value in metrics.items():
            if metric_name.lower() in _CORRECTNESS_METRIC_NAMES:
                try:
                    if float(value) == 0.0:
                        return "wrong_output"
                except (TypeError, ValueError):
                    pass

    # --- success / partial / worse -------------------------------------------
    if score_delta >= 0.001:
        return "success"

    # partial: combined score didn't improve, but some sub-metric did
    divergence = detect_sub_metric_divergence(record)
    if divergence:
        return "partial"

    return "worse"


# ---------------------------------------------------------------------------
# detect_sub_metric_divergence
# ---------------------------------------------------------------------------

def detect_sub_metric_divergence(record: dict) -> Optional[Dict[str, float]]:
    """
    Return the sub-metrics that improved even though the combined score did
    not improve (score_delta < 0.001), or ``None`` if no divergence exists.

    Improvement for a sub-metric is defined as: the sub-metric value in the
    child is strictly greater than the corresponding sub-metric value in the
    parent.  We look for per-metric deltas inside ``evaluator_metrics``.  If
    the record stores deltas directly (keys ending in ``_delta``), those are
    used; otherwise we fall back to comparing child and parent metric dicts
    stored inside ``evaluator_artifacts``.

    Parameters
    ----------
    record:
        A single JSONL record dict.

    Returns
    -------
    dict or None
        Mapping of ``{metric_name: delta}`` for metrics that improved, or
        ``None`` if the combined score already improved or no sub-metric
        divergence was detected.
    """
    # Only meaningful when combined score did NOT improve
    score_delta = record.get("score_delta")
    if score_delta is None:
        child = record.get("child_score")
        parent = record.get("parent_score")
        score_delta = (child - parent) if (child is not None and parent is not None) else 0.0

    if score_delta >= 0.001:
        # Combined score already improved — no divergence to report
        return None

    metrics: Optional[dict] = record.get("evaluator_metrics")
    improved: Dict[str, float] = {}

    # Case 1: record stores per-metric deltas directly (keys ending in _delta)
    delta_keys = {k: v for k, v in metrics.items() if k.endswith("_delta")} if metrics else {}
    if delta_keys:
        for key, delta in delta_keys.items():
            try:
                if float(delta) > 0:
                    improved[key] = float(delta)
            except (TypeError, ValueError):
                pass
        return improved or None

    # Case 2: compare child vs parent sub-metric dicts from evaluator_artifacts
    artifacts: Optional[dict] = record.get("evaluator_artifacts")
    if artifacts:
        parent_metrics = artifacts.get("parent_metrics") or {}
        child_metrics = artifacts.get("child_metrics") or {}
        for metric_name, child_val in child_metrics.items():
            parent_val = parent_metrics.get(metric_name)
            if parent_val is not None:
                try:
                    delta = float(child_val) - float(parent_val)
                    if delta > 0:
                        improved[metric_name] = delta
                except (TypeError, ValueError):
                    pass
        return improved or None

    # No sub-metric comparison data available
    return None


# ---------------------------------------------------------------------------
# cascade_stage_summary
# ---------------------------------------------------------------------------

def cascade_stage_summary(records: List[dict]) -> dict:
    """
    Summarise how many failures occurred at each cascade pipeline stage.

    Parameters
    ----------
    records:
        List of raw record dicts.

    Returns
    -------
    dict
        ``{stage_name: {"count": int, "pct": float}}`` where ``pct`` is the
        percentage of *all* records (not just failed ones) that failed at that
        stage.  An additional ``"_total"`` key stores the total record count.
        Only stages that appear in the data are included.
    """
    total = len(records)
    counts: Counter = Counter()

    for rec in records:
        stage = rec.get("cascade_stage_failed")
        if stage:  # None / empty string → not a cascade failure
            counts[stage] += 1

    result: dict = {"_total": total}
    for stage, count in counts.most_common():
        pct = (count / total * 100.0) if total > 0 else 0.0
        result[stage] = {"count": count, "pct": round(pct, 4)}

    return result


# ---------------------------------------------------------------------------
# flag_high_variance
# ---------------------------------------------------------------------------

def flag_high_variance(record: dict, std_threshold: float = 0.1) -> bool:
    """
    Return ``True`` when the standard deviation of evaluation scores across
    multiple runs exceeds *std_threshold*.

    Parameters
    ----------
    record:
        A single JSONL record dict.
    std_threshold:
        Maximum acceptable standard deviation.  Default: ``0.1``.

    Returns
    -------
    bool
    """
    score_std = record.get("score_std")
    if score_std is None:
        return False
    try:
        return float(score_std) > std_threshold
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# analyze_evaluator
# ---------------------------------------------------------------------------

def analyze_evaluator(
    records: List[dict],
    std_threshold: float = 0.1,
) -> List[dict]:
    """
    Tag every record with ``failure_mode`` and ``high_variance`` fields.

    The function modifies each record dict **in-place** and also returns the
    list for convenience.

    Parameters
    ----------
    records:
        List of raw record dicts (order is preserved).
    std_threshold:
        Threshold passed to :func:`flag_high_variance`.

    Returns
    -------
    List[dict]
        The same list with ``failure_mode`` and ``high_variance`` added to
        each element.
    """
    for rec in records:
        rec["failure_mode"] = classify_failure(rec)
        rec["high_variance"] = flag_high_variance(rec, std_threshold)
    return records
