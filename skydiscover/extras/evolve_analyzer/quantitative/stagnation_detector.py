"""
stagnation_detector.py
----------------------
Detects stagnation streaks in evolutionary optimization runs.

A stagnation "streak" is a run of consecutive iterations with no progress.
Progress is defined as: score_delta >= min_delta AND format_valid == True
AND evaluation_status != "crash".
"""
from __future__ import annotations

from collections import Counter
from typing import List, Optional, Tuple

from .bundle import IterationSummary, StagnationPeriod


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _has_progress(record: dict, min_delta: float) -> bool:
    """Return True if this record counts as making progress."""
    score_delta = record.get("score_delta")
    if score_delta is None:
        child = record.get("child_score")
        parent = record.get("parent_score")
        if child is not None and parent is not None:
            score_delta = child - parent
        else:
            score_delta = 0.0

    format_valid = record.get("format_valid", True)
    status = record.get("evaluation_status", "success")

    return (
        score_delta >= min_delta
        and bool(format_valid)
        and status != "crash"
    )


def _classify_failure(record: dict, min_delta: float) -> str:
    """
    Inline failure classifier (mirrors evaluator_analyzer logic without import).

    Categories used by stagnation: "crash", "timeout", "format_invalid",
    "no_improvement".
    """
    status = record.get("evaluation_status", "success")
    if status == "crash":
        return "crash"
    if status == "timeout":
        return "timeout"
    if not record.get("format_valid", True):
        return "format_invalid"
    return "no_improvement"


def _dominant_and_secondary(failure_types: List[str]) -> Tuple[str, List[str]]:
    """Return (most_common_type, list_of_all_other_types_present)."""
    if not failure_types:
        return "no_improvement", []
    counts = Counter(failure_types)
    dominant = counts.most_common(1)[0][0]
    secondary = sorted(
        {ft for ft in counts if ft != dominant},
        key=lambda ft: -counts[ft],
    )
    return dominant, secondary


def _compute_severity(
    length: int,
    dominant_failure_type: str,
    threshold: int,
) -> str:
    """
    Severity assignment per design doc:
      - length >= 2*threshold                                     → "critical"
      - dominant in ("identical_output", "compliance_violation",
                     "crash", "format_invalid")
        and length >= threshold                                   → "critical"
      - dominant in above set and length < threshold              → "high"
      - otherwise                                                 → "warning"
    """
    if length >= 2 * threshold:
        return "critical"
    if dominant_failure_type in (
        "identical_output", "compliance_violation", "crash", "format_invalid"
    ):
        return "critical" if length >= threshold else "high"
    return "warning"


def _make_iteration_summary(record: dict, failure_type: str) -> IterationSummary:
    """Build an IterationSummary from a raw record dict."""
    score_delta = record.get("score_delta")
    if score_delta is None:
        child = record.get("child_score")
        parent = record.get("parent_score")
        score_delta = (child - parent) if (child is not None and parent is not None) else 0.0

    # compliance_status: synthesise a simple string from available fields
    compliance_status: Optional[str] = None
    if "evolved_block_only" in record or "signature_preserved" in record:
        parts = []
        if "evolved_block_only" in record:
            parts.append("block_ok" if record["evolved_block_only"] else "block_violation")
        if "signature_preserved" in record:
            parts.append("sig_ok" if record["signature_preserved"] else "sig_violation")
        compliance_status = "|".join(parts)

    crash_error: Optional[str] = None
    if failure_type == "crash":
        crash_error = record.get("error")
        if crash_error is None:
            artifacts = record.get("evaluator_artifacts") or {}
            if isinstance(artifacts, dict):
                crash_error = artifacts.get("stderr") or artifacts.get("error")

    island_id = record.get("island_id")
    return IterationSummary(
        iteration=record.get("iteration", 0),
        mutation_type=record.get("mutation_type") or "unknown",
        failure_type=failure_type,
        score_delta=float(score_delta),
        compliance_status=compliance_status,
        format_valid=bool(record.get("format_valid", True)),
        crash_error=crash_error,
        island_id=str(island_id) if island_id is not None else None,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_stagnation(
    records: List[dict],
    threshold: int = 10,
    min_delta: float = 0.001,
) -> Tuple[List[dict], List[StagnationPeriod]]:
    """
    Detect stagnation streaks across a list of JSONL records.

    Parameters
    ----------
    records:
        Raw record dicts loaded from a JSONL log file.
    threshold:
        Minimum streak length that sets ``is_alert = True``.
    min_delta:
        Minimum score improvement required to count as progress.

    Returns
    -------
    tagged_records:
        Same records, each enriched with ``streak_id`` (str | None) and
        ``streak_position`` (int, 1-indexed within streak, 0 if not in one).
    stagnation_periods:
        One :class:`StagnationPeriod` per completed streak (length >= 1).
    """
    # Work on copies so we don't mutate caller's dicts in place unexpectedly
    sorted_records = sorted(records, key=lambda r: r.get("iteration", 0))
    tagged: List[dict] = [dict(r) for r in sorted_records]

    stagnation_periods: List[StagnationPeriod] = []

    # Initialise all records as not in a streak
    for rec in tagged:
        rec["streak_id"] = None
        rec["streak_position"] = 0

    # --- Single-pass streak detection ---
    current_streak_start_idx: Optional[int] = None
    current_streak_records: List[int] = []   # indices into `tagged`

    def _close_streak(end_idx: int, recovery_idx: Optional[int]) -> None:
        """Finalise the current streak and append a StagnationPeriod."""
        nonlocal current_streak_start_idx, current_streak_records
        if not current_streak_records:
            return

        streak_indices = current_streak_records
        start_iter = tagged[streak_indices[0]].get("iteration", 0)
        end_iter = tagged[streak_indices[-1]].get("iteration", 0)
        streak_id = f"streak_{start_iter}"
        length = len(streak_indices)

        # Tag each record in the streak
        for pos, idx in enumerate(streak_indices, start=1):
            tagged[idx]["streak_id"] = streak_id
            tagged[idx]["streak_position"] = pos

        # Build failure sequence
        failure_types: List[str] = []
        failure_sequence: List[IterationSummary] = []
        for idx in streak_indices:
            rec = tagged[idx]
            ft = _classify_failure(rec, min_delta)
            failure_types.append(ft)
            failure_sequence.append(_make_iteration_summary(rec, ft))

        dominant, secondary = _dominant_and_secondary(failure_types)
        severity = _compute_severity(length, dominant, threshold)
        is_alert = length >= threshold

        # Score at start of stagnation
        score_at_stagnation_start: float = 0.0
        first_rec = tagged[streak_indices[0]]
        if first_rec.get("parent_score") is not None:
            score_at_stagnation_start = float(first_rec["parent_score"])
        elif first_rec.get("child_score") is not None:
            score_at_stagnation_start = float(first_rec["child_score"])

        # Recovery info
        recovery_iteration: Optional[int] = None
        recovery_mutation_type: Optional[str] = None
        recovery_model: Optional[str] = None
        score_at_recovery: Optional[float] = None

        if recovery_idx is not None and recovery_idx < len(tagged):
            rec_r = tagged[recovery_idx]
            recovery_iteration = rec_r.get("iteration")
            recovery_mutation_type = rec_r.get("mutation_type")
            recovery_model = rec_r.get("model")
            cs = rec_r.get("child_score")
            score_at_recovery = float(cs) if cs is not None else None

        stagnation_periods.append(
            StagnationPeriod(
                streak_id=streak_id,
                start_iteration=start_iter,
                end_iteration=end_iter,
                length=length,
                failure_sequence=failure_sequence,
                dominant_failure_type=dominant,
                secondary_failure_types=secondary,
                is_alert=is_alert,
                severity=severity,
                recovery_iteration=recovery_iteration,
                recovery_mutation_type=recovery_mutation_type,
                recovery_model=recovery_model,
                score_at_stagnation_start=score_at_stagnation_start,
                score_at_recovery=score_at_recovery,
            )
        )

        # Reset state
        current_streak_start_idx = None
        current_streak_records = []

    for idx, rec in enumerate(tagged):
        progress = _has_progress(rec, min_delta)

        if not progress:
            # Extend (or start) the current streak
            if current_streak_start_idx is None:
                current_streak_start_idx = idx
            current_streak_records.append(idx)
        else:
            # This record makes progress → close any open streak
            if current_streak_records:
                _close_streak(end_idx=idx - 1, recovery_idx=idx)
            # The current record is NOT tagged as part of any streak
            # (streak_id = None, streak_position = 0 already set above)

    # Close any streak still open at end of list (no recovery found)
    if current_streak_records:
        _close_streak(end_idx=len(tagged) - 1, recovery_idx=None)
        # Mark end_iteration as None (still ongoing) for the last period
        if stagnation_periods:
            last = stagnation_periods[-1]
            stagnation_periods[-1] = StagnationPeriod(
                streak_id=last.streak_id,
                start_iteration=last.start_iteration,
                end_iteration=None,
                length=last.length,
                failure_sequence=last.failure_sequence,
                dominant_failure_type=last.dominant_failure_type,
                secondary_failure_types=last.secondary_failure_types,
                is_alert=last.is_alert,
                severity=last.severity,
                recovery_iteration=None,
                recovery_mutation_type=None,
                recovery_model=None,
                score_at_stagnation_start=last.score_at_stagnation_start,
                score_at_recovery=None,
            )

    return tagged, stagnation_periods
