"""
Meta-analysis quality analyzer: evaluates how well the LLM's own reasoning
traces and meta-suggestions contributed to the optimization process.
"""
from __future__ import annotations

import difflib
from typing import List, Optional

from skydiscover.extras.evolve_analyzer.quantitative.bundle import MetaAnalysisMetrics

_NULL_RESULT = MetaAnalysisMetrics(
    suggestion_follow_rate=None,
    conditional_improvement_rate=None,
    pattern_reuse_frequency=None,
    scratchpad_growth_rate=None,
    compaction_events=None,
)


def analyze_meta_quality(records: List[dict]) -> MetaAnalysisMetrics:
    """Compute meta-analysis quality metrics from iteration records."""
    if not records:
        return _NULL_RESULT

    # Determine whether the required fields exist anywhere.
    has_reasoning = any(
        r.get("reasoning_trace") is not None for r in records
    )
    has_suggestion = any(
        r.get("meta_suggestion") is not None for r in records
    )

    if not has_reasoning and not has_suggestion:
        return _NULL_RESULT

    # ── suggestion_follow_rate ────────────────────────────────────────────────
    suggestion_follow_rate: Optional[float] = None
    records_with_suggestion = [
        r for r in records if r.get("meta_suggestion") is not None
    ]
    if records_with_suggestion:
        followed_count = sum(
            1
            for r in records_with_suggestion
            if r.get("followed_suggestion") is True
        )
        suggestion_follow_rate = followed_count / len(records_with_suggestion)

    # ── conditional_improvement_rate ─────────────────────────────────────────
    conditional_improvement_rate: Optional[float] = None
    followed_records = [
        r
        for r in records_with_suggestion
        if r.get("followed_suggestion") is True
    ]
    if followed_records:
        improved_count = sum(
            1
            for r in followed_records
            if (r.get("score_delta") or 0.0) > 0.001
        )
        conditional_improvement_rate = improved_count / len(followed_records)

    # ── Reasoning-trace metrics ───────────────────────────────────────────────
    # Collect (iteration, trace) pairs for records that have a reasoning trace.
    sorted_records = sorted(records, key=lambda r: r.get("iteration", 0))
    trace_pairs = [
        (r.get("iteration", idx), r["reasoning_trace"])
        for idx, r in enumerate(sorted_records)
        if r.get("reasoning_trace") is not None
    ]

    # pattern_reuse_frequency
    pattern_reuse_frequency: Optional[float] = None
    # scratchpad_growth_rate
    scratchpad_growth_rate: Optional[float] = None
    # compaction_events
    compaction_events: Optional[int] = None

    if len(trace_pairs) >= 2:
        traces = [t for _, t in trace_pairs]
        lengths = [len(t) for t in traces]

        # pattern_reuse_frequency: fraction of consecutive pairs where
        # SequenceMatcher ratio > 0.5 (indicating heavy phrase reuse).
        reuse_count = 0
        for i in range(1, len(traces)):
            ratio = difflib.SequenceMatcher(
                None, traces[i - 1], traces[i], autojunk=False
            ).ratio()
            if ratio > 0.5:
                reuse_count += 1
        pattern_reuse_frequency = reuse_count / (len(traces) - 1)

        # scratchpad_growth_rate: mean per-iteration increase in trace length.
        length_deltas = [
            lengths[i] - lengths[i - 1] for i in range(1, len(lengths))
        ]
        scratchpad_growth_rate = sum(length_deltas) / len(length_deltas)

        # compaction_events: consecutive pairs where length dropped > 50%.
        compaction_events = sum(
            1
            for i in range(1, len(lengths))
            if lengths[i - 1] > 0
            and lengths[i] < lengths[i - 1] * 0.5
        )

    elif len(trace_pairs) == 1:
        # Only one trace — we can still report zero reuse / compaction events.
        pattern_reuse_frequency = 0.0
        scratchpad_growth_rate = None
        compaction_events = 0

    return MetaAnalysisMetrics(
        suggestion_follow_rate=suggestion_follow_rate,
        conditional_improvement_rate=conditional_improvement_rate,
        pattern_reuse_frequency=pattern_reuse_frequency,
        scratchpad_growth_rate=scratchpad_growth_rate,
        compaction_events=compaction_events,
    )
