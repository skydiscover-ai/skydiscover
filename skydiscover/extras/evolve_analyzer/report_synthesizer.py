"""
report_synthesizer.py
---------------------
Synthesizes quantitative + qualitative + historical results into a structured
EvolveLoopReport.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from skydiscover.extras.evolve_analyzer.qualitative.qualitative_analyzer import (
    SKIP_REASON_FULL_REWRITE,
    SKIP_REASON_NO_TRACE,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AggregateStats:
    total_iterations: int
    n_evaluations: int
    n_successful: int
    n_stagnation_periods: int
    total_stagnation_iterations: int
    best_score: float
    worst_score: float
    final_score: float
    total_llm_cost_usd: Optional[float]
    total_duration_hours: Optional[float]
    # External baseline (optional — populated from --baseline-score / config)
    baseline_score: Optional[float] = None
    baseline_metrics: Optional[Dict[str, float]] = None
    score_improvement_vs_baseline: Optional[float] = None


@dataclass
class StagnationPeriodReport:
    streak_id: str
    start_iteration: int
    end_iteration: Optional[int]
    length: int
    severity: str
    dominant_failure_type: str
    llm_analysis: Optional[dict]   # from qualitative Step A
    recommendation: str
    crash_samples: Optional[List[Dict[str, Any]]] = None  # [{iteration, error}] when dominant=crash


@dataclass
class DimensionReport:
    name: str
    rating: Optional[int]   # 1–5, or None when data_available=False
    rating_label: str        # "🔴 Critical" | "🟠 Poor" | "🟡 Fair" | "🟢 Good" | "✅ Excellent" | "N/A"
    summary: str
    evidence: List[str]
    historical: Optional[Any]   # HistoricalComparison
    recommendation: str
    data_available: bool


@dataclass
class LLMJudgeStatus:
    provider: str
    model: str
    base_url: Optional[str]
    status: str                     # "success" | "skipped" | "failed"
    skip_reason: Optional[str] = None   # populated when status == "skipped"
    error: Optional[str] = None         # populated when status == "failed"


@dataclass
class EvolveLoopReport:
    experiment_id: str
    executive_summary: str
    executive_summary_md: str
    dimensions: List[DimensionReport]
    cross_dimension_interactions: str
    stagnation_periods: List[StagnationPeriodReport]
    aggregate_stats: AggregateStats
    novel_observations: List[str]
    llm_judge_status: Optional[LLMJudgeStatus] = None
    run_source: Optional[str] = None
    run_path: Optional[str] = None
    run_config_path: Optional[str] = None
    algorithm_class: str = "population_evolutionary"
    algorithm_name: Optional[str] = None
    num_islands: Optional[int] = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RATING_LABELS = {
    1: "🔴 Critical",
    2: "🟠 Poor",
    3: "🟡 Fair",
    4: "🟢 Good",
    5: "✅ Excellent",
}


# ---------------------------------------------------------------------------
# Default rating thresholds (used when no algorithm_class config is present)
# ---------------------------------------------------------------------------

_DEFAULT_REGRESSION_THRESHOLDS: List[float] = [0.05, 0.15, 0.30, 0.50]
_DEFAULT_EXPLORATION_THRESHOLDS: List[float] = [0.70, 0.50, 0.30, 0.10]
# convergence: [t5, t4] for time_to_best (higher=better); [t1, t2, t3] for plateau_fraction (lower=worse)
_DEFAULT_CONVERGENCE_TTB_THRESHOLDS: List[float] = [0.8, 0.6]
_DEFAULT_CONVERGENCE_PLATEAU_THRESHOLDS: List[float] = [0.20, 0.40, 0.60]


def _rate_by_thresholds(
    value: float,
    thresholds: List[float],
    lower_is_better: bool,
) -> int:
    """Map a metric value to a 1–5 rating using a 4-element threshold list.

    thresholds = [t5, t4, t3, t2]

    lower_is_better=True : value < t5 → 5, value < t4 → 4, …, else → 1
    lower_is_better=False: value > t5 → 5, value >= t4 → 4, …, else → 1
    """
    t5, t4, t3, t2 = thresholds
    if lower_is_better:
        if value < t5:
            return 5
        if value < t4:
            return 4
        if value < t3:
            return 3
        if value < t2:
            return 2
        return 1
    else:
        if value > t5:
            return 5
        if value >= t4:
            return 4
        if value >= t3:
            return 3
        if value >= t2:
            return 2
        return 1


def _build_rating_context(algorithm_class: str, config: dict) -> dict:
    """Resolve per-algorithm-class rating thresholds from config.

    Returns a dict with keys:
      algorithm_class, regression_frequency_thresholds, exploration_sdi_thresholds,
      convergence_ttb_thresholds, convergence_plateau_thresholds
    Falls back to package defaults when the class is absent from config.
    """
    cls_cfg = (config or {}).get("algorithm_classes", {}).get(algorithm_class, {})
    conv_cfg = cls_cfg.get("convergence_thresholds", {})
    return {
        "algorithm_class": algorithm_class,
        "regression_frequency_thresholds": cls_cfg.get(
            "regression_frequency_thresholds", list(_DEFAULT_REGRESSION_THRESHOLDS)
        ),
        "exploration_sdi_thresholds": cls_cfg.get(
            "exploration_sdi_thresholds", list(_DEFAULT_EXPLORATION_THRESHOLDS)
        ),
        "convergence_ttb_thresholds": conv_cfg.get(
            "time_to_best_thresholds", list(_DEFAULT_CONVERGENCE_TTB_THRESHOLDS)
        ),
        "convergence_plateau_thresholds": conv_cfg.get(
            "plateau_fraction_thresholds", list(_DEFAULT_CONVERGENCE_PLATEAU_THRESHOLDS)
        ),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rating_label(rating: Optional[int]) -> str:
    if rating is None:
        return "N/A"
    return _RATING_LABELS.get(rating, "🟡 Fair")


def _find_historical(
    historical: List[Any],
    *keywords: str,
) -> Optional[Any]:
    """Find a HistoricalComparison whose metric_name contains any of the keywords."""
    for h in historical:
        mn = getattr(h, "metric_name", "").lower()
        for kw in keywords:
            if kw.lower() in mn:
                return h
    return None


def _compute_aggregate_stats(df: pd.DataFrame) -> AggregateStats:
    """Derive AggregateStats from the tagged records DataFrame."""
    total_iterations = len(df)

    # n_evaluations: rows whose evaluation_status is not None / not 'skipped'
    if "evaluation_status" in df.columns:
        n_evaluations = int(df["evaluation_status"].notna().sum())
    else:
        n_evaluations = total_iterations

    # n_successful: evaluation_status == 'success'
    if "evaluation_status" in df.columns:
        n_successful = int((df["evaluation_status"] == "success").sum())
    elif "child_score" in df.columns:
        n_successful = int(df["child_score"].notna().sum())
    else:
        n_successful = total_iterations

    # Stagnation info
    if "streak_id" in df.columns:
        n_stagnation_periods = int(df["streak_id"].nunique())
        total_stagnation_iterations = int(df["streak_id"].notna().sum())
    else:
        n_stagnation_periods = 0
        total_stagnation_iterations = 0

    # Score statistics
    score_col = "child_score" if "child_score" in df.columns else None
    if score_col is not None:
        valid_scores = df[score_col].dropna()
        best_score = float(valid_scores.max()) if len(valid_scores) else float("nan")
        worst_score = float(valid_scores.min()) if len(valid_scores) else float("nan")
        final_score = float(df[score_col].iloc[-1]) if len(df) else float("nan")
    else:
        best_score = worst_score = final_score = float("nan")

    # LLM cost
    total_llm_cost_usd: Optional[float] = None
    if "llm_cost_usd" in df.columns:
        costs = df["llm_cost_usd"].dropna()
        if len(costs):
            total_llm_cost_usd = float(costs.sum())

    # Duration
    total_duration_hours: Optional[float] = None
    if "timestamp" in df.columns:
        timestamps = df["timestamp"].dropna()
        if len(timestamps) >= 2:
            duration_s = float(timestamps.max() - timestamps.min())
            total_duration_hours = duration_s / 3600.0

    return AggregateStats(
        total_iterations=total_iterations,
        n_evaluations=n_evaluations,
        n_successful=n_successful,
        n_stagnation_periods=n_stagnation_periods,
        total_stagnation_iterations=total_stagnation_iterations,
        best_score=best_score,
        worst_score=worst_score,
        final_score=final_score,
        total_llm_cost_usd=total_llm_cost_usd,
        total_duration_hours=total_duration_hours,
    )


# ---------------------------------------------------------------------------
# Dimension builders
# ---------------------------------------------------------------------------

def _build_convergence_dimension(
    quant: Any,
    historical: List[Any],
    rating_context: Optional[dict] = None,
) -> DimensionReport:
    conv = getattr(quant, "convergence", None)
    if conv is None:
        return DimensionReport(
            name="Convergence",
            rating=None,
            rating_label="N/A",
            summary="No convergence data available.",
            evidence=[],
            historical=_find_historical(historical, "convergence"),
            recommendation="Ensure convergence analyzer is enabled.",
            data_available=False,
        )

    ttbf = conv.time_to_best_fraction  # lower = better
    poi = conv.plateau_onset_iteration  # None means no plateau
    n = len(conv.best_so_far_curve) if conv.best_so_far_curve else 1

    if rating_context:
        ttb_thresholds = rating_context["convergence_ttb_thresholds"]
        plateau_thresholds = rating_context["convergence_plateau_thresholds"]
    else:
        ttb_thresholds = _DEFAULT_CONVERGENCE_TTB_THRESHOLDS
        plateau_thresholds = _DEFAULT_CONVERGENCE_PLATEAU_THRESHOLDS

    plateau_fraction = (poi / n) if (poi is not None and n > 0) else None

    if poi is None:
        if ttbf >= ttb_thresholds[0]:
            rating = 5
        elif ttbf >= ttb_thresholds[1]:
            rating = 4
        else:
            rating = 3
    else:
        if plateau_fraction is not None and plateau_fraction <= plateau_thresholds[0]:
            rating = 1
        elif plateau_fraction is not None and plateau_fraction <= plateau_thresholds[1]:
            rating = 2
        elif plateau_fraction is not None and plateau_fraction <= plateau_thresholds[2]:
            rating = 3
        else:
            rating = 4

    evidence = []
    if conv.best_so_far_curve:
        curve = conv.best_so_far_curve
        waypoints = [0, len(curve) // 4, len(curve) // 2, 3 * len(curve) // 4, len(curve) - 1]
        for wp in waypoints:
            if 0 <= wp < len(curve):
                evidence.append(f"Iteration {wp}: best={curve[wp]:.4g}")
    if poi is not None:
        evidence.append(f"Plateau onset at iteration {poi} ({plateau_fraction:.1%} into run)")
    evidence.append(f"Convergence rate: {conv.convergence_rate:.4g} per iteration")
    evidence.append(f"Time to best: {ttbf:.1%} of run")
    if rating_context and (
        ttb_thresholds != _DEFAULT_CONVERGENCE_TTB_THRESHOLDS
        or plateau_thresholds != _DEFAULT_CONVERGENCE_PLATEAU_THRESHOLDS
    ):
        algo_class = rating_context.get("algorithm_class", "")
        evidence.append(
            f"Note: convergence thresholds adjusted for {algo_class} "
            f"(excellent time-to-best >= {ttb_thresholds[0]:.0%} vs "
            f"{_DEFAULT_CONVERGENCE_TTB_THRESHOLDS[0]:.0%} default)."
        )

    summaries = {
        5: "Excellent convergence — best score found late in the run with no plateau.",
        4: "Good convergence — steady improvement, late plateau or none.",
        3: "Fair convergence — moderate improvement trajectory.",
        2: "Poor convergence — plateau formed early in the run.",
        1: "Critical — plateau emerged in the first 20% of the run.",
    }
    recs = {
        5: "Convergence looks healthy; consider extending the run to find further gains.",
        4: "Good trajectory; monitor for early stopping opportunities.",
        3: "Consider adding diversity mechanisms to avoid premature convergence.",
        2: "Plateau onset is early; diversify mutation strategies or increase exploration.",
        1: "Immediate action: plateau in first 20% suggests the run stalled very quickly.",
    }

    return DimensionReport(
        name="Convergence",
        rating=rating,
        rating_label=_rating_label(rating),
        summary=summaries[rating],
        evidence=evidence,
        historical=_find_historical(historical, "convergence"),
        recommendation=recs[rating],
        data_available=True,
    )


def _build_stagnation_dimension(
    quant: Any,
    historical: List[Any],
    num_islands: Optional[int] = None,
) -> DimensionReport:
    periods = getattr(quant, "stagnation_periods", None) or []
    alert_periods = [p for p in periods if p.is_alert]
    n_alerts = len(alert_periods)
    max_len = max((p.length for p in periods), default=0)
    has_critical = any(p.severity == "critical" for p in periods)

    # Rating
    if has_critical:
        rating = 1
    elif n_alerts == 0:
        rating = 5
    elif n_alerts == 1 and max_len < 15:
        rating = 4
    elif n_alerts <= 2:
        rating = 3
    else:
        rating = 2

    show_islands = num_islands is not None and num_islands > 1

    evidence = []
    for p in alert_periods:
        end_str = str(p.end_iteration) if p.end_iteration is not None else "ongoing"
        island_part = ""
        if show_islands:
            islands = sorted({
                s.island_id for s in p.failure_sequence if s.island_id is not None
            })
            if islands:
                island_part = f", islands={islands}"
        evidence.append(
            f"Streak {p.streak_id}: iters {p.start_iteration}–{end_str}, "
            f"length={p.length}, severity={p.severity}, "
            f"dominant_failure={p.dominant_failure_type}{island_part}"
        )
    non_alert_count = len(periods) - len(alert_periods)
    if non_alert_count > 0:
        evidence.append(f"({non_alert_count} additional non-alert streaks not shown)")
    if not evidence:
        evidence.append("No stagnation periods detected.")

    summaries = {
        5: "No alert-level stagnation detected — search progressed smoothly.",
        4: "One brief stagnation episode; overall healthy progression.",
        3: "One or two stagnation periods; moderate concern.",
        2: "Multiple stagnation periods detected; search efficiency reduced.",
        1: "Critical stagnation — at least one period classified as critical severity.",
    }
    recs = {
        5: "Maintain current mutation strategy; stagnation is not a problem.",
        4: "Monitor for recurrence; consider adding recovery heuristics.",
        3: "Review failure types in stagnation periods and diversify mutation mix.",
        2: "Redesign recovery mechanisms; stagnation is recurring and impacting quality.",
        1: "Immediate intervention: critical stagnation suggests systemic failure mode.",
    }

    return DimensionReport(
        name="Stagnation",
        rating=rating,
        rating_label=_rating_label(rating),
        summary=summaries[rating],
        evidence=evidence,
        historical=_find_historical(historical, "stagnation"),
        recommendation=recs[rating],
        data_available=True,
    )


def _build_regression_dimension(
    quant: Any,
    historical: List[Any],
    rating_context: Optional[dict] = None,
) -> DimensionReport:
    reg = getattr(quant, "regression", None)
    if reg is None:
        return DimensionReport(
            name="Regression",
            rating=None,
            rating_label="N/A",
            summary="No regression data available.",
            evidence=[],
            historical=_find_historical(historical, "regression"),
            recommendation="Ensure regression analyzer is enabled.",
            data_available=False,
        )

    freq = reg.regression_frequency

    if rating_context:
        thresholds = rating_context["regression_frequency_thresholds"]
        rating = _rate_by_thresholds(freq, thresholds, lower_is_better=True)
    else:
        thresholds = _DEFAULT_REGRESSION_THRESHOLDS
        rating = _rate_by_thresholds(freq, thresholds, lower_is_better=True)

    evidence = [
        f"Regression frequency: {freq:.1%}",
        f"Mean recovery time: {reg.mean_recovery_time:.1f} iterations",
        f"Severity distribution: mild={reg.severity_distribution.get('mild', 0)}, "
        f"moderate={reg.severity_distribution.get('moderate', 0)}, "
        f"severe={reg.severity_distribution.get('severe', 0)}",
    ]
    if rating_context and thresholds != _DEFAULT_REGRESSION_THRESHOLDS:
        algo_class = rating_context.get("algorithm_class", "")
        evidence.append(
            f"Note: regression thresholds adjusted for {algo_class} "
            f"(expected range {thresholds[0]:.0%}–{thresholds[1]:.0%} vs "
            f"{_DEFAULT_REGRESSION_THRESHOLDS[0]:.0%}–{_DEFAULT_REGRESSION_THRESHOLDS[1]:.0%} default)."
        )
    if reg.death_spiral_periods:
        for start, end in reg.death_spiral_periods:
            evidence.append(f"Death spiral: iterations {start}–{end}")

    summaries = {
        5: "Excellent — regressions are rare (< 5% of iterations).",
        4: "Good — low regression rate (5–15%).",
        3: "Fair — moderate regression frequency (15–30%).",
        2: "Poor — high regression rate (30–50%) reducing search efficiency.",
        1: "Critical — over half of iterations produce regressions.",
    }
    recs = {
        5: "Regression rate is healthy; keep current acceptance criteria.",
        4: "Low regression rate; minor tuning of mutation strength may help further.",
        3: "Review mutation operators causing regressions; consider tighter acceptance.",
        2: "Add regression guard mechanisms or reduce mutation aggressiveness.",
        1: "Fundamental issue: dominant strategy produces consistent regressions.",
    }

    return DimensionReport(
        name="Regression",
        rating=rating,
        rating_label=_rating_label(rating),
        summary=summaries[rating],
        evidence=evidence,
        historical=_find_historical(historical, "regression"),
        recommendation=recs[rating],
        data_available=True,
    )


def _build_efficiency_dimension(
    quant: Any,
    historical: List[Any],
) -> DimensionReport:
    eff = getattr(quant, "efficiency", None)
    if eff is None:
        return DimensionReport(
            name="Efficiency",
            rating=None,
            rating_label="N/A",
            summary="No efficiency data available.",
            evidence=[],
            historical=_find_historical(historical, "efficiency"),
            recommendation="Ensure efficiency analyzer is enabled.",
            data_available=False,
        )

    wasted = eff.wasted_phase_fraction

    if wasted < 0.10:
        rating = 5
    elif wasted < 0.25:
        rating = 4
    elif wasted < 0.50:
        rating = 3
    elif wasted < 0.75:
        rating = 2
    else:
        rating = 1

    evidence = [
        f"Wasted phase fraction: {wasted:.1%}",
        f"Productive phase fraction: {eff.productive_phase_fraction:.1%}",
        f"Improvement per LLM call: {eff.improvement_per_llm_call:.4g}",
    ]
    if eff.improvement_per_dollar is not None:
        evidence.append(f"Improvement per dollar: {eff.improvement_per_dollar:.4g}")
    if eff.improvement_per_hour is not None:
        evidence.append(f"Improvement per hour: {eff.improvement_per_hour:.4g}")

    summaries = {
        5: "Excellent efficiency — nearly all compute produced useful improvements.",
        4: "Good efficiency — minimal wasted iterations.",
        3: "Fair efficiency — notable wasted phase but majority is productive.",
        2: "Poor efficiency — over half of iterations are in the wasted phase.",
        1: "Critical — almost all compute is wasted in a post-plateau tail.",
    }
    recs = {
        5: "Consider applying early-stopping to save cost on future runs.",
        4: "Good efficiency; slight early-stop tuning could trim the wasted tail.",
        3: "Implement early-stop signaling to reduce wasted post-plateau iterations.",
        2: "Strong early-stop criteria recommended; large wasted tail detected.",
        1: "Stop run or restart: the vast majority of the budget is being wasted.",
    }

    return DimensionReport(
        name="Efficiency",
        rating=rating,
        rating_label=_rating_label(rating),
        summary=summaries[rating],
        evidence=evidence,
        historical=_find_historical(historical, "efficiency"),
        recommendation=recs[rating],
        data_available=True,
    )


def _build_exploration_qual_evidence(qual: Any) -> List[str]:
    """Aggregate qualitative judge results (C, D, E) into evidence bullet strings."""
    evidence: List[str] = []
    if qual is None:
        return evidence

    # Step C: mutation quality
    mq = getattr(qual, "mutation_quality", None) or []
    rated = [r for r in mq if r.get("quality_rating", 0) > 0]
    if rated:
        mean_q = sum(r["quality_rating"] for r in rated) / len(rated)
        dist: dict = {}
        for r in rated:
            k = r["quality_rating"]
            dist[k] = dist.get(k, 0) + 1
        dist_str = "  ".join(f"{k}★×{v}" for k, v in sorted(dist.items()))
        evidence.append(f"Mutation quality (LLM judge, n={len(rated)}): mean={mean_q:.1f}/5  [{dist_str}]")

    # Step D: semantic compliance
    sc = getattr(qual, "semantic_compliance", None) or []
    if sc:
        counts: dict = {}
        non_fully_compliant: list = []
        for r in sc:
            lvl = r.get("compliance_level", "unknown")
            counts[lvl] = counts.get(lvl, 0) + 1
            if lvl != "fully_compliant":
                non_fully_compliant.append(r)
        total = len(sc)
        comp_str = "  ".join(f"{lvl}={v}/{total}" for lvl, v in sorted(counts.items()))
        evidence.append(f"Semantic compliance (LLM judge, n={total}): {comp_str}")
        # Group violations by description across iterations
        violation_groups: dict = {}  # desc -> {sev, fix, iterations[]}
        for r in non_fully_compliant:
            iteration = r.get("iteration", "?")
            for v in r.get("violations", []):
                if isinstance(v, dict):
                    desc = v.get("description", "")
                    sev = v.get("severity", "")
                    fix = v.get("fix_suggestion", "")
                    key = desc or str(v)
                    if key not in violation_groups:
                        violation_groups[key] = {"sev": sev, "fix": fix, "iters": []}
                    violation_groups[key]["iters"].append(iteration)
                else:
                    key = str(v)
                    if key not in violation_groups:
                        violation_groups[key] = {"sev": "", "fix": "", "iters": []}
                    violation_groups[key]["iters"].append(iteration)

        for desc, info in violation_groups.items():
            iters = info["iters"]
            freq = f"×{len(iters)}" if len(iters) > 1 else f"iter {iters[0]}"
            sev = info["sev"]
            fix = info["fix"]
            evidence.append(f"  • [{sev}] ({freq}) {desc}")
            if fix:
                evidence.append(f"    Fix: {fix}")

    # Step E: exploration structure
    es = getattr(qual, "exploration_structure", None) or {}
    if es.get("skipped_reason") == SKIP_REASON_FULL_REWRITE:
        evidence.append(
            "Exploration structure judge (E) not run — experiment used full-code rewrites; "
            "no diffs are available for diversity analysis. To enable this judge, log code diffs "
            "between parent and child programs."
        )
    elif es.get("diversity_assessment"):
        evidence.append(f"Diversity assessment (LLM judge): {es['diversity_assessment']}")
        if es.get("dominant_strategy"):
            evidence.append(f"Dominant strategy (LLM judge): {es['dominant_strategy']}")

    return evidence


def _build_exploration_dimension(
    quant: Any,
    historical: List[Any],
    qual: Any = None,
    rating_context: Optional[dict] = None,
    search_space: Any = None,
) -> DimensionReport:
    expl = getattr(quant, "exploration", None)
    if expl is None:
        return DimensionReport(
            name="Exploration",
            rating=None,
            rating_label="N/A",
            summary="No exploration data available.",
            evidence=_build_exploration_qual_evidence(qual),
            historical=_find_historical(historical, "exploration"),
            recommendation="Ensure exploration analyzer is enabled.",
            data_available=False,
        )

    sdi = expl.structural_diversity_index

    if rating_context:
        thresholds = rating_context["exploration_sdi_thresholds"]
        rating = _rate_by_thresholds(sdi, thresholds, lower_is_better=False)
    else:
        thresholds = _DEFAULT_EXPLORATION_THRESHOLDS
        rating = _rate_by_thresholds(sdi, thresholds, lower_is_better=False)

    evidence = [
        f"Structural diversity index: {sdi:.3f}",
        f"Exploit phase fraction: {expl.exploit_phase_fraction:.1%}",
        f"Explore phase fraction: {expl.explore_phase_fraction:.1%}",
        f"Distinct strategy clusters: {expl.distinct_strategy_clusters}",
        f"Revert frequency: {expl.revert_frequency:.1%}",
    ]
    if rating_context and thresholds != _DEFAULT_EXPLORATION_THRESHOLDS:
        algo_class = rating_context.get("algorithm_class", "")
        evidence.append(
            f"Note: diversity thresholds adjusted for {algo_class} "
            f"(good threshold {thresholds[1]:.2f} vs {_DEFAULT_EXPLORATION_THRESHOLDS[1]:.2f} default)."
        )

    if (
        search_space is not None
        and "island_id" in getattr(search_space, "bound_hit_params", [])
        and search_space.dominant_island_id is not None
    ):
        total_islands = int(search_space.param_distributions["island_id"].get("max", 0)) + 1
        evidence.append(
            f"Island concentration: top-k solutions skewed toward island"
            f" {search_space.dominant_island_id} (out of {total_islands}) —"
            " cross-island diversity may not be functioning as intended."
        )

    summaries = {
        5: "Excellent exploration — high structural diversity across solutions.",
        4: "Good exploration — diverse solutions generated.",
        3: "Fair exploration — moderate diversity; some repetition.",
        2: "Poor exploration — low diversity; search may be trapped locally.",
        1: "Critical — near-zero diversity; search is highly repetitive.",
    }
    recs = {
        5: "Healthy exploration; ensure exploitation is sufficient to capitalise on gains.",
        4: "Good diversity; balance explore/exploit fractions if exploits are lagging.",
        3: "Introduce more structural perturbation operators to increase diversity.",
        2: "Add restart or escape mechanisms; search is likely stuck in a local basin.",
        1: "Critical: search is producing near-identical solutions. Restart recommended.",
    }

    evidence.extend(_build_exploration_qual_evidence(qual))

    rec = recs[rating]

    if rating <= 2:
        escape_hints = []
        if expl.exploit_phase_fraction > 0.4:
            escape_hints.append(
                "increase LLM mutation temperature to generate more diverse edits"
            )
        if expl.distinct_strategy_clusters <= 3:
            escape_hints.append(
                "force edits to under-explored EVOLVE-BLOCKs or code regions"
            )
        if expl.revert_frequency > 0.15:
            escape_hints.append(
                "add random-restart rounds that begin from a different top-k parent"
            )
        if not escape_hints:
            escape_hints.append(
                "increase LLM mutation temperature or add perturbation operators"
            )
            escape_hints.append(
                "introduce random-restart rounds from diverse top-k parents"
            )
        rec += " Suggested actions: " + "; ".join(escape_hints) + "."

    if (
        search_space is not None
        and "island_id" in getattr(search_space, "bound_hit_params", [])
        and search_space.dominant_island_id is not None
    ):
        rec += " Check island initialization seeds and cross-island exchange mechanisms."

    return DimensionReport(
        name="Exploration",
        rating=rating,
        rating_label=_rating_label(rating),
        summary=summaries[rating],
        evidence=evidence,
        historical=_find_historical(historical, "exploration", "diversity"),
        recommendation=rec,
        data_available=True,
    )


def _build_search_space_dimension(
    quant: Any,
    historical: List[Any],
) -> DimensionReport:
    ss = getattr(quant, "search_space", None)
    if ss is None:
        return DimensionReport(
            name="Search Space",
            rating=None,
            rating_label="N/A",
            summary="No search space data available.",
            evidence=[],
            historical=_find_historical(historical, "search_space", "search space"),
            recommendation="Ensure search space analyzer is enabled.",
            data_available=False,
        )

    ratio = ss.trial_to_param_ratio  # None if no params
    eff_dim = ss.effective_dimensionality
    frozen = getattr(ss, "frozen_params", [])

    if ratio is None or eff_dim == 0:
        if frozen:
            # Parameters exist but none ever varied — the search collapsed.
            rating = 1
            summary = (
                "Collapsed search space — all observed parameters are frozen "
                f"(never varied): {', '.join(frozen)}."
            )
            evidence = [
                "effective_dimensionality=0: no parameter changed across iterations.",
                f"Frozen parameters (single value throughout entire run): {frozen}",
            ]
            rec = (
                "Search space completely collapsed. "
                "Enable multi-island exploration, diversify mutation strategies, "
                "or add restart mechanisms."
            )
        else:
            rating = 1
            summary = "No effective parameters detected; search space analysis unavailable."
            evidence = ["effective_dimensionality=0; no numeric or categorical parameters found."]
            rec = "Verify that iteration records include a 'parameters' field."
    else:
        if ratio > 10:
            rating = 5
        elif ratio >= 5:
            rating = 4
        elif ratio >= 2:
            rating = 3
        elif ratio >= 1:
            rating = 2
        else:
            rating = 1

        # Penalise when frozen params exist alongside varying ones: the search
        # explored some dimensions but left key parameters entirely untouched.
        if frozen:
            rating = max(1, rating - 1)

        summaries = {
            5: "Excellent parameter coverage — many trials per effective dimension.",
            4: "Good coverage — adequate trials relative to parameter count.",
            3: "Fair coverage — moderate trial-to-param ratio.",
            2: "Poor coverage — very few trials per effective dimension.",
            1: "Critical — fewer trials than effective dimensions; search is undersampled.",
        }
        recs_no_frozen = {
            5: "Search space is well-sampled; results should be reliable.",
            4: "Good coverage; minor increase in iterations could solidify findings.",
            3: "Consider reducing parameter count or increasing iteration budget.",
            2: "More iterations needed to adequately sample the parameter space.",
            1: "Significantly increase budget or reduce parameter dimensionality.",
        }
        recs_frozen = {
            5: "Coverage appears adequate, but frozen parameters indicate unexplored diversity — enable multi-island or varied mutation strategies.",
            4: "Frozen parameters limit diversity; activate unused dimensions (e.g. island exploration, mutation variety).",
            3: "Frozen parameters and moderate coverage both limit reliability; diversify mutation strategies.",
            2: "Low coverage compounded by frozen parameters; more iterations and greater diversity needed.",
            1: "Critical — activate frozen parameters and increase budget to meaningfully sample the space.",
        }
        summary = summaries[rating]
        evidence = [
            f"Effective dimensionality: {eff_dim}",
        ]
        for param, dist in sorted(ss.param_distributions.items()):
            if dist.get("type") == "numeric":
                if param == "island_id" and param in ss.bound_hit_params:
                    bound_flag = f" ⚠ top-k skewed to island {ss.dominant_island_id}"
                else:
                    bound_flag = " ⚠ hits bound" if param in ss.bound_hit_params else ""
                frozen_flag = " ⚠ frozen (never varied)" if param in frozen else ""
                evidence.append(
                    f"    {param} [numeric]: range={dist['min']:.4g}–{dist['max']:.4g},"
                    f" mean={dist['mean']:.4g}, std={dist['std']:.4g}{bound_flag}{frozen_flag}"
                )
            elif dist.get("type") == "categorical":
                counts = dist.get("counts", {})
                counts_str = ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda x: -x[1]))
                frozen_flag = " ⚠ frozen (never varied)" if param in frozen else ""
                evidence.append(f"    {param} [categorical]: {counts_str}{frozen_flag}")
        tunable_bound_hits = [p for p in ss.bound_hit_params if not p.endswith("_id")]
        evidence += [
            f"Trial-to-param ratio: {ratio:.2f}",
            f"Bound-hit params: {tunable_bound_hits or 'none'}",
        ]
        if frozen:
            evidence.append(
                f"Frozen params (present but never varied — unexplored diversity): {frozen}"
            )
        rec = recs_frozen[rating] if frozen else recs_no_frozen[rating]
        if tunable_bound_hits:
            bound_names = ", ".join(tunable_bound_hits)
            rec += (
                f" Top-k solutions consistently hit the bound for: {bound_names} —"
                " consider widening the search bounds for these parameters."
            )

    return DimensionReport(
        name="Search Space",
        rating=rating,
        rating_label=_rating_label(rating),
        summary=summary,
        evidence=evidence,
        historical=_find_historical(historical, "search_space", "dimensionality"),
        recommendation=rec,
        data_available=(ss is not None),
    )


def _build_ceiling_dimension(
    quant: Any,
    historical: List[Any],
) -> DimensionReport:
    ceil = getattr(quant, "ceiling", None)
    if ceil is None:
        return DimensionReport(
            name="Ceiling",
            rating=None,
            rating_label="N/A",
            summary="No ceiling data available.",
            evidence=[],
            historical=_find_historical(historical, "ceiling"),
            recommendation="Ensure ceiling analyzer is enabled.",
            data_available=False,
        )

    trend = ceil.marginal_improvement_trend  # "improving" | "flat" | "declining"
    gain_prob = ceil.estimated_gain_probability  # Optional[float]

    if trend == "improving":
        rating = 5
    elif trend == "flat":
        if gain_prob is not None and gain_prob > 0.3:
            rating = 4
        elif gain_prob is not None and gain_prob < 0.3:
            rating = 2
        else:
            rating = 3
    elif trend == "declining":
        rating = 1
    else:
        rating = 3

    evidence = [
        f"Marginal improvement trend: {trend}",
    ]
    if ceil.flat_trend_start_iteration is not None:
        evidence.append(f"Flat trend detected from iteration {ceil.flat_trend_start_iteration}")
    if gain_prob is not None:
        evidence.append(f"Estimated gain probability: {gain_prob:.1%}")
    if ceil.early_stop_suggested_at is not None:
        evidence.append(f"Early stop suggested at iteration {ceil.early_stop_suggested_at}")
    if ceil.wasted_iterations_after_suggestion is not None:
        evidence.append(
            f"Wasted iterations after early-stop suggestion: {ceil.wasted_iterations_after_suggestion}"
        )

    summaries = {
        5: "Marginal improvements are still growing — run has not hit ceiling.",
        4: "Flat trend but meaningful gain probability remains.",
        3: "Flat trend with uncertain gain probability.",
        2: "Flat trend with low gain probability — approaching ceiling.",
        1: "Declining marginal improvements — run has hit its performance ceiling.",
    }
    recs = {
        5: "Continue the run; gains are still accelerating.",
        4: "Consider running a few more iterations; gains may still be achievable.",
        3: "Monitor closely; early stopping may be warranted soon.",
        2: "Strongly consider stopping; further compute is unlikely to yield gains.",
        1: "Stop the run — marginal returns are declining and ceiling has been reached.",
    }

    return DimensionReport(
        name="Ceiling",
        rating=rating,
        rating_label=_rating_label(rating),
        summary=summaries[rating],
        evidence=evidence,
        historical=_find_historical(historical, "ceiling", "gain"),
        recommendation=recs[rating],
        data_available=True,
    )


def _build_sub_metrics_dimension(
    quant: Any,
    historical: List[Any],
    baseline_metrics: Optional[Dict[str, float]] = None,
) -> DimensionReport:
    sm = getattr(quant, "sub_metrics", None)
    if sm is None or not sm.metrics:
        return DimensionReport(
            name="Evaluator Metrics",
            rating=None,
            rating_label="N/A",
            summary="No evaluator_metrics found in records; evaluator metric analysis unavailable.",
            evidence=[],
            historical=None,
            recommendation="Ensure evaluator_metrics is populated in iteration records.",
            data_available=False,
        )

    # Determine rating from primary driver's improvement vs seed
    primary = sm.primary_driver
    primary_stat = sm.metrics.get(primary) if primary else None
    driver_improvement = primary_stat.improvement_vs_seed if primary_stat else None

    if driver_improvement is None:
        rating = 3
    elif abs(driver_improvement) >= 0.20:
        rating = 5
    elif abs(driver_improvement) >= 0.10:
        rating = 4
    elif abs(driver_improvement) >= 0.03:
        rating = 3
    elif abs(driver_improvement) >= 0.01:
        rating = 2
    else:
        rating = 2

    # Build evidence lines — show metrics sorted by |improvement vs seed|, top 10
    def _sort_key(s: Any) -> float:
        return abs(s.improvement_vs_seed) if s.improvement_vs_seed is not None else 0.0

    sorted_stats = sorted(sm.metrics.values(), key=_sort_key, reverse=True)[:10]

    evidence: List[str] = []
    for s in sorted_stats:
        direction = "↓ better" if s.lower_is_better else "↑ better"
        if s.seed_value is not None and s.improvement_vs_seed is not None:
            sign = "−" if s.lower_is_better and s.best < s.seed_value else (
                "−" if not s.lower_is_better and s.best < s.seed_value else "+"
            )
            pct = abs(s.improvement_vs_seed) * 100
            seed_str = f"seed={s.seed_value:.4g}"
        else:
            sign = ""
            pct = 0.0
            seed_str = "seed=N/A"

        baseline_str = ""
        if baseline_metrics and s.name in baseline_metrics:
            bv = baseline_metrics[s.name]
            if s.lower_is_better:
                bv_imp = (bv - s.best) / abs(bv) * 100 if abs(bv) > 1e-9 else 0.0
                bv_sign = "−" if s.best < bv else "+"
            else:
                bv_imp = (s.best - bv) / abs(bv) * 100 if abs(bv) > 1e-9 else 0.0
                bv_sign = "+" if s.best > bv else "−"
            baseline_str = f"  baseline={bv:.4g} ({bv_sign}{abs(bv_imp):.1f}% vs baseline)"

        evidence.append(
            f"{s.name:<30} {seed_str}  best={s.best:.4g} (iter {s.best_iteration})"
            f"  {sign}{pct:.1f}% vs seed  [{direction}]{baseline_str}"
        )

    # Primary driver line
    if primary and primary_stat and primary_stat.improvement_vs_seed is not None:
        pct = abs(primary_stat.improvement_vs_seed) * 100
        direction_word = "decrease" if primary_stat.lower_is_better else "increase"
        evidence.insert(0, f"Primary driver: {primary} ({pct:.1f}% {direction_word} vs seed)")

    driver_label = primary if primary else "unknown"
    summaries = {
        5: f"Strong evaluator metric improvement — primary driver ({driver_label}) improved ≥20% vs seed.",
        4: f"Good evaluator metric improvement — primary driver ({driver_label}) improved ≥10% vs seed.",
        3: f"Moderate evaluator metric improvement — primary driver ({driver_label}) improved ≥3% vs seed.",
        2: f"Weak evaluator metric improvement — gains are marginal across all metrics.",
    }
    recs = {
        5: "Evaluator metric gains are substantial; validate with multi-run measurements.",
        4: "Good gains; check whether the scoring formula weights align with the actual improvement.",
        3: "Moderate gains; consider reweighting the scoring formula toward the primary driver.",
        2: "Gains are near measurement noise; consider longer runs or a different scoring formula.",
    }

    return DimensionReport(
        name="Evaluator Metrics",
        rating=rating,
        rating_label=_rating_label(rating),
        summary=summaries.get(rating, summaries[2]),
        evidence=evidence,
        historical=None,
        recommendation=recs.get(rating, recs[2]),
        data_available=True,
    )


def _build_meta_qual_evidence(qual: Any) -> List[str]:
    """Aggregate Step F (meta_quality) judge result into evidence bullet strings."""
    evidence: List[str] = []
    if qual is None:
        return evidence
    mq = getattr(qual, "meta_quality", None) or {}
    if mq.get("skipped_reason") == SKIP_REASON_NO_TRACE:
        evidence.append(
            "Meta quality judge (F) not run — reasoning trace was not logged by this framework. "
            "Judge F requires the LLM's chain-of-thought or scratchpad output to evaluate "
            "reasoning coherence."
        )
    elif mq.get("coherence_rating"):
        evidence.append(f"Reasoning coherence (LLM judge): {mq['coherence_rating']}/5")
        for issue in (mq.get("issues") or [])[:3]:
            evidence.append(f"  Issue: {str(issue)[:150]}")
        if mq.get("recommendation"):
            evidence.append(f"  LLM recommendation: {mq['recommendation'][:150]}")
    return evidence


def _build_meta_analysis_dimension(
    quant: Any,
    historical: List[Any],
    qual: Any = None,
) -> DimensionReport:
    meta = getattr(quant, "meta_analysis", None)
    if meta is None:
        return DimensionReport(
            name="Meta-Analysis",
            rating=None,
            rating_label="N/A",
            summary="No meta-analysis data available (reasoning traces not present).",
            evidence=[],
            historical=_find_historical(historical, "meta"),
            recommendation="Enable reasoning trace logging for meta-analysis.",
            data_available=False,
        )

    sfr = meta.suggestion_follow_rate  # Optional[float]
    cir = meta.conditional_improvement_rate  # Optional[float]

    if sfr is None and cir is None:
        return DimensionReport(
            name="Meta-Analysis",
            rating=None,
            rating_label="N/A",
            summary="Meta-analysis fields not logged by this framework — no data to rate.",
            evidence=["suggestion_follow_rate and conditional_improvement_rate are both None.",
                      "Meta quality judge (F) not run — reasoning trace was not logged by this framework."] + _build_meta_qual_evidence(qual),
            historical=_find_historical(historical, "meta"),
            recommendation="Ensure meta_suggestion and followed_suggestion fields are logged.",
            data_available=False,
        )

    # Rating based on combination of sfr and cir
    sfr_val = sfr if sfr is not None else 0.5
    cir_val = cir if cir is not None else 0.5
    combined = (sfr_val + cir_val) / 2.0

    if combined >= 0.7:
        rating = 5
    elif combined >= 0.5:
        rating = 4
    elif combined >= 0.35:
        rating = 3
    elif combined >= 0.2:
        rating = 2
    else:
        rating = 1

    evidence = []
    if sfr is not None:
        evidence.append(f"Suggestion follow rate: {sfr:.1%}")
    if cir is not None:
        evidence.append(f"Conditional improvement rate: {cir:.1%}")
    if meta.pattern_reuse_frequency is not None:
        evidence.append(f"Pattern reuse frequency: {meta.pattern_reuse_frequency:.1%}")
    if meta.scratchpad_growth_rate is not None:
        evidence.append(f"Scratchpad growth rate: {meta.scratchpad_growth_rate:.1f} chars/iter")
    if meta.compaction_events is not None:
        evidence.append(f"Compaction events: {meta.compaction_events}")

    evidence.extend(_build_meta_qual_evidence(qual))

    summaries = {
        5: "Excellent meta-analysis quality — LLM suggestions are well-followed and effective.",
        4: "Good meta-analysis — suggestions frequently followed with good success rate.",
        3: "Fair meta-analysis — moderate suggestion adherence or conditional improvement.",
        2: "Poor meta-analysis — suggestions rarely followed or rarely productive.",
        1: "Critical — meta-analysis suggests LLM self-guidance is largely ineffective.",
    }
    recs = {
        5: "Meta-guidance is effective; maintain current reasoning trace approach.",
        4: "Good meta-guidance; explore ways to increase suggestion acceptance.",
        3: "Review types of suggestions being generated and ignored.",
        2: "Audit reasoning traces for quality; consider prompt engineering improvements.",
        1: "Fundamental issue with LLM self-guidance; review prompting strategy entirely.",
    }

    return DimensionReport(
        name="Meta-Analysis",
        rating=rating,
        rating_label=_rating_label(rating),
        summary=summaries[rating],
        evidence=evidence,
        historical=_find_historical(historical, "meta", "suggestion"),
        recommendation=recs[rating],
        data_available=True,
    )


# ---------------------------------------------------------------------------
# Stagnation period reports
# ---------------------------------------------------------------------------

def _build_stagnation_period_reports(
    quant: Any,
    qual: Any,
) -> List[StagnationPeriodReport]:
    periods = getattr(quant, "stagnation_periods", None) or []
    qual_analyses: dict = {}

    # Extract LLM analyses from qualitative bundle keyed by streak_id
    if qual is not None:
        raw = getattr(qual, "stagnation_analyses", None)
        if raw is not None:
            if isinstance(raw, dict):
                qual_analyses = raw
            elif isinstance(raw, list):
                for item in raw:
                    sid = getattr(item, "streak_id", None) or (
                        item.get("streak_id") if isinstance(item, dict) else None
                    )
                    if sid:
                        qual_analyses[sid] = (
                            item if isinstance(item, dict) else item.__dict__
                        )

    reports: List[StagnationPeriodReport] = []
    for p in periods:
        llm_analysis = qual_analyses.get(p.streak_id)

        # Build recommendation based on severity and dominant failure type
        if p.severity == "critical":
            if p.dominant_failure_type == "crash":
                # Inspect crash errors for a more specific message
                crash_errors = " ".join(
                    s.crash_error or "" for s in p.failure_sequence if s.failure_type == "crash"
                ).lower()
                if "budget" in crash_errors or "budget_exceeded" in crash_errors:
                    rec = "LLM API budget exceeded — top up the budget or switch to a cheaper model to resume."
                elif "auth" in crash_errors or "401" in crash_errors or "403" in crash_errors:
                    rec = "LLM API authentication failure — verify API key and credentials."
                elif "rate" in crash_errors or "429" in crash_errors:
                    rec = "LLM API rate limit hit — reduce parallelism or add retry backoff."
                elif "timeout" in crash_errors or "timed out" in crash_errors:
                    rec = "LLM calls are timing out — increase timeout or reduce prompt size."
                else:
                    rec = (
                        f"Critical streak driven by '{p.dominant_failure_type}'. "
                        "Review crash logs above and fix the underlying error before continuing."
                    )
            else:
                rec = (
                    f"Critical streak driven by '{p.dominant_failure_type}'. "
                    "Immediate review of mutation strategy required."
                )
        elif p.severity == "high":
            rec = (
                f"High-severity stagnation from '{p.dominant_failure_type}'. "
                "Consider injecting diversity or changing mutation operator."
            )
        else:
            rec = (
                f"Warning-level stagnation ({p.dominant_failure_type}). "
                "Monitor and apply recovery if streak persists."
            )

        crash_samples = None
        if p.dominant_failure_type == "crash":
            seen_errors: dict = {}
            for s in p.failure_sequence:
                if s.failure_type == "crash" and s.crash_error:
                    err = s.crash_error.strip()
                    if err not in seen_errors:
                        seen_errors[err] = s.iteration
            crash_samples = [
                {"iteration": it, "error": err}
                for err, it in sorted(seen_errors.items(), key=lambda x: x[1])
            ][:5]

        reports.append(
            StagnationPeriodReport(
                streak_id=p.streak_id,
                start_iteration=p.start_iteration,
                end_iteration=p.end_iteration,
                length=p.length,
                severity=p.severity,
                dominant_failure_type=p.dominant_failure_type,
                llm_analysis=llm_analysis,
                recommendation=rec,
                crash_samples=crash_samples,
            )
        )

    return reports


# ---------------------------------------------------------------------------
# Cross-dimension narrative
# ---------------------------------------------------------------------------

def _generate_cross_dimension_narrative(dimensions: List[DimensionReport]) -> str:
    by_name = {d.name: d for d in dimensions}

    lines: List[str] = []

    # Regression vs Exploration
    reg = by_name.get("Regression")
    expl = by_name.get("Exploration")
    if reg and expl and reg.data_available and expl.data_available:
        if reg.rating <= 2 and expl.rating <= 2:
            lines.append(
                "High regression frequency correlates with low exploration diversity, "
                "suggesting the search is trapped and unable to escape regression cycles."
            )
        elif reg.rating <= 2 and expl.rating >= 4:
            lines.append(
                "Despite high structural diversity, the regression rate remains elevated, "
                "indicating that diverse mutations are not translating into score improvements."
            )
        elif reg.rating >= 4 and expl.rating >= 4:
            lines.append(
                "Strong exploration diversity and low regression rate together indicate "
                "a healthy search dynamic with effective mutation strategies."
            )

    # Stagnation vs Efficiency
    stag = by_name.get("Stagnation")
    eff = by_name.get("Efficiency")
    if stag and eff and stag.data_available and eff.data_available:
        if stag.rating <= 2 and eff.rating <= 2:
            lines.append(
                "Frequent stagnation periods are directly contributing to wasted compute: "
                "a large fraction of iterations occur in post-plateau or stalled phases."
            )
        elif stag.rating >= 4 and eff.rating >= 4:
            lines.append(
                "Minimal stagnation and low wasted-phase fraction reinforce each other, "
                "producing a high-efficiency run."
            )

    # Convergence vs Ceiling
    conv = by_name.get("Convergence")
    ceil = by_name.get("Ceiling")
    if conv and ceil and conv.data_available and ceil.data_available:
        if conv.rating >= 4 and ceil.rating <= 2:
            lines.append(
                "Good convergence speed combined with a declining marginal trend suggests "
                "the run converged quickly but is now hitting a performance ceiling."
            )
        elif conv.rating <= 2 and ceil.rating >= 4:
            lines.append(
                "Slow convergence alongside improving marginal trends indicates the search "
                "is still exploring productively — extending the run may be worthwhile."
            )

    # Search Space vs Exploration
    ss = by_name.get("Search Space")
    if ss and expl and ss.data_available and expl.data_available:
        if ss.rating <= 2 and expl.rating >= 4:
            lines.append(
                "High structural diversity despite low trial-to-param ratio suggests "
                "the search is broadly exploring but without sufficient density to "
                "characterise the parameter space reliably."
            )

    if not lines:
        # Generic fallback
        avg_rating = sum(d.rating for d in dimensions if d.data_available) / max(
            1, sum(1 for d in dimensions if d.data_available)
        )
        if avg_rating >= 4:
            lines.append(
                "Dimension ratings are uniformly high, indicating a well-functioning "
                "optimization loop with no major cross-cutting issues."
            )
        elif avg_rating >= 3:
            lines.append(
                "Mixed dimension ratings suggest selective weaknesses; "
                "addressing the lowest-rated dimensions should yield the most benefit."
            )
        else:
            lines.append(
                "Multiple dimensions are underperforming; the optimization loop "
                "shows systemic issues that may require a fundamental review."
            )
        lines.append(
            "No strong cross-dimension interactions were detected based on current metrics."
        )

    return " ".join(lines)


# ---------------------------------------------------------------------------
# Novel observations
# ---------------------------------------------------------------------------

def _generate_novel_observations(
    dimensions: List[DimensionReport],
    quant: Any,
) -> List[str]:
    observations: List[str] = []
    by_name = {d.name: d for d in dimensions}

    expl = by_name.get("Exploration")
    stag = by_name.get("Stagnation")
    reg = by_name.get("Regression")
    eff = by_name.get("Efficiency")
    conv = by_name.get("Convergence")
    ceil = by_name.get("Ceiling")
    ss = by_name.get("Search Space")

    # High diversity but high stagnation
    if (
        expl and stag
        and expl.data_available and stag.data_available
        and expl.rating >= 4 and stag.rating <= 2
    ):
        observations.append(
            "Unusual: high structural diversity co-occurs with high stagnation frequency. "
            "The search explores broadly but fails to convert variety into improvements."
        )

    # Very low regression but also very low improvement (efficiency poor)
    if (
        reg and eff
        and reg.data_available and eff.data_available
        and reg.rating == 5 and eff.rating <= 2
    ):
        observations.append(
            "Unusual: near-zero regression rate but poor efficiency. "
            "The search avoids regressions by seldom attempting bold mutations, "
            "resulting in a long unproductive plateau rather than active backtracking."
        )

    # Convergence fast but ceiling not yet reached
    if (
        conv and ceil
        and conv.data_available and ceil.data_available
        and conv.rating >= 4 and ceil.rating >= 4
    ):
        observations.append(
            "Positive signal: rapid convergence combined with ongoing gain probability "
            "suggests the search found a productive region early and continues to mine it."
        )

    # Search space undercoverage with good results
    if ss and ss.data_available and ss.rating <= 2:
        conv_quant = getattr(quant, "convergence", None)
        if conv_quant and conv_quant.convergence_rate > 0.01:
            observations.append(
                "Notable: low trial-to-param ratio (sparse search space coverage) "
                "despite a healthy convergence rate. Results may not generalise "
                "to the full parameter space."
            )

    # All dimensions fair or above — positive
    available_ratings = [d.rating for d in dimensions if d.data_available]
    if available_ratings and min(available_ratings) >= 3 and sum(available_ratings) / len(available_ratings) >= 4:
        observations.append(
            "All dimensions rate at least Fair with a high average — this run is performing "
            "consistently well across all measured axes."
        )

    return observations


# ---------------------------------------------------------------------------
# Executive summary
# ---------------------------------------------------------------------------

def _generate_executive_summary(
    dimensions: List[DimensionReport],
    agg: AggregateStats,
) -> str:
    available = [d for d in dimensions if d.data_available]
    if not available:
        return (
            f"Experiment completed {agg.total_iterations} iterations "
            f"(best score: {agg.best_score:.4g}). "
            "Insufficient data for dimensional analysis."
        )

    if agg.total_iterations > 0 and agg.n_successful == 0:
        return (
            f"Run failed completely — 0 of {agg.total_iterations} iterations "
            "produced a valid evaluation. All iterations crashed before scoring. "
            "Review the crash details in the Stagnation section below."
        )

    avg_rating = sum(d.rating for d in available) / len(available)
    worst_dim = min(available, key=lambda d: d.rating)

    if avg_rating >= 4.5:
        overall = "excellent"
    elif avg_rating >= 3.5:
        overall = "good"
    elif avg_rating >= 2.5:
        overall = "fair"
    elif avg_rating >= 1.5:
        overall = "poor"
    else:
        overall = "critical"

    score_str = f"best score of {agg.best_score:.4g}"
    if agg.baseline_score is not None and agg.score_improvement_vs_baseline is not None:
        sign = "+" if agg.score_improvement_vs_baseline >= 0 else ""
        score_str += (
            f" ({sign}{agg.score_improvement_vs_baseline:.2%} vs external baseline"
            f" of {agg.baseline_score:.4g})"
        )

    sentence1 = (
        f"Overall run quality is {overall} (average dimension rating "
        f"{avg_rating:.1f}/5) across {agg.total_iterations} iterations "
        f"reaching a {score_str}."
    )

    critical_dims = [d for d in available if d.rating == 1]
    poor_dims = [d for d in available if d.rating == 2]

    if len(critical_dims) > 1:
        parts = [sentence1, "", "Critical dimensions:"]
        for d in critical_dims:
            parts.append(f"  {d.name} ({d.rating_label}): {d.summary}")
            parts.append(f"    Recommendation: {d.recommendation}")
        if poor_dims:
            poor_names = ", ".join(d.name for d in poor_dims)
            parts.append(f"\nAlso poor: {poor_names}.")
        return "\n".join(parts)
    elif len(critical_dims) == 1:
        sentence2 = (
            f"The biggest concern is the {critical_dims[0].name} dimension "
            f"({critical_dims[0].rating_label}): {critical_dims[0].summary}"
        )
        sentence3 = f"Top recommendation: {critical_dims[0].recommendation}"
        if poor_dims:
            poor_names = ", ".join(d.name for d in poor_dims)
            sentence3 += f" Also poor: {poor_names}."
    else:
        sentence2 = (
            f"The biggest concern is the {worst_dim.name} dimension "
            f"({worst_dim.rating_label}): {worst_dim.summary}"
        )
        sentence3 = f"Top recommendation: {worst_dim.recommendation}"

    return f"{sentence1} {sentence2} {sentence3}"


def _generate_executive_summary_md(
    dimensions: List[DimensionReport],
    agg: AggregateStats,
) -> str:
    available = [d for d in dimensions if d.data_available]
    if not available:
        return (
            f"Experiment completed {agg.total_iterations} iterations "
            f"(best score: {agg.best_score:.4g}). "
            "Insufficient data for dimensional analysis."
        )

    if agg.total_iterations > 0 and agg.n_successful == 0:
        return (
            f"**Run failed completely** — 0 of {agg.total_iterations} iterations "
            "produced a valid evaluation. All iterations crashed before scoring. "
            "Review the crash details in the Stagnation section below."
        )

    avg_rating = sum(d.rating for d in available) / len(available)
    worst_dim = min(available, key=lambda d: d.rating)

    if avg_rating >= 4.5:
        overall = "excellent"
    elif avg_rating >= 3.5:
        overall = "good"
    elif avg_rating >= 2.5:
        overall = "fair"
    elif avg_rating >= 1.5:
        overall = "poor"
    else:
        overall = "critical"

    score_str = f"best score of {agg.best_score:.4g}"
    if agg.baseline_score is not None and agg.score_improvement_vs_baseline is not None:
        sign = "+" if agg.score_improvement_vs_baseline >= 0 else ""
        score_str += (
            f" ({sign}{agg.score_improvement_vs_baseline:.2%} vs external baseline"
            f" of {agg.baseline_score:.4g})"
        )

    lines: List[str] = []
    lines.append(
        f"Overall run quality is **{overall}** (average dimension rating "
        f"**{avg_rating:.1f}/5**) across {agg.total_iterations} iterations "
        f"reaching a {score_str}."
    )
    lines.append("")

    critical_dims = [d for d in available if d.rating == 1]
    poor_dims = [d for d in available if d.rating == 2]

    if len(critical_dims) > 1:
        lines.append("**Critical dimensions:**")
        lines.append("")
        for d in critical_dims:
            lines.append(f"- **{d.name}** {d.rating_label}: {d.summary}")
            lines.append(f"  - *Recommendation:* {d.recommendation}")
        if poor_dims:
            lines.append("")
            poor_names = ", ".join(f"**{d.name}**" for d in poor_dims)
            lines.append(f"Also poor: {poor_names}.")
    elif len(critical_dims) == 1:
        d = critical_dims[0]
        lines.append(
            f"**Biggest concern — {d.name}** {d.rating_label}: {d.summary}"
        )
        lines.append(f"*Recommendation:* {d.recommendation}")
        if poor_dims:
            poor_names = ", ".join(f"**{d.name}**" for d in poor_dims)
            lines.append(f"\nAlso poor: {poor_names}.")
    else:
        lines.append(
            f"**Biggest concern — {worst_dim.name}** {worst_dim.rating_label}: "
            f"{worst_dim.summary}"
        )
        lines.append(f"*Recommendation:* {worst_dim.recommendation}")

    return "\n".join(lines)


def _build_infrastructure_dimension(
    quant: Any,
    historical: List[Any],
) -> DimensionReport:
    infra = getattr(quant, "infrastructure", None)
    if infra is None:
        return DimensionReport(
            name="Infrastructure",
            rating=None,
            rating_label="N/A",
            summary="No infrastructure analysis available.",
            evidence=[],
            historical=_find_historical(historical, "infrastructure"),
            recommendation="Ensure infrastructure analyzer is enabled.",
            data_available=False,
        )

    cause = infra.failure_cause
    sf = infra.sentinel_fraction

    if cause == "NONE":
        rating = 5
        summary = "No infrastructure failures detected — all evaluations completed normally."
        rec = "Infrastructure looks healthy; no action needed."
    elif cause == "INFRA_DEGRADATION":
        rating = 3
        summary = (
            f"Evaluator degradation detected (eval-time spike ×{infra.eval_time_spike_ratio:.1f}) "
            "before first sentinel. Server was slow but recovered."
        )
        rec = (
            "Monitor evaluator server health. Consider adding retry logic or "
            "health-check probes before long runs."
        )
    elif cause == "INFRA_CRASH":
        if sf >= 0.75:
            rating = 1
            summary = (
                f"Critical: server crash corrupted {sf:.1%} of the run "
                f"({infra.sentinel_count} sentinel iterations). "
                "Results after the crash are invalid."
            )
        elif sf >= 0.25:
            rating = 1
            summary = (
                f"Server crash corrupted {sf:.1%} of the run "
                f"({infra.sentinel_count} sentinel iterations). "
                "Analysis results are unreliable."
            )
        else:
            rating = 2
            summary = (
                f"Server crash detected ({infra.sentinel_count} sentinel iterations, "
                f"{sf:.1%} of run). Early results may be valid."
            )
        rec = (
            "Rerun the experiment with a stable evaluator server. "
            "Filter sentinel iterations before drawing conclusions."
        )
    elif cause == "EVALUATOR_NOISE":
        rating = 3
        summary = (
            f"Isolated evaluator failures detected ({infra.sentinel_count} sentinel "
            "iterations, not a contiguous block). Likely transient noise."
        )
        rec = "Add retry logic around evaluator calls to handle transient failures."
    else:
        rating = 2
        summary = f"Unexpected evaluator failure pattern: {cause}."
        rec = "Investigate evaluator logs for root cause."

    evidence: List[str] = [
        f"{infra.sentinel_count} sentinel iteration(s) ({sf:.1%} of run)",
    ]
    if infra.first_sentinel_iteration is not None:
        evidence.append(f"First sentinel at iteration {infra.first_sentinel_iteration}")
    if infra.crash_onset_iteration is not None:
        evidence.append(f"Server crash onset at iteration {infra.crash_onset_iteration} (eval_time=0.0s)")
    if infra.degradation_window is not None:
        dw = infra.degradation_window
        ratio_str = f" (×{infra.eval_time_spike_ratio:.1f} baseline)" if infra.eval_time_spike_ratio else ""
        evidence.append(f"Degradation window: iterations {dw[0]}–{dw[1]}{ratio_str}")
    if infra.affected_iterations:
        sample = infra.affected_iterations[:10]
        suffix = "..." if len(infra.affected_iterations) > 10 else ""
        evidence.append(f"Sentinel iterations: {sample}{suffix}")

    if infra.log_evidence:
        le = infra.log_evidence
        if le.get("http_error_burst_start"):
            types = "/".join(le.get("http_error_types") or [])
            evidence.append(
                f"Log: HTTP {types} burst started at {le['http_error_burst_start']}"
                f" ({le.get('http_error_burst_count', '?')} error lines)"
            )
        if le.get("crash_timestamp"):
            host = le.get("crash_host", "unknown host")
            evidence.append(
                f"Log: ConnectionRefusedError at {le['crash_timestamp']}"
                f" — server {host} became unreachable"
            )
        for raw_line in (le.get("sample_error_lines") or []):
            evidence.append(f"  › {raw_line.strip()}")

    return DimensionReport(
        name="Infrastructure",
        rating=rating,
        rating_label=_rating_label(rating),
        summary=summary,
        evidence=evidence,
        historical=_find_historical(historical, "infrastructure"),
        recommendation=rec,
        data_available=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_num_islands(df: "pd.DataFrame") -> Optional[int]:
    if "island_id" not in df.columns:
        return None
    unique = df["island_id"].dropna().unique()
    return int(len(unique)) if len(unique) > 0 else None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def synthesize_report(
    quant: Any,
    qual: Any,
    historical: List[Any],
    config: dict,
    experiment_id: str = "",
    llm_judge_status: Optional[LLMJudgeStatus] = None,
    algorithm_class: str = "population_evolutionary",
    algorithm_name: Optional[str] = None,
) -> EvolveLoopReport:
    """
    Synthesize a full EvolveLoopReport from quantitative, qualitative, and
    historical analysis results.

    Parameters
    ----------
    quant:
        QuantitativeBundle produced by the quantitative analysis pass.
    qual:
        QualitativeBundle produced by the qualitative analysis pass (may be None).
    historical:
        List of HistoricalComparison objects from HistoricalDB.compare calls.
    config:
        Experiment configuration dict (unused in computation but preserved for
        future extensibility).
    experiment_id:
        Identifier for this experiment.
    """
    df: pd.DataFrame = getattr(quant, "df", pd.DataFrame())

    # --- Aggregate stats ---
    agg = _compute_aggregate_stats(df)

    # --- Baseline context (from config) ---
    baseline_cfg = config.get("baseline", {}) if config else {}
    baseline_score: Optional[float] = baseline_cfg.get("score")
    baseline_metrics: Optional[Dict[str, float]] = baseline_cfg.get("metrics") or None
    if baseline_score is not None:
        agg.baseline_score = float(baseline_score)
        agg.baseline_metrics = baseline_metrics
        import math
        if not math.isnan(agg.best_score) and abs(float(baseline_score)) > 1e-9:
            agg.score_improvement_vs_baseline = (
                (agg.best_score - float(baseline_score)) / abs(float(baseline_score))
            )

    # --- Rating context (algorithm-class-aware thresholds) ---
    rating_context = _build_rating_context(algorithm_class, config or {})

    # --- Dimension reports ---
    num_islands = _compute_num_islands(df)
    dimensions: List[DimensionReport] = [
        _build_convergence_dimension(quant, historical, rating_context=rating_context),
        _build_stagnation_dimension(quant, historical, num_islands=num_islands),
        _build_regression_dimension(quant, historical, rating_context=rating_context),
        _build_efficiency_dimension(quant, historical),
        _build_exploration_dimension(quant, historical, qual=qual, rating_context=rating_context, search_space=quant.search_space),
        _build_search_space_dimension(quant, historical),
        _build_ceiling_dimension(quant, historical),
        _build_meta_analysis_dimension(quant, historical, qual=qual),
        _build_sub_metrics_dimension(quant, historical, baseline_metrics=baseline_metrics),
        _build_infrastructure_dimension(quant, historical),
    ]

    # --- Stagnation period reports ---
    stagnation_periods = _build_stagnation_period_reports(quant, qual)

    # --- Cross-dimension narrative ---
    cross_dimension_interactions = _generate_cross_dimension_narrative(dimensions)

    # --- Executive summary ---
    executive_summary = _generate_executive_summary(dimensions, agg)
    executive_summary_md = _generate_executive_summary_md(dimensions, agg)

    # --- Novel observations ---
    novel_observations = _generate_novel_observations(dimensions, quant)

    ingestion_cfg = config.get("ingestion", {}) if config else {}
    return EvolveLoopReport(
        experiment_id=experiment_id,
        executive_summary=executive_summary,
        executive_summary_md=executive_summary_md,
        dimensions=dimensions,
        cross_dimension_interactions=cross_dimension_interactions,
        stagnation_periods=stagnation_periods,
        aggregate_stats=agg,
        novel_observations=novel_observations,
        llm_judge_status=llm_judge_status,
        run_source=ingestion_cfg.get("source") or None,
        run_path=ingestion_cfg.get("path") or None,
        run_config_path=config.get("_config_path") or None,
        algorithm_class=algorithm_class,
        algorithm_name=algorithm_name,
        num_islands=_compute_num_islands(df),
    )
