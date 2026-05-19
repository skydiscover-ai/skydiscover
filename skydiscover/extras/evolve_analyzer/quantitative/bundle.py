"""
QuantitativeBundle and all supporting dataclasses produced by the quantitative analyzers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ── Stagnation ────────────────────────────────────────────────────────────────

@dataclass
class IterationSummary:
    iteration: int
    mutation_type: str
    failure_type: str
    score_delta: float
    compliance_status: Optional[str]
    format_valid: bool
    crash_error: Optional[str] = None
    island_id: Optional[str] = None


@dataclass
class StagnationPeriod:
    streak_id: str
    start_iteration: int
    end_iteration: Optional[int]          # None if still ongoing at run end
    length: int
    failure_sequence: List[IterationSummary]
    dominant_failure_type: str
    secondary_failure_types: List[str]
    is_alert: bool                        # length >= threshold
    severity: str                         # "warning" | "high" | "critical"
    recovery_iteration: Optional[int]
    recovery_mutation_type: Optional[str]
    recovery_model: Optional[str]
    score_at_stagnation_start: float
    score_at_recovery: Optional[float]


# ── Convergence ───────────────────────────────────────────────────────────────

@dataclass
class ConvergenceMetrics:
    best_so_far_curve: List[float]
    rolling_mean: List[float]
    rolling_variance: List[float]
    change_points: List[int]
    convergence_rate: float               # improvement per iteration
    improvement_per_eval: float           # improvement per evaluator call
    time_to_best_fraction: float          # fraction of compute before best found
    plateau_onset_iteration: Optional[int]


# ── Exploration ───────────────────────────────────────────────────────────────

@dataclass
class ExplorationMetrics:
    structural_diversity_index: float
    exploit_phase_fraction: float
    explore_phase_fraction: float
    distinct_strategy_clusters: int
    revert_frequency: float


# ── Regression ────────────────────────────────────────────────────────────────

@dataclass
class RegressionMetrics:
    regression_frequency: float
    severity_distribution: dict
    mean_recovery_time: float
    death_spiral_periods: List[Tuple[int, int]]


# ── Efficiency ────────────────────────────────────────────────────────────────

@dataclass
class EfficiencyMetrics:
    improvement_per_llm_call: float
    improvement_per_dollar: Optional[float]
    improvement_per_hour: Optional[float]
    improvement_per_eval_call: Optional[float]
    productive_phase_fraction: float
    wasted_phase_fraction: float
    pareto_frontier: List[Tuple[float, float]]   # (cumulative_cost, best_score)


# ── Search Space ──────────────────────────────────────────────────────────────

@dataclass
class SearchSpaceMetrics:
    param_distributions: dict
    bound_hit_params: List[str]
    frozen_params: List[str]
    effective_dimensionality: int
    trial_to_param_ratio: Optional[float]
    dominant_island_id: Optional[int] = None   # set when island_id hits a bound


# ── Evaluator Metrics ─────────────────────────────────────────────────────────

@dataclass
class SubMetricStats:
    name: str
    best: float
    worst: float
    final: Optional[float]
    best_iteration: int
    seed_value: Optional[float]          # from parent_metrics of first iteration
    improvement_vs_seed: Optional[float] # positive = better; (best-seed)/|seed| adjusted for direction
    lower_is_better: bool


@dataclass
class SubMetricAnalysis:
    metrics: Dict[str, SubMetricStats]
    primary_driver: Optional[str]        # metric with highest |improvement_vs_seed|


# ── Meta-Analysis ─────────────────────────────────────────────────────────────

@dataclass
class MetaAnalysisMetrics:
    suggestion_follow_rate: Optional[float]
    conditional_improvement_rate: Optional[float]
    pattern_reuse_frequency: Optional[float]
    scratchpad_growth_rate: Optional[float]
    compaction_events: Optional[int]


# ── Ceiling ───────────────────────────────────────────────────────────────────

@dataclass
class CeilingMetrics:
    marginal_improvement_trend: str       # "declining" | "flat" | "improving"
    plateau_p_value: Optional[float]
    estimated_gain_probability: Optional[float]
    flat_trend_start_iteration: Optional[int]  # iteration after last best-score improvement
    early_stop_suggested_at: Optional[int]
    early_stop_actual_at: Optional[int]
    wasted_iterations_after_suggestion: Optional[int]


# ── Infrastructure ────────────────────────────────────────────────────────────

@dataclass
class InfrastructureMetrics:
    sentinel_count: int                     # iterations with combined_score <= -9999
    sentinel_fraction: float                # sentinel_count / total_iterations
    first_sentinel_iteration: Optional[int]
    crash_onset_iteration: Optional[int]    # first sentinel with eval_time == 0.0
    degradation_window: Optional[Tuple[int, int]]  # (start_iter, end_iter) of pre-crash eval-time spike
    failure_cause: str                      # "INFRA_CRASH" | "INFRA_DEGRADATION" | "EVALUATOR_NOISE" | "CODE_REGRESSION" | "NONE"
    affected_iterations: List[int]          # all sentinel iteration numbers
    eval_time_spike_ratio: Optional[float]  # peak / baseline eval_time during degradation
    log_evidence: Optional[dict] = None     # structured signals extracted from logs/*.log


# ── Bundle ────────────────────────────────────────────────────────────────────

@dataclass
class QuantitativeBundle:
    """
    Output of the full quantitative analysis pass.

    df is the original records DataFrame with additional tag columns added by
    each analyzer (streak_id, streak_position, failure_mode, evolved_block_only,
    format_valid, signature_preserved, high_variance, etc.).
    """
    df: pd.DataFrame

    stagnation_periods: List[StagnationPeriod] = field(default_factory=list)
    convergence: Optional[ConvergenceMetrics] = None
    exploration: Optional[ExplorationMetrics] = None
    regression: Optional[RegressionMetrics] = None
    efficiency: Optional[EfficiencyMetrics] = None
    search_space: Optional[SearchSpaceMetrics] = None
    meta_analysis: Optional[MetaAnalysisMetrics] = None
    ceiling: Optional[CeilingMetrics] = None
    sub_metrics: Optional[SubMetricAnalysis] = None
    infrastructure: Optional[InfrastructureMetrics] = None

    # Which dimensions had sufficient data
    data_availability: dict = field(default_factory=dict)
