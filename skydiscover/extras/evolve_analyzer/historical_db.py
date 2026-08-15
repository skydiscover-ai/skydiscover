"""
historical_db.py
----------------
Standalone SQLite database for cross-experiment comparative analysis.

Stores per-experiment quantitative metrics and tracks recurring patterns
across experiments to enable percentile-based and heuristic comparisons.
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class HistoricalComparison:
    metric_name: str
    current_value: float
    percentile: Optional[float]
    historical_median: Optional[float]
    n_experiments: int
    rating_basis: str   # "historical" | "heuristic"
    summary: str


# ---------------------------------------------------------------------------
# Heuristic thresholds (used when n_experiments < min_experiments)
# ---------------------------------------------------------------------------

_HEURISTIC_THRESHOLDS: Dict[str, List] = {
    # (threshold, direction, label)
    # direction "lower_better": value below threshold[0] is good, etc.
    # direction "higher_better": value above threshold[0] is good, etc.
    "convergence_rate": [
        ("higher_better", 0.01, "good", 0.001, "moderate", "poor"),
    ],
    "stagnation_count": [
        ("lower_better_int", 0, "excellent", 3, "good", "concerning"),
    ],
    "regression_frequency": [
        ("lower_better", 0.1, "good", 0.3, "moderate", "poor"),
    ],
    "improvement_per_llm_call": [
        ("higher_better", 0.01, "good", 0.001, "moderate", "poor"),
    ],
    "improvement_per_dollar": [
        ("higher_better", 1.0, "good", 0.1, "moderate", "poor"),
    ],
    "productive_phase_fraction": [
        ("higher_better", 0.7, "good", 0.4, "moderate", "poor"),
    ],
    "structural_diversity_index": [
        ("higher_better", 0.5, "good", 0.2, "moderate", "poor"),
    ],
    "exploit_phase_fraction": [
        ("lower_better", 0.3, "balanced", 0.7, "moderate", "over-exploiting"),
    ],
    "estimated_gain_probability": [
        ("higher_better", 0.3, "good", 0.1, "moderate", "poor"),
    ],
    "mean_recovery_time": [
        ("lower_better", 5.0, "good", 15.0, "moderate", "poor"),
    ],
    "max_stagnation_length": [
        ("lower_better", 10, "good", 20, "moderate", "poor"),
    ],
    "time_to_best_fraction": [
        ("lower_better", 0.5, "good", 0.8, "moderate", "poor"),
    ],
}


def _heuristic_summary(metric_name: str, value: float) -> str:
    """Return a heuristic rating string for the given metric and value."""
    spec = _HEURISTIC_THRESHOLDS.get(metric_name)
    if spec is None:
        return f"Value {value:.4g} (no heuristic available)"

    rule = spec[0]
    direction = rule[0]

    if direction == "higher_better":
        _, hi_thresh, hi_label, lo_thresh, mid_label, bad_label = rule
        if value > hi_thresh:
            return f"Value {value:.4g} — {hi_label} (above threshold {hi_thresh})"
        elif value >= lo_thresh:
            return f"Value {value:.4g} — {mid_label} (between {lo_thresh} and {hi_thresh})"
        else:
            return f"Value {value:.4g} — {bad_label} (below threshold {lo_thresh})"

    elif direction == "lower_better":
        _, lo_thresh, lo_label, hi_thresh, mid_label, bad_label = rule
        if value < lo_thresh:
            return f"Value {value:.4g} — {lo_label} (below threshold {lo_thresh})"
        elif value <= hi_thresh:
            return f"Value {value:.4g} — {mid_label} (between {lo_thresh} and {hi_thresh})"
        else:
            return f"Value {value:.4g} — {bad_label} (above threshold {hi_thresh})"

    elif direction == "lower_better_int":
        # stagnation_count: 0 = excellent, 1-2 = good, 3+ = concerning
        _, zero_thresh, zero_label, hi_thresh, mid_label, bad_label = rule
        int_val = int(value)
        if int_val == zero_thresh:
            return f"Value {int_val} — {zero_label}"
        elif int_val < hi_thresh:
            return f"Value {int_val} — {mid_label}"
        else:
            return f"Value {int_val} — {bad_label}"

    return f"Value {value:.4g}"


# ---------------------------------------------------------------------------
# HistoricalDB
# ---------------------------------------------------------------------------

class HistoricalDB:
    """SQLite-backed store for cross-experiment comparative analysis."""

    def __init__(
        self,
        db_path: str,
        min_experiments: int = 5,
        pattern_promotion_threshold: int = 3,
    ) -> None:
        self.db_path = db_path
        self.min_experiments = min_experiments
        self.pattern_promotion_threshold = pattern_promotion_threshold
        # Ensure tables exist at construction time.
        with self._connect() as conn:
            conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_experiment(
        self,
        experiment_id: str,
        metrics: Any,
        tool: str = "",
        benchmark: str = "",
    ) -> None:
        """
        Persists metrics for a completed experiment.

        *metrics* is expected to be a QuantitativeBundle; scalar values are
        extracted from its sub-objects.  Any missing sub-object silently
        contributes NULLs for its columns.
        """
        now = time.time()

        # --- Extract scalar values from QuantitativeBundle sub-objects ---

        # Top-level / df-derived
        df = getattr(metrics, "df", None)
        n_iterations: Optional[int] = None
        final_best_score: Optional[float] = None
        if df is not None and hasattr(df, "__len__"):
            n_iterations = len(df)
            # Try to get the best child_score from the dataframe
            if hasattr(df, "columns") and "child_score" in df.columns:
                final_best_score = float(df["child_score"].max())

        # Convergence
        conv = getattr(metrics, "convergence", None)
        convergence_rate: Optional[float] = getattr(conv, "convergence_rate", None)
        plateau_onset_iteration: Optional[int] = getattr(conv, "plateau_onset_iteration", None)
        time_to_best_fraction: Optional[float] = getattr(conv, "time_to_best_fraction", None)

        # Stagnation
        stag_periods = getattr(metrics, "stagnation_periods", None) or []
        stagnation_count: int = len(stag_periods)
        max_stagnation_length: int = max(
            (sp.length for sp in stag_periods), default=0
        )

        # Regression
        reg = getattr(metrics, "regression", None)
        regression_frequency: Optional[float] = getattr(reg, "regression_frequency", None)
        mean_recovery_time: Optional[float] = getattr(reg, "mean_recovery_time", None)

        # Efficiency
        eff = getattr(metrics, "efficiency", None)
        improvement_per_llm_call: Optional[float] = getattr(eff, "improvement_per_llm_call", None)
        improvement_per_dollar: Optional[float] = getattr(eff, "improvement_per_dollar", None)
        productive_phase_fraction: Optional[float] = getattr(eff, "productive_phase_fraction", None)

        # Exploration
        expl = getattr(metrics, "exploration", None)
        structural_diversity_index: Optional[float] = getattr(expl, "structural_diversity_index", None)
        exploit_phase_fraction: Optional[float] = getattr(expl, "exploit_phase_fraction", None)

        # Ceiling
        ceil = getattr(metrics, "ceiling", None)
        marginal_improvement_trend: Optional[str] = getattr(ceil, "marginal_improvement_trend", None)
        estimated_gain_probability: Optional[float] = getattr(ceil, "estimated_gain_probability", None)

        sql = """
            INSERT OR REPLACE INTO experiments (
                experiment_id, tool, benchmark, recorded_at, n_iterations,
                final_best_score,
                convergence_rate, plateau_onset_iteration, time_to_best_fraction,
                stagnation_count, max_stagnation_length,
                regression_frequency, mean_recovery_time,
                improvement_per_llm_call, improvement_per_dollar, productive_phase_fraction,
                structural_diversity_index, exploit_phase_fraction,
                marginal_improvement_trend, estimated_gain_probability
            ) VALUES (
                ?, ?, ?, ?, ?,
                ?,
                ?, ?, ?,
                ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?
            )
        """
        params = (
            experiment_id, tool, benchmark, now, n_iterations,
            final_best_score,
            convergence_rate, plateau_onset_iteration, time_to_best_fraction,
            stagnation_count, max_stagnation_length,
            regression_frequency, mean_recovery_time,
            improvement_per_llm_call, improvement_per_dollar, productive_phase_fraction,
            structural_diversity_index, exploit_phase_fraction,
            marginal_improvement_trend, estimated_gain_probability,
        )

        with self._connect() as conn:
            conn.execute(sql, params)
            conn.commit()

    def compare(
        self,
        metric_name: str,
        value: float,
        filters: Optional[Dict[str, str]] = None,
    ) -> HistoricalComparison:
        """
        Returns a HistoricalComparison for *metric_name* vs historical data.

        If sufficient experiments exist (>= min_experiments), compute the
        percentile rank and historical median.  Otherwise fall back to
        heuristic thresholds.
        """
        # Build WHERE clause from optional filters
        where_clauses = [f"{metric_name} IS NOT NULL"]
        params: list = []
        if filters:
            for col, val in filters.items():
                where_clauses.append(f"{col} = ?")
                params.append(val)

        where_sql = " AND ".join(where_clauses)
        query = f"SELECT {metric_name} FROM experiments WHERE {where_sql}"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        historical_values = [row[0] for row in rows if row[0] is not None]
        n_experiments = len(historical_values)

        if n_experiments >= self.min_experiments:
            # Compute percentile: fraction of historical values <= current
            leq = sum(1 for v in historical_values if v <= value)
            percentile = leq / n_experiments

            sorted_vals = sorted(historical_values)
            mid = n_experiments // 2
            if n_experiments % 2 == 0:
                historical_median = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
            else:
                historical_median = sorted_vals[mid]

            pct_int = int(round(percentile * 100))
            if pct_int >= 80:
                summary = f"Better than {pct_int}% of past runs (top {100 - pct_int}%)"
            elif pct_int >= 50:
                summary = f"Above median — better than {pct_int}% of past runs"
            elif pct_int >= 20:
                summary = f"Below median — worse than {100 - pct_int}% of past runs"
            else:
                summary = f"Worse than {100 - pct_int}% of past runs (bottom {pct_int}%)"

            return HistoricalComparison(
                metric_name=metric_name,
                current_value=value,
                percentile=percentile,
                historical_median=historical_median,
                n_experiments=n_experiments,
                rating_basis="historical",
                summary=summary,
            )
        else:
            # Heuristic fallback
            summary = _heuristic_summary(metric_name, value)
            return HistoricalComparison(
                metric_name=metric_name,
                current_value=value,
                percentile=None,
                historical_median=None,
                n_experiments=n_experiments,
                rating_basis="heuristic",
                summary=summary,
            )

    def record_pattern(self, pattern: str) -> None:
        """Upsert a pattern, incrementing its experiment_count."""
        now = time.time()
        sql = """
            INSERT INTO recurring_patterns (pattern, experiment_count, last_seen)
            VALUES (?, 1, ?)
            ON CONFLICT(pattern) DO UPDATE SET
                experiment_count = experiment_count + 1,
                last_seen = excluded.last_seen
        """
        with self._connect() as conn:
            conn.execute(sql, (pattern, now))
            conn.commit()

    def get_recurring_patterns(self) -> List[str]:
        """Return patterns with experiment_count >= pattern_promotion_threshold."""
        sql = "SELECT pattern FROM recurring_patterns WHERE experiment_count >= ?"
        with self._connect() as conn:
            rows = conn.execute(sql, (self.pattern_promotion_threshold,)).fetchall()
        return [row[0] for row in rows]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a sqlite3 connection, committing on success and closing always."""
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(db_file))
        conn.execute("PRAGMA journal_mode=WAL")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                tool TEXT,
                benchmark TEXT,
                recorded_at REAL,
                n_iterations INTEGER,
                final_best_score REAL,
                convergence_rate REAL,
                plateau_onset_iteration INTEGER,
                time_to_best_fraction REAL,
                stagnation_count INTEGER,
                max_stagnation_length INTEGER,
                regression_frequency REAL,
                mean_recovery_time REAL,
                improvement_per_llm_call REAL,
                improvement_per_dollar REAL,
                productive_phase_fraction REAL,
                structural_diversity_index REAL,
                exploit_phase_fraction REAL,
                marginal_improvement_trend TEXT,
                estimated_gain_probability REAL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS recurring_patterns (
                pattern TEXT PRIMARY KEY,
                experiment_count INTEGER DEFAULT 0,
                last_seen REAL
            )
        """)

        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
