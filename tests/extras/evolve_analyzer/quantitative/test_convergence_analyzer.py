"""
Tests for convergence_analyzer.py
"""
import math

import pytest

from skydiscover.extras.evolve_analyzer.quantitative.convergence_analyzer import analyze_convergence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_record(
    iteration,
    child_score,
    parent_score=None,
    score_delta=None,
    evaluation_status="success",
    format_valid=True,
    **kwargs,
):
    r = {
        "iteration": iteration,
        "child_score": child_score,
        "evaluation_status": evaluation_status,
        "format_valid": format_valid,
    }
    if parent_score is not None:
        r["parent_score"] = parent_score
    if score_delta is not None:
        r["score_delta"] = score_delta
    elif parent_score is not None:
        r["score_delta"] = child_score - parent_score
    else:
        r["score_delta"] = 0.0
    r.update(kwargs)
    return r


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

class TestEmptyInput:
    def test_empty_records_returns_zero_metrics(self):
        m = analyze_convergence([])
        assert m.best_so_far_curve == []
        assert m.rolling_mean == []
        assert m.rolling_variance == []
        assert m.change_points == []
        assert m.convergence_rate == 0.0
        assert m.improvement_per_eval == 0.0
        assert m.time_to_best_fraction == 0.0
        assert m.plateau_onset_iteration is None


# ---------------------------------------------------------------------------
# Single record
# ---------------------------------------------------------------------------

class TestSingleRecord:
    def test_single_record_best_curve_has_one_entry(self):
        rec = make_record(1, child_score=0.7)
        m = analyze_convergence([rec])
        assert len(m.best_so_far_curve) == 1
        assert m.best_so_far_curve[0] == pytest.approx(0.7)

    def test_single_record_rolling_stats(self):
        rec = make_record(1, child_score=0.7)
        m = analyze_convergence([rec])
        assert len(m.rolling_mean) == 1
        assert m.rolling_mean[0] == pytest.approx(0.7)
        assert m.rolling_variance[0] == pytest.approx(0.0)

    def test_single_record_no_change_points(self):
        rec = make_record(1, child_score=0.7)
        m = analyze_convergence([rec])
        assert m.change_points == []

    def test_single_record_no_plateau(self):
        rec = make_record(1, child_score=0.7)
        m = analyze_convergence([rec])
        assert m.plateau_onset_iteration is None

    def test_single_record_convergence_rate(self):
        # initial_best == final_best → rate = 0 / 1 = 0.0
        rec = make_record(1, child_score=0.7)
        m = analyze_convergence([rec])
        assert m.convergence_rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Best-so-far curve
# ---------------------------------------------------------------------------

class TestBestSoFarCurve:
    def test_monotonically_increasing_scores(self):
        records = [make_record(i, child_score=i * 0.1) for i in range(1, 6)]
        m = analyze_convergence(records)
        # Each step should be the new high
        for i, val in enumerate(m.best_so_far_curve):
            assert val == pytest.approx((i + 1) * 0.1)

    def test_curve_never_decreases(self):
        scores = [0.9, 0.3, 0.8, 0.5, 0.7]
        records = [make_record(i + 1, child_score=s) for i, s in enumerate(scores)]
        m = analyze_convergence(records)
        for a, b in zip(m.best_so_far_curve, m.best_so_far_curve[1:]):
            assert b >= a

    def test_crash_records_excluded_from_best(self):
        records = [
            make_record(1, child_score=0.5),
            make_record(2, child_score=0.9, evaluation_status="crash"),
            make_record(3, child_score=0.6),
        ]
        m = analyze_convergence(records)
        # Crash at iteration 2 should not update best
        assert m.best_so_far_curve[1] == pytest.approx(0.5)
        assert m.best_so_far_curve[2] == pytest.approx(0.6)

    def test_all_crash_records_produce_nan_curve(self):
        records = [
            make_record(i, child_score=0.9, evaluation_status="crash")
            for i in range(1, 4)
        ]
        m = analyze_convergence(records)
        for val in m.best_so_far_curve:
            assert math.isnan(val)

    def test_curve_length_matches_record_count(self):
        records = [make_record(i, child_score=0.5) for i in range(1, 11)]
        m = analyze_convergence(records)
        assert len(m.best_so_far_curve) == 10

    def test_records_sorted_by_iteration_before_curve(self):
        # Supply records out of order
        records = [
            make_record(3, child_score=0.9),
            make_record(1, child_score=0.3),
            make_record(2, child_score=0.6),
        ]
        m = analyze_convergence(records)
        # After sorting: 0.3, 0.6, 0.9 → curve should be monotone
        assert m.best_so_far_curve[0] == pytest.approx(0.3)
        assert m.best_so_far_curve[1] == pytest.approx(0.6)
        assert m.best_so_far_curve[2] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Rolling mean and variance
# ---------------------------------------------------------------------------

class TestRollingStats:
    def test_rolling_mean_within_window(self):
        scores = [0.2, 0.4, 0.6]
        records = [make_record(i + 1, child_score=s) for i, s in enumerate(scores)]
        m = analyze_convergence(records, window=10)
        # All records fit in one window
        assert m.rolling_mean[0] == pytest.approx(0.2)
        assert m.rolling_mean[1] == pytest.approx((0.2 + 0.4) / 2)
        assert m.rolling_mean[2] == pytest.approx((0.2 + 0.4 + 0.6) / 3)

    def test_rolling_variance_with_uniform_scores(self):
        records = [make_record(i + 1, child_score=0.5) for i in range(5)]
        m = analyze_convergence(records, window=5)
        for v in m.rolling_variance:
            assert v == pytest.approx(0.0)

    def test_rolling_window_slides_correctly(self):
        # window=2; each position uses at most last 2 scores
        scores = [0.0, 1.0, 0.0, 1.0]
        records = [make_record(i + 1, child_score=s) for i, s in enumerate(scores)]
        m = analyze_convergence(records, window=2)
        # At index 1: mean of [0.0, 1.0] = 0.5
        assert m.rolling_mean[1] == pytest.approx(0.5)
        # At index 2: mean of [1.0, 0.0] = 0.5
        assert m.rolling_mean[2] == pytest.approx(0.5)

    def test_nan_scores_excluded_from_rolling_stats(self):
        # Records without child_score produce NaN scores
        records = [
            make_record(1, child_score=0.4),
            {"iteration": 2, "evaluation_status": "crash"},  # no child_score key → NaN
            make_record(3, child_score=0.6),
        ]
        m = analyze_convergence(records, window=3)
        # Window at index 2: valid scores are 0.4 and 0.6
        assert m.rolling_mean[2] == pytest.approx((0.4 + 0.6) / 2)

    def test_rolling_lists_same_length_as_records(self):
        records = [make_record(i, child_score=0.5) for i in range(1, 8)]
        m = analyze_convergence(records)
        assert len(m.rolling_mean) == 7
        assert len(m.rolling_variance) == 7


# ---------------------------------------------------------------------------
# Change points
# ---------------------------------------------------------------------------

class TestChangePoints:
    def test_no_improvement_no_change_points(self):
        records = [make_record(i, child_score=0.5) for i in range(1, 5)]
        m = analyze_convergence(records)
        assert m.change_points == []

    def test_every_step_improves_all_are_change_points(self):
        records = [make_record(i, child_score=i * 0.1) for i in range(1, 5)]
        m = analyze_convergence(records)
        # Change points at iterations 2, 3, 4 (each improves over previous)
        assert m.change_points == [2, 3, 4]

    def test_single_improvement_one_change_point(self):
        records = [
            make_record(1, child_score=0.3),
            make_record(2, child_score=0.3),
            make_record(3, child_score=0.8),
            make_record(4, child_score=0.7),
        ]
        m = analyze_convergence(records)
        assert m.change_points == [3]

    def test_change_points_use_iteration_value_not_index(self):
        # Records with non-consecutive iterations
        records = [
            make_record(10, child_score=0.3),
            make_record(20, child_score=0.7),
        ]
        m = analyze_convergence(records)
        assert 20 in m.change_points


# ---------------------------------------------------------------------------
# Scalar summary metrics
# ---------------------------------------------------------------------------

class TestScalarMetrics:
    def test_convergence_rate_positive_improvement(self):
        records = [
            make_record(1, child_score=0.3),
            make_record(2, child_score=0.7),
        ]
        m = analyze_convergence(records)
        # (0.7 - 0.3) / 2 = 0.2
        assert m.convergence_rate == pytest.approx(0.2)

    def test_improvement_per_eval_excludes_crashes(self):
        records = [
            make_record(1, child_score=0.3),
            make_record(2, child_score=0.9, evaluation_status="crash"),
            make_record(3, child_score=0.7),
        ]
        m = analyze_convergence(records)
        # final_best = 0.7, initial_best = 0.3, non_crash_count = 2
        assert m.improvement_per_eval == pytest.approx((0.7 - 0.3) / 2)

    def test_improvement_per_eval_zero_when_all_crash(self):
        records = [
            make_record(i, child_score=0.9, evaluation_status="crash")
            for i in range(1, 4)
        ]
        m = analyze_convergence(records)
        assert m.improvement_per_eval == pytest.approx(0.0)

    def test_time_to_best_fraction_immediate_best(self):
        # Best is at the very first record
        records = [
            make_record(1, child_score=0.9),
            make_record(2, child_score=0.5),
            make_record(3, child_score=0.4),
        ]
        m = analyze_convergence(records)
        # Best found at index 0 → 0/(3-1) = 0.0 (found immediately)
        assert m.time_to_best_fraction == pytest.approx(0.0)

    def test_time_to_best_fraction_last_record(self):
        records = [
            make_record(1, child_score=0.3),
            make_record(2, child_score=0.5),
            make_record(3, child_score=0.9),
        ]
        m = analyze_convergence(records)
        # Best at iteration 3, total = 3 → 3/3 = 1.0
        assert m.time_to_best_fraction == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Plateau onset
# ---------------------------------------------------------------------------

class TestPlateauOnset:
    def test_plateau_detected_when_improvement_stops_early(self):
        records = [
            make_record(1, child_score=0.3),
            make_record(2, child_score=0.7),   # last improvement
            make_record(3, child_score=0.6),
            make_record(4, child_score=0.65),
        ]
        m = analyze_convergence(records)
        # Improvement last at index 1 (iteration 2); plateau starts at iteration 3
        assert m.plateau_onset_iteration == 3

    def test_no_plateau_when_improvement_at_last_step(self):
        records = [
            make_record(1, child_score=0.3),
            make_record(2, child_score=0.5),
            make_record(3, child_score=0.9),
        ]
        m = analyze_convergence(records)
        # Last improvement is at the final step → no plateau
        assert m.plateau_onset_iteration is None

    def test_no_plateau_when_all_same_score(self):
        records = [make_record(i, child_score=0.5) for i in range(1, 5)]
        m = analyze_convergence(records)
        assert m.plateau_onset_iteration is None

    def test_plateau_uses_iteration_value(self):
        records = [
            make_record(100, child_score=0.3),
            make_record(200, child_score=0.7),
            make_record(300, child_score=0.65),
        ]
        m = analyze_convergence(records)
        assert m.plateau_onset_iteration == 300

    def test_plateau_onset_none_for_single_record(self):
        m = analyze_convergence([make_record(1, child_score=0.5)])
        assert m.plateau_onset_iteration is None


# ---------------------------------------------------------------------------
# Output structure completeness
# ---------------------------------------------------------------------------

class TestOutputStructure:
    def test_all_list_fields_same_length(self):
        records = [make_record(i, child_score=i * 0.05) for i in range(1, 11)]
        m = analyze_convergence(records)
        n = len(records)
        assert len(m.best_so_far_curve) == n
        assert len(m.rolling_mean) == n
        assert len(m.rolling_variance) == n

    def test_returned_type_is_convergence_metrics(self):
        from skydiscover.extras.evolve_analyzer.quantitative.bundle import ConvergenceMetrics
        m = analyze_convergence([make_record(1, child_score=0.5)])
        assert isinstance(m, ConvergenceMetrics)

    def test_change_points_are_ints(self):
        records = [
            make_record(1, child_score=0.3),
            make_record(2, child_score=0.9),
        ]
        m = analyze_convergence(records)
        for cp in m.change_points:
            assert isinstance(cp, int)
