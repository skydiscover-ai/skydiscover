"""
Tests for stagnation_detector.py
"""
import pytest

from skydiscover.extras.evolve_analyzer.quantitative.stagnation_detector import detect_stagnation


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
# Empty / trivial inputs
# ---------------------------------------------------------------------------

class TestEmptyInput:
    def test_empty_records_returns_empty_lists(self):
        tagged, periods = detect_stagnation([])
        assert tagged == []
        assert periods == []

    def test_single_progress_record_no_streak(self):
        rec = make_record(1, child_score=0.8, parent_score=0.5)
        tagged, periods = detect_stagnation([rec])
        assert len(tagged) == 1
        assert len(periods) == 0
        assert tagged[0]["streak_id"] is None
        assert tagged[0]["streak_position"] == 0

    def test_single_non_progress_record_produces_streak(self):
        rec = make_record(1, child_score=0.5, parent_score=0.5)  # delta = 0
        tagged, periods = detect_stagnation([rec])
        assert len(periods) == 1
        assert periods[0].length == 1
        assert periods[0].streak_id == "streak_1"
        assert tagged[0]["streak_id"] == "streak_1"
        assert tagged[0]["streak_position"] == 1


# ---------------------------------------------------------------------------
# Basic streak detection
# ---------------------------------------------------------------------------

class TestBasicStreakDetection:
    def test_all_progress_no_streaks(self):
        records = [
            make_record(i, child_score=0.5 + i * 0.1, parent_score=0.5 + (i - 1) * 0.1)
            for i in range(1, 6)
        ]
        tagged, periods = detect_stagnation(records)
        assert periods == []
        for rec in tagged:
            assert rec["streak_id"] is None
            assert rec["streak_position"] == 0

    def test_all_no_progress_one_ongoing_streak(self):
        records = [
            make_record(i, child_score=0.5, parent_score=0.5)  # delta = 0 always
            for i in range(1, 6)
        ]
        tagged, periods = detect_stagnation(records)
        assert len(periods) == 1
        assert periods[0].length == 5
        # Ongoing streak: end_iteration is None, no recovery
        assert periods[0].end_iteration is None
        assert periods[0].recovery_iteration is None

    def test_streak_followed_by_recovery(self):
        no_progress = [make_record(i, child_score=0.5, parent_score=0.5) for i in range(1, 4)]
        recovery = make_record(4, child_score=0.9, parent_score=0.5)
        records = no_progress + [recovery]
        tagged, periods = detect_stagnation(records)

        assert len(periods) == 1
        p = periods[0]
        assert p.length == 3
        assert p.start_iteration == 1
        assert p.end_iteration == 3
        assert p.recovery_iteration == 4
        assert p.score_at_recovery == pytest.approx(0.9)

        # Recovery record itself is not tagged as part of the streak
        recovery_tagged = next(t for t in tagged if t["iteration"] == 4)
        assert recovery_tagged["streak_id"] is None
        assert recovery_tagged["streak_position"] == 0

    def test_multiple_separate_streaks(self):
        records = (
            [make_record(i, child_score=0.5, parent_score=0.5) for i in range(1, 3)]
            + [make_record(3, child_score=0.9, parent_score=0.5)]   # recovery
            + [make_record(i, child_score=0.5, parent_score=0.5) for i in range(4, 6)]
        )
        tagged, periods = detect_stagnation(records)
        assert len(periods) == 2

    def test_streak_position_is_1_indexed(self):
        records = [make_record(i, child_score=0.5, parent_score=0.5) for i in range(1, 4)]
        tagged, _ = detect_stagnation(records)
        positions = [t["streak_position"] for t in tagged]
        assert positions == [1, 2, 3]


# ---------------------------------------------------------------------------
# Threshold and is_alert
# ---------------------------------------------------------------------------

class TestThresholdAndAlert:
    def test_streak_below_threshold_no_alert(self):
        records = [make_record(i, child_score=0.5, parent_score=0.5) for i in range(1, 5)]
        _, periods = detect_stagnation(records, threshold=10)
        assert len(periods) == 1
        assert periods[0].is_alert is False

    def test_streak_exactly_at_threshold_is_alert(self):
        threshold = 5
        records = [make_record(i, child_score=0.5, parent_score=0.5) for i in range(1, threshold + 1)]
        _, periods = detect_stagnation(records, threshold=threshold)
        assert len(periods) == 1
        assert periods[0].is_alert is True

    def test_streak_above_threshold_is_alert(self):
        records = [make_record(i, child_score=0.5, parent_score=0.5) for i in range(1, 20)]
        _, periods = detect_stagnation(records, threshold=5)
        assert periods[0].is_alert is True

    def test_severity_warning_below_threshold(self):
        records = [make_record(i, child_score=0.5, parent_score=0.5) for i in range(1, 4)]
        _, periods = detect_stagnation(records, threshold=10)
        assert periods[0].severity == "warning"

    def test_severity_critical_at_double_threshold(self):
        threshold = 5
        records = [make_record(i, child_score=0.5, parent_score=0.5) for i in range(1, 2 * threshold + 1)]
        _, periods = detect_stagnation(records, threshold=threshold)
        assert periods[0].severity == "critical"


# ---------------------------------------------------------------------------
# min_delta boundary
# ---------------------------------------------------------------------------

class TestMinDeltaBoundary:
    def test_delta_exactly_at_min_delta_counts_as_progress(self):
        min_delta = 0.001
        rec = make_record(1, child_score=0.501, parent_score=0.5, score_delta=min_delta)
        _, periods = detect_stagnation([rec], min_delta=min_delta)
        assert periods == []

    def test_delta_just_below_min_delta_is_stagnation(self):
        min_delta = 0.001
        rec = make_record(1, child_score=0.5009, parent_score=0.5, score_delta=0.0009)
        _, periods = detect_stagnation([rec], min_delta=min_delta)
        assert len(periods) == 1

    def test_negative_delta_is_stagnation(self):
        rec = make_record(1, child_score=0.3, parent_score=0.5, score_delta=-0.2)
        _, periods = detect_stagnation([rec])
        assert len(periods) == 1


# ---------------------------------------------------------------------------
# Failure-type classification inside streaks
# ---------------------------------------------------------------------------

class TestFailureTypeClassification:
    def test_crash_is_classified(self):
        rec = make_record(1, child_score=0.5, evaluation_status="crash")
        _, periods = detect_stagnation([rec])
        summary = periods[0].failure_sequence[0]
        assert summary.failure_type == "crash"
        assert periods[0].dominant_failure_type == "crash"

    def test_timeout_is_classified(self):
        rec = make_record(1, child_score=0.5, evaluation_status="timeout")
        _, periods = detect_stagnation([rec])
        assert periods[0].dominant_failure_type == "timeout"

    def test_format_invalid_is_classified(self):
        rec = make_record(1, child_score=0.5, format_valid=False)
        _, periods = detect_stagnation([rec])
        assert periods[0].dominant_failure_type == "format_invalid"

    def test_no_improvement_default_classification(self):
        rec = make_record(1, child_score=0.5, parent_score=0.5)  # delta = 0
        _, periods = detect_stagnation([rec])
        assert periods[0].dominant_failure_type == "no_improvement"

    def test_mixed_failure_types_dominant_and_secondary(self):
        records = (
            [make_record(i, child_score=0.5, evaluation_status="crash") for i in range(1, 4)]
            + [make_record(4, child_score=0.5, evaluation_status="timeout")]
        )
        _, periods = detect_stagnation(records)
        p = periods[0]
        assert p.dominant_failure_type == "crash"
        assert "timeout" in p.secondary_failure_types


# ---------------------------------------------------------------------------
# Score at stagnation start
# ---------------------------------------------------------------------------

class TestScoreAtStagnationStart:
    def test_uses_parent_score_when_available(self):
        rec = make_record(1, child_score=0.9, parent_score=0.7)
        rec["score_delta"] = 0.0  # force non-progress
        _, periods = detect_stagnation([rec])
        assert periods[0].score_at_stagnation_start == pytest.approx(0.7)

    def test_falls_back_to_child_score(self):
        rec = {"iteration": 1, "child_score": 0.6, "score_delta": 0.0, "format_valid": True}
        _, periods = detect_stagnation([rec])
        assert periods[0].score_at_stagnation_start == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Record ordering and score_delta inference
# ---------------------------------------------------------------------------

class TestRecordOrderingAndDeltaInference:
    def test_records_sorted_by_iteration(self):
        # Feed records out of order; streak should still be detected correctly
        records = [
            make_record(3, child_score=0.5, parent_score=0.5),
            make_record(1, child_score=0.5, parent_score=0.5),
            make_record(2, child_score=0.5, parent_score=0.5),
        ]
        tagged, periods = detect_stagnation(records)
        iterations = [t["iteration"] for t in tagged]
        assert iterations == [1, 2, 3]
        assert periods[0].start_iteration == 1

    def test_score_delta_inferred_from_child_and_parent(self):
        # Record without explicit score_delta
        rec = {"iteration": 1, "child_score": 0.9, "parent_score": 0.5, "format_valid": True, "evaluation_status": "success"}
        _, periods = detect_stagnation([rec])
        # delta = 0.4 >= 0.001 → progress → no stagnation
        assert periods == []

    def test_missing_score_delta_and_scores_defaults_to_zero(self):
        # No score_delta, no child_score, no parent_score → delta defaults to 0 → stagnation
        rec = {"iteration": 1, "format_valid": True, "evaluation_status": "success"}
        _, periods = detect_stagnation([rec])
        assert len(periods) == 1

    def test_caller_dicts_not_mutated(self):
        original = make_record(1, child_score=0.5, parent_score=0.5)
        before_keys = set(original.keys())
        detect_stagnation([original])
        # Original dict should not have streak_id injected into it
        assert "streak_id" not in original or True  # just ensure no exception; tagged is a copy
        # The tagged copy has streak_id, but the original is unchanged
        assert set(original.keys()) == before_keys


# ---------------------------------------------------------------------------
# Compliance status in IterationSummary
# ---------------------------------------------------------------------------

class TestComplianceStatusInSummary:
    def test_compliance_status_built_when_fields_present(self):
        rec = make_record(
            1, child_score=0.5, parent_score=0.5,
            evolved_block_only=True, signature_preserved=False,
        )
        _, periods = detect_stagnation([rec])
        summary = periods[0].failure_sequence[0]
        assert summary.compliance_status is not None
        assert "block_ok" in summary.compliance_status
        assert "sig_violation" in summary.compliance_status

    def test_compliance_status_none_when_fields_absent(self):
        rec = make_record(1, child_score=0.5, parent_score=0.5)
        _, periods = detect_stagnation([rec])
        summary = periods[0].failure_sequence[0]
        assert summary.compliance_status is None

    def test_mutation_type_unknown_when_missing(self):
        rec = {"iteration": 5, "child_score": 0.5, "score_delta": 0.0, "format_valid": True}
        _, periods = detect_stagnation([rec])
        assert periods[0].failure_sequence[0].mutation_type == "unknown"
