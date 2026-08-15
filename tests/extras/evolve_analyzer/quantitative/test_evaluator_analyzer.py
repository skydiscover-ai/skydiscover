"""
Tests for evaluator_analyzer.py
"""
import pytest

from skydiscover.extras.evolve_analyzer.quantitative.evaluator_analyzer import (
    analyze_evaluator,
    cascade_stage_summary,
    classify_failure,
    detect_sub_metric_divergence,
    flag_high_variance,
)


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
# classify_failure — priority ordering
# ---------------------------------------------------------------------------

class TestClassifyFailurePriority:
    def test_crash_has_highest_priority(self):
        # Even if format_valid is False, crash wins
        rec = make_record(1, child_score=0.5, evaluation_status="crash", format_valid=False)
        assert classify_failure(rec) == "crash"

    def test_timeout_beats_format_invalid(self):
        rec = make_record(1, child_score=0.5, evaluation_status="timeout", format_valid=False)
        assert classify_failure(rec) == "timeout"

    def test_format_invalid_beats_wrong_output(self):
        # format_valid=False should return format_invalid before wrong_output check
        rec = make_record(
            1, child_score=0.5, format_valid=False,
            evaluator_metrics={"correctness": 0},
        )
        assert classify_failure(rec) == "format_invalid"

    def test_success_requires_score_delta_ge_threshold(self):
        rec = make_record(1, child_score=0.6, parent_score=0.5)  # delta = 0.1
        assert classify_failure(rec) == "success"

    def test_success_boundary_exactly_at_threshold(self):
        rec = make_record(1, child_score=0.501, parent_score=0.5, score_delta=0.001)
        assert classify_failure(rec) == "success"

    def test_score_just_below_threshold_not_success(self):
        rec = make_record(1, child_score=0.5009, parent_score=0.5, score_delta=0.0009)
        # No sub-metric divergence → "worse"
        result = classify_failure(rec)
        assert result in ("worse", "partial")

    def test_worse_when_nothing_matches(self):
        rec = make_record(1, child_score=0.5, parent_score=0.5)  # delta = 0, no sub-metrics
        assert classify_failure(rec) == "worse"


# ---------------------------------------------------------------------------
# classify_failure — wrong_output via sub-metrics
# ---------------------------------------------------------------------------

class TestClassifyFailureWrongOutput:
    def test_zero_correctness_metric_is_wrong_output(self):
        rec = make_record(1, child_score=0.5, evaluator_metrics={"correctness": 0})
        assert classify_failure(rec) == "wrong_output"

    def test_zero_accuracy_metric_is_wrong_output(self):
        rec = make_record(1, child_score=0.5, evaluator_metrics={"accuracy": 0.0})
        assert classify_failure(rec) == "wrong_output"

    def test_zero_pass_rate_is_wrong_output(self):
        rec = make_record(1, child_score=0.5, evaluator_metrics={"pass_rate": 0})
        assert classify_failure(rec) == "wrong_output"

    def test_nonzero_correctness_does_not_trigger_wrong_output(self):
        rec = make_record(1, child_score=0.5, evaluator_metrics={"correctness": 0.8})
        # delta = 0 so should be "worse" or "partial", not "wrong_output"
        result = classify_failure(rec)
        assert result != "wrong_output"

    def test_unknown_metric_name_not_matched(self):
        # "score" is not in the correctness set
        rec = make_record(1, child_score=0.5, evaluator_metrics={"score": 0})
        assert classify_failure(rec) != "wrong_output"

    def test_correctness_metric_name_case_insensitive(self):
        rec = make_record(1, child_score=0.5, evaluator_metrics={"CORRECTNESS": 0})
        assert classify_failure(rec) == "wrong_output"

    def test_non_numeric_metric_value_skipped(self):
        rec = make_record(1, child_score=0.5, evaluator_metrics={"correctness": "n/a"})
        # Should not raise; should skip and fall through
        result = classify_failure(rec)
        assert result in ("worse", "partial")


# ---------------------------------------------------------------------------
# classify_failure — partial (sub-metric divergence)
# ---------------------------------------------------------------------------

class TestClassifyFailurePartial:
    def test_partial_when_submetric_improved_despite_no_combined_improvement(self):
        rec = make_record(
            1, child_score=0.5, parent_score=0.5,
            evaluator_metrics={"speed_delta": 0.2},  # a positive _delta key
        )
        assert classify_failure(rec) == "partial"

    def test_score_delta_inferred_from_child_and_parent_when_missing(self):
        rec = {
            "child_score": 0.9,
            "parent_score": 0.5,
            "format_valid": True,
            "evaluation_status": "success",
        }
        assert classify_failure(rec) == "success"


# ---------------------------------------------------------------------------
# detect_sub_metric_divergence
# ---------------------------------------------------------------------------

class TestDetectSubMetricDivergence:
    def test_returns_none_when_combined_score_improved(self):
        rec = make_record(1, child_score=0.9, parent_score=0.5)  # delta = 0.4
        assert detect_sub_metric_divergence(rec) is None

    def test_returns_none_when_no_metrics(self):
        rec = make_record(1, child_score=0.5, parent_score=0.5)
        assert detect_sub_metric_divergence(rec) is None

    def test_detects_positive_delta_key(self):
        rec = make_record(
            1, child_score=0.5, parent_score=0.5,
            evaluator_metrics={"latency_delta": 0.3, "memory_delta": -0.1},
        )
        divergence = detect_sub_metric_divergence(rec)
        assert divergence is not None
        assert "latency_delta" in divergence
        assert divergence["latency_delta"] == pytest.approx(0.3)
        assert "memory_delta" not in divergence  # negative delta not included

    def test_returns_none_when_all_delta_keys_non_positive(self):
        rec = make_record(
            1, child_score=0.5, parent_score=0.5,
            evaluator_metrics={"latency_delta": -0.1},
        )
        assert detect_sub_metric_divergence(rec) is None

    def test_detects_improvement_via_evaluator_artifacts(self):
        rec = make_record(1, child_score=0.5, parent_score=0.5)
        rec["evaluator_artifacts"] = {
            "parent_metrics": {"precision": 0.5, "recall": 0.6},
            "child_metrics": {"precision": 0.7, "recall": 0.5},
        }
        divergence = detect_sub_metric_divergence(rec)
        assert divergence is not None
        assert "precision" in divergence
        assert divergence["precision"] == pytest.approx(0.2)
        assert "recall" not in divergence  # recall went down

    def test_artifacts_with_no_improvement_returns_none(self):
        rec = make_record(1, child_score=0.5, parent_score=0.5)
        rec["evaluator_artifacts"] = {
            "parent_metrics": {"precision": 0.8},
            "child_metrics": {"precision": 0.7},
        }
        assert detect_sub_metric_divergence(rec) is None

    def test_delta_keys_take_precedence_over_artifacts(self):
        # When _delta keys are present, artifacts are NOT consulted
        rec = make_record(1, child_score=0.5, parent_score=0.5)
        rec["evaluator_metrics"] = {"speed_delta": 0.1}
        rec["evaluator_artifacts"] = {
            "parent_metrics": {"precision": 0.9},
            "child_metrics": {"precision": 0.1},
        }
        divergence = detect_sub_metric_divergence(rec)
        assert divergence is not None
        assert "speed_delta" in divergence
        assert "precision" not in divergence

    def test_non_numeric_artifact_values_skipped(self):
        rec = make_record(1, child_score=0.5, parent_score=0.5)
        rec["evaluator_artifacts"] = {
            "parent_metrics": {"tag": "old"},
            "child_metrics": {"tag": "new"},
        }
        # Should not raise
        result = detect_sub_metric_divergence(rec)
        assert result is None


# ---------------------------------------------------------------------------
# cascade_stage_summary
# ---------------------------------------------------------------------------

class TestCascadeStageSummary:
    def test_empty_records(self):
        result = cascade_stage_summary([])
        assert result == {"_total": 0}

    def test_no_cascade_failures(self):
        records = [make_record(i, child_score=0.5) for i in range(1, 4)]
        result = cascade_stage_summary(records)
        assert result == {"_total": 3}

    def test_counts_and_percentages(self):
        records = [
            {**make_record(1, 0.5), "cascade_stage_failed": "stage_a"},
            {**make_record(2, 0.5), "cascade_stage_failed": "stage_a"},
            {**make_record(3, 0.5), "cascade_stage_failed": "stage_b"},
            make_record(4, 0.9),  # no cascade failure
        ]
        result = cascade_stage_summary(records)
        assert result["_total"] == 4
        assert result["stage_a"]["count"] == 2
        assert result["stage_a"]["pct"] == pytest.approx(50.0)
        assert result["stage_b"]["count"] == 1
        assert result["stage_b"]["pct"] == pytest.approx(25.0)

    def test_empty_string_stage_ignored(self):
        records = [
            {**make_record(1, 0.5), "cascade_stage_failed": ""},
            {**make_record(2, 0.5), "cascade_stage_failed": "real_stage"},
        ]
        result = cascade_stage_summary(records)
        assert "" not in result
        assert "real_stage" in result

    def test_single_record_with_cascade_failure(self):
        records = [{**make_record(1, 0.5), "cascade_stage_failed": "parse"}]
        result = cascade_stage_summary(records)
        assert result["_total"] == 1
        assert result["parse"]["count"] == 1
        assert result["parse"]["pct"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# flag_high_variance
# ---------------------------------------------------------------------------

class TestFlagHighVariance:
    def test_returns_false_when_score_std_absent(self):
        rec = make_record(1, child_score=0.5)
        assert flag_high_variance(rec) is False

    def test_returns_false_when_score_std_below_threshold(self):
        rec = make_record(1, child_score=0.5, score_std=0.05)
        assert flag_high_variance(rec, std_threshold=0.1) is False

    def test_returns_false_when_score_std_exactly_at_threshold(self):
        rec = make_record(1, child_score=0.5, score_std=0.1)
        assert flag_high_variance(rec, std_threshold=0.1) is False

    def test_returns_true_when_score_std_above_threshold(self):
        rec = make_record(1, child_score=0.5, score_std=0.15)
        assert flag_high_variance(rec, std_threshold=0.1) is True

    def test_custom_threshold(self):
        rec = make_record(1, child_score=0.5, score_std=0.05)
        assert flag_high_variance(rec, std_threshold=0.03) is True

    def test_non_numeric_score_std_returns_false(self):
        rec = make_record(1, child_score=0.5, score_std="high")
        assert flag_high_variance(rec) is False


# ---------------------------------------------------------------------------
# analyze_evaluator (integration)
# ---------------------------------------------------------------------------

class TestAnalyzeEvaluator:
    def test_tags_failure_mode_and_high_variance(self):
        records = [
            make_record(1, child_score=0.9, parent_score=0.5, score_std=0.2),
            make_record(2, child_score=0.5, evaluation_status="crash"),
        ]
        result = analyze_evaluator(records)
        assert result[0]["failure_mode"] == "success"
        assert result[0]["high_variance"] is True
        assert result[1]["failure_mode"] == "crash"
        assert result[1]["high_variance"] is False

    def test_modifies_records_in_place_and_returns_same_list(self):
        records = [make_record(1, child_score=0.5)]
        returned = analyze_evaluator(records)
        assert returned is records
        assert "failure_mode" in records[0]
        assert "high_variance" in records[0]

    def test_empty_records(self):
        result = analyze_evaluator([])
        assert result == []

    def test_custom_std_threshold_forwarded(self):
        records = [make_record(1, child_score=0.5, score_std=0.05)]
        analyze_evaluator(records, std_threshold=0.03)
        assert records[0]["high_variance"] is True

    def test_all_failure_modes_assigned(self):
        records = [
            make_record(1, child_score=0.5, evaluation_status="crash"),
            make_record(2, child_score=0.5, evaluation_status="timeout"),
            make_record(3, child_score=0.5, format_valid=False),
            make_record(4, child_score=0.5, evaluator_metrics={"correctness": 0}),
            make_record(5, child_score=0.9, parent_score=0.5),
            make_record(6, child_score=0.5, parent_score=0.5),
        ]
        analyze_evaluator(records)
        modes = [r["failure_mode"] for r in records]
        assert modes[0] == "crash"
        assert modes[1] == "timeout"
        assert modes[2] == "format_invalid"
        assert modes[3] == "wrong_output"
        assert modes[4] == "success"
        assert modes[5] in ("worse", "partial")
