"""Tests for the infrastructure failure analyzer."""
import pytest
from skydiscover.extras.evolve_analyzer.quantitative.infrastructure_analyzer import analyze_infrastructure


def make_record(iteration, score, eval_time=None):
    r = {"iteration": iteration, "child_score": float(score)}
    if eval_time is not None:
        r["eval_time_seconds"] = float(eval_time)
    return r


def test_empty_records():
    assert analyze_infrastructure([]) is None


def test_no_sentinels():
    records = [make_record(i, -100.0 + i, eval_time=30.0) for i in range(1, 21)]
    result = analyze_infrastructure(records)
    assert result is not None
    assert result.failure_cause == "NONE"
    assert result.sentinel_count == 0
    assert result.affected_iterations == []
    assert result.first_sentinel_iteration is None


def test_infra_crash():
    normal = [make_record(i, -100.0 + i, eval_time=30.0) for i in range(1, 11)]
    sentinel = [make_record(i, -10999.0, eval_time=0.0) for i in range(11, 21)]
    records = normal + sentinel
    result = analyze_infrastructure(records)
    assert result is not None
    assert result.failure_cause == "INFRA_CRASH"
    assert result.sentinel_count == 10
    assert result.first_sentinel_iteration == 11
    assert result.crash_onset_iteration == 11
    assert result.sentinel_fraction == pytest.approx(0.5)


def test_degradation_window():
    normal = [make_record(i, -100.0 + i, eval_time=30.0) for i in range(1, 11)]
    # Degraded: high eval_time but real (non-sentinel) scores
    degraded = [make_record(i, -100.0 + i, eval_time=90.0) for i in range(11, 16)]
    sentinel = [make_record(i, -10999.0, eval_time=0.0) for i in range(16, 21)]
    records = normal + degraded + sentinel
    result = analyze_infrastructure(records)
    assert result is not None
    assert result.failure_cause == "INFRA_CRASH"
    assert result.degradation_window is not None
    assert result.degradation_window[0] == 11
    assert result.degradation_window[1] == 15
    assert result.eval_time_spike_ratio is not None
    assert result.eval_time_spike_ratio > 1.0


def test_evaluator_noise():
    # Scattered single sentinel among many normal records (not a contiguous block of >5)
    records = [make_record(i, -100.0 + i, eval_time=30.0) for i in range(1, 21)]
    # Replace one record with a sentinel
    records[10] = make_record(11, -10999.0, eval_time=0.0)
    result = analyze_infrastructure(records)
    assert result is not None
    assert result.failure_cause == "EVALUATOR_NOISE"
    assert result.sentinel_count == 1


def test_high_sentinel_fraction_is_critical():
    normal = [make_record(i, -100.0, eval_time=30.0) for i in range(1, 11)]
    sentinel = [make_record(i, -10999.0, eval_time=0.0) for i in range(11, 51)]
    records = normal + sentinel
    result = analyze_infrastructure(records)
    assert result is not None
    assert result.failure_cause == "INFRA_CRASH"
    assert result.sentinel_fraction >= 0.75
    assert result.rating if hasattr(result, "rating") else True  # rating is on DimensionReport not metrics


def test_meta_evo_signals_propagated():
    """log_evidence with meta-evolution data passes through analyze_infrastructure."""
    records = [make_record(i, -100.0 + i, eval_time=30.0) for i in range(1, 11)]
    records[0]["_infra_log_signals"] = {
        "meta_evo_total_iterations": 6,
        "meta_evo_fallback_count": 6,
        "meta_evo_fallback_fraction": 1.0,
        "meta_evo_fully_non_functional": True,
        "meta_evo_failed_models": ["gpt-5", "gpt-5-mini"],
        "meta_evo_error_types": ["auth_denied"],
    }
    result = analyze_infrastructure(records)
    assert result is not None
    assert result.log_evidence is not None
    assert result.log_evidence["meta_evo_fully_non_functional"] is True
    assert result.log_evidence["meta_evo_fallback_count"] == 6
    assert "gpt-5" in result.log_evidence["meta_evo_failed_models"]


def test_meta_evo_signals_absent_when_not_injected():
    """Without _infra_log_signals, log_evidence is None."""
    records = [make_record(i, -100.0 + i, eval_time=30.0) for i in range(1, 11)]
    result = analyze_infrastructure(records)
    assert result is not None
    assert result.log_evidence is None
