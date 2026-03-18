"""Tests for Evaluator — result normalization and cascade threshold logic."""

from skydiscover.evaluation.evaluation_result import EvaluationResult
from skydiscover.evaluation.evaluator import Evaluator


def _make_evaluator():
    """Create an Evaluator without loading an evaluation file."""
    inst = object.__new__(Evaluator)
    return inst


class TestNormalizeResult:
    def test_evaluation_result_passthrough(self):
        inst = _make_evaluator()
        original = EvaluationResult(metrics={"score": 1.0})
        assert inst._normalize_result(original) is original

    def test_dict_converted(self):
        inst = _make_evaluator()
        result = inst._normalize_result({"accuracy": 0.9, "speed": 0.8})
        assert isinstance(result, EvaluationResult)
        assert result.metrics == {"accuracy": 0.9, "speed": 0.8}

    def test_unexpected_type_returns_error(self):
        inst = _make_evaluator()
        result = inst._normalize_result("unexpected string")
        assert result.metrics["error"] == 0.0

    def test_none_returns_error(self):
        inst = _make_evaluator()
        result = inst._normalize_result(None)
        assert result.metrics["error"] == 0.0


class TestPassesThreshold:
    def test_combined_score_above(self):
        inst = _make_evaluator()
        assert inst._passes_threshold({"combined_score": 0.8}, 0.5) is True

    def test_combined_score_below(self):
        inst = _make_evaluator()
        assert inst._passes_threshold({"combined_score": 0.3}, 0.5) is False

    def test_combined_score_exact(self):
        inst = _make_evaluator()
        assert inst._passes_threshold({"combined_score": 0.5}, 0.5) is True

    def test_average_fallback(self):
        inst = _make_evaluator()
        # No combined_score — should average all numeric values.
        assert inst._passes_threshold({"a": 0.6, "b": 0.8}, 0.5) is True

    def test_empty_metrics(self):
        inst = _make_evaluator()
        assert inst._passes_threshold({}, 0.5) is False
