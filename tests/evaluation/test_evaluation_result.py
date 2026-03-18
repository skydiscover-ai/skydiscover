"""Tests for EvaluationResult dataclass."""

from skydiscover.evaluation.evaluation_result import EvaluationResult


class TestEvaluationResult:
    def test_from_dict(self):
        result = EvaluationResult.from_dict({"score": 0.5})
        assert result.metrics == {"score": 0.5}
        assert result.artifacts == {}

    def test_to_dict_without_artifacts(self):
        result = EvaluationResult(metrics={"score": 0.5})
        assert result.to_dict() == {"score": 0.5}

    def test_to_dict_with_artifacts(self):
        result = EvaluationResult(
            metrics={"score": 0.5},
            artifacts={"log": "ok"},
        )
        d = result.to_dict()
        assert d["score"] == 0.5
        assert d["artifacts"] == {"log": "ok"}

    def test_default_artifacts_empty(self):
        result = EvaluationResult(metrics={})
        assert result.artifacts == {}
