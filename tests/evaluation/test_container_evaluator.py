"""Tests for ContainerizedEvaluator — pure logic that doesn't need Docker."""

import json
from unittest.mock import patch

import pytest

from skydiscover.evaluation.container_evaluator import ContainerizedEvaluator


@pytest.fixture
def parse_output():
    """Return a bound _parse_output method without starting a real container."""
    inst = object.__new__(ContainerizedEvaluator)
    return inst._parse_output


# ------------------------------------------------------------------
# _parse_output: success cases
# ------------------------------------------------------------------


class TestParseOutputSuccess:
    def test_full_valid_response(self, parse_output):
        stdout = json.dumps({
            "status": "success",
            "combined_score": 0.85,
            "metrics": {"combined_score": 0.85, "accuracy": 0.9, "speed": 0.8},
            "artifacts": {"feedback": "good job"},
        })
        result = parse_output(stdout)
        assert result.metrics["combined_score"] == 0.85
        assert result.metrics["accuracy"] == 0.9
        assert result.metrics["speed"] == 0.8
        assert result.artifacts["feedback"] == "good job"
        assert "status" not in result.artifacts

    def test_combined_score_promoted_to_metrics(self, parse_output):
        stdout = json.dumps({
            "status": "success",
            "combined_score": 0.5,
            "metrics": {"accuracy": 0.5},
        })
        result = parse_output(stdout)
        assert result.metrics["combined_score"] == 0.5
        assert result.metrics["accuracy"] == 0.5

    def test_no_artifacts(self, parse_output):
        stdout = json.dumps({
            "status": "success",
            "combined_score": 1.0,
            "metrics": {"combined_score": 1.0},
        })
        assert parse_output(stdout).artifacts == {}

    def test_integer_metrics_converted_to_float(self, parse_output):
        stdout = json.dumps({
            "status": "success",
            "combined_score": 1,
            "metrics": {"n_correct": 5, "n_total": 5},
        })
        result = parse_output(stdout)
        assert result.metrics["n_correct"] == 5.0
        assert isinstance(result.metrics["n_correct"], float)

    def test_non_numeric_metrics_filtered(self, parse_output):
        stdout = json.dumps({
            "status": "success",
            "combined_score": 0.5,
            "metrics": {"combined_score": 0.5, "label": "fast", "count": 3},
        })
        result = parse_output(stdout)
        assert "label" not in result.metrics
        assert result.metrics["count"] == 3.0

    def test_trailing_whitespace_stripped(self, parse_output):
        stdout = json.dumps({"status": "success", "combined_score": 0.7, "metrics": {}}) + "\n\n"
        assert parse_output(stdout).metrics["combined_score"] == 0.7


# ------------------------------------------------------------------
# _parse_output: error / edge cases
# ------------------------------------------------------------------


class TestParseOutputErrors:
    def test_malformed_json(self, parse_output):
        result = parse_output("not json at all")
        assert result.metrics["error"] == 0.0
        assert "raw_output" in result.artifacts

    def test_empty_string(self, parse_output):
        result = parse_output("")
        assert result.metrics["error"] == 0.0
        assert "raw_output" in result.artifacts

    def test_error_status_surfaces_in_artifacts(self, parse_output):
        stdout = json.dumps({
            "status": "error",
            "combined_score": 0.0,
            "metrics": {"combined_score": 0.0},
            "artifacts": {"error": "segfault"},
        })
        result = parse_output(stdout)
        assert result.metrics["combined_score"] == 0.0
        assert result.artifacts["status"] == "error"
        assert result.artifacts["error"] == "segfault"

    def test_timeout_status(self, parse_output):
        stdout = json.dumps({"status": "timeout", "combined_score": 0.0, "metrics": {}})
        assert parse_output(stdout).artifacts["status"] == "timeout"

    def test_missing_status_defaults_to_error(self, parse_output):
        stdout = json.dumps({"combined_score": 0.5, "metrics": {"combined_score": 0.5}})
        assert parse_output(stdout).artifacts["status"] == "error"

    def test_missing_combined_score_defaults_to_zero(self, parse_output):
        stdout = json.dumps({"status": "success", "metrics": {}})
        assert parse_output(stdout).metrics["combined_score"] == 0.0

    def test_missing_metrics_dict(self, parse_output):
        stdout = json.dumps({"status": "success", "combined_score": 0.3})
        assert parse_output(stdout).metrics["combined_score"] == 0.3

    def test_partial_json_truncated(self, parse_output):
        result = parse_output('{"status": "suc')
        assert result.metrics["error"] == 0.0
        assert "raw_output" in result.artifacts


# ------------------------------------------------------------------
# llm_judge attribute
# ------------------------------------------------------------------


class TestLlmJudgeAttribute:
    def test_init_sets_llm_judge_to_none(self):
        """ContainerizedEvaluator.__init__ must set self.llm_judge before Docker calls."""
        with patch.object(ContainerizedEvaluator, "_build_image", return_value="fake:latest"), \
             patch.object(ContainerizedEvaluator, "_start_container", return_value="abc123"):
            from skydiscover.config import EvaluatorConfig

            inst = ContainerizedEvaluator.__new__(ContainerizedEvaluator)
            ContainerizedEvaluator.__init__(inst, "/tmp/fake", EvaluatorConfig())
            assert inst.llm_judge is None
