"""Unit tests for AlphaEvolve score bridging functions."""

import pytest

from skydiscover.extras.external.alphaevolve_backend import (
    _ae_scores_to_metrics,
    _metrics_to_ae_scores,
)


class TestMetricsToAeScores:
    """Tests for _metrics_to_ae_scores (skydiscover metrics -> AE format)."""

    def test_single_metric(self) -> None:
        result = _metrics_to_ae_scores({"combined_score": 0.85})

        scores_list = result["scores"]["scores"]
        assert len(scores_list) == 1
        assert scores_list[0]["metric"] == "combined_score"
        assert scores_list[0]["score"] == 0.85
        assert result["insights"] == {}

    def test_multiple_metrics(self) -> None:
        result = _metrics_to_ae_scores({"accuracy": 0.9, "speed": 0.7})

        scores_list = result["scores"]["scores"]
        assert len(scores_list) == 2
        metrics_found = {s["metric"]: s["score"] for s in scores_list}
        assert metrics_found["accuracy"] == 0.9
        assert metrics_found["speed"] == 0.7

    def test_filters_non_numeric(self) -> None:
        result = _metrics_to_ae_scores(
            {"score": 0.5, "name": "test", "passed": True}
        )

        scores_list = result["scores"]["scores"]
        assert len(scores_list) == 1
        assert scores_list[0]["metric"] == "score"
        assert scores_list[0]["score"] == 0.5

    def test_empty_metrics(self) -> None:
        result = _metrics_to_ae_scores({})

        assert result == {"scores": {"scores": []}, "insights": {}}


class TestAeScoresToMetrics:
    """Tests for _ae_scores_to_metrics (AE format -> skydiscover metrics)."""

    def test_extracts_combined_score(self) -> None:
        evaluation = {
            "scores": {
                "scores": [{"metric": "combined_score", "score": 0.9}]
            }
        }

        result = _ae_scores_to_metrics(evaluation)
        assert result["combined_score"] == 0.9

    def test_auto_adds_combined_score_from_first(self) -> None:
        evaluation = {
            "scores": {
                "scores": [{"metric": "accuracy", "score": 0.8}]
            }
        }

        result = _ae_scores_to_metrics(evaluation)
        assert result["accuracy"] == 0.8
        assert result["combined_score"] == 0.8

    def test_empty_scores(self) -> None:
        evaluation = {"scores": {"scores": []}}

        result = _ae_scores_to_metrics(evaluation)
        assert result == {}

    def test_none_score_becomes_zero(self) -> None:
        evaluation = {
            "scores": {
                "scores": [{"metric": "combined_score", "score": None}]
            }
        }

        result = _ae_scores_to_metrics(evaluation)
        assert result["combined_score"] == 0.0
