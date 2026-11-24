"""Regression tests for the rectangular circle-packing benchmark."""

from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.math.circle_packing_rect.evaluator import evaluator


@pytest.fixture
def circles(monkeypatch):
    packing = np.array([(x * 0.12, y * 0.12, 0.05) for x in range(7) for y in range(3)])
    program = SimpleNamespace(circle_packing21=lambda: packing)
    monkeypatch.setattr(evaluator, "__import__", lambda _: program)
    return packing


@pytest.mark.parametrize(("column", "value"), [(0, np.nan), (1, np.inf), (2, -np.inf)])
def test_rejects_nonfinite_circle_fields(circles, column, value):
    circles[0, column] = value
    result = evaluator.evaluate("packing.py")
    assert result["combined_score"] == 0.0
    assert result["error"] == "Circle coordinates and radii must be finite."


def test_scores_finite_packing(circles):
    result = evaluator.evaluate("packing.py")
    assert "error" not in result
    assert result["radii_sum"] == pytest.approx(21 * 0.05)
    assert result["combined_score"] == pytest.approx(21 * 0.05 / evaluator.BENCHMARK)
