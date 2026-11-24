"""Validation checks for the autocorrelation and hexagon-packing benchmarks."""

from importlib import import_module

import numpy as np
import pytest

from benchmarks.math.second_autocorr_ineq.evaluator.evaluator import verify_c2_solution
from benchmarks.math.third_autocorr_ineq.evaluator.evaluator import verify_c3_solution


@pytest.mark.parametrize("verify", [verify_c2_solution, verify_c3_solution])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_autocorrelation_rejects_nonfinite_samples(verify, value):
    samples = np.ones(4)
    samples[0] = value
    with pytest.raises(ValueError, match="Function values must be finite"):
        verify(samples, 2.0, len(samples))


def test_autocorrelation_accepts_finite_samples():
    samples = np.ones(4)
    assert verify_c2_solution(samples, 2 / 3, len(samples)) == pytest.approx(2 / 3)
    verify_c3_solution(samples, 2.0, len(samples))


@pytest.fixture(params=[11, 12])
def hexagon_case(request):
    size = request.param
    package = f"benchmarks.math.hexagon_packing.{size}"
    evaluator = import_module(f"{package}.evaluator.evaluator")
    initial = import_module(f"{package}.initial_program")
    inner, outer, side = getattr(initial, f"hexagon_packing_{size}")()
    return evaluator, inner.astype(float), outer.astype(float), side


def test_hexagons_accept_finite_construction(hexagon_case):
    evaluator, inner, outer, side = hexagon_case
    evaluator.verify_construction(inner, outer[:2], side, outer[2])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("inner", np.nan, "Inner hexagon parameters must be finite"),
        ("center", np.inf, "Outer hexagon center and angle must be finite"),
        ("angle", np.nan, "Outer hexagon center and angle must be finite"),
        ("side", np.nan, "Outer hexagon side length must be finite and positive"),
        ("side", np.inf, "Outer hexagon side length must be finite and positive"),
        ("side", -np.inf, "Outer hexagon side length must be finite and positive"),
        ("side", 0.0, "Outer hexagon side length must be finite and positive"),
        ("side", -1.0, "Outer hexagon side length must be finite and positive"),
    ],
)
def test_hexagons_reject_invalid_parameters(hexagon_case, field, value, message):
    evaluator, inner, outer, side = hexagon_case
    if field == "inner":
        inner[0, 0] = value
    elif field == "center":
        outer[0] = value
    elif field == "angle":
        outer[2] = value
    else:
        side = value

    with pytest.raises(ValueError, match=message):
        evaluator.verify_construction(inner, outer[:2], side, outer[2])
