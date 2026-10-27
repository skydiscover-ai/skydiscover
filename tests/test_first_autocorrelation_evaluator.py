"""Invalid samples must never earn a score in the first autocorrelation benchmark."""

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/math/first_autocorr_ineq/evaluator/evaluator.py"
)
_SPEC = importlib.util.spec_from_file_location("_first_autocorrelation_evaluator", _PATH)
evaluator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluator)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
@pytest.mark.parametrize("all_samples", [True, False], ids=["all_samples", "one_sample"])
def test_non_finite_samples_receive_no_score(tmp_path, value, all_samples):
    program = tmp_path / "_first_autocorrelation_candidate.py"
    samples = f"np.full(10, float('{value}'))" if all_samples else "np.ones(10)"
    program.write_text(
        "import numpy as np\n"
        "def run():\n"
        f"    samples = {samples}\n"
        f"    samples[0] = float('{value}')\n"
        "    return samples, 0.5, 0.0, 10\n",
        encoding="utf-8",
    )
    try:
        result = evaluator.evaluate(str(program))
    finally:
        sys.modules.pop(program.stem, None)

    assert result["combined_score"] == 0.0
    assert "finite" in result["error"]
    assert "c1" not in result


def test_finite_solution_still_receives_its_score(tmp_path):
    program = tmp_path / "_first_autocorrelation_candidate.py"
    program.write_text(
        "import numpy as np\n" "def run():\n" "    return np.ones(10), 2.0, 0.0, 10\n",
        encoding="utf-8",
    )
    try:
        result = evaluator.evaluate(str(program))
    finally:
        sys.modules.pop(program.stem, None)

    assert "error" not in result
    assert result["c1"] == 2.0
    assert result["combined_score"] == pytest.approx(evaluator.BENCHMARK / 2.0)
