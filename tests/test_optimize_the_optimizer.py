"""Regression tests for optimize-the-optimizer harness integrity."""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "optimize_the_optimizer"


def _load_evaluate():
    path = ROOT / "evaluator.py"
    name = "optimize_the_optimizer_evaluator"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.evaluate


evaluate = _load_evaluate()

CHEAT_RUN_CONTROLLER = textwrap.dedent(
    """\
    import numpy as np

    class SearchController:
        def __init__(self, dim, bounds, rng):
            self.dim = dim
            self.lo, self.hi = bounds
            self.rng = rng

        def initial_population(self):
            # Stay at the far corner — a real search should score far below 1.0.
            return [np.full(self.dim, self.hi)]

        def ask(self, population, n):
            return [np.full(self.dim, self.hi) for _ in range(n)]

    def run_controller(*args, **kwargs):
        # Per-problem maximize ceiling. If the evaluator getattr's this off the
        # candidate, every normalized score is 1.0 with no search.
        return 0.0
    """
)

EMPTY_ASK = textwrap.dedent(
    """\
    import numpy as np

    class SearchController:
        def __init__(self, dim, bounds, rng):
            self.dim = dim
            self.lo, self.hi = bounds
            self.rng = rng

        def initial_population(self):
            return [self.rng.uniform(self.lo, self.hi, size=self.dim)]

        def ask(self, population, n):
            return []
    """
)

ORIGIN_NOOP = textwrap.dedent(
    """\
    import numpy as np

    class SearchController:
        def __init__(self, dim, bounds, rng):
            self.dim = dim

        def initial_population(self):
            return [np.zeros(self.dim)]

        def ask(self, population, n):
            return []
    """
)


def test_evaluate_baseline_program():
    metrics = evaluate(str(ROOT / "initial_program.py"))
    assert "combined_score" in metrics
    assert "error" not in metrics
    assert 0.0 < metrics["combined_score"] < 1.0


def test_candidate_run_controller_stub_does_not_score_perfect(tmp_path):
    program = tmp_path / "cheat_run_controller.py"
    program.write_text(CHEAT_RUN_CONTROLLER)
    metrics = evaluate(str(program))
    # Stub harness would yield exactly 1.0; a corner-stuck controller must not.
    assert metrics["combined_score"] < 1.0
    assert metrics["combined_score"] < 0.5


def test_empty_ask_does_not_hang(tmp_path):
    program = tmp_path / "empty_ask.py"
    program.write_text(EMPTY_ASK)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(evaluate, str(program))
        metrics = future.result(timeout=15)
    assert "combined_score" in metrics
    assert metrics["latency_s"] < 10.0


def test_origin_noop_does_not_beat_baseline(tmp_path):
    baseline = evaluate(str(ROOT / "initial_program.py"))
    program = tmp_path / "origin_noop.py"
    program.write_text(ORIGIN_NOOP)
    metrics = evaluate(str(program))
    assert "error" not in metrics
    # Pre-shift, zeros + empty ask scored ~0.998 and beat the baseline (~0.959).
    assert metrics["combined_score"] < baseline["combined_score"]
    assert metrics["combined_score"] < 0.95

