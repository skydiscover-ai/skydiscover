"""Evaluate hypothesis→experiment discovery loops on a hidden nonlinear target."""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import traceback
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# Undisclosed evaluation entropy. Candidate code never sees this generator, and
# the held-out test RNG is spawned independently so snapshotting the RNG handed
# to design_experiments cannot reconstruct x_test.
_EVAL_ENTROPY = 0xA5C1D15C07E57


def _hidden_world() -> dict[str, Any]:
    """Load world.py into a private dict — never registered as sys.modules['world']."""
    path = os.path.join(_HERE, "world.py")
    ns: dict[str, Any] = {"__name__": "_skydiscover_hidden_world", "__file__": path}
    with open(path, encoding="utf-8") as f:
        exec(compile(f.read(), path, "exec"), ns)
    return ns


def _independent_rngs() -> tuple[np.random.Generator, np.random.Generator, np.random.Generator]:
    """Return (candidate, oracle-noise, test) generators that do not share state."""
    master = np.random.default_rng(_EVAL_ENTROPY)
    spawn = getattr(master, "spawn", None)
    if callable(spawn):
        cand_rng, noise_rng, test_rng = spawn(3)
        return cand_rng, noise_rng, test_rng
    ss = np.random.SeedSequence(_EVAL_ENTROPY)
    cand_ss, noise_ss, test_ss = ss.spawn(3)
    return (
        np.random.default_rng(cand_ss),
        np.random.default_rng(noise_ss),
        np.random.default_rng(test_ss),
    )


@contextmanager
def _sandbox_candidate_imports() -> Iterator[None]:
    """Keep the evaluator dir (and `world`) unreachable while candidate code runs."""
    saved_path = list(sys.path)
    here = os.path.abspath(_HERE)
    sys.path[:] = [p for p in sys.path if os.path.abspath(p) != here]
    sys.modules.pop("world", None)
    try:
        yield
    finally:
        sys.modules.pop("world", None)
        sys.path[:] = saved_path


def _load(program_path: str):
    spec = importlib.util.spec_from_file_location("candidate_science", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {program_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in ("design_experiments", "fit_hypothesis"):
        if not hasattr(module, name):
            raise AttributeError(f"program must define {name}")
    return module.design_experiments, module.fit_hypothesis


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def evaluate(program_path: str) -> dict[str, Any]:
    t0 = time.time()
    try:
        world = _hidden_world()
        n_features = world["N_FEATURES"]
        query_budget = world["QUERY_BUDGET"]
        test_n = world["TEST_N"]
        true_function = world["true_function"]
        oracle_observe = world["oracle_observe"]
        sample_inputs = world["sample_inputs"]

        cand_rng, noise_rng, test_rng = _independent_rngs()

        with _sandbox_candidate_imports():
            design_experiments, fit_hypothesis = _load(program_path)
            queries = np.asarray(
                design_experiments(query_budget, n_features, cand_rng), dtype=np.float64
            )
            if queries.ndim != 2 or queries.shape[1] != n_features:
                raise ValueError(
                    f"design_experiments must return (budget, {n_features}), got {queries.shape}"
                )
            if len(queries) > query_budget:
                queries = queries[:query_budget]
            queries = np.clip(queries, -2.0, 2.0)

            y_obs = oracle_observe(queries, noise_rng)
            predict = fit_hypothesis(queries, y_obs)

            x_test = sample_inputs(test_n, test_rng)
            y_test = true_function(x_test)  # noiseless held-out truth
            y_hat = np.asarray(predict(x_test), dtype=np.float64).reshape(-1)
            if y_hat.shape != y_test.shape:
                raise ValueError("predict() must return a 1-D array matching y")

        r2 = _r2(y_test, y_hat)
        # Mild reward for using the budget efficiently (not under-querying).
        coverage = min(1.0, len(queries) / float(query_budget))
        mse = float(np.mean((y_test - y_hat) ** 2))
        combined = 0.9 * max(0.0, r2) + 0.1 * coverage

        return {
            "combined_score": float(combined),
            "r2": float(r2),
            "mse": mse,
            "n_queries": float(len(queries)),
            "coverage": float(coverage),
            "latency_s": float(time.time() - t0),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "combined_score": 0.0,
            "r2": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "latency_s": float(time.time() - t0),
        }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "initial_program.py")
    print(evaluate(path))
