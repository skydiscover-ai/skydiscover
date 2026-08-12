"""Evaluate hypothesis→experiment discovery loops on a hidden nonlinear target."""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import traceback
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from world import (  # noqa: E402
    N_FEATURES,
    QUERY_BUDGET,
    TEST_N,
    true_function,
    oracle_observe,
    sample_inputs,
)


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
        design_experiments, fit_hypothesis = _load(program_path)
        rng = np.random.default_rng(0)

        queries = np.asarray(
            design_experiments(QUERY_BUDGET, N_FEATURES, rng), dtype=np.float64
        )
        if queries.ndim != 2 or queries.shape[1] != N_FEATURES:
            raise ValueError(
                f"design_experiments must return (budget, {N_FEATURES}), got {queries.shape}"
            )
        if len(queries) > QUERY_BUDGET:
            queries = queries[:QUERY_BUDGET]
        queries = np.clip(queries, -2.0, 2.0)

        y_obs = oracle_observe(queries, rng)
        predict = fit_hypothesis(queries, y_obs)

        x_test = sample_inputs(TEST_N, rng)
        y_test = true_function(x_test)  # noiseless held-out truth
        y_hat = np.asarray(predict(x_test), dtype=np.float64).reshape(-1)
        if y_hat.shape != y_test.shape:
            raise ValueError("predict() must return a 1-D array matching y")

        r2 = _r2(y_test, y_hat)
        # Mild reward for using the budget efficiently (not under-querying).
        coverage = min(1.0, len(queries) / float(QUERY_BUDGET))
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
