"""Score a search controller on train + held-out black-box problems."""

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

from initial_program import run_controller  # noqa: E402
from problems import TEST_PROBLEMS, TRAIN_PROBLEMS  # noqa: E402

BUDGET = 64
SEEDS = (0, 1, 2)


def _load_controller(program_path: str):
    spec = importlib.util.spec_from_file_location("candidate_controller", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {program_path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses (3.14+) look up cls.__module__ in sys.modules during decoration.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "SearchController"):
        raise AttributeError("program must define SearchController")
    # Prefer the candidate's run_controller if present; else shared harness.
    run_fn = getattr(module, "run_controller", run_controller)
    return module.SearchController, run_fn


def _normalize_score(raw: float, problem_name: str) -> float:
    """Map raw maximize-scores into roughly [0, 1] per problem family."""
    # Heuristic ceilings from random search baselines on these bounds/budgets.
    floors = {
        "sphere_2d": -20.0,
        "sphere_4d": -20.0,
        "rastrigin_2d": -80.0,
        "ackley_2d": -15.0,
        "rosenbrock_2d": -200.0,
    }
    ceilings = {
        "sphere_2d": 0.0,
        "sphere_4d": 0.0,
        "rastrigin_2d": 0.0,
        "ackley_2d": 0.0,
        "rosenbrock_2d": 0.0,
    }
    lo = floors.get(problem_name, -50.0)
    hi = ceilings.get(problem_name, 0.0)
    if hi <= lo:
        return 0.0
    return float(np.clip((raw - lo) / (hi - lo), 0.0, 1.0))


def _eval_suite(controller_cls, run_fn, problems) -> dict[str, float]:
    scores = []
    per: dict[str, float] = {}
    for problem in problems:
        seed_scores = []
        for seed in SEEDS:
            raw = run_fn(
                controller_cls,
                problem.fn,
                problem.dim,
                problem.bounds,
                BUDGET,
                seed,
            )
            seed_scores.append(_normalize_score(raw, problem.name))
        mean = float(np.mean(seed_scores))
        per[problem.name] = mean
        scores.append(mean)
    return {"mean": float(np.mean(scores)) if scores else 0.0, "per_problem": per}


def evaluate(program_path: str) -> dict[str, Any]:
    t0 = time.time()
    try:
        controller_cls, run_fn = _load_controller(program_path)
        train = _eval_suite(controller_cls, run_fn, TRAIN_PROBLEMS)
        test = _eval_suite(controller_cls, run_fn, TEST_PROBLEMS)
        combined = 0.65 * test["mean"] + 0.35 * train["mean"]
        return {
            "combined_score": float(combined),
            "test_score": float(test["mean"]),
            "train_score": float(train["mean"]),
            "latency_s": float(time.time() - t0),
            "artifacts": {
                "train": train["per_problem"],
                "test": test["per_problem"],
            },
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "combined_score": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "latency_s": float(time.time() - t0),
        }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "initial_program.py")
    print(evaluate(path))
