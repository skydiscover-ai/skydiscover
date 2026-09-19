"""Score a search controller on train + held-out black-box problems."""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from problems import Problem, TEST_PROBLEMS as _RAW_TEST, TRAIN_PROBLEMS as _RAW_TRAIN  # noqa: E402

BUDGET = 64
SEEDS = (0, 1, 2)

# Hidden per-problem offsets so a do-nothing controller at the origin cannot
# farm a near-perfect score. Not imported by candidate programs.
_OFFSET_SEED = 0x51DE5A1D


def _offset_for(dim: int, bounds: tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    lo, hi = bounds
    span = hi - lo
    offset = np.empty(dim, dtype=np.float64)
    for i in range(dim):
        if rng.random() < 0.5:
            offset[i] = rng.uniform(lo + 0.15 * span, lo + 0.35 * span)
        else:
            offset[i] = rng.uniform(hi - 0.35 * span, hi - 0.15 * span)
    return offset


def _shift_objective(base_fn, dim: int, bounds: tuple[float, float], name: str, rng: np.random.Generator):
    offset = _offset_for(dim, bounds, rng)
    if name.startswith("rosenbrock"):
        # Canonical Rosenbrock is minimized at ones; move that point to `offset`.
        def fn(x, _base=base_fn, _o=offset):
            return _base(np.asarray(x, dtype=np.float64) - _o + 1.0)
    else:
        def fn(x, _base=base_fn, _o=offset):
            return _base(np.asarray(x, dtype=np.float64) - _o)
    return fn


def _shifted_suite(problems: list[Problem], rng: np.random.Generator) -> list[Problem]:
    return [
        Problem(p.name, p.dim, p.bounds, _shift_objective(p.fn, p.dim, p.bounds, p.name, rng))
        for p in problems
    ]


_offset_rng = np.random.default_rng(_OFFSET_SEED)
TRAIN_PROBLEMS = _shifted_suite(_RAW_TRAIN, _offset_rng)
TEST_PROBLEMS = _shifted_suite(_RAW_TEST, _offset_rng)


@dataclass
class _Member:
    """Harness-owned population record. Duck-typed as `.x` / `.score` for candidates."""

    x: np.ndarray
    score: float  # higher is better (negated objective)


def run_controller(
    controller_cls: type,
    objective: Callable[[np.ndarray], float],
    dim: int,
    bounds: tuple[float, float],
    budget: int,
    seed: int,
) -> float:
    """Run ``controller_cls`` for ``budget`` evaluations; return best maximize-score.

    Lives in the evaluator, not the candidate. Candidates may define their own
    ``run_controller`` (copies of the seed did); it is never called.
    """
    rng = np.random.default_rng(seed)
    ctrl = controller_cls(dim, bounds, rng)
    population: list[_Member] = []
    best = -float("inf")
    remaining = budget

    init = ctrl.initial_population()
    for x in init:
        if remaining <= 0:
            break
        score = -float(objective(x))
        population.append(_Member(x=np.asarray(x, dtype=np.float64), score=score))
        best = max(best, score)
        remaining -= 1

    while remaining > 0:
        batch = min(len(population) or 1, remaining)
        proposals = ctrl.ask(population, batch)
        if not proposals:
            break
        for x in proposals:
            if remaining <= 0:
                break
            score = -float(objective(np.asarray(x, dtype=np.float64)))
            population.append(_Member(x=np.asarray(x, dtype=np.float64), score=score))
            best = max(best, score)
            remaining -= 1
            if len(population) > 40:
                population.sort(key=lambda c: c.score, reverse=True)
                population = population[:24]
    return best


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
    # Never take run_controller from the candidate — the seed copies one, and a
    # stub that returns the per-problem ceiling (0.0) would score 1.0 with no search.
    return module.SearchController


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


def _eval_suite(controller_cls, problems) -> dict[str, float]:
    scores = []
    per: dict[str, float] = {}
    for problem in problems:
        seed_scores = []
        for seed in SEEDS:
            raw = run_controller(
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
        controller_cls = _load_controller(program_path)
        train = _eval_suite(controller_cls, TRAIN_PROBLEMS)
        test = _eval_suite(controller_cls, TEST_PROBLEMS)
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
