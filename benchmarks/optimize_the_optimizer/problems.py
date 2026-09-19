"""Black-box toy suite used to score search controllers (optimize-the-optimizer)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Problem:
    name: str
    dim: int
    bounds: tuple[float, float]
    # Minimize; evaluator will negate into a maximize score.
    fn: Callable[[np.ndarray], float]


def _sphere(x: np.ndarray) -> float:
    return float(np.sum(x * x))


def _rastrigin(x: np.ndarray) -> float:
    n = x.size
    return float(10 * n + np.sum(x * x - 10 * np.cos(2 * np.pi * x)))


def _ackley(x: np.ndarray) -> float:
    n = x.size
    a = -20 * np.exp(-0.2 * np.sqrt(np.sum(x * x) / n))
    b = -np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
    return float(a + b + 20 + np.e)


def _rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


# Train problems (used during search). Hold out ackley + rosenbrock.
TRAIN_PROBLEMS: list[Problem] = [
    Problem("sphere_2d", 2, (-3.0, 3.0), _sphere),
    Problem("sphere_4d", 4, (-2.0, 2.0), _sphere),
    Problem("rastrigin_2d", 2, (-3.0, 3.0), _rastrigin),
]

TEST_PROBLEMS: list[Problem] = [
    Problem("ackley_2d", 2, (-3.0, 3.0), _ackley),
    Problem("rosenbrock_2d", 2, (-2.0, 2.0), _rosenbrock),
]
