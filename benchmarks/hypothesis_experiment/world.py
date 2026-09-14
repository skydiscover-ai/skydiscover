"""Hidden ground-truth process for the hypothesis→experiment benchmark."""

from __future__ import annotations

import numpy as np

N_FEATURES = 3
QUERY_BUDGET = 40
TRAIN_HOLDOUT = 80
TEST_N = 120
NOISE = 0.08


def true_function(x: np.ndarray) -> np.ndarray:
    """y = sin(x0) + 0.5 x1 x2 - 0.25 x0^2  (column-wise)."""
    x = np.asarray(x, dtype=np.float64)
    return np.sin(x[:, 0]) + 0.5 * x[:, 1] * x[:, 2] - 0.25 * x[:, 0] ** 2


def sample_inputs(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-2.0, 2.0, size=(n, N_FEATURES))


def oracle_observe(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = true_function(x)
    return y + NOISE * rng.normal(size=y.shape)
