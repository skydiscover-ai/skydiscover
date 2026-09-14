"""Baseline scientific discovery loop: design queries, then fit a hypothesis.

Evolve the experiment design and the fitted model. The evaluator's oracle is
hidden — you only see noisy observations of the points you query.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


# EVOLVE-BLOCK-START
def design_experiments(budget: int, n_features: int, rng: np.random.Generator) -> np.ndarray:
    """Return a ``(budget, n_features)`` matrix of query points in [-2, 2]."""
    # Naive space-filling: stratified random.
    return rng.uniform(-2.0, 2.0, size=(budget, n_features))


def fit_hypothesis(
    x: np.ndarray,
    y: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit a predictive model from observations; return ``predict(x)->y``.

    Baseline: degree-2 polynomial features + least squares.
    """
    def _features(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        cols = [np.ones(len(z)), z[:, 0], z[:, 1], z[:, 2]]
        cols.extend(
            [
                z[:, 0] ** 2,
                z[:, 1] ** 2,
                z[:, 2] ** 2,
                z[:, 0] * z[:, 1],
                z[:, 0] * z[:, 2],
                z[:, 1] * z[:, 2],
            ]
        )
        return np.column_stack(cols)

    phi = _features(x)
    coef, *_ = np.linalg.lstsq(phi, y, rcond=None)

    def predict(z: np.ndarray) -> np.ndarray:
        return _features(z) @ coef

    return predict


# EVOLVE-BLOCK-END
