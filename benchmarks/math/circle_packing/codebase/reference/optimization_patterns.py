"""
Common patterns for constrained geometric optimization using scipy.

This module shows how to use scipy.optimize.minimize with inequality
constraints and the SLSQP solver — useful for any problem where you
need to maximize/minimize an objective subject to geometric constraints.
"""

import numpy as np
from scipy.optimize import minimize


def example_constrained_optimization():
    """
    Template: pack n objects by optimizing positions + sizes jointly.

    Decision vector:  x = [pos_0, pos_1, ..., pos_{n-1}, size_0, ..., size_{n-1}]
    Objective:        maximize sum(sizes)  =>  minimize -sum(sizes)
    Constraints:      non-overlap + boundary containment (all >= 0)
    """
    n = 10  # number of objects

    # --- Objective: negative sum of sizes (we minimize, so negate to maximize) ---
    def objective(x):
        sizes = x[2 * n :]
        return -np.sum(sizes)

    # --- Constraints as a single function returning array of values >= 0 ---
    def constraints_fn(x):
        positions = x[: 2 * n].reshape(n, 2)
        sizes = x[2 * n :]

        c = []
        # Pairwise non-overlap: dist(i,j) - size_i - size_j >= 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(positions[i] - positions[j])
                c.append(dist - sizes[i] - sizes[j])

        # Boundary: each object stays inside [0, 1] x [0, 1]
        for i in range(n):
            c.append(positions[i, 0] - sizes[i])  # left
            c.append(1 - positions[i, 0] - sizes[i])  # right
            c.append(positions[i, 1] - sizes[i])  # bottom
            c.append(1 - positions[i, 1] - sizes[i])  # top

        return np.array(c)

    # --- Initial guess ---
    x0_pos = np.random.rand(n, 2) * 0.6 + 0.2  # avoid edges
    x0_sizes = np.full(n, 0.05)
    x0 = np.concatenate([x0_pos.flatten(), x0_sizes])

    # --- Bounds ---
    pos_bounds = [(0, 1)] * (2 * n)
    size_bounds = [(0.01, 0.25)] * n
    bounds = pos_bounds + size_bounds

    # --- Solve ---
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "ineq", "fun": constraints_fn},
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    opt_positions = result.x[: 2 * n].reshape(n, 2)
    opt_sizes = result.x[2 * n :]
    return opt_positions, opt_sizes, -result.fun  # return positive sum


def multi_start_optimization(objective, constraint_fn, bounds, n_starts=5):
    """
    Run SLSQP from multiple random starts and keep the best.

    This helps escape local optima — the solver is gradient-based
    and sensitive to the initial guess.
    """
    best_result = None
    for _ in range(n_starts):
        x0 = np.array([np.random.uniform(lo, hi) for lo, hi in bounds])
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "ineq", "fun": constraint_fn},
            options={"maxiter": 500, "ftol": 1e-8},
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result
    return best_result
