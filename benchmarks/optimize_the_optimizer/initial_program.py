"""Baseline search controller for the optimize-the-optimizer benchmark.

Evolve ``SearchController`` — parent selection + mutation under a fixed eval
budget on held-out black-box problems. This is the EvoX² wedge: the candidate
*is* a search strategy, scored by how well it optimizes other tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Candidate:
    x: np.ndarray
    score: float  # higher is better (negated objective)


# EVOLVE-BLOCK-START
class SearchController:
    """Population search strategy evaluated under a fixed black-box budget."""

    def __init__(self, dim: int, bounds: tuple[float, float], rng: np.random.Generator):
        self.dim = dim
        self.lo, self.hi = bounds
        self.rng = rng
        self.population_size = 8
        self.mutation_scale = 0.25

    def initial_population(self) -> list[np.ndarray]:
        return [
            self.rng.uniform(self.lo, self.hi, size=self.dim)
            for _ in range(self.population_size)
        ]

    def select_parents(self, population: list[Candidate], k: int = 2) -> list[Candidate]:
        """Tournament selection."""
        parents = []
        for _ in range(k):
            a, b = self.rng.choice(len(population), size=2, replace=False)
            parents.append(population[a] if population[a].score >= population[b].score else population[b])
        return parents

    def mutate(self, parent: Candidate) -> np.ndarray:
        noise = self.rng.normal(scale=self.mutation_scale, size=self.dim)
        child = parent.x + noise * (self.hi - self.lo)
        return np.clip(child, self.lo, self.hi)

    def ask(self, population: list[Candidate], n: int) -> list[np.ndarray]:
        """Propose ``n`` new points given the current population."""
        if not population:
            return self.initial_population()[:n]
        proposals = []
        for _ in range(n):
            parents = self.select_parents(population, k=2)
            # Blend + mutate.
            alpha = float(self.rng.random())
            blend = alpha * parents[0].x + (1 - alpha) * parents[1].x
            child = self.mutate(Candidate(blend, 0.0))
            proposals.append(child)
        return proposals


# EVOLVE-BLOCK-END


def run_controller(
    controller_cls: type,
    objective: Callable[[np.ndarray], float],
    dim: int,
    bounds: tuple[float, float],
    budget: int,
    seed: int,
) -> float:
    """Run ``controller_cls`` for ``budget`` evaluations; return best maximize-score."""
    rng = np.random.default_rng(seed)
    ctrl = controller_cls(dim, bounds, rng)
    population: list[Candidate] = []
    best = -float("inf")
    remaining = budget

    init = ctrl.initial_population()
    for x in init:
        if remaining <= 0:
            break
        score = -float(objective(x))
        population.append(Candidate(x=np.asarray(x, dtype=np.float64), score=score))
        best = max(best, score)
        remaining -= 1

    while remaining > 0:
        batch = min(len(population) or 1, remaining)
        proposals = ctrl.ask(population, batch)
        for x in proposals:
            if remaining <= 0:
                break
            score = -float(objective(np.asarray(x, dtype=np.float64)))
            population.append(Candidate(x=np.asarray(x, dtype=np.float64), score=score))
            best = max(best, score)
            remaining -= 1
            # Keep population bounded.
            if len(population) > 40:
                population.sort(key=lambda c: c.score, reverse=True)
                population = population[:24]
    return best
