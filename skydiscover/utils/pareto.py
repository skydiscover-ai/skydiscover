"""Pareto-front helpers shared by multiobjective search modes.

Vectors are assumed to already be in **maximization** space (callers should
run :func:`skydiscover.utils.metrics.normalize_metric_value` first).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")


def dominates(vec_a: Sequence[float], vec_b: Sequence[float]) -> bool:
    """True if ``vec_a`` Pareto-dominates ``vec_b`` (all ≥ and at least one >)."""
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Objective vectors must have equal length, got {len(vec_a)} vs {len(vec_b)}"
        )
    at_least_one_better = False
    for a, b in zip(vec_a, vec_b):
        if a < b:
            return False
        if a > b:
            at_least_one_better = True
    return at_least_one_better


def nondominated_indices(vectors: Sequence[Sequence[float]]) -> List[int]:
    """Return indices of the non-dominated subset of ``vectors``."""
    n = len(vectors)
    front: List[int] = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if dominates(vectors[j], vectors[i]):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


def nondominated_items(
    items: Sequence[T],
    vectors: Sequence[Sequence[float]],
) -> List[T]:
    """Filter ``items`` to the non-dominated front given parallel ``vectors``."""
    if len(items) != len(vectors):
        raise ValueError("items and vectors must have the same length")
    return [items[i] for i in nondominated_indices(vectors)]


def nsga2_ranks_and_crowding(
    vectors: Sequence[Sequence[float]],
) -> Tuple[Dict[int, int], Dict[int, float]]:
    """NSGA-II non-dominated ranks + crowding distances keyed by index."""
    n = len(vectors)
    if n == 0:
        return {}, {}

    remaining = set(range(n))
    ranks: Dict[int, int] = {}
    layers: List[List[int]] = []
    rank = 0
    while remaining:
        front = []
        for i in list(remaining):
            dominated = False
            for j in remaining:
                if i == j:
                    continue
                if dominates(vectors[j], vectors[i]):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        for i in front:
            ranks[i] = rank
            remaining.discard(i)
        layers.append(front)
        rank += 1

    crowding: Dict[int, float] = {i: 0.0 for i in range(n)}
    num_objectives = len(vectors[0]) if vectors else 0
    for layer in layers:
        if len(layer) <= 2:
            for i in layer:
                crowding[i] = float("inf")
            continue
        for m in range(num_objectives):
            sorted_layer = sorted(layer, key=lambda idx: vectors[idx][m])
            crowding[sorted_layer[0]] = float("inf")
            crowding[sorted_layer[-1]] = float("inf")
            obj_range = vectors[sorted_layer[-1]][m] - vectors[sorted_layer[0]][m]
            if abs(obj_range) < 1e-10:
                continue
            for k in range(1, len(sorted_layer) - 1):
                crowding[sorted_layer[k]] += (
                    vectors[sorted_layer[k + 1]][m] - vectors[sorted_layer[k - 1]][m]
                ) / obj_range

    return ranks, crowding


def objective_vector(
    metrics: dict,
    objectives: Iterable[str],
    higher_is_better: Optional[dict] = None,
) -> Optional[List[float]]:
    """Build a maximization-space objective vector from a metrics dict.

    Missing / non-numeric objectives become ``-inf`` so incomplete metric
    vectors cannot dominate fully evaluated ones.
    """
    from skydiscover.utils.metrics import normalize_metric_value

    objs = list(objectives)
    if not objs:
        return None
    hib = higher_is_better or {}
    vector: List[float] = []
    for objective in objs:
        normalized = normalize_metric_value(objective, metrics.get(objective), hib)
        vector.append(normalized if normalized is not None else float("-inf"))
    return vector
