"""
Exploration / exploitation balance analysis for evolutionary optimization runs.
"""
from __future__ import annotations

import difflib
import random
from typing import List

from skydiscover.extras.evolve_analyzer.quantitative.bundle import ExplorationMetrics

_EXPLOIT_MUTATION_TYPES = {"exploit", "local_search", "refine", "hill_climb"}

_MAX_PAIRS = 50


def _normalized_edit_distance(a: str, b: str) -> float:
    """
    Return 1 - SequenceMatcher ratio, giving a value in [0, 1] where
    0 means identical and 1 means completely different.
    """
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    return 1.0 - ratio


def analyze_exploration(records: List[dict]) -> ExplorationMetrics:
    """
    Analyse the exploration / exploitation balance of an evolutionary run.

    Parameters
    ----------
    records:
        Raw JSONL records (each a dict) from the run.

    Returns
    -------
    ExplorationMetrics
    """
    if not records:
        return ExplorationMetrics(
            structural_diversity_index=0.0,
            exploit_phase_fraction=0.0,
            explore_phase_fraction=1.0,
            distinct_strategy_clusters=1,
            revert_frequency=0.0,
        )

    sorted_records = sorted(records, key=lambda r: r.get("iteration", 0))
    total = len(sorted_records)

    # ── structural diversity index ─────────────────────────────────────────────
    codes = [r.get("child_code") for r in sorted_records if r.get("child_code")]
    structural_diversity_index = 0.0

    if len(codes) >= 2:
        # Build candidate pairs; sample if there are too many
        all_pairs = [(i, j) for i in range(len(codes)) for j in range(i + 1, len(codes))]
        if len(all_pairs) > _MAX_PAIRS:
            rng = random.Random(42)
            sampled_pairs = rng.sample(all_pairs, _MAX_PAIRS)
        else:
            sampled_pairs = all_pairs

        distances = [
            _normalized_edit_distance(codes[i], codes[j]) for i, j in sampled_pairs
        ]
        structural_diversity_index = sum(distances) / len(distances) if distances else 0.0

    # ── exploit / explore phase fractions ─────────────────────────────────────
    has_mutation_type = any(r.get("mutation_type") is not None for r in sorted_records)

    exploit_count = 0
    if has_mutation_type:
        for rec in sorted_records:
            mt = rec.get("mutation_type")
            if mt is not None and mt in _EXPLOIT_MUTATION_TYPES:
                exploit_count += 1
    else:
        # Heuristic: small positive delta signals exploitation
        for rec in sorted_records:
            delta = rec.get("score_delta", 0.0)
            if delta is not None and 0.0 < delta < 0.05:
                exploit_count += 1

    exploit_phase_fraction = exploit_count / total if total > 0 else 0.0
    explore_phase_fraction = 1.0 - exploit_phase_fraction

    # ── distinct strategy clusters ────────────────────────────────────────────
    mutation_types = {
        r.get("mutation_type")
        for r in sorted_records
        if r.get("mutation_type") is not None
    }
    distinct_strategy_clusters = len(mutation_types) if mutation_types else 1

    # ── revert frequency ──────────────────────────────────────────────────────
    # Approximate: count iterations where score_delta < -0.01
    revert_count = sum(
        1
        for rec in sorted_records
        if (rec.get("score_delta") is not None and rec.get("score_delta") < -0.01)
    )
    revert_frequency = revert_count / total if total > 0 else 0.0

    return ExplorationMetrics(
        structural_diversity_index=structural_diversity_index,
        exploit_phase_fraction=exploit_phase_fraction,
        explore_phase_fraction=explore_phase_fraction,
        distinct_strategy_clusters=distinct_strategy_clusters,
        revert_frequency=revert_frequency,
    )
