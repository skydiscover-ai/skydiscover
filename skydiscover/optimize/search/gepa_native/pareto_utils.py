# Ported from gepa/src/gepa/gepa_utils.py — Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import random
from typing import Any, Mapping


def is_dominated(y, programs, program_at_pareto_front_valset):
    y_fronts = [front for front in program_at_pareto_front_valset.values() if y in front]
    for front in y_fronts:
        found_dominator_in_front = False
        for other_prog in front:
            if other_prog in programs:
                found_dominator_in_front = True
                break
        if not found_dominator_in_front:
            return False
    return True


def remove_dominated_programs(program_at_pareto_front_valset, scores=None):
    freq = {}
    for front in program_at_pareto_front_valset.values():
        for p in front:
            freq[p] = freq.get(p, 0) + 1

    dominated = set()
    programs = list(freq.keys())

    if scores is None:
        scores = dict.fromkeys(programs, 1)

    programs = sorted(programs, key=lambda x: scores[x], reverse=False)

    found_to_remove = True
    while found_to_remove:
        found_to_remove = False
        for y in programs:
            if y in dominated:
                continue
            if is_dominated(
                y,
                set(programs).difference({y}).difference(dominated),
                program_at_pareto_front_valset,
            ):
                dominated.add(y)
                found_to_remove = True
                break

    dominators = [p for p in programs if p not in dominated]
    return {
        val_id: {prog_idx for prog_idx in front if prog_idx in dominators}
        for val_id, front in program_at_pareto_front_valset.items()
    }


def select_program_candidate_from_pareto_front(
    pareto_front_programs: Mapping[Any, set],
    scores: Mapping[Any, float],
    rng: random.Random,
):
    new_front = remove_dominated_programs(pareto_front_programs, scores=scores)
    freq = {}
    for front in new_front.values():
        for prog_id in front:
            freq[prog_id] = freq.get(prog_id, 0) + 1
    sampling_list = [prog_id for prog_id, f in freq.items() for _ in range(f)]
    assert len(sampling_list) > 0
    return rng.choice(sampling_list)
