"""Tests for EvoX multiobjective support (issue #42)."""

from __future__ import annotations

from skydiscover.config import EvoxDatabaseConfig
from skydiscover.context_builder.default.builder import DefaultContextBuilder
from skydiscover.config import Config, SearchConfig
from skydiscover.search.base_database import Program
from skydiscover.search.evox.database.initial_search_strategy import EvolvedProgramDatabase
from skydiscover.utils.pareto import dominates, nondominated_indices


def _make_program(program_id: str, **metrics) -> Program:
    return Program(
        id=program_id,
        solution=f"def solve():\n    return '{program_id}'\n",
        metrics=metrics,
    )


def _pareto_db(**extra) -> EvolvedProgramDatabase:
    config = EvoxDatabaseConfig(
        pareto_objectives=extra.pop("pareto_objectives", ["accuracy", "latency"]),
        higher_is_better=extra.pop(
            "higher_is_better", {"accuracy": True, "latency": False}
        ),
        fitness_key=extra.pop("fitness_key", "accuracy"),
        pareto_objectives_weight=extra.pop("pareto_objectives_weight", 0.4),
        **extra,
    )
    return EvolvedProgramDatabase("evox", config)


def test_dominates_and_front_helpers():
    assert dominates([1.0, 1.0], [0.5, 0.5])
    assert not dominates([1.0, 0.0], [0.0, 1.0])
    assert nondominated_indices([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]) == [0, 1]


def test_pareto_front_and_representative_best():
    db = _pareto_db()
    high_acc = _make_program("p1", accuracy=0.95, latency=90.0, combined_score=0.5)
    low_lat = _make_program("p2", accuracy=0.90, latency=10.0, combined_score=0.9)
    dominated = _make_program("p3", accuracy=0.80, latency=120.0, combined_score=0.99)

    db.add(high_acc)
    db.add(low_lat)
    db.add(dominated)

    front_ids = {p.id for p in db.get_pareto_front()}
    assert front_ids == {"p1", "p2"}

    best = db.get_best_program()
    assert best is not None
    assert best.id == "p1"  # fitness_key=accuracy

    top_ids = [p.id for p in db.get_top_programs(3)]
    assert top_ids[0] in {"p1", "p2"}
    assert "p3" == top_ids[-1]


def test_scalar_mode_unchanged():
    db = EvolvedProgramDatabase("evox", EvoxDatabaseConfig())
    worse = _make_program("p1", combined_score=0.1)
    better = _make_program("p2", combined_score=0.9)
    db.add(worse)
    db.add(better)
    assert db.get_best_program().id == "p2"
    assert [p.id for p in db.get_pareto_front()] == ["p2"]


def test_pareto_cache_rebuilds_after_add():
    db = _pareto_db()
    db.add(_make_program("p1", accuracy=0.9, latency=50.0))
    front_a = db.get_pareto_front()
    front_b = db.get_pareto_front()
    assert [p.id for p in front_a] == [p.id for p in front_b]
    assert db._pareto_front_cache_valid is True

    db.add(_make_program("p2", accuracy=0.95, latency=10.0))
    # add() invalidates then rebuilds via _update_best_program → get_pareto_front
    front_c = db.get_pareto_front()
    assert {p.id for p in front_c} == {"p2"}
    assert db._pareto_front_cache_valid is True


def test_sample_prefers_pareto_parent():
    db = _pareto_db()
    db.add(_make_program("p1", accuracy=0.95, latency=90.0))
    db.add(_make_program("p2", accuracy=0.90, latency=10.0))
    db.add(_make_program("p3", accuracy=0.50, latency=200.0))

    parents = set()
    for _ in range(40):
        parent_dict, _ = db.sample(num_context_programs=1)
        parents.add(next(iter(parent_dict.values())).id)
    assert parents <= {"p1", "p2"}
    assert "p3" not in parents


def test_default_context_builder_mentions_objectives():
    config = Config(
        search=SearchConfig(
            type="evox",
            database=EvoxDatabaseConfig(
                pareto_objectives=["accuracy", "latency"],
                higher_is_better={"accuracy": True, "latency": False},
            ),
        )
    )
    builder = DefaultContextBuilder(config)
    text = builder._identify_improvement_areas(
        "code",
        {"accuracy": 0.9, "latency": 12.0, "combined_score": 0.5},
        [],
    )
    assert "Pareto trade-offs" in text
    assert "accuracy" in text
    assert "latency" in text
