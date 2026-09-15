"""`search.database.random_seed` must actually make selection reproducible.

Every shipped template set a seed, and the flagship engine ignored it: adaevolve
called the module-global `random`, and `AdaEvolveDatabaseConfig` did not even
declare the field, so the key was swallowed as a passthrough extra. Two runs with
the same seed diverged. Each database now owns a `random.Random` and shares it
with its archives and adapter.
"""

from __future__ import annotations

import dataclasses

import pytest

from skydiscover.optimize import config as C
from skydiscover.optimize.search.adaevolve.database import AdaEvolveDatabase
from skydiscover.optimize.search.base_database import Program

ENGINE_CONFIGS = [
    C.DatabaseConfig,
    C.AdaEvolveDatabaseConfig,
    C.BeamSearchDatabaseConfig,
    C.BestOfNDatabaseConfig,
    C.EvoxDatabaseConfig,
    C.GEPANativeDatabaseConfig,
    C.OpenEvolveNativeDatabaseConfig,
]


@pytest.mark.parametrize("cls", ENGINE_CONFIGS, ids=lambda c: c.__name__)
def test_every_engine_declares_the_seed(cls):
    """A seed set in YAML must land on a real field, not a silently-kept extra."""
    assert "random_seed" in {f.name for f in dataclasses.fields(cls)}


def _db(seed=None):
    return AdaEvolveDatabase(
        "seeded",
        C.AdaEvolveDatabaseConfig(
            population_size=6,
            num_islands=2,
            use_dynamic_islands=False,
            use_paradigm_breakthrough=False,
            random_seed=seed,
        ),
    )


def test_same_seed_gives_the_same_stream():
    a, b = _db(7), _db(7)
    assert [a.rng.random() for _ in range(5)] == [b.rng.random() for _ in range(5)]


def test_different_seeds_diverge():
    assert [_db(1).rng.random() for _ in range(5)] != [_db(2).rng.random() for _ in range(5)]


def test_unseeded_databases_are_independent():
    """No seed must keep the previous behaviour: fresh OS entropy, not a fixed stream."""
    assert [_db().rng.random() for _ in range(5)] != [_db().rng.random() for _ in range(5)]


def test_archive_and_adapter_share_the_database_rng():
    """One seed has to govern the whole engine, not just the database's own picks."""
    db = _db(11)
    assert db.adapter.rng is db.rng
    for archive in db.archives or []:
        assert archive.rng is db.rng


def test_seeded_sampling_is_reproducible():
    """The user-visible promise: same seed, same sequence of sampled parents."""

    def run():
        db = _db(3)
        for i in range(8):
            db.add(Program(id=f"p{i}", solution=f"# {i}\n", metrics={"combined_score": i / 10}))
        picks = []
        for _ in range(6):
            parents, inspirations = db.sample()
            picks.append(
                (
                    tuple(sorted(parents)),
                    tuple(sorted((k, tuple(p.id for p in v)) for k, v in inspirations.items())),
                )
            )
        return picks

    assert run() == run()


def test_seeding_does_not_touch_the_global_rng():
    """Seeding one database must not reach every other component in the process."""
    import random

    random.seed(0)
    expected = [random.random() for _ in range(3)]
    random.seed(0)
    _db(99)  # constructing a seeded database must not disturb the global stream
    assert [random.random() for _ in range(3)] == expected


# The AdaEvolve cases above only sampled AdaEvolveDatabase. The other sampling
# engines each had a selection path that reached for the module-global `random`
# instead of their own `self.rng`, so `random_seed` silently did nothing there.
# These cases actually SAMPLE each engine, which is what caught the regression.
from skydiscover.optimize.search.beam_search.database import BeamSearchDatabase  # noqa: E402
from skydiscover.optimize.search.best_of_n.database import BestOfNDatabase  # noqa: E402
from skydiscover.optimize.search.evox.database.initial_search_strategy import (  # noqa: E402
    EvolvedProgram,
    EvolvedProgramDatabase,
)
from skydiscover.optimize.search.evox.database.search_strategy_db import (  # noqa: E402
    SearchStrategy,
    SearchStrategyDatabase,
)
from skydiscover.optimize.search.openevolve_native.database import (  # noqa: E402
    OpenEvolveNativeDatabase,
)


def _sampling_engine(seed, engine):
    if engine == "best_of_n":
        return BestOfNDatabase("t", C.BestOfNDatabaseConfig(random_seed=seed)), Program
    if engine == "beam_search":
        return (
            BeamSearchDatabase(
                "t",
                C.BeamSearchDatabaseConfig(
                    random_seed=seed, beam_selection_strategy="stochastic", beam_temperature=1.0
                ),
            ),
            Program,
        )
    if engine == "evox_init":
        return EvolvedProgramDatabase("t", C.DatabaseConfig(random_seed=seed)), EvolvedProgram
    if engine == "openevolve_native":
        return (
            OpenEvolveNativeDatabase("t", C.OpenEvolveNativeDatabaseConfig(random_seed=seed)),
            Program,
        )
    return SearchStrategyDatabase("t", C.EvoxDatabaseConfig(random_seed=seed)), SearchStrategy


def _sample_signature(seed, engine):
    db, prog_cls = _sampling_engine(seed, engine)
    for i in range(12):
        db.add(prog_cls(id=f"p{i}", solution=f"# {i}\n", metrics={"combined_score": i / 12.0}))
    picks = []
    for _ in range(6):
        parent, context = db.sample()
        parent_id = (
            parent.id if hasattr(parent, "id") else tuple(sorted(v.id for v in parent.values()))
        )
        context_list = (
            context if isinstance(context, list) else [p for v in context.values() for p in v]
        )
        picks.append((parent_id, tuple(sorted(p.id for p in context_list))))
    return picks


# openevolve_native is omitted here: its RNG is seed-clean (see below) but _seed_empty_island
# mints a uuid.uuid4() id, a separate nondeterminism source, so its pick-stream isn't reproducible.
@pytest.mark.parametrize("engine", ["best_of_n", "beam_search", "evox", "evox_init"])
def test_sampling_engine_honours_the_seed(engine):
    """Same seed -> same sampled stream for engines that previously ignored it."""
    assert _sample_signature(3, engine) == _sample_signature(3, engine)


@pytest.mark.parametrize(
    "engine", ["best_of_n", "beam_search", "evox", "evox_init", "openevolve_native"]
)
def test_sampling_engine_does_not_touch_the_global_rng(engine):
    """Sampling must draw from the database's own rng, never the module-global one."""
    import random

    random.seed(0)
    expected = [random.random() for _ in range(3)]
    random.seed(0)
    _sample_signature(3, engine)
    assert [random.random() for _ in range(3)] == expected
