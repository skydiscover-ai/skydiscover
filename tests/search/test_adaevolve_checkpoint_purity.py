"""Writing a checkpoint must not change the run.

`AdaEvolveDatabase.save()` used to rebuild `self.programs` from the archives and
re-add an evicted best program to `archives[0]`. Both are mutations, so how the
search behaved depended on how often a checkpoint happened to land -- a run with
`checkpoint_interval: 10` and the same run with `checkpoint_interval: 1000`
diverged. Pruning now happens every iteration in `end_iteration()`; saving is
read-only.
"""

from __future__ import annotations

import copy

import pytest

from skydiscover.optimize.config import AdaEvolveDatabaseConfig
from skydiscover.optimize.search.adaevolve.database import AdaEvolveDatabase
from skydiscover.optimize.search.base_database import Program


def _program(pid: str, score: float) -> Program:
    return Program(id=pid, solution=f"# {pid}\n", metrics={"combined_score": score})


def _db(tmp_path, **extra) -> AdaEvolveDatabase:
    config = AdaEvolveDatabaseConfig(
        population_size=4,
        num_islands=2,
        use_dynamic_islands=False,
        use_paradigm_breakthrough=False,
        db_path=str(tmp_path / "ckpt"),
        **extra,
    )
    return AdaEvolveDatabase("purity", config)


def _snapshot(db):
    """The in-memory state a save must leave untouched."""
    return (
        sorted(db.programs),
        [sorted(p.id for p in a.get_all()) for a in (db.archives or [])],
        [sorted(p.id for p in i) for i in (db.islands or [])],
        db.best_program_id,
    )


def test_save_does_not_mutate_in_memory_state(tmp_path):
    db = _db(tmp_path)
    for i in range(12):
        db.add(_program(f"p{i}", score=i / 100.0), iteration=i)
    db.end_iteration(1)

    before = _snapshot(db)
    db.save(str(tmp_path / "ckpt"), iteration=1)
    assert _snapshot(db) == before


def test_repeated_saves_are_idempotent(tmp_path):
    db = _db(tmp_path)
    for i in range(12):
        db.add(_program(f"p{i}", score=i / 100.0), iteration=i)
    db.end_iteration(1)

    db.save(str(tmp_path / "a"), iteration=1)
    once = _snapshot(db)
    for n in range(3):
        db.save(str(tmp_path / f"b{n}"), iteration=1)
    assert _snapshot(db) == once


def test_saving_mid_run_never_perturbs_the_next_step(tmp_path):
    """Interleaving checkpoints into a run leaves the live set untouched each time.

    Compared within a single database, not across two: archive eviction and island
    selection use unseeded `random`, so two instances given identical adds diverge
    on their own and could not isolate what saving does.
    """
    db = _db(tmp_path)
    for i in range(20):
        db.add(_program(f"p{i}", score=i / 100.0), iteration=i)
        db.end_iteration(i)
        if i % 3 == 0:
            before = _snapshot(db)
            db.save(str(tmp_path / f"c{i}"), iteration=i)
            assert _snapshot(db) == before, f"save at iteration {i} mutated live state"


def test_registry_tracks_the_population_without_a_save(tmp_path):
    """Pruning is driven by end_iteration(), not by whether a checkpoint landed."""
    db = _db(tmp_path)
    for i in range(30):
        db.add(_program(f"p{i}", score=i / 100.0), iteration=i)
        db.end_iteration(i)

    live = {p.id for a in db.archives for p in a.get_all()} if db.archives else set()
    if db.best_program_id:
        live.add(db.best_program_id)
    # No save has happened; the registry must already be trimmed to the population.
    assert set(db.programs) == live


def test_evicted_best_program_still_reaches_the_checkpoint(tmp_path):
    """A best program pushed out of the population must survive a resume."""
    db = _db(tmp_path)
    best = _program("champion", score=99.0)
    db.add(best, iteration=0)
    db.best_program_id = "champion"
    for i in range(20):
        db.add(_program(f"filler{i}", score=0.5), iteration=i)
    db.end_iteration(1)

    out = tmp_path / "ckpt"
    db.save(str(out), iteration=1)

    saved_ids = {p.stem for p in (out / "programs").glob("*.json")}
    assert "champion" in saved_ids


def test_every_database_accepts_the_save_snapshot_override():
    """`save(programs=...)` must work on every database, not just the base class.

    Adding the parameter to ProgramDatabase alone left four subclasses overriding
    `save()` with the old signature, so `db.save(path, i, programs=...)` raised
    TypeError depending on which engine you happened to be running.
    """
    import inspect

    from skydiscover.optimize.cli import _SEARCH_CHOICES
    from skydiscover.optimize.config import Config, apply_overrides
    from skydiscover.optimize.extras.external import KNOWN_EXTERNAL
    from skydiscover.optimize.search.registry import create_database

    checked = 0
    for search_type in _SEARCH_CHOICES:
        if search_type in KNOWN_EXTERNAL:
            continue  # dispatched to a third-party backend, not a ProgramDatabase
        config = Config()
        # apply_overrides swaps in the DatabaseConfig each engine expects, the same
        # way `--search` does; a bare Config() carries the default one.
        apply_overrides(config, search=search_type)
        db = create_database(search_type, config.search.database)
        params = inspect.signature(type(db).save).parameters
        assert "programs" in params, f"{type(db).__name__}.save() lost the programs override"
        checked += 1
    assert checked >= 6
