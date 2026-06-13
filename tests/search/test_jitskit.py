"""Unit + contract guards for the ``jitskit`` agentic strategy wrapper.

These tests do NOT run the real runtime (that needs the claude CLI + bare-metal
hardware). They lock in the wrapper contract the integration invariants depend
on:

  * I1  — the controller reports the runtime's own peak and the Runner's
          ``mode="test"`` re-score is skipped (``skip_test_rescore``).
  * I3  — every runtime knob is reachable from ``search.database``.
  * I4  — budget / threads LISTs pass through verbatim.

The real "byte-identical vs ``bash run.sh``" acceptance test is documented in
``skydiscover/search/jitskit/README.md`` and is gated on hardware.
"""

import json
from types import SimpleNamespace

import pytest

from skydiscover.config import Config, JitsKitConfig
from skydiscover.search.jitskit.controller import JitsKitController
from skydiscover.search.jitskit.database import JitsKitDatabase
from skydiscover.search.registry import _CONTROLLER_REGISTRY, _DATABASE_REGISTRY


def _ctrl() -> JitsKitController:
    """A controller instance without running __init__ (which builds an evaluator)."""
    return object.__new__(JitsKitController)


def _spec(**overrides) -> JitsKitConfig:
    """A minimal valid JitsKitConfig (workload + a workload source)."""
    base = dict(workload="50:50", distribution="zipf(0.99)")
    base.update(overrides)
    return JitsKitConfig(**base)


# ---------------------------------------------------------------------------
# Registration + the I1 opt-out flag
# ---------------------------------------------------------------------------


def test_registered():
    import skydiscover.search.route  # noqa: F401  (importing wires the registries)

    assert _CONTROLLER_REGISTRY["jitskit"].__name__ == "JitsKitController"
    assert _DATABASE_REGISTRY["jitskit"].__name__ == "JitsKitDatabase"


def test_skip_test_rescore_is_set():
    assert JitsKitController.skip_test_rescore is True


# ---------------------------------------------------------------------------
# __init__ contract (real construction, evaluator mocked) — guards close()
# ---------------------------------------------------------------------------


def test_init_builds_evaluator_and_defaults_suffix(monkeypatch):
    """A real controller must build an evaluator (so base close() is safe) and
    default file_suffix to '.cc'. Exercised with a stub evaluator, no hardware."""
    import skydiscover.search.jitskit.controller as mod

    sentinel = object()
    monkeypatch.setattr(mod, "create_evaluator", lambda cfg, env_vars=None: sentinel)

    cfg = Config.from_dict({"language": "cpp", "search": {"type": "jitskit"}})
    ci = SimpleNamespace(
        config=cfg,
        evaluation_file="/tmp/evaluator",
        database=JitsKitDatabase("jitskit", cfg.search.database),
        file_suffix=None,  # must fall back to ".cc"
        output_dir=None,
        evaluator_env_vars=None,
    )
    ctrl = JitsKitController(ci)
    assert ctrl.file_suffix == ".cc"
    assert ctrl.evaluator is sentinel
    assert cfg.evaluator.evaluation_file == "/tmp/evaluator"
    ctrl.close()  # base close() dereferences self.evaluator; must not raise


# ---------------------------------------------------------------------------
# Config knobs ride on search.database (I3) and lists survive (I4)
# ---------------------------------------------------------------------------


def test_knobs_ride_on_search_database_not_dropped():
    cfg = Config.from_dict(
        {
            "language": "cpp",
            "search": {
                "type": "jitskit",
                "database": {
                    "backend": "claude",
                    "mode": "ltm",
                    "threads": [16],
                    "mem_budget_gb": [8, 32],
                    "critique_mode": "full",
                    "workload": "50:50",
                    "some_future_flag": "x",  # tolerated, not crashing
                },
            },
        }
    )
    db = cfg.search.database
    assert isinstance(db, JitsKitConfig)
    assert db.backend == "claude"
    assert db.threads == [16]
    assert db.mem_budget_gb == [8, 32]
    assert db.critique_mode == "full"
    assert getattr(db, "some_future_flag") == "x"


# ---------------------------------------------------------------------------
# Flag translation must stay faithful to run.sh
# ---------------------------------------------------------------------------


def test_build_flags_translation():
    db = _spec(
        backend="claude",
        mode="ltm",
        value_size=100,
        mem_budget_gb=[8, 32],
        threads=[16, 64],
        critique_mode="full",
        feedback_level="rich",
        audit_every=15,
        parallel_eval=True,
        no_planner=True,
    )
    flags = _ctrl()._build_flags(db, max_iterations=50)
    joined = " ".join(flags)

    assert "--backend claude" in joined
    assert "--mode ltm" in joined
    assert "--iterations 50" in joined
    assert "--setup 50:50" in joined  # workload -> --setup
    # theta is stripped — run.sh's --distribution takes a bare token, not zipf(0.99)
    assert flags[flags.index("--distribution") + 1] == "zipf"
    assert "--value-size 100" in joined
    assert "--mem-budget 8,32" in joined  # CSV: run.sh converts comma->space
    assert "--critique full" in joined
    assert "--parallel-eval" in joined
    assert "--no-planner" in joined
    assert "--no-leaderboard" not in joined  # absent switch not emitted

    # --threads must be ONE space-separated argv token, NOT CSV: the orchestrator
    # parses it with str.split(), so "16,64" would crash the run.
    assert flags[flags.index("--threads") + 1] == "16 64"


def test_trace_files_override_distribution():
    db = _spec(distribution="zipf", trace_load="/data/load.dat", trace_run="/data/run.dat")
    joined = " ".join(_ctrl()._build_flags(db, 10))
    assert "--trace-load /data/load.dat" in joined
    assert "--trace-run /data/run.dat" in joined
    assert "--distribution" not in joined


def test_emitted_flags_are_all_accepted_by_run_sh():
    """Every flag the wrapper can emit must be a real run.sh option, else run.sh
    exits 1. Guards against whitelist/runtime drift (the most likely regression
    for a 'wrap as-is' adapter)."""
    # The exact set run.sh's case statement accepts (run.sh:74-117).
    accepted = {
        "--backend",
        "--mode",
        "--distribution",
        "--setup",
        "--value-size",
        "--mem-budget",
        "--iterations",
        "--max-turns",
        "--model",
        "--threads",
        "--delete-rate",
        "--critique",
        "--no-fast-exit",
        "--fast-exit",
        "--summary",
        "--show-baseline",
        "--seed",
        "--trace-load",
        "--trace-run",
        "--audit-every",
        "--feedback-level",
        "--parallel-eval",
        "--num-workers",
        "--no-planner",
        "--no-leaderboard",
        "--audit-checks-dir",
    }
    db = _spec(
        backend="codex",
        mode="inmem",
        model="m",
        value_size=1,
        mem_budget_gb=[8],
        threads=[16],
        max_turns=1,
        critique_mode="full",
        feedback_level="rich",
        audit_every=1,
        seed=1,
        num_workers=1,
        audit_checks_dir="/x",
        parallel_eval=True,
        no_planner=True,
        no_leaderboard=True,
        show_baseline=True,
        trace_load="/l",
        trace_run="/r",
    )
    emitted = {tok for tok in _ctrl()._build_flags(db, 5) if tok.startswith("--")}
    assert emitted <= accepted, f"unknown run.sh flags: {emitted - accepted}"


# ---------------------------------------------------------------------------
# Spec validation: fail fast with a clear message, not a buried run.sh exit
# ---------------------------------------------------------------------------


def test_validate_spec_requires_workload():
    with pytest.raises(ValueError, match="workload"):
        JitsKitController._validate_spec(JitsKitConfig(distribution="zipf"))


def test_validate_spec_requires_workload_source():
    with pytest.raises(ValueError, match="workload source"):
        JitsKitController._validate_spec(JitsKitConfig(workload="50:50"))


def test_validate_spec_accepts_trace_pair():
    JitsKitController._validate_spec(
        JitsKitConfig(workload="rmw", trace_load="/l", trace_run="/r")
    )  # must not raise


# ---------------------------------------------------------------------------
# Score selection: global peak across ALL iterations, paired with best_impl.cc
# ---------------------------------------------------------------------------


def test_best_entry_scans_all_iterations_for_global_peak():
    """best_impl.cc is the peak-Mops winner, but leaderboard[0] is ranked by
    avg %-of-baseline. The score must be the GLOBAL peak across all entries so it
    matches best_impl.cc — not leaderboard[0]'s peak."""
    leaderboard = [
        # leaderboard[0]: best avg %, but NOT the global peak
        {"workload_peaks": {"w": {"peak_mops": 12.0}}, "avg_pct_of_faster": 1.5},
        # a later entry holds the true global peak (this is best_impl.cc)
        {"workload_peaks": {"w": {"peak_mops": 20.0}}, "avg_pct_of_faster": 1.1},
    ]
    entry, peak = JitsKitController._best_entry(leaderboard)
    assert peak == 20.0
    assert entry["avg_pct_of_faster"] == 1.1


def test_read_best_pairs_global_peak_with_solution(tmp_path):
    (tmp_path / "best_impl.cc").write_text("// winning source\n")
    (tmp_path / "leaderboard.json").write_text(
        json.dumps(
            [
                {
                    "workload_peaks": {"a": {"peak_mops": 9.0}, "b": {"peak_mops": 18.0}},
                    "best_pct_of_faster": 1.4,
                    "all_validation_passed": True,
                },
            ]
        )
    )
    solution, mops, indicators = _ctrl()._read_best(tmp_path)
    assert solution == "// winning source\n"
    assert mops == 18.0  # max across this entry's workloads
    assert indicators["jitskit_best_pct_of_faster"] == 1.4
    assert indicators["jitskit_all_validation_passed"] is True


def test_read_best_none_when_incomplete(tmp_path):
    (tmp_path / "best_impl.cc").write_text("x")  # leaderboard.json missing
    assert _ctrl()._read_best(tmp_path) is None
    (tmp_path / "leaderboard.json").write_text("{ not json")  # mid-write
    assert _ctrl()._read_best(tmp_path) is None


# ---------------------------------------------------------------------------
# _add_program writes a clean, well-formed best into a real database
# ---------------------------------------------------------------------------


def test_add_program_records_best_in_database():
    cfg = Config.from_dict({"language": "cpp", "search": {"type": "jitskit"}})
    ctrl = _ctrl()
    ctrl.database = JitsKitDatabase("jitskit", cfg.search.database)

    ctrl._add_program(("// impl\n", 14.5, {"jitskit_status": "COMPLETED"}), iteration=3)

    best = ctrl.database.get_best_program()
    assert best is not None
    assert best.solution == "// impl\n"
    assert best.metrics["combined_score"] == 14.5
    assert best.metrics["validity"] == 1.0
    assert best.metrics["jitskit_status"] == "COMPLETED"
    assert best.iteration_found == 3
    assert len(ctrl.database.programs) == 1  # exactly one row, no duplicate


# ---------------------------------------------------------------------------
# Runtime resolution + the missing-runtime error
# ---------------------------------------------------------------------------


def test_runtime_dir_default_and_override():
    ctrl = _ctrl()
    default = ctrl._runtime_dir(JitsKitConfig())
    assert default.name == "runtime"
    override = ctrl._runtime_dir(JitsKitConfig(runtime_dir="~/somewhere/skykv-claude"))
    assert str(override).endswith("somewhere/skykv-claude")
    assert "~" not in str(override)  # expanduser applied
