"""Unit + acceptance guards for the ``jitskit`` agentic strategy wrapper.

These tests do NOT run the real runtime (that needs the claude CLI + bare-metal
hardware).  They lock in the *wrapper contract* that the integration plan's
invariants depend on:

  * I1  — the controller reports the runtime's own leaderboard peak and the
          Runner's ``mode="test"`` re-score is skipped (``skip_test_rescore``).
  * I3  — every runtime knob is reachable from ``search.database`` (and the
          bespoke top-level blocks the first draft used are NOT silently dropped,
          because knobs now ride on the parsed ``search.database`` section).
  * I4  — a budget/threads LIST passes through verbatim.

The real "byte-identical vs ``bash run.sh``" acceptance test is documented in
``skydiscover/search/jitskit/README.md`` and is gated on hardware, so it is not
part of CI.
"""

import json

import pytest

from skydiscover.config import Config, JitsKitConfig
from skydiscover.search.jitskit.controller import JitsKitController
from skydiscover.search.registry import _CONTROLLER_REGISTRY, _DATABASE_REGISTRY


def _ctrl() -> JitsKitController:
    """A controller instance without running __init__ (which builds an evaluator)."""
    return object.__new__(JitsKitController)


def test_registered():
    # Importing route wires the registries.
    import skydiscover.search.route  # noqa: F401

    assert _CONTROLLER_REGISTRY["jitskit"].__name__ == "JitsKitController"
    assert _DATABASE_REGISTRY["jitskit"].__name__ == "JitsKitDatabase"


def test_skip_test_rescore_is_set():
    # The Runner reads this to bypass its authoritative re-score (I1).
    assert JitsKitController.skip_test_rescore is True


def test_knobs_ride_on_search_database_not_dropped():
    """Knobs under search.database are parsed (the D1 fix); lists survive (I4)."""
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
                    # an unknown extra key must be tolerated, not crash
                    "some_future_flag": "x",
                },
            },
        }
    )
    db = cfg.search.database
    assert isinstance(db, JitsKitConfig)
    assert db.backend == "claude"
    assert db.mode == "ltm"
    assert db.threads == [16]
    assert db.mem_budget_gb == [8, 32]  # list preserved (I4)
    assert db.critique_mode == "full"
    assert db.workload == "50:50"
    assert getattr(db, "some_future_flag") == "x"


def test_build_flags_translation():
    db = JitsKitConfig(
        backend="claude",
        mode="ltm",
        workload="50:50",
        distribution="zipf(0.99)",
        value_size=100,
        mem_budget_gb=[8, 32],
        threads=[16],
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
    assert "--distribution zipf(0.99)" in joined
    assert "--value-size 100" in joined
    assert "--mem-budget 8,32" in joined  # csv list
    assert "--threads 16" in joined
    assert "--critique full" in joined
    assert "--feedback-level rich" in joined
    assert "--audit-every 15" in joined
    assert "--parallel-eval" in joined  # bool switch present
    assert "--no-planner" in joined
    assert "--no-leaderboard" not in joined  # bool switch absent


def test_trace_files_override_distribution():
    db = JitsKitConfig(distribution="zipf", trace_load="/data/load.dat", trace_run="/data/run.dat")
    joined = " ".join(_ctrl()._build_flags(db, 10))
    assert "--trace-load /data/load.dat" in joined
    assert "--trace-run /data/run.dat" in joined
    assert "--distribution" not in joined  # trace files take precedence


def test_extract_peak_mops_from_workload_peaks():
    lb = [
        {
            "workload_peaks": {"50:50": {"peak_mops": 12.5}, "rmw": {"peak_mops": 18.0}},
            "best_pct_of_faster": 1.4,
            "all_validation_passed": True,
        }
    ]
    assert JitsKitController._extract_peak_mops(lb) == 18.0
    ind = JitsKitController._leading_indicators(lb)
    assert ind["jitskit_best_pct_of_faster"] == 1.4
    assert ind["jitskit_all_validation_passed"] is True


def test_read_best_uses_leaderboard_number(tmp_path):
    """The reported score IS the leaderboard peak — no re-measurement (I1)."""
    (tmp_path / "best_impl.cc").write_text("// winning source\n")
    (tmp_path / "leaderboard.json").write_text(
        json.dumps([{"workload_peaks": {"w": {"peak_mops": 9.99}}}])
    )
    solution, mops, indicators = _ctrl()._read_best(tmp_path)
    assert solution == "// winning source\n"
    assert mops == 9.99


def test_read_best_none_when_incomplete(tmp_path):
    # Missing leaderboard.json -> nothing to report yet.
    (tmp_path / "best_impl.cc").write_text("x")
    assert _ctrl()._read_best(tmp_path) is None
