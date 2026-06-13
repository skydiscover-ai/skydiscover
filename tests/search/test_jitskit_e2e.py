"""Hardware-gated end-to-end faithfulness test for the ``jitskit`` strategy.

This is caveat (c) from the plan review: the one check that closes the real gap
between "the wrapper looks right + is unit-tested with mocks" and "the integrated
run IS standalone Jitskit." Everything in ``test_jitskit.py`` runs with mocks; this
file runs the **actual runtime** once and verifies the wrapper reported it faithfully.

WHY IT IS NOT A "RUN TWICE AND DIFF" TEST
-----------------------------------------
The Jitskit loop is an LLM agent — it is *stochastic*. Two independent ``run.sh``
invocations of the same spec produce *different* ``best_impl.cc``, so a cross-run
byte-diff would fail for the wrong reason. The integration guarantee is **faithful
passthrough of a single run**, not cross-run determinism. So we do ONE real run and
assert the wrapper's database output matches *that same run's* on-disk artifacts:

  1. ``database.get_best_program().solution`` is **byte-identical** to the run dir's
     ``best_impl.cc`` (the wrapper did not mutate the source).
  2. ``combined_score`` equals the **global peak Mops/s** across ALL leaderboard
     entries — recomputed here *independently* (not via the controller's own
     ``_best_entry``), so it is a genuine cross-check that the score and the source
     come from the same iteration (the paper's "best across all iterations").

Invariant I1 (no Runner re-measurement) is NOT re-checked here: this test drives the
controller directly, so the Runner's ``mode="test"`` re-score path never executes and
a "no ``test_*`` metric" assertion would be tautological. I1 is enforced by the
one-line guard at ``runner.py:194`` (``getattr(controller, "skip_test_rescore",
False)`` — non-jitskit strategies default to False, so they are unaffected) and
asserted at the flag level in ``test_jitskit.py``. This file proves only what a real
run can prove: faithful passthrough (1) and correct peak selection (2).

GATING
------
Requires a real box: the runtime submodule checked out, the backend CLI on PATH, and
``SKYKV_E2E=1`` set deliberately. It is green only on a machine that can actually run
the agent + build/benchmark the C++ harness. It never runs in normal CI.

    SKYKV_E2E=1 uv run pytest tests/search/test_jitskit_e2e.py -q -s
"""

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import List, Optional

import pytest

from skydiscover.config import load_config
from skydiscover.search.default_discovery_controller import DiscoveryControllerInput
from skydiscover.search.jitskit.controller import JitsKitController
from skydiscover.search.jitskit.database import JitsKitDatabase

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_DIR = _REPO_ROOT / "skydiscover" / "search" / "jitskit" / "runtime"
_RUN_SH = _RUNTIME_DIR / "agent-pipeline" / "run.sh"
_TASK_CONFIG = _REPO_ROOT / "benchmarks" / "kvstore" / "0001_ycsb50_zipf_8gb" / "config.yaml"

# Keep the hardware run bounded — we only need one valid best to verify passthrough,
# not a full 50-iteration search. Overridable for debugging.
_E2E_ITERATIONS = int(os.environ.get("SKYKV_E2E_ITERATIONS", "1"))

pytestmark = pytest.mark.skipif(
    os.environ.get("SKYKV_E2E") != "1"
    or not _RUN_SH.exists()
    or not _TASK_CONFIG.exists()
    or (shutil.which("claude") is None and shutil.which("codex") is None),
    reason=(
        "hardware e2e: set SKYKV_E2E=1, init the runtime submodule "
        "(git submodule update --init --recursive), and have the backend CLI "
        "(claude/codex) on PATH. Green only on the measurement box."
    ),
)


def _independent_peak(leaderboard) -> Optional[float]:
    """Global peak Mops/s across ALL entries — recomputed independently of the
    controller, so this is a real cross-check rather than a tautology.

    Mirrors the runtime's BestTracker definition: the max single peak over every
    leaderboard entry's workloads (plus any entry-level peak/best fields).
    """
    entries = leaderboard if isinstance(leaderboard, list) else [leaderboard]
    peak: Optional[float] = None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        candidates: List[float] = []
        workload_peaks = entry.get("workload_peaks")
        if isinstance(workload_peaks, dict):
            for w in workload_peaks.values():
                if isinstance(w, dict) and isinstance(w.get("peak_mops"), (int, float)):
                    candidates.append(float(w["peak_mops"]))
        for key in ("peak_mops", "best_mops"):
            if isinstance(entry.get(key), (int, float)):
                candidates.append(float(entry[key]))
        if candidates:
            entry_peak = max(candidates)
            peak = entry_peak if peak is None else max(peak, entry_peak)
    return peak


def test_wrapper_reports_a_single_run_faithfully(tmp_path):
    """One real Jitskit run; the wrapper's DB best must equal that run's artifacts."""
    # Load the SHIPPED task config so the spec/flags are exactly what users run.
    # from_dict already swaps search.database to JitsKitConfig because type==jitskit.
    config = load_config(str(_TASK_CONFIG))
    assert config.search.type == "jitskit", "task config must select -s jitskit"

    # A trivial host evaluator file (never invoked: jitskit self-evaluates and the
    # Runner-side re-score is skipped). Pointing create_evaluator at a .py keeps the
    # controller's mandatory evaluator construction Docker-free — the agent itself
    # needs no Docker, and neither should this test.
    stub_eval = tmp_path / "stub_eval.py"
    stub_eval.write_text(
        "def evaluate(path):\n" "    return {'combined_score': 0.0, 'validity': 1.0}\n"
    )

    output_dir = tmp_path / "out"
    ci = DiscoveryControllerInput(
        config=config,
        evaluation_file=str(stub_eval),
        database=JitsKitDatabase("jitskit", config.search.database),
        file_suffix=".cc",
        output_dir=str(output_dir),
        evaluator_env_vars=None,
    )
    controller = JitsKitController(ci)

    best = asyncio.run(controller.run_discovery(0, max_iterations=_E2E_ITERATIONS))

    # The run must have produced a usable best. If a 1-iteration stochastic agent
    # produced nothing valid, that is an env/agent issue — re-run or raise iterations.
    assert best is not None, (
        "Jitskit produced no valid best — cannot verify passthrough. "
        f"See {output_dir}/jitskit.log; try SKYKV_E2E_ITERATIONS=2."
    )

    # Locate the exact run dir the wrapper tracked (it records it for us).
    summary = json.loads((output_dir / "run_summary.json").read_text())
    run_dir = Path(summary["run_dir"])
    assert run_dir.is_dir(), f"run_summary.json points at a missing run dir: {run_dir}"

    on_disk_impl = (run_dir / "best_impl.cc").read_text()
    leaderboard = json.loads((run_dir / "leaderboard.json").read_text())

    # (1) Byte-identical source — the wrapper passed through, did not mutate.
    assert best.solution == on_disk_impl, "reported solution differs from run's best_impl.cc"

    # (2) Score == independently recomputed global peak (same iteration as the source).
    expected_peak = _independent_peak(leaderboard)
    assert expected_peak is not None, "leaderboard.json carried no peak Mops/s"
    assert best.metrics["combined_score"] == pytest.approx(expected_peak), (
        f"reported score {best.metrics['combined_score']} != global peak {expected_peak} "
        "— score and source may be from different iterations (the leaderboard[0] bug)"
    )
