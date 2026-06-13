"""Jitskit agentic strategy controller.

Wraps the Jitskit multi-agent KV-store synthesis runtime **as-is** and exposes
it to SkyDiscover as a search strategy, mirroring ``search/claude_code/``.

Design contract (see ``JITSKIT_SKYDISCOVER_INTEGRATION_PLAN.md`` §1, §5.1):

  * The runtime is NEVER edited and NEVER containerized.  ``-s jitskit`` runs
    ``bash <runtime>/agent-pipeline/run.sh <flags>`` in a **subprocess on the
    host**, so the agent keeps full host + filesystem access (numactl, perf,
    cgroup, /mnt/ssd) — unlike ``claude_code`` which is always Dockerized.
  * The reported score IS the runtime's own peak Mops/s from
    ``leaderboard.json``.  SkyDiscover does **not** re-measure: this controller
    sets ``skip_test_rescore = True`` so the Runner's ``mode="test"`` re-eval
    (``runner.py``) is bypassed for agentic strategies — that is invariant I1,
    and it requires the small additive guard in ``runner.py``.
  * Config knobs ride on ``search.database`` extras (a ``JitsKitConfig``); the
    task spec rides on ``benchmark.params`` resolved into ``run.sh`` flags.

The adapter surface is exactly "translate config -> env/flags, launch run.sh,
read back files" — no ``orchestrator.py`` edits are required.
"""

import asyncio
import json
import logging
import multiprocessing as mp
import os
import shlex
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, List, Optional

from skydiscover.evaluation import create_evaluator
from skydiscover.search.base_database import Program
from skydiscover.search.default_discovery_controller import (
    DiscoveryController,
    DiscoveryControllerInput,
)

logger = logging.getLogger(__name__)

# The vendored runtime is a git submodule of the private `skykv-claude` project
# ROOT (= PROJECT_DIR), not `agent-pipeline/` alone — run.sh reads setups/,
# interface/ and traces/ that live above agent-pipeline/.
_DEFAULT_RUNTIME_DIR = Path(__file__).parent / "runtime"
_RUN_SH = "agent-pipeline/run.sh"


class JitsKitController(DiscoveryController):
    """Discovery controller that delegates iteration to the Jitskit runtime."""

    # The Runner checks this to skip its authoritative ``mode="test"`` re-score
    # (invariant I1): Jitskit's own bare-metal leaderboard number is the score,
    # and a Dockerized re-measure would be weaker and non-comparable.
    skip_test_rescore = True

    def __init__(self, controller_input: DiscoveryControllerInput):
        self.config = controller_input.config
        self.evaluation_file = controller_input.evaluation_file
        self.database = controller_input.database
        self.file_suffix = controller_input.file_suffix or ".cc"
        self.output_dir = controller_input.output_dir
        self.evaluator_env_vars = controller_input.evaluator_env_vars

        self.config.evaluator.evaluation_file = self.evaluation_file
        self.config.evaluator.file_suffix = self.file_suffix

        # An evaluator is constructed so the base ``close()`` (which dereferences
        # ``self.evaluator``) is safe and so *other* strategies can score the
        # same task for comparability.  Jitskit itself never uses it — the score
        # comes from leaderboard.json and ``skip_test_rescore`` keeps the Runner
        # from invoking it.
        self.evaluator = create_evaluator(self.config.evaluator, env_vars=self.evaluator_env_vars)

        self.monitor_callback: Optional[Callable] = None
        self.feedback_reader = None
        self.early_stopping_triggered = False
        self.shutdown_event = mp.Event()

    # ------------------------------------------------------------------
    # Config -> run.sh flag translation
    # ------------------------------------------------------------------

    def _runtime_dir(self, db) -> Path:
        rt = getattr(db, "runtime_dir", None)
        return Path(rt).expanduser() if rt else _DEFAULT_RUNTIME_DIR

    def _build_flags(self, db, max_iterations: int) -> List[str]:
        """Translate JitsKitConfig knobs into run.sh flags (Appendix B)."""

        def _csv(v) -> str:
            return ",".join(str(x) for x in v) if isinstance(v, (list, tuple)) else str(v)

        flags: List[str] = [
            "--backend",
            str(getattr(db, "backend", "claude")),
            "--mode",
            str(getattr(db, "mode", "ltm")),
            "--iterations",
            str(max_iterations),
        ]

        # Workload signature.  --setup is the workload-mix key (kept verbatim;
        # the runtime is not edited, so the historical flag name stays).
        workload = getattr(db, "workload", None)
        if workload is not None:
            flags += ["--setup", str(workload)]

        # Distribution vs explicit trace files (mutually exclusive in run.sh).
        trace_load = getattr(db, "trace_load", None)
        trace_run = getattr(db, "trace_run", None)
        if trace_load and trace_run:
            flags += ["--trace-load", str(trace_load), "--trace-run", str(trace_run)]
        else:
            dist = getattr(db, "distribution", None)
            if dist is not None:
                flags += ["--distribution", str(dist)]

        # Simple scalar / list pass-throughs: (attr, flag, is_csv).
        passthrough = [
            ("value_size", "--value-size", False),
            ("mem_budget_gb", "--mem-budget", True),  # list allowed (I4)
            ("max_turns", "--max-turns", False),
            ("model", "--model", False),
            ("threads", "--threads", True),
            ("critique_mode", "--critique", False),
            ("feedback_level", "--feedback-level", False),
            ("audit_every", "--audit-every", False),
            ("seed", "--seed", False),
            ("num_workers", "--num-workers", False),
            ("audit_checks_dir", "--audit-checks-dir", False),
        ]
        for attr, flag, is_csv in passthrough:
            val = getattr(db, attr, None)
            if val is None or val == "":
                continue
            flags += [flag, _csv(val) if is_csv else str(val)]

        # Boolean switches.
        for attr, flag in [
            ("parallel_eval", "--parallel-eval"),
            ("no_planner", "--no-planner"),
            ("no_leaderboard", "--no-leaderboard"),
            ("show_baseline", "--show-baseline"),
        ]:
            if getattr(db, attr, False):
                flags.append(flag)

        return flags

    # ------------------------------------------------------------------
    # leaderboard.json parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_peak_mops(leaderboard: dict | list) -> Optional[float]:
        """Best-effort peak Mops/s from a leaderboard.json (ranked best-first)."""
        entry = leaderboard[0] if isinstance(leaderboard, list) and leaderboard else leaderboard
        if not isinstance(entry, dict):
            return None
        peaks = entry.get("workload_peaks")
        if isinstance(peaks, dict):
            mops = [
                w.get("peak_mops")
                for w in peaks.values()
                if isinstance(w, dict) and isinstance(w.get("peak_mops"), (int, float))
            ]
            if mops:
                return float(max(mops))
        for key in ("peak_mops", "best_mops"):
            if isinstance(entry.get(key), (int, float)):
                return float(entry[key])
        return None

    @staticmethod
    def _leading_indicators(leaderboard: dict | list) -> dict:
        entry = leaderboard[0] if isinstance(leaderboard, list) and leaderboard else leaderboard
        if not isinstance(entry, dict):
            return {}
        out = {}
        for k in ("avg_pct_of_faster", "best_pct_of_faster", "all_validation_passed", "status"):
            if k in entry:
                out[f"jitskit_{k}"] = entry[k]
        return out

    def _read_best(self, run_dir: Path) -> Optional[tuple]:
        """Return (solution, mops, indicators) from a run dir, or None."""
        lb_path = run_dir / "leaderboard.json"
        impl_path = run_dir / "best_impl.cc"
        if not (lb_path.exists() and impl_path.exists()):
            return None
        try:
            leaderboard = json.loads(lb_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        mops = self._extract_peak_mops(leaderboard)
        if mops is None:
            return None
        try:
            solution = impl_path.read_text()
        except OSError:
            return None
        return solution, mops, self._leading_indicators(leaderboard)

    # ------------------------------------------------------------------
    # Main discovery loop
    # ------------------------------------------------------------------

    async def run_discovery(
        self,
        start_iteration: int,
        max_iterations: int,
        checkpoint_callback: Optional[Callable] = None,
        **kwargs,
    ) -> Optional[Program]:
        db = self.database.config
        runtime_dir = self._runtime_dir(db)
        run_sh = runtime_dir / _RUN_SH
        if not run_sh.exists():
            raise FileNotFoundError(
                f"Jitskit runtime not found at {run_sh}. The runtime is a git "
                f"submodule of the private skykv-claude project root; run "
                f"`git submodule update --init --recursive` or set "
                f"`search.database.runtime_dir` to a local checkout."
            )

        if not os.environ.get("ANTHROPIC_API_KEY"):
            logger.warning("ANTHROPIC_API_KEY is not set; the Jitskit claude backend will fail.")

        flags = self._build_flags(db, max_iterations)

        # Collision pre-flight (Appendix B): the runtime keys its workspace by
        # {backend}_{mode}_{run_key} WITHOUT a timestamp (run.sh:327 derives
        # RUN_KEY from the spec and cannot be overridden via env).  Concurrent
        # same-spec runs would clobber each other — warn rather than silently
        # corrupt.  Isolate by pointing `runtime_dir` at distinct checkouts.
        ws_dir = runtime_dir / "workspaces"
        if ws_dir.is_dir() and any(ws_dir.iterdir()):
            logger.warning(
                "Jitskit workspace dir %s is non-empty — a same-spec run may be "
                "in progress. Concurrent identical-spec runs share a workspace "
                "(run.sh:327); use a distinct runtime_dir to isolate them.",
                ws_dir,
            )

        runs_dir = runtime_dir / "runs"
        before = set(p.name for p in runs_dir.iterdir()) if runs_dir.is_dir() else set()

        out = Path(self.output_dir) if self.output_dir else None
        if out:
            out.mkdir(parents=True, exist_ok=True)
        log_path = (
            (out / "jitskit.log") if out else (runtime_dir / f".jitskit_{uuid.uuid4().hex[:8]}.log")
        )

        cmd = ["bash", str(run_sh), *flags]
        logger.info("Launching Jitskit on host: %s (cwd=%s)", shlex.join(cmd), runtime_dir)

        loop = asyncio.get_running_loop()
        done = threading.Event()

        def _run() -> int:
            with open(log_path, "w") as lf:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(runtime_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=os.environ.copy(),
                    text=True,
                )
                try:
                    for line in proc.stdout:  # tee: preserve the live console
                        lf.write(line)
                        lf.flush()
                        logger.info("[jitskit] %s", line.rstrip())
                        if self.shutdown_event.is_set():
                            proc.terminate()
                            break
                finally:
                    proc.wait()
                done.set()
                return proc.returncode

        run_future = loop.run_in_executor(None, _run)

        # Resolve the run dir (created by orchestrator: {backend}_{mode}_{run_key}_{ts}).
        run_dir: Optional[Path] = None
        last_mops = float("-inf")
        ckpt = 0
        ckpt_interval = max(1, getattr(self.config, "checkpoint_interval", 1))

        while not run_future.done():
            await asyncio.sleep(5)
            if run_dir is None and runs_dir.is_dir():
                new = [p for p in runs_dir.iterdir() if p.name not in before and p.is_dir()]
                if new:
                    run_dir = max(new, key=lambda p: p.stat().st_mtime)
                    logger.info("Jitskit run dir: %s", run_dir)
            if run_dir is not None:
                best = self._read_best(run_dir)
                if best and best[1] > last_mops:
                    last_mops = best[1]
                    ckpt += 1
                    self._add_program(best, ckpt, parent=None)
                    if self.monitor_callback:
                        try:
                            self.monitor_callback(self.database.get_best_program(), ckpt)
                        except Exception:
                            logger.debug("monitor_callback failed", exc_info=True)
                    if checkpoint_callback and ckpt % ckpt_interval == 0:
                        checkpoint_callback(ckpt)

        rc = await run_future
        logger.info("Jitskit exited with code %s", rc)

        # Final best (in case the last improvement landed after the last poll).
        if run_dir is not None:
            best = self._read_best(run_dir)
            if best and best[1] >= last_mops:
                ckpt += 1
                self._add_program(best, ckpt, parent=None)

        if out and log_path.exists():
            try:
                summary = {
                    "runtime_dir": str(runtime_dir),
                    "run_dir": str(run_dir) if run_dir else None,
                    "flags": flags,
                    "peak_mops": None if last_mops == float("-inf") else last_mops,
                    "returncode": rc,
                }
                (out / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            except OSError:
                pass

        return self.database.get_best_program()

    def _add_program(self, best: tuple, iteration: int, parent: Optional[str]) -> None:
        solution, mops, indicators = best
        prog = Program(
            id=str(uuid.uuid4()),
            solution=solution,
            language="cpp",
            metrics={"combined_score": mops, "validity": 1.0, **indicators},
            iteration_found=iteration,
            parent_id=parent,
            metadata={"strategy": "jitskit"},
        )
        self.database.add(prog, iteration=iteration)
        logger.info("[jitskit] new best @ iter %d: %.3f Mops/s", iteration, mops)
