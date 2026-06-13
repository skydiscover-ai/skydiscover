"""Jitskit agentic strategy controller.

Wraps the Jitskit multi-agent KV-store synthesis runtime **as-is** and exposes
it to SkyDiscover as a search strategy — the multi-agent sibling of
``search/claude_code/``.

What "as-is" means, concretely:

* The runtime is never edited. ``-s jitskit`` runs
  ``bash <runtime>/agent-pipeline/run.sh <flags>`` in a subprocess **on the
  host** (not in Docker, unlike ``claude_code``) so the agent keeps the full
  host access it needs to measure real hardware: ``numactl``, ``perf``,
  cgroup memory caps, and the NVMe scratch dir.
* The reported score IS the runtime's own peak Mops/s read back from
  ``leaderboard.json``. SkyDiscover does not re-measure — the controller sets
  ``skip_test_rescore = True`` so the Runner's ``mode="test"`` re-evaluation is
  bypassed (integration invariant I1). A Docker re-measure would run off the
  measurement hardware and report a weaker, non-comparable number.
* Every knob rides on ``search.database`` (a ``JitsKitConfig``); the controller
  translates it to the equivalent ``run.sh`` flags. There is no
  ``orchestrator.py`` edit anywhere in the path.

The whole adapter surface is therefore: translate config -> flags, launch
``run.sh``, and read back ``best_impl.cc`` + ``leaderboard.json``.
"""

import asyncio
import json
import logging
import multiprocessing as mp
import os
import shlex
import shutil
import signal
import subprocess
import uuid
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from skydiscover.evaluation import create_evaluator
from skydiscover.search.base_database import Program
from skydiscover.search.default_discovery_controller import (
    DiscoveryController,
    DiscoveryControllerInput,
)

logger = logging.getLogger(__name__)

# The runtime is vendored in-tree (committed, not a submodule) at ``runtime/`` — the
# clean run-essentials of the private ``skykv-claude`` project ROOT (= the runtime's
# ``PROJECT_DIR``). It must be the project root, not ``agent-pipeline/`` alone, because
# run.sh reads ``setups/``, ``interface/`` and traces one level above ``agent-pipeline/``.
_DEFAULT_RUNTIME_DIR = Path(__file__).parent / "runtime"
_RUN_SH = "agent-pipeline/run.sh"

# How often the poll loop wakes to look for new bests / honor a shutdown request.
_POLL_SECONDS = 5
# Grace period after SIGTERM before we escalate to SIGKILL on shutdown.
_SHUTDOWN_GRACE_SECONDS = 30


class JitsKitController(DiscoveryController):
    """Discovery controller that delegates iteration to the Jitskit runtime."""

    # Read by the Runner to skip its authoritative ``mode="test"`` re-score
    # (invariant I1): Jitskit owns its bare-metal measurement, and a Dockerized
    # re-measure would be weaker and non-comparable.
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

        # We build an evaluator even though Jitskit never calls it: the base
        # ``close()`` dereferences ``self.evaluator``, and *other* strategies use
        # it to score the same task for comparability. Jitskit's own score comes
        # from ``leaderboard.json``, and ``skip_test_rescore`` keeps the Runner
        # from ever invoking this evaluator on a Jitskit result.
        self.evaluator = create_evaluator(self.config.evaluator, env_vars=self.evaluator_env_vars)

        self.monitor_callback: Optional[Callable] = None
        self.feedback_reader = None
        self.early_stopping_triggered = False
        self.shutdown_event = mp.Event()

    # ------------------------------------------------------------------
    # Config -> run.sh translation
    # ------------------------------------------------------------------

    def _runtime_dir(self, db) -> Path:
        """Resolve the runtime dir (the vendored in-tree ``runtime/`` by default)."""
        configured = getattr(db, "runtime_dir", None)
        return Path(configured).expanduser() if configured else _DEFAULT_RUNTIME_DIR

    @staticmethod
    def _validate_spec(db) -> None:
        """Fail fast, with a clear message, on configs run.sh would reject.

        run.sh exits 1 (buried in a subprocess log) if the workload mix or a
        workload source is missing. We surface that as a plain ValueError before
        launching anything.
        """
        if not getattr(db, "workload", None):
            raise ValueError(
                "jitskit requires search.database.workload (the workload mix, "
                "e.g. '50:50', 'rmw', '100:0', '0:100') — it maps to run.sh --setup."
            )
        has_distribution = bool(getattr(db, "distribution", None))
        has_trace = bool(getattr(db, "trace_load", None)) and bool(getattr(db, "trace_run", None))
        if not (has_distribution or has_trace):
            raise ValueError(
                "jitskit requires a workload source: set search.database.distribution "
                "(e.g. 'zipf') OR both trace_load and trace_run."
            )

    def _build_flags(self, db, max_iterations: int) -> List[str]:
        """Translate a JitsKitConfig into the exact run.sh flags (Appendix B)."""
        flags: List[str] = [
            "--backend",
            str(getattr(db, "backend", "claude")),
            "--mode",
            str(getattr(db, "mode", "ltm")),
            "--iterations",
            str(max_iterations),
            "--setup",
            str(db.workload),
        ]

        # Distribution and explicit trace files are mutually exclusive in run.sh;
        # trace files take precedence (they bypass the runtime's default data dir).
        if getattr(db, "trace_load", None) and getattr(db, "trace_run", None):
            flags += ["--trace-load", str(db.trace_load), "--trace-run", str(db.trace_run)]
        elif getattr(db, "distribution", None):
            # run.sh's --distribution takes a bare token (zipf/uniform/...); a theta in
            # e.g. "zipf(0.99)" is carried by the trace file, not the flag — strip it.
            flags += ["--distribution", str(db.distribution).split("(", 1)[0].strip()]

        # Scalar pass-throughs: (attribute, flag).
        for attr, flag in [
            ("value_size", "--value-size"),
            ("max_turns", "--max-turns"),
            ("model", "--model"),
            ("critique_mode", "--critique"),
            ("feedback_level", "--feedback-level"),
            ("audit_every", "--audit-every"),
            ("seed", "--seed"),
            ("num_workers", "--num-workers"),
            ("audit_checks_dir", "--audit-checks-dir"),
        ]:
            value = getattr(db, attr, None)
            if value not in (None, ""):
                flags += [flag, str(value)]

        # List-valued knobs need the format run.sh expects for each one:
        #   --mem-budget takes comma-separated ("8,32"); run.sh splits it itself.
        #   --threads takes ONE space-separated argv token ("16 64"); the
        #     orchestrator parses it with str.split(), so a comma would crash it.
        mem_budget = getattr(db, "mem_budget_gb", None)
        if mem_budget:
            flags += ["--mem-budget", ",".join(str(x) for x in mem_budget)]
        threads = getattr(db, "threads", None)
        if threads:
            flags += ["--threads", " ".join(str(x) for x in threads)]

        # Zero-argument boolean switches.
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
    # leaderboard.json -> (solution, score, indicators)
    # ------------------------------------------------------------------

    @staticmethod
    def _best_entry(leaderboard) -> Optional[Tuple[dict, float]]:
        """Return the (entry, peak_mops) with the highest peak Mops/s, or None.

        The runtime writes ``best_impl.cc`` for the iteration with the highest
        PEAK Mops/s (BestTracker), whereas ``leaderboard.json`` is sorted by
        average % of the baseline. So we must NOT trust ``leaderboard[0]``'s
        score for ``best_impl.cc`` — we scan every entry for the global peak,
        which is exactly the iteration ``best_impl.cc`` came from and is also the
        paper's definition of the score ("best throughput across all
        iterations"). This keeps the reported score and the reported source code
        from the same iteration.
        """
        entries = leaderboard if isinstance(leaderboard, list) else [leaderboard]
        best: Optional[Tuple[dict, float]] = None
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            peaks = entry.get("workload_peaks")
            candidates: List[float] = []
            if isinstance(peaks, dict):
                candidates = [
                    w["peak_mops"]
                    for w in peaks.values()
                    if isinstance(w, dict) and isinstance(w.get("peak_mops"), (int, float))
                ]
            for key in ("peak_mops", "best_mops"):
                if isinstance(entry.get(key), (int, float)):
                    candidates.append(entry[key])
            if not candidates:
                continue
            entry_peak = float(max(candidates))
            if best is None or entry_peak > best[1]:
                best = (entry, entry_peak)
        return best

    @staticmethod
    def _leading_indicators(entry: dict) -> dict:
        """Advisory metrics from the same entry the score came from."""
        out = {}
        for key in ("avg_pct_of_faster", "best_pct_of_faster", "all_validation_passed", "status"):
            if key in entry:
                out[f"jitskit_{key}"] = entry[key]
        return out

    def _read_best(self, run_dir: Path) -> Optional[Tuple[str, float, dict]]:
        """Return (solution, peak_mops, indicators) for a run dir, or None.

        None means "no usable best yet" — either file is missing, the JSON is
        mid-write, or no entry carries a peak number.
        """
        leaderboard_path = run_dir / "leaderboard.json"
        impl_path = run_dir / "best_impl.cc"
        if not (leaderboard_path.exists() and impl_path.exists()):
            return None
        try:
            leaderboard = json.loads(leaderboard_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        best = self._best_entry(leaderboard)
        if best is None:
            return None
        entry, peak_mops = best
        try:
            solution = impl_path.read_text()
        except OSError:
            return None
        return solution, peak_mops, self._leading_indicators(entry)

    # ------------------------------------------------------------------
    # Subprocess lifecycle
    # ------------------------------------------------------------------

    def _host_preflight(self, db) -> None:
        """Warn about missing host prerequisites the runtime needs (non-fatal).

        These are warnings, not errors: the runtime degrades on its own (e.g. it
        runs without a memory cap if sudo is unavailable), but a missing tool is
        otherwise an opaque mid-run failure, so we surface it up front.
        """
        missing = [tool for tool in ("cmake", "make", "numactl") if shutil.which(tool) is None]
        if missing:
            logger.warning(
                "Host tools not found on PATH: %s — the runtime build/benchmark may fail.", missing
            )

        backend = getattr(db, "backend", "claude") or "claude"
        if shutil.which(backend) is None:
            logger.warning("Backend CLI %r not found on PATH — the agent cannot start.", backend)
        if backend == "claude" and not os.environ.get("ANTHROPIC_API_KEY"):
            logger.warning("ANTHROPIC_API_KEY is not set — the claude backend will fail.")

        budgets = getattr(db, "mem_budget_gb", None) or []
        if (getattr(db, "mode", "ltm") or "ltm") == "ltm" and any(b > 0 for b in budgets):
            probe = subprocess.run(["sudo", "-n", "true"], capture_output=True)
            if probe.returncode != 0:
                logger.warning(
                    "Passwordless sudo unavailable: ltm memory-budget cgroups will be "
                    "skipped and the benchmark will run WITHOUT a memory cap, so the "
                    "reported Mops/s will not be comparable to a capped run."
                )

    def _terminate(self, proc: subprocess.Popen) -> None:
        """Stop the run.sh process tree, giving the runtime time to clean up.

        run.sh ``exec``s the orchestrator, whose SIGTERM handler releases the
        cgroup — so we SIGTERM the whole process group (reaching the benchmark
        child too) and only escalate to SIGKILL if it overruns the grace period.
        A direct SIGKILL would orphan the cgroup.
        """
        if proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
        try:
            proc.wait(timeout=_SHUTDOWN_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            logger.warning(
                "Jitskit did not exit within %ss — sending SIGKILL.", _SHUTDOWN_GRACE_SECONDS
            )
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()

    # ------------------------------------------------------------------
    # Main discovery loop
    # ------------------------------------------------------------------

    async def run_discovery(
        self,
        start_iteration: int,
        max_iterations: int,
        checkpoint_callback: Optional[Callable] = None,
        **kwargs,  # post_process_result / retry_times: not applicable to a wrapped runtime
    ) -> Optional[Program]:
        db = self.database.config
        self._validate_spec(db)

        runtime_dir = self._runtime_dir(db)
        run_sh = runtime_dir / _RUN_SH
        if not run_sh.exists():
            raise FileNotFoundError(
                f"Jitskit runtime not found at {run_sh}. The runtime is vendored in-tree at "
                f"skydiscover/search/jitskit/runtime/; if it is missing, set "
                f"`search.database.runtime_dir` to a local skykv-claude checkout."
            )

        self._host_preflight(db)
        flags = self._build_flags(db, max_iterations)

        # The orchestrator names each run dir ``{backend}_{mode}_{run_key}_{ts}``;
        # we only ever care about dirs for THIS backend/mode, which lets us ignore
        # unrelated concurrent runs sharing the checkout.
        backend = getattr(db, "backend", "claude")
        mode = getattr(db, "mode", "ltm")
        run_prefix = f"{backend}_{mode}_"

        # Collision pre-flight: run.sh derives RUN_KEY from the spec with no
        # timestamp (run.sh:327) and cannot be overridden via env, so two
        # same-spec runs in one checkout share a workspace and clobber each other.
        # Isolate by pointing runtime_dir at distinct checkouts.
        workspaces = runtime_dir / "workspaces"
        if workspaces.is_dir() and any(p.name.startswith(run_prefix) for p in workspaces.iterdir()):
            logger.warning(
                "A %s* workspace already exists under %s — a same-spec run may be in "
                "progress. Concurrent identical-spec runs share a workspace; use a "
                "distinct search.database.runtime_dir to isolate them.",
                run_prefix,
                workspaces,
            )

        runs_dir = runtime_dir / "runs"
        existing_runs = {p.name for p in runs_dir.iterdir()} if runs_dir.is_dir() else set()

        out = Path(self.output_dir) if self.output_dir else None
        if out:
            out.mkdir(parents=True, exist_ok=True)
        log_path = (
            (out / "jitskit.log") if out else (runtime_dir / f".jitskit_{uuid.uuid4().hex[:8]}.log")
        )

        cmd = ["bash", str(run_sh), *flags]
        logger.info("Launching Jitskit on host: %s (cwd=%s)", shlex.join(cmd), runtime_dir)

        # Created in the async scope so both the stdout-draining thread and this
        # poll loop can see it; start_new_session puts it in its own process group
        # so _terminate() can signal the whole tree.
        proc = subprocess.Popen(
            cmd,
            cwd=str(runtime_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            text=True,
            start_new_session=True,
        )

        def _drain_stdout() -> int:
            """Tee the merged stdout/stderr to the log and the live console."""
            with open(log_path, "w") as log_file:
                for line in proc.stdout:
                    log_file.write(line)
                    log_file.flush()
                    logger.info("[jitskit] %s", line.rstrip())
            proc.wait()
            return proc.returncode

        loop = asyncio.get_running_loop()
        drain = loop.run_in_executor(None, _drain_stdout)

        run_dir: Optional[Path] = None
        best_mops_added = float("-inf")
        checkpoints = 0
        checkpoint_interval = max(1, getattr(self.config, "checkpoint_interval", 1))

        while not drain.done():
            if self.shutdown_event.is_set():
                logger.info("Shutdown requested — stopping the Jitskit run.")
                self._terminate(proc)
                break
            await asyncio.sleep(_POLL_SECONDS)

            if run_dir is None and runs_dir.is_dir():
                fresh = [
                    p
                    for p in runs_dir.iterdir()
                    if p.is_dir() and p.name.startswith(run_prefix) and p.name not in existing_runs
                ]
                if fresh:
                    run_dir = max(fresh, key=lambda p: p.stat().st_mtime)
                    logger.info("Tracking Jitskit run dir: %s", run_dir)

            if run_dir is not None:
                best = self._read_best(run_dir)
                if best and best[1] > best_mops_added:
                    best_mops_added = best[1]
                    checkpoints += 1
                    self._add_program(best, checkpoints)
                    if self.monitor_callback:
                        try:
                            self.monitor_callback(self.database.get_best_program(), checkpoints)
                        except Exception:
                            logger.debug("monitor_callback failed", exc_info=True)
                    if checkpoint_callback and checkpoints % checkpoint_interval == 0:
                        checkpoint_callback(checkpoints)

        rc = await drain
        logger.info("Jitskit exited with code %s", rc)

        # One last read in case the final improvement landed between polls.
        if run_dir is not None:
            best = self._read_best(run_dir)
            if best and best[1] > best_mops_added:
                best_mops_added = best[1]
                checkpoints += 1
                self._add_program(best, checkpoints)

        # Distinguish "the runtime refused to start / failed" from "the agent
        # genuinely found nothing", which otherwise look identical to the caller.
        if best_mops_added == float("-inf"):
            if rc != 0:
                raise RuntimeError(
                    f"Jitskit run.sh exited {rc} without producing a result; see {log_path} "
                    f"for the error (common causes: missing --setup/--distribution, "
                    f"absent trace files, /mnt scratch dir, or backend CLI)."
                )
            if run_dir is None:
                raise RuntimeError(
                    f"Jitskit exited 0 but created no {run_prefix}* run dir under {runs_dir}; "
                    f"see {log_path}."
                )
            logger.warning("Jitskit completed without a valid best (no peak Mops in leaderboard).")

        if out:
            summary = {
                "runtime_dir": str(runtime_dir),
                "run_dir": str(run_dir) if run_dir else None,
                "flags": flags,
                "peak_mops": None if best_mops_added == float("-inf") else best_mops_added,
                "returncode": rc,
            }
            (out / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

        return self.database.get_best_program()

    def _add_program(self, best: Tuple[str, float, dict], iteration: int) -> None:
        solution, mops, indicators = best
        # Derive validity from the runtime's own report. A peak the runtime marked NOT
        # validated gets score+validity zeroed, so an unvalidated win is never reported as
        # valid (and score+source stay on the same iteration). If the runtime didn't report
        # the flag at all, default to valid — preserve prior behavior, don't zero on missing data.
        valid = bool(indicators.get("jitskit_all_validation_passed", True))
        program = Program(
            id=str(uuid.uuid4()),
            solution=solution,
            language="cpp",
            metrics={
                "combined_score": mops if valid else 0.0,
                "validity": 1.0 if valid else 0.0,
                **indicators,
            },
            iteration_found=iteration,
            parent_id=None,
            metadata={"strategy": "jitskit"},
        )
        self.database.add(program, iteration=iteration)
        logger.info("[jitskit] new best @ iteration %d: %.3f Mops/s", iteration, mops)
