#!/usr/bin/env python3
"""
agent-pipeline/orchestrator.py -- Evolution loop orchestrator.

Runs N iterations of KV store evolution: call agent, build, test, benchmark,
track best. Called by run.sh which handles arg parsing and
environment setup.

All config is passed via SKYKV_* environment variables (set by run.sh).
"""

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

_VALID_CRITIQUE_MODES = ("off", "review", "audit", "full")


def _detect_agent_failure(log_path: str) -> str | None:
    """Scan the agent's JSON event log for failures that left the workspace
    impl file untouched. Returns a short reason string, or None if the agent
    completed normally with at least one file_change.

    Without this check, a rate-limited codex session (429) that never wrote
    to kvstore_impl.cc would cascade into 10+ IMPL_UNCHANGED_REUSED_* statuses
    because the orchestrator only checks file existence, not whether THIS
    iteration produced it. The workspace impl from a prior iter is enough
    to trip the "unchanged" shortcut.

    Detection rule: if any `turn.failed`/`error` event with a retry-limit or
    status-code message appears AND no `file_change`-kind events were
    completed, treat it as an agent failure. Otherwise pass-through.
    """
    if not os.path.isfile(log_path):
        return None

    saw_file_change = False
    last_error: str | None = None
    turn_failed = False
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = event.get("type", "")
                if etype == "item.completed":
                    item = event.get("item") or {}
                    if item.get("type") == "file_change":
                        saw_file_change = True
                elif etype in ("error", "turn.failed"):
                    msg = event.get("message") or (
                        (event.get("error") or {}).get("message") or ""
                    )
                    last_error = msg
                    if etype == "turn.failed":
                        turn_failed = True
    except OSError:
        return None

    if turn_failed and not saw_file_change:
        # Summarize the failure reason
        if last_error and "429" in last_error:
            return f"rate_limited ({last_error[:120]})"
        if last_error:
            return f"turn_failed ({last_error[:120]})"
        return "turn_failed (no error message)"
    return None


def _normalize_critique_mode(mode: str, legacy_enabled: bool) -> str:
    """Resolve SKYKV_CRITIQUE_MODE to one of {off, review, audit, full}.

    If the env var is unset but the legacy SKYKV_CRITIQUE_ENABLED=1 is set,
    default to "review" to preserve prior behavior.
    """
    mode = (mode or "").strip().lower()
    if mode in _VALID_CRITIQUE_MODES:
        return mode
    return "review" if legacy_enabled else "off"


# Prompt text for the generator is kept in agent-pipeline/prompts/generator_prompt.txt and
# loaded here (same pattern as critique.py / critique_audit.py). Python code
# in this module must NOT embed prompt-facing strings — changes to what the
# agent reads belong in the .txt, not in orchestrator.py.
_GENERATOR_PROMPT_SECTION_RE = re.compile(r"^##\s+([A-Z][A-Z0-9_]*)\s*$")
_GENERATOR_PROMPT_CACHE: dict[str, str] | None = None

# Stage-A turn budget used by --critique off (Alex's three-stage fallback).
# Alex's tested 5+15 split (cf. PR #20 result table) is the default.
STAGE_A_TURNS_ALEX_MODE = 5


def _load_generator_prompt_sections() -> dict[str, str]:
    """Parse generator_prompt.txt into {SECTION_NAME: text}.

    Section delimiters match `^##\\s+NAME` where NAME is UPPER_SNAKE_CASE,
    so section bodies can freely use `##`-prefixed Markdown subheadings
    without being treated as new top-level sections.
    """
    global _GENERATOR_PROMPT_CACHE
    if _GENERATOR_PROMPT_CACHE is not None:
        return _GENERATOR_PROMPT_CACHE
    path = os.path.join(os.path.dirname(__file__), "prompts", "generator_prompt.txt")
    with open(path) as f:
        content = f.read()
    sections: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in content.splitlines():
        m = _GENERATOR_PROMPT_SECTION_RE.match(line)
        if m:
            if current is not None:
                sections[current] = "\n".join(buf).strip() + "\n"
            current = m.group(1)
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip() + "\n"
    _GENERATOR_PROMPT_CACHE = sections
    return sections


_PLANNING_PROMPT_CACHE: dict[str, str] | None = None


def _load_planning_prompt_sections() -> dict[str, str]:
    """Parse planning_prompt.txt into {SECTION_NAME: text}."""
    global _PLANNING_PROMPT_CACHE
    if _PLANNING_PROMPT_CACHE is not None:
        return _PLANNING_PROMPT_CACHE
    path = os.path.join(os.path.dirname(__file__), "prompts", "planning_prompt.txt")
    with open(path) as f:
        content = f.read()
    sections: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in content.splitlines():
        m = _GENERATOR_PROMPT_SECTION_RE.match(line)
        if m:
            if current is not None:
                sections[current] = "\n".join(buf).strip() + "\n"
            current = m.group(1)
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip() + "\n"
    _PLANNING_PROMPT_CACHE = sections
    return sections


_DESIGN_PROMPT_CACHE: dict[str, str] | None = None


def _load_design_prompt_sections() -> dict[str, str]:
    """Parse design_prompt.txt into {SECTION_NAME: text}."""
    global _DESIGN_PROMPT_CACHE
    if _DESIGN_PROMPT_CACHE is not None:
        return _DESIGN_PROMPT_CACHE
    path = os.path.join(os.path.dirname(__file__), "prompts", "design_prompt.txt")
    with open(path) as f:
        content = f.read()
    sections: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in content.splitlines():
        m = _GENERATOR_PROMPT_SECTION_RE.match(line)
        if m:
            if current is not None:
                sections[current] = "\n".join(buf).strip() + "\n"
            current = m.group(1)
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip() + "\n"
    _DESIGN_PROMPT_CACHE = sections
    return sections


@dataclass
class RunConfig:
    """All parameters for a single evolution run."""
    backend: str
    mode: str
    dist_key: str
    dist_desc: str
    load_file: str
    run_file: str
    setup_key: str
    setup_name: str
    setup_desc: str
    wl_id: int
    value_size: int
    mem_budget_gb: list[int]
    iterations: int
    max_turns: int
    model: str
    threads: list[int]
    card_set: str
    baseline_name: str
    storage_dir: str
    project_dir: str
    workspace: str
    experiments_dir: str
    checkpoints_dir: str
    correctness_storage: str
    correctness_mem: int
    instr_file: str
    critique_enabled: bool
    critique_mode: str  # one of "off", "review", "audit", "full"
    bench_fast_exit: bool  # skip store destructor on bench exit (Mops/s already written)
    summary_enabled: bool  # rotate codex sessions + inject a summary when history gets huge
    show_baseline: bool  # show "X% of FASTER" in agent feedback (off by default)
    audit_every: int  # auditor fires every N iters with the last N iters' artifacts; 1 = per-iter
    seed_name: str    # name of seed dir under seeds/ to pre-populate iter 1 with (empty = off)
    feedback_level: str  # "minimal" (Mops/s only) or "rich" (all leading indicators + perf)
    parallel_eval: bool  # fan out benchmarks + correctness to GCP worker VMs
    num_workers: int     # 0 = auto (budgets × threads)
    # Bypass switches for "off-the-shelf Claude Code" baseline runs. Each
    # disables one paper contribution without otherwise altering the loop.
    no_planner: bool         # skip Phase 0 planning + whiteboard injection
    no_leaderboard: bool     # skip best-impl tracking + reference injection
    audit_checks_dir: str    # path to frozen audit suite; if set, skip live
                             # auditor agent and use these tests as the gate

    @classmethod
    def from_env(cls) -> "RunConfig":
        """Read SKYKV_* env vars set by run.sh."""
        def env(key: str) -> str:
            val = os.environ.get(f"SKYKV_{key}", "")
            if not val:
                sys.exit(f"ERROR: SKYKV_{key} not set")
            return val

        def env_list_int(key: str) -> list[int]:
            return [int(x) for x in env(key).split()]

        project_dir = env("PROJECT_DIR")
        run_key = env("RUN_KEY")
        backend = env("BACKEND")
        mode = env("MODE")

        experiments_dir = os.path.join(
            project_dir, "runs",
            f"{backend}_{mode}_{run_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        return cls(
            backend=backend,
            mode=mode,
            dist_key=env("DIST_KEY"),
            dist_desc=env("DIST_DESC"),
            load_file=env("LOAD"),
            run_file=env("RUN"),
            setup_key=env("SETUP_KEY"),
            setup_name=env("SETUP_NAME"),
            setup_desc=env("SETUP_DESC"),
            wl_id=int(env("WL_ID")),
            value_size=int(env("VALUE_SIZE")),
            mem_budget_gb=env_list_int("MEM_BUDGET_GB"),
            iterations=int(env("ITERATIONS")),
            max_turns=int(env("MAX_TURNS")),
            model=env("MODEL"),
            threads=env_list_int("THREADS"),
            card_set=env("CARD_SET"),
            baseline_name=env("BASELINE_NAME"),
            storage_dir=env("STORAGE_DIR"),
            project_dir=project_dir,
            workspace=os.path.join(project_dir, "workspaces", f"{backend}_{mode}_{run_key}"),
            experiments_dir=experiments_dir,
            # Leaderboard now lives at the top of experiments_dir (no more
            # checkpoints/iter_* duplication). Field retained for back-compat.
            checkpoints_dir=experiments_dir,
            correctness_storage=os.path.join(
                "/mnt/ssd", f"kvstore_consistency_{backend}_{mode}_{run_key}"
            ),
            correctness_mem=4 * 1024 * 1024 * 1024,
            instr_file="CLAUDE.md" if backend == "claude" else "AGENTS.md",
            critique_enabled=os.environ.get("SKYKV_CRITIQUE_ENABLED", "0") == "1",
            critique_mode=_normalize_critique_mode(
                os.environ.get("SKYKV_CRITIQUE_MODE", ""),
                legacy_enabled=os.environ.get("SKYKV_CRITIQUE_ENABLED", "0") == "1",
            ),
            # Default ON — the iteration loop doesn't care about destructor
            # hygiene and the OS reclaims memory either way. Opt out only for
            # final eval runs that want a clean shutdown for some reason.
            bench_fast_exit=os.environ.get("SKYKV_BENCH_FAST_EXIT", "1") == "1",
            summary_enabled=os.environ.get("SKYKV_SUMMARY", "0") == "1",
            show_baseline=os.environ.get("SKYKV_SHOW_BASELINE", "0") == "1",
            audit_every=max(1, int(os.environ.get("SKYKV_AUDIT_EVERY", "1") or "1")),
            seed_name=os.environ.get("SKYKV_SEED", "").strip(),
            feedback_level=(os.environ.get("SKYKV_FEEDBACK_LEVEL", "rich").strip().lower()
                            if os.environ.get("SKYKV_FEEDBACK_LEVEL", "").strip().lower()
                            in ("minimal", "rich") else "rich"),
            parallel_eval=os.environ.get("SKYKV_PARALLEL_EVAL", "0") == "1",
            num_workers=int(os.environ.get("SKYKV_NUM_WORKERS", "0") or "0"),
            no_planner=os.environ.get("SKYKV_NO_PLANNER", "0") == "1",
            no_leaderboard=os.environ.get("SKYKV_NO_LEADERBOARD", "0") == "1",
            audit_checks_dir=os.environ.get("SKYKV_AUDIT_CHECKS_DIR", "").strip(),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Cgroup v2 memory enforcement
# ═══════════════════════════════════════════════════════════════════════════════

class CgroupManager:
    def __init__(self):
        self.name = f"skykv_bench_{os.getpid()}"
        self.path = f"/sys/fs/cgroup/{self.name}"
        self.active = False

    def set_limit(self, budget_gb: int) -> None:
        if budget_gb == 0:
            if os.path.isdir(self.path):
                self._sudo_write(f"{self.path}/memory.max", "max")
            self.active = False
            return
        overhead_gb = 12
        limit_bytes = (budget_gb + overhead_gb) * 1024 * 1024 * 1024
        try:
            subprocess.run(["sudo", "mkdir", "-p", self.path], check=True,
                           capture_output=True)
            self._sudo_write(f"{os.path.dirname(self.path)}/cgroup.subtree_control",
                             "+memory")
            self._sudo_write(f"{self.path}/memory.max", str(limit_bytes))
            self.active = True
        except (subprocess.CalledProcessError, OSError):
            self.active = False
            print(f"[WARN] cgroup setup failed for budget={budget_gb}GB "
                  "-- benchmark will run WITHOUT memory limit.", file=sys.stderr)

    def exec(self, cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        if self.active:
            self._sudo_write(f"{self.path}/cgroup.procs", str(os.getpid()))
        try:
            return subprocess.run(cmd, **kwargs)
        finally:
            if self.active:
                parent = os.path.dirname(self.path)
                try:
                    self._sudo_write(f"{parent}/cgroup.procs", str(os.getpid()))
                except (subprocess.CalledProcessError, OSError) as e:
                    print(f"[WARN] Failed to migrate orchestrator out of "
                          f"sub-cgroup: {e}", file=sys.stderr)

    def cleanup(self) -> None:
        subprocess.run(["sudo", "rmdir", self.path],
                       capture_output=True, check=False)

    @staticmethod
    def _sudo_write(path: str, value: str) -> None:
        subprocess.run(["sudo", "tee", path], input=value.encode(),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=True)




# ═══════════════════════════════════════════════════════════════════════════════
# Workspace setup
# ═══════════════════════════════════════════════════════════════════════════════

def strip_comments(obj):
    """Recursively remove _comment keys from a JSON-like object."""
    if isinstance(obj, dict):
        return {k: strip_comments(v) for k, v in obj.items() if k != "_comment"}
    if isinstance(obj, list):
        return [strip_comments(x) for x in obj]
    return obj


class WorkspaceManager:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.ws = Path(cfg.workspace)

    def setup(self) -> None:
        (self.ws / "interface" / "generated").mkdir(parents=True, exist_ok=True)
        (self.ws / "interface" / "build").mkdir(parents=True, exist_ok=True)
        self._copy_cards()
        self._link_docs()
        self._copy_harness()
        self._patch_harness()
        self._patch_cards()
        self._write_instructions()
        print(f"[setup] Workspace ready: {self.ws}")

    def _copy_cards(self) -> None:
        cards_dst = self.ws / "cards"
        cards_dst.mkdir(exist_ok=True)
        cards_src = Path(self.cfg.project_dir) / "setups" / self.cfg.card_set / "cards"
        for f in cards_src.glob("*.json"):
            data = json.loads(f.read_text())
            (cards_dst / f.name).write_text(json.dumps(strip_comments(data), indent=2))

    def _link_docs(self) -> None:
        docs_src = Path(self.cfg.project_dir).parent / "docs"
        if not docs_src.is_dir():
            return
        docs_dst = self.ws / "docs"
        docs_dst.mkdir(exist_ok=True)
        for name in ["faster-sigmod.txt", "faster-sigmod.pdf"]:
            src = docs_src / name
            dst = docs_dst / name
            if src.is_file() and not dst.exists():
                dst.symlink_to(src)

    def _copy_harness(self) -> None:
        iface_src = Path(self.cfg.project_dir) / "interface"
        eval_src = Path(self.cfg.project_dir) / "setups" / self.cfg.card_set / "evaluators"
        iface_dst = self.ws / "interface"
        # Shared harness files from interface/
        for name in ["kvstore_interface.h", "benchmark_harness.cc", "CMakeLists.txt"]:
            shutil.copy2(iface_src / name, iface_dst / name)
        # Setup-specific evaluator (consistency harness)
        if (eval_src / "consistency_harness.cc").is_file():
            shutil.copy2(eval_src / "consistency_harness.cc", iface_dst / "consistency_harness.cc")

    def _patch_harness(self) -> None:
        cfg = self.cfg
        if cfg.value_size != 8:
            for name in ["benchmark_harness.cc", "consistency_harness.cc"]:
                p = self.ws / "interface" / name
                text = p.read_text()
                text = text.replace("kValueSize  = 8;",
                                    f"kValueSize  = {cfg.value_size};")
                p.write_text(text)
        if cfg.mode == "ltm":
            for name in ["benchmark_harness.cc", "consistency_harness.cc"]:
                p = self.ws / "interface" / name
                text = p.read_text()
                text = text.replace("kHashTableSize = 1ULL << 27;",
                                    "kHashTableSize = 1ULL << 25;")
                p.write_text(text)
            # NOTE: previously bumped kConsistencyKeys=50M for LTM here so the
            # 5GB load would force disk spill, exercising the recovery-under-
            # spill path. Disabled because it added ~10min/iter to correctness
            # which is unacceptable for the evolution loop. The benchmark phase
            # at 25GB still exercises spill; correctness only validates the
            # algorithm. Re-enable for paper-quality final correctness eval.
            #
            # p = self.ws / "interface" / "consistency_harness.cc"
            # text = p.read_text().replace(
            #     "kConsistencyKeys = 1'000'000ULL;",
            #     "kConsistencyKeys = 50'000'000ULL;",
            # )
            # p.write_text(text)

        # Fast-bench mode: reduce benchmark dataset size so load phase +
        # destructor cleanup fit in ~2 min instead of ~10+ min. Relative
        # Mops/s comparison across iterations is preserved; absolute numbers
        # are smaller. Intended for the evolution loop; final paper-quality
        # numbers should always use the full dataset.
        #
        # Implementation: set env vars at bench-invocation time (see the
        # `env = {...}` block near the run command). The harness caps its
        # runtime-derived kInitCount / kTxnCount against these envs — this
        # replaces an older text-replace strategy that no longer applies
        # now that the harness auto-detects trace sizes from file stat.

    def _patch_cards(self) -> None:
        cfg = self.cfg
        cards = self.ws / "cards"

        # Patch trace card: value_size, workload, distribution, file paths,
        # and key counts derived from the actual trace file sizes. (Historic
        # 250M/1B defaults in the JSON are correct for zipf/uniform/scan/
        # belady/stride, but wrong for real traces like metakv (17-82M keys)
        # or twitter<N> where counts vary per cluster. Deriving from file
        # size matches what the harness does at startup — so the agent sees
        # the same numbers the harness actually uses.)
        tc = cards / "trace_card.json"
        data = json.loads(tc.read_text())
        data["dataset"]["value_size"] = cfg.value_size
        data["dataset"]["load_file"] = cfg.load_file
        data["dataset"]["run_file"] = cfg.run_file
        try:
            data["dataset"]["load_keys"] = os.path.getsize(cfg.load_file) // 8
            data["dataset"]["run_keys"] = os.path.getsize(cfg.run_file) // 8
        except OSError as e:
            print(f"[WARN] could not stat trace files for card update: {e}",
                  file=sys.stderr)
        data["workload"] = cfg.setup_name
        data["distribution"] = cfg.dist_desc
        tc.write_text(json.dumps(data, indent=2))

        # Patch memory budget into system_card (machine config, not trace).
        # Uses primary_scenario_gb (list of benchmark targets) instead of a
        # single budget_bytes — the agent should optimize FOR these budgets
        # but derive all sizing from the runtime mem_budget_bytes parameter.
        if cfg.mode == "ltm":
            p = cards / "system_card.json"
            data = json.loads(p.read_text())
            if "memory_constraint" in data:
                data["memory_constraint"]["primary_scenario_gb"] = cfg.mem_budget_gb
            p.write_text(json.dumps(data, indent=2))

    def _write_instructions(self) -> None:
        cfg = self.cfg
        threads_desc = ", ".join(f"t={t}" for t in cfg.threads)

        text = textwrap.dedent(f"""\
        # KV Store Generation Task

        You are implementing a high-performance concurrent key-value store from scratch.
        Your goal is to maximize throughput on the **{cfg.setup_name}** workload ({cfg.setup_desc})
        with **{cfg.dist_desc}** key distribution and **{cfg.value_size}-byte values** while satisfying
        the monotonicity property for crash recovery.

        ## Specification

        Read these files:
        - `cards/system_card.json` -- API, consistency guarantees, optimization objective
        - `cards/trace_card.json` -- Dataset, workload, and distribution parameters
        - `interface/kvstore_interface.h` -- The `IKVStore` abstract class you must implement

        Design freely -- any architecture is valid. If you need more context, research relevant
        papers, blog posts, or reference implementations via web search.

        ## Rules
        **Do NOT modify any file outside `interface/generated/`.**

        ## What to Implement
        Write `interface/generated/kvstore_impl.cc`:
        1. `#include "../kvstore_interface.h"` at the top
        2. Class inheriting `IKVStore`, implementing all pure virtual methods
        3. Factory: `IKVStore* create_kvstore() {{ return new YourClass(); }}`
        4. C++17 + `<pthread.h>` only -- no external libraries
        5. `Read`, `Upsert`, `RMW` must be thread-safe; concurrent `RMW` on the same key must be atomic (no lost updates)
        6. Must retain all loaded keys
        7. **Override `InitExtended()`** -- `storage_path` is ALWAYS provided. Your store must support `Checkpoint()` and crash recovery at all times. You cannot disable persistence to gain throughput.
        """)

        if cfg.mode == "ltm":
            text += "8. Must stay within the memory budget (`mem_budget_bytes`) -- pure in-memory will fail\n"
        text += (
            "9. **All sizing must be runtime-parameterized.** Hash table buckets, cache entries, "
            "buffer pool pages, arena sizes -- ALL must be computed from `mem_budget_bytes` inside "
            "`InitExtended()`, not hardcoded as compile-time constants. Your code must work "
            "correctly at ANY `mem_budget_bytes` value.\n"
            "10. **Values are variable-sized per key.** Store and return exactly `value.size` bytes "
            "-- never truncate or assume a fixed size. The workload's typical value size is an "
            "optimization hint for layout, not a hard constraint. For fixed-slot pre-allocation "
            "use `GenValue::kMaxSize` (4096); for better efficiency use a variable-length arena.\n"
        )

        text += textwrap.dedent(f"""\

        ## Goal

        Maximize throughput (Mops/s) at thread counts: {threads_desc}.
        The orchestrator benchmarks your implementation after each iteration and reports results.

        ## C++ Reminders
        - `std::mutex` is NOT copyable/movable -- use `new[]` or `unique_ptr<T[]>`
        - `std::atomic<T>` requires trivially copyable T
        """)

        (self.ws / cfg.instr_file).write_text(text)




# ═══════════════════════════════════════════════════════════════════════════════
# Agent runner
#
# Session model -- both backends maintain persistent sessions:
#   Claude: --session-id <UUID> on first call, --resume <UUID> on subsequent
#           calls. UUID is deterministic per (experiment, agent role) so the
#           generator, critic, and auditor each have distinct addressable
#           sessions — avoiding `--continue`'s "most-recent in cwd" footgun
#           when multiple Claude agents share a workspace.
#   Codex:  `codex exec --json` captures thread_id on first iteration,
#           then `codex exec resume <thread_id>` on subsequent iterations.
#           Stress tested to 1.13M tokens with full memory retention.
#
#   IMPORTANT: do NOT set model_context_window or model_auto_compact_token_limit
#   in ~/.codex/config.toml. With explicit custom values, auto-compaction breaks
#   (openai/codex#16033). With defaults, sessions run indefinitely.
# ═══════════════════════════════════════════════════════════════════════════════

class AgentRunner:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.iter_count = 0
        self.session_name = f"exp-{os.path.basename(cfg.experiments_dir)}"
        self.codex_thread_id: str | None = None
        # Claude session UUID — use --session-id/--resume so generator,
        # critic, and auditor have distinct, addressable sessions even
        # when running from the same workspace cwd. (Plain --continue
        # would resume the most-recent session in the cwd, which can
        # collide with the critic's session that was created last.)
        from critique import _deterministic_session_id
        self.claude_session_id: str = _deterministic_session_id(
            cfg.experiments_dir, agent="generator"
        )

    def run(self, prompt: str, log_file: str, max_turns: int | None = None) -> int:
        self.iter_count += 1
        self._max_turns_override = max_turns
        if self.cfg.backend == "claude":
            return self._run_claude(prompt, log_file)
        return self._run_codex(prompt, log_file)

    def _run_claude(self, prompt: str, log_file: str) -> int:
        # First call: --session-id creates the session with our UUID.
        # Subsequent calls: --resume <UUID> continues that exact session.
        session_flag = (
            ["--session-id", self.claude_session_id] if self.iter_count == 1
            else ["--resume", self.claude_session_id]
        )
        turns = self._max_turns_override or self.cfg.max_turns
        cmd = ["claude", "-p", prompt, *session_flag,
               "--max-turns", str(turns),
               "--dangerously-skip-permissions",
               "--model", self.cfg.model, "--verbose"]
        with open(log_file, "w") as lf:
            r = subprocess.run(cmd, cwd=self.cfg.workspace,
                               stdout=lf, stderr=subprocess.STDOUT, check=False)
        return r.returncode

    def _run_codex(self, prompt: str, log_file: str) -> int:
        # If --summary is on and this session's history is above the threshold,
        # one-shot the dying thread for a handoff memo, clear thread_id, and
        # prepend the memo to the next prompt so the fresh session continues
        # seamlessly. Prevents the unbounded resume-replay that drives 429s.
        if self.cfg.summary_enabled:
            from memory import maybe_rotate_codex
            memo = maybe_rotate_codex(
                self, "codex_thread_id", self.cfg.workspace,
                self.cfg.model, os.path.dirname(log_file), "generator",
            )
            if memo is not None:
                prompt = (
                    "## Carryover summary (from prior session)\n\n"
                    f"{memo}\n\n---\n\n{prompt}"
                )

        if self.codex_thread_id is not None:
            # Resume existing session
            cmd = ["codex", "exec",
                   "--model", self.cfg.model,
                   "--sandbox", "danger-full-access",
                   "-C", self.cfg.workspace,
                   "--skip-git-repo-check",
                   "--json",
                   "resume", self.codex_thread_id, prompt]
        else:
            # First iteration: start new session
            cmd = ["codex", "exec",
                   "--model", self.cfg.model,
                   "--sandbox", "danger-full-access",
                   "-C", self.cfg.workspace,
                   "--skip-git-repo-check",
                   "--json",
                   prompt]

        with open(log_file, "w") as lf:
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                    stdout=lf, stderr=subprocess.STDOUT)
            # Print progress while codex runs
            import time as _time
            last_size = 0
            tool_calls = 0
            while proc.poll() is None:
                _time.sleep(5)
                try:
                    cur_size = os.path.getsize(log_file)
                    if cur_size > last_size:
                        # Count JSON events as proxy for tool calls
                        with open(log_file, "r") as peek:
                            peek.seek(last_size)
                            chunk = peek.read()
                        new_items = chunk.count('"type":"item.completed"')
                        tool_calls += new_items
                        last_size = cur_size
                        if new_items > 0:
                            print(f"    ... codex: {tool_calls} steps completed", flush=True)
                except OSError:
                    pass
            r = subprocess.CompletedProcess(cmd, proc.returncode)

        # Capture thread_id from JSON output for session persistence
        if self.codex_thread_id is None:
            self._extract_thread_id(log_file)
            if self.codex_thread_id is None:
                raise RuntimeError(
                    f"Failed to extract codex thread_id from {log_file}. "
                    "Cannot resume sessions — aborting to avoid silent "
                    "fallback to fresh-session-per-iteration."
                )

        return r.returncode

    def _extract_thread_id(self, log_file: str) -> None:
        """Parse the thread_id from codex exec --json output."""
        try:
            with open(log_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        if event.get("type") == "thread.started" and "thread_id" in event:
                            self.codex_thread_id = event["thread_id"]
                            print(f"  Codex session: {self.codex_thread_id}")
                            return
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt builder
# ═══════════════════════════════════════════════════════════════════════════════

class PromptBuilder:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg

    def build_initial(self) -> str:
        cfg = self.cfg
        plan_block = self._whiteboard_block(1)
        design_block = self._design_handoff_block(1)
        return textwrap.dedent(f"""\
        Read {cfg.instr_file} for the task specification.

        You have {cfg.max_turns} tool calls for this iteration. Use them wisely.
        {plan_block}
        {design_block}

        ## Your Task
        Implement a high-performance concurrent KV store in interface/generated/kvstore_impl.cc.
        Optimize for {cfg.setup_name} ({cfg.setup_desc}).

        - Read the spec files (cards/*.json, interface/kvstore_interface.h)
        - If you need background, research via web search -- any approach is valid
        - Implement, build, and verify correctness

        {self._specialization_block()}

        {self._correctness_instructions()}

        IMPORTANT: Correctness is a GATE. Both crash recovery tests MUST pass.""")

    def build_subsequent(self, iteration: int, prev_status: str,
                         prev_iter_dir: str, impl_lines: int,
                         best_iter: int | None = None,
                         best_mops: float = 0.0) -> str:
        cfg = self.cfg
        parts = [
            f"Read {cfg.instr_file} for the task specification.",
            f"\nYou have {cfg.max_turns} tool calls for this iteration. Use them wisely.",
            f"\n## Iteration {iteration} -- Improve Your Implementation",
            f"\nYour previous implementation ({impl_lines} lines) "
            "is at interface/generated/kvstore_impl.cc.",
        ]

        # Best-so-far reference: without this, the agent only sees iter N-1's
        # (possibly regressed) code and can silently drift away from its best
        # known architecture. The mirrored reference file lets it compare —
        # and, when it's regressed, recover — without a hard revert.
        best_block = self._best_reference_block(best_iter, best_mops)
        if best_block:
            parts.append(best_block)

        # Status-specific feedback
        if prev_status == "BUILD_FAILED":
            parts.append(self._build_failed_feedback(prev_iter_dir))
        elif prev_status == "CORRECTNESS_FAILED":
            parts.append(self._correctness_failed_feedback(prev_iter_dir))
        elif prev_status == "AGENT_FAILED":
            parts.append(self._agent_failed_feedback(prev_iter_dir))
        elif prev_status == "AUDIT_FAILED":
            parts.append(self._audit_failed_feedback(prev_iter_dir))
        elif prev_status == "CANARY_BUDGET_FAILED":
            parts.append(self._canary_budget_failed_feedback(prev_iter_dir))
        else:
            parts.append(self._benchmark_feedback(prev_iter_dir))

        # Stateful reviewer feedback (only when critique is enabled and produced text)
        if cfg.critique_enabled:
            critique_block = self._prev_critique_block(prev_iter_dir)
            if critique_block:
                parts.append(critique_block)

        # Whiteboard context from planning phase
        plan_block = self._whiteboard_block(iteration)
        if plan_block:
            parts.append(plan_block)

        # Design handoff: the critic's (or Phase 0's) spec for this iter.
        # When present, this is the authoritative spec — implement it.
        design_block = self._design_handoff_block(iteration)
        if design_block:
            parts.append(design_block)

        parts.append(self._specialization_block())

        # Seed-specific exploration guidance. Only active when a seed was used
        # (iter 1 was the seed impl itself, measured as the per-machine
        # baseline). Keeps non-seeded runs' prompts unchanged.
        seed_block = self._seed_exploration_block()
        if seed_block:
            parts.append(seed_block)

        parts.append(f"\n{self._correctness_instructions()}")
        parts.append("\nIMPORTANT: Correctness is a GATE. Both crash recovery tests MUST pass.")
        return "\n".join(parts)

    def _seed_exploration_block(self) -> str:
        """Seeded-run guidance: nudge toward architectural changes, not tweaks.

        Only rendered when cfg.seed_name is set. The seed's iter-1 measurement
        is the per-machine reference; we ask the agent to target ~1.25-1.5×
        that. Lists concrete directions (caching policy, memory management,
        concurrency, batching, data structure, layout) without prescribing any
        single design, and flags single-digit-percent micro-optimizations as
        ineffective by themselves.
        """
        cfg = self.cfg
        if not getattr(cfg, "seed_name", ""):
            return ""
        bn = cfg.baseline_name
        return (
            f"\n## Seeded run — iterating on top of {bn}\n\n"
            f"Iter-1 measured the {bn} adapter as your per-machine "
            f"baseline. Aim to beat it by ~1.25-1.5× on this workload "
            f"signature. Gains of that size on KV stores typically come "
            f"from architectural choices (caching policy, memory management, "
            f"concurrency topology, batching / pipelining, data structure, "
            f"layout) rather than small tweaks.\n\n"
            f"Useful to understand for this workload:\n"
            f"  - working-set vs memory budget ratio\n"
            f"  - which op path dominates the mix "
            f"(read / upsert / RMW; hot vs cold)\n"
            f"  - how {bn}'s design (HybridLog, index, thread model) "
            f"fits or doesn't fit this signature\n\n"
            f"You can modify anything under `generated/` (including "
            f"vendored `faster_src/`), add new files, or rewrite "
            f"`kvstore_impl.cc` from scratch. The only hard constraint is "
            f"correct IKVStore + passing all 6 consistency tests.\n"
        )

    def _best_reference_block(self, best_iter: int | None,
                              best_mops: float) -> str:
        """Point the generator at its own best-known impl.

        Silent no-op before any COMPLETED iter (best_iter is None / best_mops
        is 0.0). Otherwise loads the BEST_REFERENCE section from
        agent-pipeline/prompts/generator_prompt.txt and formats in the iter + Mops/s.
        """
        # --no-leaderboard bypass: do not inject any best-impl reference into
        # the generator prompt (off-the-shelf baseline has no leaderboard).
        if self.cfg.no_leaderboard:
            return ""
        if best_iter is None or best_mops <= 0:
            return ""
        section = _load_generator_prompt_sections()["BEST_REFERENCE"]
        return "\n" + section.format(
            best_iter=best_iter,
            best_mops=best_mops,
        )

    def _specialization_block(self) -> str:
        """Load the SPECIALIZATION section from agent-pipeline/prompts/generator_prompt.txt
        and inject this run's requirement + workload signature.

        Keeping the prompt text in a .txt file (same convention as
        prompts/critique_prompt.txt and prompts/critique_audit_prompt.txt) means prompt
        tweaks don't require Python edits and the diff history separates
        content changes from code changes.
        """
        cfg = self.cfg
        budget_str = (
            "unlimited" if cfg.mem_budget_gb == [0]
            else " / ".join(f"{b} GB" for b in cfg.mem_budget_gb)
        )
        threads_str = ", ".join(str(t) for t in cfg.threads)
        # Rough dataset footprint: 250M keys × value_size bytes.
        dataset_gb = round(250 * cfg.value_size / 1024, 1)
        section = _load_generator_prompt_sections()["SPECIALIZATION"]
        return "\n" + section.format(
            baseline_name=cfg.baseline_name,
            dist_desc=cfg.dist_desc,
            setup_desc=cfg.setup_desc,
            value_size=cfg.value_size,
            budget_str=budget_str,
            dataset_gb=dataset_gb,
            threads_str=threads_str,
        )

    def _correctness_instructions(self) -> str:
        cfg = self.cfg
        return textwrap.dedent(f"""\
        ## Build & Correctness
        cd interface/build
        cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc) kvstore_bench consistency_test
        # All 6 tests must pass (the orchestrator runs them in parallel):
        #   1=RMW crash recovery (fuzzy ckpt)  2=RMW crash (no ckpt)
        #   3=Upsert crash recovery (fuzzy)    4=Upsert crash (no ckpt)
        #   5=Read correctness (no torn values) 6=Read freshness (cross-thread visibility)
        numactl --cpunodebind=0 --membind=0 ./build/consistency_test 1 8 {cfg.load_file} {cfg.correctness_mem} {cfg.correctness_storage}
        numactl --cpunodebind=0 --membind=0 ./build/consistency_test 2 8 {cfg.load_file} {cfg.correctness_mem} {cfg.correctness_storage}

        Fix any failures. Do NOT run benchmarks.""")

    def _build_failed_feedback(self, prev_iter_dir: str) -> str:
        log_path = os.path.join(prev_iter_dir, "build.log")
        if not os.path.isfile(log_path):
            return "\n## Previous iteration: BUILD FAILED\n(no build log)"
        lines = open(log_path).readlines()
        # Find context around first error: line — that's what the agent needs
        error_snippet = ""
        for idx, line in enumerate(lines):
            if "error:" in line.lower():
                start = max(0, idx - 2)
                end = min(len(lines), idx + 10)
                error_snippet = "".join(lines[start:end])
                break
        # Fall back to tail if no error: line found
        if not error_snippet:
            error_snippet = "".join(lines[-30:])
        return f"\n## Previous iteration: BUILD FAILED\n```\n{error_snippet}\n```"

    def _correctness_failed_feedback(self, prev_iter_dir: str) -> str:
        parts = ["\n## Previous iteration: CORRECTNESS FAILED"]
        for tid in range(1, 7):
            tf = os.path.join(prev_iter_dir, f"consistency_test{tid}.txt")
            if os.path.isfile(tf):
                text = open(tf).read()
                if "PASSED" in text and "FAILED" not in text:
                    parts.append(f"Test {tid}: PASSED")
                else:
                    tail = "\n".join(text.splitlines()[-20:])
                    parts.append(f"Test {tid}: FAILED\n```\n{tail}\n```")
        return "\n".join(parts)

    def _agent_failed_feedback(self, prev_iter_dir: str) -> str:
        log_path = os.path.join(prev_iter_dir, "agent_log.txt")
        extra = ""
        if os.path.isfile(log_path):
            text = open(log_path).read()
            if re.search(r"Reconnecting|500 Internal Server|401 Unauthorized", text):
                extra = " (API connectivity issue)"
            elif "AGENT_EXIT_CODE" in text:
                m = re.search(r"AGENT_EXIT_CODE=(\d+)", text)
                extra = f" (agent crashed: exit {m.group(1)})" if m else ""
        return f"\n## Previous iteration: no implementation was produced{extra}. Start fresh."

    def _audit_failed_feedback(self, prev_iter_dir: str) -> str:
        """Format audit-gate failure as feedback for the generator.

        Reads the per-iteration audit_results.txt for compact gate output,
        plus surfaces the test names + AUDIT/HACK fields. The benchmark was
        skipped, so we deliberately do NOT call _benchmark_feedback here.
        """
        parts = ["\n## Previous iteration: AUDIT GATE FAILED",
                 "(benchmark was skipped — fix the audit failures below)"]
        results_path = os.path.join(prev_iter_dir, "audit_results.txt")
        if os.path.isfile(results_path):
            text = open(results_path).read()
            # Trim deprecated/passing tests from the verbose results, keep failures
            parts.append("```")
            parts.append(text[-3000:] if len(text) > 3000 else text)
            parts.append("```")
        else:
            parts.append("(audit_results.txt missing — pipeline state corrupt)")
        return "\n".join(parts)

    def _canary_budget_failed_feedback(self, prev_iter_dir: str) -> str:
        """Format canary budget failure as feedback for the generator."""
        canary_path = os.path.join(prev_iter_dir, "canary_budget_test.txt")
        parts = [
            "\n## Previous iteration: CANARY BUDGET TEST FAILED",
            "",
            "Your implementation was tested at a memory budget DIFFERENT from the",
            "primary benchmark budgets, and it crashed or failed. This means your",
            "code has **hardcoded data-structure sizes** instead of computing them",
            "from `mem_budget_bytes` at runtime.",
            "",
            "**You MUST fix this.** All sizing -- hash table buckets, cache entries,",
            "buffer pool pages, arena sizes -- must be derived from `mem_budget_bytes`",
            "inside `InitExtended()`. Do not use compile-time constants like:",
            "  `constexpr size_t kCacheEntries = 8192 * 256;`",
            "Instead compute from budget:",
            "  `size_t cache_entries = (mem_budget_bytes * 6 / 10) / sizeof(CacheEntry);`",
            "",
            "The benchmark was skipped. Fix the parameterization and rebuild.",
        ]
        if os.path.isfile(canary_path):
            with open(canary_path) as f:
                text = f.read()
            tail = "\n".join(text.splitlines()[-20:])
            parts.append(f"\nCanary test output:\n```\n{tail}\n```")
        return "\n".join(parts)

    def _prev_critique_block(self, prev_iter_dir: str) -> str:
        """Inject the previous iter's stateful-reviewer critique into this iter's prompt."""
        critique_path = os.path.join(prev_iter_dir, "critique.txt")
        if not os.path.isfile(critique_path):
            return ""
        try:
            with open(critique_path) as f:
                text = f.read().strip()
        except OSError:
            return ""
        if not text or text == "(no critique produced)":
            return ""
        return (
            "\n## Previous iteration review (from a stateful reviewer agent)\n\n"
            "The reviewer has read your code, bench results, and correctness output, and\n"
            "is tracking what you did across iterations. Read this and act on it:\n\n"
            f"{text}\n"
        )

    def build_planning(self) -> str:
        """Build the phase-0 planning prompt from planning_prompt.txt."""
        cfg = self.cfg
        budget_str = (
            "unlimited" if cfg.mem_budget_gb == [0]
            else " / ".join(f"{b} GB" for b in cfg.mem_budget_gb)
        )
        threads_str = ", ".join(str(t) for t in cfg.threads)
        dataset_gb = round(250 * cfg.value_size / 1024, 1)
        section = _load_planning_prompt_sections()["PLANNING"]
        body = section.format(
            baseline_name=cfg.baseline_name,
            dist_desc=cfg.dist_desc,
            setup_desc=cfg.setup_desc,
            value_size=cfg.value_size,
            budget_str=budget_str,
            dataset_gb=dataset_gb,
            threads_str=threads_str,
            iterations=cfg.iterations,
            max_turns=cfg.max_turns,
        )
        return f"Read {cfg.instr_file} for the task specification.\n\n{body}"

    def _whiteboard_block(self, iteration: int) -> str:
        """Inject whiteboard context if planning/whiteboard.md exists.

        Read-only pointer from the generator's perspective. The `## Learnings`
        section is maintained by the critic (default --critique review/full
        mode) or by the generator's own Stage A design call (legacy --critique
        off mode, via design_prompt.txt::DESIGN_SUBSEQUENT). Either way, the
        instruction to *update* Learnings lives in the writer's prompt, not
        here.
        """
        # --no-planner bypass: do not inject whiteboard reference.
        if self.cfg.no_planner:
            return ""
        wb_path = os.path.join(self.cfg.workspace, "planning", "whiteboard.md")
        if not os.path.isfile(wb_path):
            return ""
        total = self.cfg.iterations
        return (
            f"\nRead your whiteboard at `planning/whiteboard.md` for the overall "
            f"exploration plan and accumulated Learnings. You are on iteration "
            f"{iteration} of {total}."
        )

    def _design_handoff_block(self, iteration: int) -> str:
        """Inline the current planning/design.md as a handoff spec.

        Primary input to the generator in the default critic-as-designer flow.
        Critic writes design.md at the end of each iteration (Phase 0 seeds it
        for iter 1). The generator reads this block and implements the spec —
        no redesign in the generator's turn.

        Returns empty string when design.md is missing, which should only
        happen transiently (iter 1 before Phase 0 has run, or a critic
        failure between iters). The caller's prompt still functions — the
        agent falls back to its feedback + whiteboard for direction.
        """
        design_path = os.path.join(self.cfg.workspace, "planning", "design.md")
        if not os.path.isfile(design_path):
            if iteration >= 2:
                print(f"  [WARN] planning/design.md missing at iter {iteration} — "
                      "critic (or Stage A) did not write a handoff spec. "
                      "Generator proceeding without design.md inlined.",
                      file=sys.stderr)
            return ""
        try:
            with open(design_path) as f:
                design_text = f.read().strip()
        except OSError:
            return ""
        if not design_text:
            print(f"  [WARN] planning/design.md is empty at iter {iteration} — "
                  "previous writer (critic or Stage A) produced an empty file.",
                  file=sys.stderr)
            return ""
        return (
            f"\n## Design spec for iter {iteration} — implement this\n\n"
            f"The spec below was produced by the critic at the end of iter "
            f"{iteration - 1} (or Phase 0 for iter 1) based on the latest bench "
            f"results and whiteboard learnings. Implement it. Do not redesign "
            f"in this turn — if you disagree with a choice, note it in your "
            f"impl comments and the next critique can revise.\n\n"
            f"```md\n{design_text}\n```\n"
        )

    def build_design(self, iteration: int, prev_status: str,
                     prev_iter_dir: str, impl_lines: int,
                     best_iter: int | None = None,
                     best_mops: float = 0.0) -> str:
        """Build the Stage A (design) prompt from design_prompt.txt.

        Only called by the legacy --critique off flow (Alex's three-stage
        fallback) at iteration >= 2. At iter 1 Phase 0 has already seeded
        planning/design.md, so no Stage A design call is needed.

        DESIGN_INITIAL was deleted; iter-1 bootstrap is handled in
        planning_prompt.txt under "Two artifacts to produce".
        """
        cfg = self.cfg
        section = _load_design_prompt_sections()["DESIGN_SUBSEQUENT"]

        # Build status feedback (same logic as build_subsequent). The
        # CANARY_BUDGET_FAILED branch mirrors the one in build_subsequent so
        # --critique off Stage A also sees the parameterization diagnostic
        # instead of a bare benchmark-results header on an empty bench_output.
        if prev_status == "BUILD_FAILED":
            status_feedback = self._build_failed_feedback(prev_iter_dir)
        elif prev_status == "CORRECTNESS_FAILED":
            status_feedback = self._correctness_failed_feedback(prev_iter_dir)
        elif prev_status == "AGENT_FAILED":
            status_feedback = self._agent_failed_feedback(prev_iter_dir)
        elif prev_status == "AUDIT_FAILED":
            status_feedback = self._audit_failed_feedback(prev_iter_dir)
        elif prev_status == "CANARY_BUDGET_FAILED":
            status_feedback = self._canary_budget_failed_feedback(prev_iter_dir)
        else:
            status_feedback = self._benchmark_feedback(prev_iter_dir)

        best_block = self._best_reference_block(best_iter, best_mops)
        critique_block = ""
        if cfg.critique_enabled:
            critique_block = self._prev_critique_block(prev_iter_dir)

        body = section.format(
            iteration=iteration,
            total_iters=cfg.iterations,
            baseline_name=cfg.baseline_name,
            status_feedback=status_feedback,
            best_block=best_block,
            critique_block=critique_block,
        )
        plan_block = self._whiteboard_block(iteration)
        return f"Read {cfg.instr_file} for the task specification.\n{plan_block}\n\n{body}"

    def build_implement(self, iteration: int) -> str:
        """Build the Stage B (implement) prompt from design_prompt.txt."""
        cfg = self.cfg
        section = _load_design_prompt_sections()["IMPLEMENT"]
        parts = [section]
        parts.append(self._specialization_block())
        parts.append(f"\n{self._correctness_instructions()}")
        parts.append("\nIMPORTANT: Correctness is a GATE. Both crash recovery tests MUST pass.")
        return "\n".join(parts)

    def _benchmark_feedback(self, prev_iter_dir: str) -> str:
        cfg = self.cfg
        bench_dir = os.path.join(prev_iter_dir, "bench_output")
        # Import format_feedback directly (same directory)
        sys.path.insert(0, os.path.join(cfg.project_dir, "agent-pipeline"))
        from format_feedback import load_baselines, format_feedback

        # Baseline comparison is OFF by default — the agent should optimize
        # purely from its own trajectory, not anchor to a specific number.
        # Enable via SKYKV_SHOW_BASELINE=1 or --show-baseline flag.
        baselines = {}
        if getattr(cfg, "show_baseline", False):
            baseline_csv = os.path.join(cfg.project_dir, "setups", cfg.card_set,
                                        "baselines", "adapter_baseline_results.csv")
            col_map = {"50_50": 1, "0_100": 3}
            col = col_map.get(cfg.setup_key)
            if col is not None:
                baselines = load_baselines(baseline_csv, col)

            # Seed upgrade: when show_baseline is ON and a seed IS the
            # reference impl, the seed's iter-1 measurement IS "baseline on
            # THIS machine right now" — more honest than a portable CSV.
            if cfg.seed_name:
                iter1_dir = os.path.join(cfg.experiments_dir, "iter_001")
                iter1_bench = os.path.join(iter1_dir, "bench_output")
                try:
                    for fn in os.listdir(iter1_bench):
                        text = open(os.path.join(iter1_bench, fn)).read()
                        m = re.search(r'Total:\s*([\d.]+)\s*Mops/s', text)
                        mem = re.search(r'_mem(\d+)g', fn)
                        if m and mem:
                            iter1_mops = float(m.group(1))
                            gb = mem.group(1)
                            prev = float(baselines.get(gb, 0.0))
                            if iter1_mops > prev:
                                baselines[gb] = iter1_mops
                except (OSError, FileNotFoundError):
                    pass

        try:
            return "\n" + format_feedback(
                bench_dir, cfg.wl_id, cfg.setup_name,
                [str(t) for t in cfg.threads],
                [str(g) for g in cfg.mem_budget_gb],
                baselines, cfg.baseline_name,
                level=cfg.feedback_level,
            )
        except Exception as e:
            return f"\n## Benchmark Results ({cfg.setup_name})\n  (feedback formatting failed: {e})"


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline steps: build, correctness, benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def numa_prefix(threads: int) -> list[str]:
    if threads > 32:
        return []
    return ["numactl", "--cpunodebind=0", "--membind=0"]


class PipelineSteps:
    def __init__(self, cfg: RunConfig, cgroup: CgroupManager):
        self.cfg = cfg
        self.cgroup = cgroup
        self.build_dir = os.path.join(cfg.workspace, "interface", "build")
        self._cmake_done = False
        self.pool = None  # WorkerPool, set by EvolutionRunner if --parallel-eval

    def build(self, iter_dir: str) -> bool:
        log_path = os.path.join(iter_dir, "build.log")
        with open(log_path, "w") as lf:
            if not self._cmake_done:
                r1 = subprocess.run(
                    ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"],
                    cwd=self.build_dir, stdout=lf, stderr=subprocess.STDOUT,
                    check=False,
                )
                if r1.returncode != 0:
                    print(f"  BUILD FAILED (cmake exit {r1.returncode})")
                    return False
                self._cmake_done = True
            r2 = subprocess.run(
                ["make", f"-j{os.cpu_count()}", "kvstore_bench", "consistency_test"],
                cwd=self.build_dir, stdout=lf, stderr=subprocess.STDOUT, check=False,
            )
            if r2.returncode != 0:
                print(f"  BUILD FAILED (make exit {r2.returncode})")
                return False
        # Ensure binaries are executable (codex sandbox can strip permissions)
        for name in ["kvstore_bench", "consistency_test"]:
            p = os.path.join(self.build_dir, name)
            if os.path.isfile(p):
                os.chmod(p, 0o755)
        print("  Build OK")
        return True

    def run_correctness(self, iter_dir: str) -> bool:
        """Run the 6 consistency tests.

        LTM mode: serial — at 50M keys x 100B values, each test loads 5GB that
        exceeds the 4GB correctness budget and spills to disk. Six parallel
        processes would thrash the NVMe and also pin 48 threads to 8 physical
        cores (pin_thread maps t in 0..7 to the same set of CPUs).

        Inmem mode: parallel — data fits in RAM, overhead is CPU-bound and
        short, core contention is tolerable.
        """
        cfg = self.cfg
        test_ids = [1, 2, 3, 4, 5, 6]
        out_paths = {
            tid: os.path.join(iter_dir, f"consistency_test{tid}.txt")
            for tid in test_ids
        }
        returncodes: dict[int, int] = {}

        if self.pool and self.pool.ready:
            # ── Run correctness on workers ───────────────────────────
            self.pool.ship(self.build_dir)
            fast_exit = "KVSTORE_BENCH_FAST_EXIT=1 " if cfg.bench_fast_exit else ""
            jobs = []
            for tid in test_ids:
                remote_stor = f"/mnt/ssd/kvstore_consistency_t{tid}"
                cmd = (f"mkdir -p {remote_stor} && rm -rf {remote_stor}/* && "
                       f"touch /tmp/skykv_heartbeat && "
                       f"{fast_exit}"
                       f"timeout 300 numactl --cpunodebind=0 --membind=0 "
                       f"/tmp/skykv_eval/consistency_test {tid} 8 "
                       f"{cfg.load_file} {cfg.correctness_mem} {remote_stor}")
                jobs.append((cmd, out_paths[tid]))
            exit_codes = self.pool.fan_out(jobs)
            returncodes = dict(zip(test_ids, exit_codes))
        else:
            # ── Run correctness locally (parallel) ───────────────────
            storage_dirs = {}
            for tid in test_ids:
                d = cfg.correctness_storage + f"_t{tid}"
                os.makedirs(d, exist_ok=True)
                self._clean_dir(d)
                storage_dirs[tid] = d

            def cmd_for(tid: int) -> list[str]:
                return [*numa_prefix(8),
                        os.path.join(self.build_dir, "consistency_test"),
                        str(tid), "8", cfg.load_file,
                        str(cfg.correctness_mem), storage_dirs[tid]]

            # All 6 tests run in parallel. Each test uses kConsistencyKeys=1M
        # keys (not 250M), so even at 100B values the data per test is
        # ~100 MB — no NVMe thrashing risk from concurrent instances.
        # Timeout: 300s per test is plenty for both inmem and LTM.
        #
        # Fast-exit on test completion: PASS/FAIL is decided in the test body
        # (including mid-test crash+recover verify, which IS the persistence
        # test). The consistency_harness's post-test Checkpoint+delete is pure
        # teardown — for LTM stores with ~4GB HybridLog, the destructor flushes
        # to NVMe for minutes, and 6× concurrent teardown serializes on disk.
        # Skipping teardown is safe because orchestrator cleans storage_dirs
        # between iters.
        test_env = {**os.environ}
        if cfg.bench_fast_exit:
            test_env["KVSTORE_BENCH_FAST_EXIT"] = "1"
        procs: dict[int, subprocess.Popen] = {}
        open_files = []
        for tid in test_ids:
            f = open(out_paths[tid], "w")
            open_files.append(f)
            procs[tid] = subprocess.Popen(cmd_for(tid), stdout=f,
                                          stderr=subprocess.STDOUT,
                                          env=test_env)
        try:
            for tid in test_ids:
                p = procs[tid]
                try:
                    p.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
                returncodes[tid] = p.returncode
        finally:
            for f in open_files:
                    f.close()

        all_passed = True
        for tid in test_ids:
            text = open(out_paths[tid]).read() if os.path.isfile(out_paths[tid]) else ""
            rc = returncodes.get(tid, -1)
            if "PASSED" in text and "FAILED" not in text:
                print(f"  Test {tid}: PASSED")
            elif rc != 0:
                print(f"  Test {tid}: FAILED (exit code {rc})")
                all_passed = False
            else:
                print(f"  Test {tid}: FAILED")
                all_passed = False

        if not (self.pool and self.pool.ready):
            for d in storage_dirs.values():
                self._clean_dir(d)

        return all_passed

    def canary_budget(self) -> int:
        """Return a canary budget in GB for portability testing, or 0 to skip.

        Picks a budget different from all primary benchmarks to catch
        hardcoded sizing. Uses half the smallest primary budget (min 1 GB).
        Returns 0 if there are no constrained budgets (inmem mode).
        """
        cfg = self.cfg
        constrained = [b for b in cfg.mem_budget_gb if b > 0]
        if not constrained:
            return 0
        candidate = max(1, min(constrained) // 2)
        # Only useful if it's actually different from every primary budget
        if candidate in cfg.mem_budget_gb:
            candidate = max(1, candidate - 1)
        if candidate in cfg.mem_budget_gb:
            return 0
        return candidate

    def run_canary_budget_test(self, iter_dir: str) -> bool:
        """Quick portability canary: run one consistency test at a budget the
        agent didn't see, to catch hardcoded sizing constants.

        Runs test 5 (read correctness — fast, exercises Init+Load+Read)
        at self.canary_budget(). If the store crashes or fails, the agent
        hardcoded data-structure sizes for the primary budget.
        """
        cfg = self.cfg
        canary_gb = self.canary_budget()
        if canary_gb == 0:
            return True  # nothing to test (inmem mode or no distinct budget)

        canary_bytes = canary_gb * 1024 * 1024 * 1024
        out_path = os.path.join(iter_dir, "canary_budget_test.txt")
        storage_dir = cfg.correctness_storage + "_canary"
        os.makedirs(storage_dir, exist_ok=True)
        self._clean_dir(storage_dir)

        cmd = [*numa_prefix(8),
               os.path.join(self.build_dir, "consistency_test"),
               "5", "8", cfg.load_file,
               str(canary_bytes), storage_dir]

        test_env = {**os.environ}
        if cfg.bench_fast_exit:
            test_env["KVSTORE_BENCH_FAST_EXIT"] = "1"

        print(f"  Canary budget test ({canary_gb} GB): ", end="", flush=True)
        try:
            with open(out_path, "w") as f:
                r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                   env=test_env, timeout=300, check=False)
            text = Path(out_path).read_text()
            if r.returncode == 0 and "PASSED" in text and "FAILED" not in text:
                print("PASSED")
                return True
            else:
                print(f"FAILED (exit {r.returncode})")
                print(f"  *** Your code does not work at {canary_gb} GB. "
                      f"Data-structure sizes are likely hardcoded instead of "
                      f"derived from mem_budget_bytes. ***")
                return False
        except subprocess.TimeoutExpired:
            print("TIMED OUT")
            return False
        finally:
            self._clean_dir(storage_dir)

    def run_benchmark(self, iter_dir: str) -> tuple[bool, float]:
        cfg = self.cfg
        all_ok = True
        best_mops = 0.0

        # ── Pre-run benchmarks on workers if available ───────────────
        prefetched_exits: dict[str, int] = {}
        if self.pool and self.pool.ready:
            if self.pool.has_traces(cfg.load_file, cfg.run_file):
                jobs = []
                keys = []
                for mem_gb in cfg.mem_budget_gb:
                    mem_bytes = str(mem_gb * 1024 * 1024 * 1024) if mem_gb else "0"
                    for t in cfg.threads:
                        suffix = f"_mem{mem_gb}g" if mem_gb != 0 else ""
                        out_file = os.path.join(
                            iter_dir, "bench_output", f"wl{cfg.wl_id}_t{t}{suffix}.txt")
                        cmd = (f"KVSTORE_VALIDATE_LOAD_KEYS=1 "
                               f"KVSTORE_VALIDATE_LOAD_STRIDE=1 "
                               f"timeout 900 bash /tmp/skykv_eval/worker_bench.sh "
                               f"{cfg.wl_id} {t} {cfg.load_file} {cfg.run_file} "
                               f"{mem_bytes} /mnt/ssd/kvstore_eval_worker")
                        if cfg.bench_fast_exit:
                            cmd = "KVSTORE_BENCH_FAST_EXIT=1 " + cmd
                        jobs.append((cmd, out_file))
                        keys.append(f"{mem_gb}_{t}")
                n_workers = len(self.pool.workers)
                print(f"  [pool] Running {len(jobs)} benchmark(s) across "
                      f"{n_workers} worker(s)...")
                exit_codes = self.pool.fan_out(jobs)
                prefetched_exits = dict(zip(keys, exit_codes))
            else:
                print("  [pool] Trace files not available on workers — running locally")

        for mem_gb in cfg.mem_budget_gb:
            if not prefetched_exits:
                self.cgroup.set_limit(mem_gb)
            if mem_gb == 0:
                bench_extra = ["0", cfg.storage_dir]
                mem_label = ""
            else:
                bench_extra = [str(mem_gb * 1024 * 1024 * 1024), cfg.storage_dir]
                mem_label = f" mem={mem_gb}GB"

            for t in cfg.threads:
                print(f"  {cfg.setup_name} t={t}{mem_label}: ", end="", flush=True)

                suffix = f"_mem{mem_gb}g" if mem_gb != 0 else ""
                out_file = os.path.join(iter_dir, "bench_output",
                                        f"wl{cfg.wl_id}_t{t}{suffix}.txt")

                key = f"{mem_gb}_{t}"
                if key in prefetched_exits:
                    # Results already collected from worker pool
                    exit_code = prefetched_exits[key]
                    timed_out = (exit_code == 124)
                else:
                    # Run locally
                    self._clean_dir(cfg.storage_dir)
                    bench_cmd = [*numa_prefix(t),
                                 os.path.join(self.build_dir, "kvstore_bench"),
                                 str(cfg.wl_id), str(t), cfg.load_file, cfg.run_file,
                                 *bench_extra]
                    env = {**os.environ,
                           "KVSTORE_VALIDATE_LOAD_KEYS": "1",
                           "KVSTORE_VALIDATE_LOAD_STRIDE": "1"}
                    if cfg.bench_fast_exit:
                        env["KVSTORE_BENCH_FAST_EXIT"] = "1"

                    # Wrap with perf stat when feedback level is rich
                    perf_out = None
                    if cfg.feedback_level == "rich" and shutil.which("perf"):
                        perf_out = os.path.join(
                            iter_dir, "bench_output",
                            f"perf_wl{cfg.wl_id}_t{t}{suffix}.txt")
                        perf_events = ("context-switches,cpu-migrations,"
                                       "page-faults,major-faults")
                        # Try hardware counters too (gracefully ignored if unavailable)
                        perf_events += (",instructions,cycles,cache-misses,"
                                        "cache-references,LLC-load-misses,"
                                        "LLC-loads,branch-misses,branches")
                        cmd = ["perf", "stat", "-x", ";",
                               "-e", perf_events, "-o", perf_out,
                               "--"] + bench_cmd
                    else:
                        cmd = bench_cmd

                    timed_out = False
                    exit_code = 0
                    try:
                        with open(out_file, "w") as f:
                            # 900s: load (~70s) + stride=1 validation (~120-180s
                            # under disk-heavy distributions like belady) + 30s run
                            r = self.cgroup.exec(cmd, stdout=f, stderr=subprocess.STDOUT,
                                                 env=env, timeout=900, check=False)
                            exit_code = r.returncode
                    except subprocess.TimeoutExpired:
                        timed_out = True

                text = open(out_file).read() if os.path.isfile(out_file) else ""
                mops_match = re.search(r'([\d.]+)\s+Mops/s', text)
                mops_val = float(mops_match.group(1)) if mops_match else 0.0

                if "VALIDATION FAILED" in text:
                    wsz = re.search(r'wrong_size=(\d+)', text)
                    msk = re.search(r'missing=(\d+)', text)
                    wval = re.search(r'wrong_value=(\d+)', text)
                    detail = []
                    if wsz and int(wsz.group(1)) > 0:
                        detail.append(f"{wsz.group(1)} keys wrong size")
                    if wval and int(wval.group(1)) > 0:
                        detail.append(f"{wval.group(1)} keys wrong content")
                    if msk and int(msk.group(1)) > 0:
                        detail.append(f"{msk.group(1)} keys missing")
                    print(f"VALIDATION FAILED ({', '.join(detail) or 'unknown'})")
                    all_ok = False
                elif timed_out:
                    if mops_val > 0:
                        print(f"{mops_val:.2f} Mops/s (killed during cleanup after 300s)")
                        best_mops = max(best_mops, mops_val)
                    else:
                        print("TIMED OUT (benchmark exceeded 300s)")
                        with open(out_file, "a") as f:
                            f.write("BENCH_TIMED_OUT\n")
                        all_ok = False
                elif exit_code == 0 and mops_val > 0:
                    print(f"{mops_val:.2f} Mops/s")
                    best_mops = max(best_mops, mops_val)
                elif mops_val > 0:
                    print(f"{mops_val:.2f} Mops/s (crashed during cleanup, exit {exit_code})")
                    best_mops = max(best_mops, mops_val)
                elif re.search(r'Killed|SIGKILL|Cannot allocate|oom', text, re.I):
                    print(f"OOM KILLED (exceeded {mem_gb}GB memory budget)")
                    all_ok = False
                else:
                    # Diagnose killed-during-load and other non-obvious failures
                    # so the agent gets actionable feedback, not just "FAILED (exit -15)".
                    diagnosis = self._diagnose_bench_failure(
                        text, exit_code, timed_out, mem_gb, out_file)
                    print(diagnosis)
                    all_ok = False

        return all_ok, best_mops

    @staticmethod
    def _diagnose_bench_failure(text: str, exit_code: int,
                                timed_out: bool, mem_gb: int,
                                out_file: str) -> str:
        """Diagnose non-obvious benchmark failures by reporting the phase
        reached and raw facts. No prescriptions — just statistics."""
        has_load_done = "Load done in" in text
        has_run_start = "Running workload" in text
        is_signal_kill = exit_code in (-9, -15, 137, 143)

        # Check cgroup memory events
        oom_tag = ""
        try:
            cgroup_events = f"/sys/fs/cgroup/skykv_bench_{os.getpid()}/memory.events"
            if os.path.isfile(cgroup_events):
                with open(cgroup_events) as f:
                    events_text = f.read()
                if "oom_kill" in events_text:
                    oom_tag = ", cgroup_oom_kill=true"
        except OSError:
            pass

        if not has_load_done and is_signal_kill:
            progress = [l.strip() for l in text.splitlines()
                        if any(kw in l for kw in
                               ("Loading", "Load done", "Correctness preflight",
                                "Memory-constrained"))]
            last = progress[-1] if progress else "no progress"
            return (f"KILLED DURING LOAD PHASE "
                    f"(exit={exit_code}, budget={mem_gb}GB{oom_tag})\n"
                    f"    last_progress: {last}")

        if has_load_done and not has_run_start and is_signal_kill:
            return (f"KILLED DURING VALIDATION "
                    f"(exit={exit_code}, budget={mem_gb}GB{oom_tag})")

        if has_run_start and is_signal_kill:
            return (f"KILLED DURING RUN PHASE "
                    f"(exit={exit_code}, budget={mem_gb}GB{oom_tag})")

        tail_lines = text.strip().splitlines()[-5:]
        tail = "\n    ".join(tail_lines) if tail_lines else "(no output)"
        return f"FAILED (exit={exit_code})\n    last_output:\n    {tail}"

    @staticmethod
    def _clean_dir(path: str) -> None:
        if path and os.path.isdir(path):
            for f in os.listdir(path):
                fp = os.path.join(path, f)
                if os.path.isfile(fp) or os.path.islink(fp):
                    os.unlink(fp)
                elif os.path.isdir(fp):
                    shutil.rmtree(fp)


# ═══════════════════════════════════════════════════════════════════════════════
# Regression tracker
# ═══════════════════════════════════════════════════════════════════════════════

class BestTracker:
    """Tracks the best throughput, saves the best impl to the experiments
    dir, and mirrors it into the workspace as `reference/best_impl.cc` so
    the generator can read it in later iterations.

    Mirroring is deliberate: without it, iter N+1 only sees iter N's impl
    (whether regressed or not), and there is no way for the agent to
    compare its current direction against its best-known architecture.
    The mirrored file is NOT part of the CMake build (which is pinned to
    `interface/generated/kvstore_impl.cc`) — it's reference only.
    """

    def __init__(self, experiments_dir: str, workspace: str):
        self.best_mops = 0.0
        self.best_iter: int | None = None
        self.experiments_dir = experiments_dir
        self.workspace = workspace

    def update(self, iteration: int, iter_mops: float, iter_dir: str) -> str:
        if iter_mops > self.best_mops:
            self.best_mops = iter_mops
            self.best_iter = iteration
            impl_src = os.path.join(iter_dir, "kvstore_impl.cc")
            if os.path.isfile(impl_src):
                impl_dst = os.path.join(self.experiments_dir, "best_impl.cc")
                shutil.copy2(impl_src, impl_dst)
                self._mirror_to_workspace(impl_src, iteration, iter_mops)
            print(f"  ★ New best: {self.best_mops:.2f} Mops/s")
            return "★ NEW BEST"
        return ""

    def _mirror_to_workspace(self, impl_src: str, iteration: int,
                             mops: float) -> None:
        """Copy best_impl.cc + write best_meta.json under workspace/reference/.

        Kept outside `interface/generated/` so it can never accidentally be
        picked up by CMake (which globs nothing and pins `kvstore_impl.cc`).
        """
        ref_dir = os.path.join(self.workspace, "reference")
        try:
            os.makedirs(ref_dir, exist_ok=True)
            shutil.copy2(impl_src, os.path.join(ref_dir, "best_impl.cc"))
            with open(os.path.join(ref_dir, "best_meta.json"), "w") as f:
                json.dump({
                    "iter": iteration,
                    "mops": round(mops, 4),
                    "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                }, f, indent=2)
        except OSError as e:
            print(f"  [tracker] mirror to workspace failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main evolution loop
# ═══════════════════════════════════════════════════════════════════════════════

class EvolutionRunner:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.cgroup = CgroupManager()
        self.workspace_mgr = WorkspaceManager(cfg)
        self.agent = AgentRunner(cfg)
        self.prompt_builder = PromptBuilder(cfg)
        self.steps = PipelineSteps(cfg, self.cgroup)
        self.tracker = BestTracker(cfg.experiments_dir, cfg.workspace)
        self._prev_impl_hash: str | None = None
        self._prev_iter_dir: str | None = None
        # Critique runners. Instantiated based on mode:
        #   off    -> both None
        #   review -> reviewer only
        #   audit  -> auditor only
        #   full   -> both
        # When both are None, the orchestrator's behavior is byte-identical
        # to running without critique support.
        self.critiquer = None
        self.auditor = None
        # frozen_audit_mode: gate ON (run tests from a static dir) but
        # auditor agent OFF (no new tests written). Set when SKYKV_AUDIT_CHECKS_DIR
        # is provided. Mutually exclusive with the live AuditRunner.
        self.frozen_audit_mode = bool(cfg.audit_checks_dir)
        if cfg.critique_mode in ("review", "full"):
            from critique import CritiqueRunner
            self.critiquer = CritiqueRunner(cfg)
        if cfg.critique_mode in ("audit", "full") and not self.frozen_audit_mode:
            from critique_audit import AuditRunner
            self.auditor = AuditRunner(cfg)
        if self.frozen_audit_mode:
            # Copy frozen audit tests into the run dir so run_audit_gate finds them.
            src = cfg.audit_checks_dir
            dst = os.path.join(cfg.experiments_dir, "audit_checks")
            os.makedirs(dst, exist_ok=True)
            copied = 0
            if os.path.isdir(src):
                for name in os.listdir(src):
                    if name.startswith("check_") and (name.endswith(".py") or name.endswith(".cc")):
                        shutil.copy2(os.path.join(src, name), dst)
                        copied += 1
            print(f"  Frozen audit suite: copied {copied} tests from {src}")
            if copied == 0:
                print(f"  WARNING: SKYKV_AUDIT_CHECKS_DIR='{src}' contained no check_*.py / .cc tests")
        # Queue of (iter_dir, status, impl_path) waiting for the auditor to fire.
        # Flushed every cfg.audit_every iters and once at end-of-run.
        self._audit_pending: list[tuple[str, str, str]] = []

        # Worker pool for parallel eval
        self.pool = None
        if cfg.parallel_eval:
            from worker_pool import WorkerPool, PoolConfig
            num = cfg.num_workers or len(cfg.mem_budget_gb) * len(cfg.threads)
            max_budget = max(cfg.mem_budget_gb) if cfg.mem_budget_gb else 0
            min_ram = (max_budget + 12 + 4) if max_budget > 0 else 30
            machine = "n2-highcpu-64" if min_ram <= 56 else "n2-standard-64"
            print(f"[pool] Auto-sized: {num} workers "
                  f"({' '.join(str(b) for b in cfg.mem_budget_gb)} budgets "
                  f"× {' '.join(str(t) for t in cfg.threads)} threads)")
            print(f"[pool] Machine type: {machine} (needs ~{min_ram}GB RAM)")
            pcfg = PoolConfig.auto(machine_type=machine)
            self.pool = WorkerPool(pcfg)
            if not self.pool.init(num):
                print("[WARN] Worker pool init failed — falling back to local eval")
                self.pool = None
                cfg.parallel_eval = False
            self.steps.pool = self.pool

    def run(self) -> None:
        cfg = self.cfg
        self.workspace_mgr.setup()
        os.makedirs(cfg.experiments_dir, exist_ok=True)
        # Local storage dirs only needed when NOT using parallel eval
        if not cfg.parallel_eval:
            os.makedirs(cfg.correctness_storage, exist_ok=True)
            if cfg.storage_dir:
                os.makedirs(cfg.storage_dir, exist_ok=True)

        self._print_banner()

        # Register signal handlers
        def on_signal(signum, frame):
            print(f"\n[interrupt] caught signal -- artifacts at {cfg.workspace}",
                  file=sys.stderr)
            self.cgroup.cleanup()
            sys.exit(130)
        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)

        # Phase 0 produces planning/whiteboard.md (exploration plan) and
        # planning/design.md (first-approach spec used by iter 1). Bypass
        # when --no-planner is set (off-the-shelf baseline).
        if not cfg.no_planner:
            self._run_planning()
        else:
            print("  Planning: SKIPPED (--no-planner)")

        try:
            for i in range(1, cfg.iterations + 1):
                self._run_iteration(i)
            # End-of-run flush: if auditor is batched (audit_every > 1) and the
            # total iteration count didn't divide evenly, any leftover pending
            # iters would otherwise go un-audited. Force a final fire.
            self._flush_audit_batch(reason="end-of-run")
        finally:
            self.cgroup.cleanup()

        self._print_final()

    def _run_iteration(self, i: int) -> None:
        cfg = self.cfg
        iter_dir = os.path.join(cfg.experiments_dir, f"iter_{i:03d}")
        os.makedirs(os.path.join(iter_dir, "bench_output"), exist_ok=True)

        print(f"\n{'═' * 60}")
        print(f"  ITERATION {i}/{cfg.iterations} -- {cfg.setup_name} ({cfg.dist_key}, {cfg.mode})")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.tracker.best_iter is not None:
            print(f"  Best so far: {self.tracker.best_mops:.2f} Mops/s (iter {self.tracker.best_iter})")
        print("═" * 60)

        # Step 1: Build prompt and run agent
        prompt = self._build_prompt(i)
        print(f"▶ Step 1: {cfg.backend} ({cfg.model}) -- "
              f"{'generating' if i == 1 else 'evolving'}...")

        self.steps._clean_dir(cfg.correctness_storage)
        self.steps._clean_dir(cfg.storage_dir)

        agent_log_path = os.path.join(iter_dir, "agent_log.txt")

        # Iter 1 + seed: skip the agent call and drop seeds/<name>/* into the
        # workspace's interface/generated/ so iter 1 measures the seed baseline
        # as-is. Agent is invoked normally from iter 2 onward — it sees iter 1
        # as the starting impl and may modify anything under generated/ (incl.
        # vendored trees like faster_src/).
        if i == 1 and cfg.seed_name:
            seed_src = os.path.join(cfg.project_dir, "seeds", cfg.seed_name)
            if not os.path.isdir(seed_src):
                raise RuntimeError(
                    f"--seed '{cfg.seed_name}' requested but {seed_src} does "
                    f"not exist. Available seeds: "
                    f"{os.listdir(os.path.join(cfg.project_dir, 'seeds')) if os.path.isdir(os.path.join(cfg.project_dir, 'seeds')) else '(none)'}")
            gen_dir = os.path.join(cfg.workspace, "interface", "generated")
            # Clean slate so re-runs don't inherit stale agent edits from
            # previous runs.
            if os.path.isdir(gen_dir):
                for item in os.listdir(gen_dir):
                    p = os.path.join(gen_dir, item)
                    if os.path.isdir(p):
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        os.remove(p)
            else:
                os.makedirs(gen_dir)
            for item in os.listdir(seed_src):
                src = os.path.join(seed_src, item)
                dst = os.path.join(gen_dir, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            print(f"  Seed '{cfg.seed_name}' copied into workspace/interface/generated/")
            print(f"  (skipping agent call on iter 1 — measuring seed baseline)")
            # Placeholder log so _detect_agent_failure and any other readers
            # see a non-empty file with a clear "this was a seed iter" signal.
            with open(agent_log_path, "w") as f:
                f.write('{"type":"seed","name":"' + cfg.seed_name + '"}\n')
            agent_rc = 0
        elif cfg.critique_mode == "off":
            # Legacy Alex three-stage fallback: generator is self-sufficient
            # (no critic runs). Phase 0 seeded planning/design.md with the
            # first-approach spec, so iter 1 skips Stage A and implements
            # directly. Iter 2+ runs Stage A (reflect on prev iter + rewrite
            # design.md) then Stage B (implement).
            if i >= 2:
                design_prompt = self._build_design_prompt(i)
                design_log = os.path.join(iter_dir, "design_log.txt")
                print(f"  Stage A: design ({STAGE_A_TURNS_ALEX_MODE} turns)...")
                self.agent.run(design_prompt, design_log,
                               max_turns=STAGE_A_TURNS_ALEX_MODE)
            impl_prompt = self.prompt_builder.build_implement(i)
            print(f"  Stage B: implement ({cfg.max_turns} turns)...")
            agent_rc = self.agent.run(impl_prompt, agent_log_path,
                                      max_turns=cfg.max_turns)
        else:
            # Default flow (--critique review / full): critic owns design.
            # The generator's prompt (built by _build_prompt above) already
            # includes planning/design.md as a handoff block via
            # PromptBuilder._design_handoff_block. One agent call per iter.
            agent_rc = self.agent.run(prompt, agent_log_path)

        # Archive planning/design.md (whoever wrote it — Phase 0 at iter 1,
        # critic at iter N>1, or Alex's Stage A at iter N>1 in off mode)
        # into the iteration directory so the audit trail per-iter is
        # complete regardless of which flow wrote it.
        design_src = os.path.join(cfg.workspace, "planning", "design.md")
        if os.path.isfile(design_src):
            shutil.copy2(design_src, os.path.join(iter_dir, "design.md"))

        # Detect agent-side failures that left the workspace impl untouched.
        # Common case: codex hit a rate limit (429) or auth error, the agent
        # log ends with a turn.failed event, and no file_change events were
        # emitted. Without this check we'd silently treat it as "unchanged"
        # and IMPL_UNCHANGED_REUSED_* the pipeline forever.
        agent_failure = _detect_agent_failure(agent_log_path)
        if agent_failure is not None:
            print(f"  ERROR: agent did not produce changes ({agent_failure})")
            self._record(iter_dir, "AGENT_FAILED")
            self._prev_iter_dir = iter_dir
            return

        # Check for impl
        impl_path = os.path.join(cfg.workspace, "interface", "generated", "kvstore_impl.cc")
        if not os.path.isfile(impl_path):
            print(f"  ERROR: No implementation generated (agent exit {agent_rc})")
            self._record(iter_dir, "AGENT_FAILED")

            return

        # Copy impl to iter dir
        shutil.copy2(impl_path, os.path.join(iter_dir, "kvstore_impl.cc"))
        lines = sum(1 for _ in open(impl_path))
        print(f"  Code: {lines} lines")

        # Check if impl is unchanged from previous iteration
        impl_hash = hashlib.md5(open(impl_path, "rb").read()).hexdigest()
        if (self._prev_impl_hash is not None
                and impl_hash == self._prev_impl_hash
                and self._prev_iter_dir is not None):
            prev_status_file = os.path.join(self._prev_iter_dir, "status.txt")
            prev_status = ""
            if os.path.isfile(prev_status_file):
                prev_status = open(prev_status_file).read().strip()
            # Record as a distinct status — downstream code (PromptBuilder,
            # critique, analyze.py, throughput plots) can detect reused iters
            # and avoid treating them as independent data points.
            reused_status = f"IMPL_UNCHANGED_REUSED_{prev_status}"
            print(f"  Impl unchanged (hash {impl_hash[:8]}...) -- "
                  f"skipping rebuild/bench, status: {reused_status}")
            self._record(iter_dir, reused_status)
            # Write a pointer file so downstream code knows where the real
            # results live. Do NOT copy bench_output — duplicate files confuse
            # analysis and make throughput plots show invisible flat segments.
            with open(os.path.join(iter_dir, "REUSED_FROM.txt"), "w") as f:
                f.write(os.path.basename(self._prev_iter_dir) + "\n")
            # Also skip critique for reused iters — there's nothing new to review.
            # Keep _prev_iter_dir pointing at the last REAL iteration (not this
            # reused one) so if iter N+1 also produces the same impl, it will
            # correctly REUSED_FROM the original real iter, not another reused one.
            return

        # Step 2: Build
        print("▶ Step 2: Building...")
        if not self.steps.build(iter_dir):
            self._record(iter_dir, "BUILD_FAILED")
            self._prev_impl_hash = impl_hash
            self._prev_iter_dir = iter_dir
            self._maybe_critique(iter_dir, "BUILD_FAILED")
            return

        # Step 3: Correctness
        print("▶ Step 3: Crash recovery tests...")
        if not self.steps.run_correctness(iter_dir):
            print("  CORRECTNESS FAILED -- skipping benchmarks")
            self._record(iter_dir, "CORRECTNESS_FAILED")
            self._prev_impl_hash = impl_hash
            self._prev_iter_dir = iter_dir
            self._maybe_critique(iter_dir, "CORRECTNESS_FAILED")
            self._maybe_audit(iter_dir, "CORRECTNESS_FAILED", impl_path)
            return

        # Step 3b: Canary budget test (LTM only).
        # Quick portability check at a budget the agent didn't see, to catch
        # hardcoded sizing constants. Failure here means the agent baked in
        # data-structure sizes for the primary budget instead of computing
        # them from mem_budget_bytes at runtime.
        if cfg.mode == "ltm":
            canary_gb = self.steps.canary_budget()
            if canary_gb > 0:
                print(f"▶ Step 3b: Canary budget portability test...")
                if not self.steps.run_canary_budget_test(iter_dir):
                    print("  CANARY BUDGET FAILED -- code has hardcoded sizing")
                    self._record(iter_dir, "CANARY_BUDGET_FAILED")
                    self._prev_impl_hash = impl_hash
                    self._prev_iter_dir = iter_dir
                    self._maybe_critique(iter_dir, "CANARY_BUDGET_FAILED")
                    return

        # Step 4: Audit check (advisory — never blocks the benchmark).
        # Accumulated audit tests run for coverage; failures are recorded
        # in the iter dir and flow to next iter's critic as advisory.
        # Promotion from log-only to gate is automatic after 2 ignored
        # iters (see critique_audit_prompt.txt AUDIT_ADVISORY section).
        if self.auditor is not None or self.frozen_audit_mode:
            audit_ok = self._run_audit_gate(iter_dir, impl_path)
            if not audit_ok:
                print("  Audit: exploit indicators present — "
                      "recorded as advisory; benchmark still runs")

        # Step 5: Benchmark
        print(f"▶ Step 5: Benchmark {cfg.setup_name}...")
        bench_ok, iter_best_mops = self.steps.run_benchmark(iter_dir)
        status = "COMPLETED" if bench_ok else "BENCH_FAILED"
        self._record(iter_dir, status)

        # Step 6: Track best
        annotation = self.tracker.update(i, iter_best_mops, iter_dir)

        # Step 7: Analyze + checkpoint
        self._analyze(iter_dir, i)

        self._prev_impl_hash = impl_hash
        self._prev_iter_dir = iter_dir

        # Step 8: Stateful review critique (review or full mode).
        self._maybe_critique(iter_dir, status)

        # Step 9: Adversarial audit critique (audit or full mode).
        # Runs AFTER review so the auditor can see anything the reviewer said.
        self._maybe_audit(iter_dir, status, impl_path)

        print(f"  Peak: {iter_best_mops:.2f} Mops/s | "
              f"Best ever: {self.tracker.best_mops:.2f} Mops/s "
              f"(iter {self.tracker.best_iter or 'none'})")


    def _build_prompt(self, i: int) -> str:
        if i == 1:
            return self.prompt_builder.build_initial()
        cfg = self.cfg
        prev_dir = os.path.join(cfg.experiments_dir, f"iter_{i - 1:03d}")
        prev_status = "UNKNOWN"
        status_file = os.path.join(prev_dir, "status.txt")
        if os.path.isfile(status_file):
            prev_status = open(status_file).read().strip()

        # If the previous iteration was a reused no-op (IMPL_UNCHANGED_REUSED_*),
        # follow REUSED_FROM.txt to the real iter and use its feedback. The
        # status prefix is stripped so build_subsequent sees the underlying
        # result (e.g., COMPLETED) and formats feedback accordingly.
        if prev_status.startswith("IMPL_UNCHANGED_REUSED_"):
            reused_from_file = os.path.join(prev_dir, "REUSED_FROM.txt")
            if os.path.isfile(reused_from_file):
                reused_basename = open(reused_from_file).read().strip()
                candidate = os.path.join(cfg.experiments_dir, reused_basename)
                if os.path.isdir(candidate):
                    prev_dir = candidate
                    prev_status = prev_status[len("IMPL_UNCHANGED_REUSED_"):]

        impl_path = os.path.join(cfg.workspace, "interface", "generated", "kvstore_impl.cc")
        impl_lines = sum(1 for _ in open(impl_path)) if os.path.isfile(impl_path) else 0
        return self.prompt_builder.build_subsequent(
            i, prev_status, prev_dir, impl_lines,
            best_iter=self.tracker.best_iter,
            best_mops=self.tracker.best_mops,
        )

    def _build_design_prompt(self, i: int) -> str:
        """Build design-stage prompt — mirrors _build_prompt but calls build_design."""
        if i == 1:
            return self.prompt_builder.build_design(
                1, "", "", 0,
                best_iter=None, best_mops=0.0,
            )
        cfg = self.cfg
        prev_dir = os.path.join(cfg.experiments_dir, f"iter_{i - 1:03d}")
        prev_status = "UNKNOWN"
        status_file = os.path.join(prev_dir, "status.txt")
        if os.path.isfile(status_file):
            prev_status = open(status_file).read().strip()

        if prev_status.startswith("IMPL_UNCHANGED_REUSED_"):
            reused_from_file = os.path.join(prev_dir, "REUSED_FROM.txt")
            if os.path.isfile(reused_from_file):
                reused_basename = open(reused_from_file).read().strip()
                candidate = os.path.join(cfg.experiments_dir, reused_basename)
                if os.path.isdir(candidate):
                    prev_dir = candidate
                    prev_status = prev_status[len("IMPL_UNCHANGED_REUSED_"):]

        impl_path = os.path.join(cfg.workspace, "interface", "generated", "kvstore_impl.cc")
        impl_lines = sum(1 for _ in open(impl_path)) if os.path.isfile(impl_path) else 0
        return self.prompt_builder.build_design(
            i, prev_status, prev_dir, impl_lines,
            best_iter=self.tracker.best_iter,
            best_mops=self.tracker.best_mops,
        )

    def _run_planning(self) -> None:
        """Phase 0: exploration planning — creates whiteboard.md + design.md.

        Produces two artifacts under {workspace}/planning/:
          - whiteboard.md — the exploration plan (3-5 distinct approaches,
            each with research anchor + first-principles rationale)
          - design.md    — first-approach design spec used by iter 1 in both
            --critique review/full and --critique off flows

        Both are archived as *_initial.md under experiments_dir for the
        run's audit trail.
        """
        cfg = self.cfg
        print(f"\n{'═' * 60}")
        print(f"  PHASE 0: PLANNING -- {cfg.setup_name} ({cfg.dist_key}, {cfg.mode})")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("═" * 60)

        planning_dir = os.path.join(cfg.workspace, "planning")
        os.makedirs(planning_dir, exist_ok=True)

        prompt = self.prompt_builder.build_planning()
        planning_log = os.path.join(cfg.experiments_dir, "planning_log.txt")
        self.agent.run(prompt, planning_log)

        wb_path = os.path.join(planning_dir, "whiteboard.md")
        design_path = os.path.join(planning_dir, "design.md")
        if os.path.isfile(wb_path):
            print("  Planning: whiteboard.md created")
            shutil.copy2(wb_path,
                         os.path.join(cfg.experiments_dir, "whiteboard_initial.md"))
        else:
            print("  Planning: WARNING — no whiteboard.md produced")
        if os.path.isfile(design_path):
            print("  Planning: design.md (initial spec for iter 1) created")
            shutil.copy2(design_path,
                         os.path.join(cfg.experiments_dir, "design_initial.md"))
        else:
            print("  Planning: WARNING — no design.md produced; iter 1 will "
                  "run without a seeded spec")

    def _analyze(self, iter_dir: str, iteration: int) -> None:
        try:
            sys.path.insert(0, os.path.join(self.cfg.project_dir, "agent-pipeline"))
            from analyze import analyze
            # Pass active budget so analyze can look up the FAIR adapter-
            # through-our-harness baseline (adapter_baseline_results.csv).
            # Falls back to native FASTER files when the CSV lacks the row.
            bud = self.cfg.mem_budget_gb[0] if self.cfg.mem_budget_gb else None
            a = analyze(os.path.join(iter_dir, "bench_output"),
                        mode=self.cfg.mode, budget_gb=bud)
            a["iteration"] = f"iter_{iteration:03d}"
            with open(os.path.join(iter_dir, "analysis.json"), "w") as f:
                json.dump(a, f, indent=2)
            s = a.get("summary", {})
            print(f"  Avg: {s.get('avg_pct_of_faster', 0):.1f}% of baseline")
            print(f"  Best: {s.get('best_pct_of_faster', 0):.1f}% of baseline")
        except Exception:
            print("  (analysis failed -- continuing)")

        subprocess.run(
            [sys.executable, os.path.join(self.cfg.project_dir, "agent-pipeline", "checkpoint.py"),
             self.cfg.experiments_dir, self.cfg.experiments_dir],
            capture_output=True, check=False,
        )

    def _print_banner(self) -> None:
        cfg = self.cfg
        print("═" * 64)
        print(f"  SkyKV -- {cfg.backend}")
        print(f"  Mode:         {cfg.mode}")
        print(f"  Distribution: {cfg.dist_desc}")
        print(f"  Setup:        {cfg.setup_name} ({cfg.setup_desc})")
        print(f"  Value size:   {cfg.value_size} bytes")
        if cfg.mem_budget_gb != [0]:
            print(f"  Mem budgets:  {' '.join(str(g) for g in cfg.mem_budget_gb)} GB")
        print(f"  Model:        {cfg.model}")
        if cfg.backend == "claude":
            print(f"  Max turns:    {cfg.max_turns}")
        print(f"  Iterations:   {cfg.iterations}")
        print(f"  Threads:      {' '.join(str(t) for t in cfg.threads)}")
        print(f"  Feedback:     {cfg.feedback_level}")
        print(f"  Critique:     {cfg.critique_mode}")
        print(f"  Summary:      {'on (rotate at >200K tokens)' if cfg.summary_enabled else 'off'}")
        if cfg.critique_mode == "off":
            print(f"  Flow:         Alex three-stage (generator self-designs; "
                  f"Stage A = {STAGE_A_TURNS_ALEX_MODE} turns from iter 2)")
        else:
            print(f"  Flow:         critic-as-designer (critic writes planning/design.md each iter)")
        print(f"  Planning:     phase 0 always runs (seeds whiteboard.md + design.md)")
        print(f"  Results:      {cfg.experiments_dir}")
        print("═" * 64)

    def _print_final(self) -> None:
        cfg = self.cfg
        # Archive final whiteboard + final design (critic's last spec) if they exist
        wb_path = os.path.join(cfg.workspace, "planning", "whiteboard.md")
        if os.path.isfile(wb_path):
            shutil.copy2(wb_path,
                         os.path.join(cfg.experiments_dir, "whiteboard_final.md"))
        design_path = os.path.join(cfg.workspace, "planning", "design.md")
        if os.path.isfile(design_path):
            shutil.copy2(design_path,
                         os.path.join(cfg.experiments_dir, "design_final.md"))
        print(f"\n{'═' * 60}")
        print(f"  DONE -- {cfg.backend} {cfg.mode} {cfg.dist_key} {cfg.setup_name}")
        print(f"  Best: {self.tracker.best_mops:.2f} Mops/s "
              f"(iteration {self.tracker.best_iter or 'none'})")
        print(f"  Results:     {cfg.experiments_dir}")
        print(f"  Leaderboard: {cfg.experiments_dir}/leaderboard.json")
        print("═" * 60)
        subprocess.run(
            [sys.executable, os.path.join(cfg.project_dir, "agent-pipeline", "checkpoint.py"),
             cfg.experiments_dir, cfg.experiments_dir],
            check=False,
        )

    @staticmethod
    def _record(iter_dir: str, status: str) -> None:
        with open(os.path.join(iter_dir, "status.txt"), "w") as f:
            f.write(status)

    def _maybe_critique(self, iter_dir: str, status: str) -> None:
        """Invoke the stateful reviewer if review (or full) mode is active."""
        if self.critiquer is None:
            return
        impl_path = os.path.join(iter_dir, "kvstore_impl.cc")
        if not os.path.isfile(impl_path):
            return
        self.critiquer.run(iter_dir, status, impl_path)

    def _maybe_audit(self, iter_dir: str, status: str, impl_path: str) -> None:
        """Queue this iter for the auditor; fire the batched auditor every
        cfg.audit_every iters.

        When audit_every == 1 (default) the queue is flushed on every call,
        making the behavior byte-identical to the pre-batching version.
        For audit_every > 1, iters accumulate and the auditor gets the whole
        batch's impls + bench results in a single prompt — the agent can
        reason across the evolution instead of just the latest snapshot.

        End-of-run leftovers are flushed by `_flush_audit_batch` from `run()`.
        """
        if self.auditor is None:
            return
        if not os.path.isfile(impl_path):
            return
        self._audit_pending.append((iter_dir, status, impl_path))
        if len(self._audit_pending) >= self.cfg.audit_every:
            self._flush_audit_batch(reason=f"batch of {self.cfg.audit_every}")
        else:
            print(f"  [audit] batched ({len(self._audit_pending)}/"
                  f"{self.cfg.audit_every}); will fire on next flush")

    def _flush_audit_batch(self, reason: str) -> None:
        """Fire the auditor on the queued iters and clear the queue.

        No-op if the queue is empty (e.g., end-of-run after a natural flush,
        or audit mode is off).
        """
        if self.auditor is None or not self._audit_pending:
            return
        batch = list(self._audit_pending)
        self._audit_pending.clear()
        n = len(batch)
        first_iter = os.path.basename(batch[0][0])
        last_iter = os.path.basename(batch[-1][0])
        print(f"  [audit] firing on {n} iter(s) [{first_iter}..{last_iter}] "
              f"({reason})")
        try:
            self.auditor.run(batch, self.cfg.experiments_dir)
        except Exception as e:
            print(f"  [audit] critique failed: {e}")

    def _run_audit_gate(self, iter_dir: str, impl_path: str) -> bool:
        """Compile + run accumulated audit tests. Returns True if gate passed.

        Gate = all gate-severity tests must pass. Log-severity tests may fail
        without blocking. Results are written to iter_dir/audit_results.txt;
        gate-failure feedback is propagated to the generator via the next
        iteration's prompt.
        """
        from critique_audit import run_audit_gate
        cfg = self.cfg
        print("▶ Step 4: Audit gate (accumulated tests)...")
        gate_ok, results = run_audit_gate(
            run_dir=cfg.experiments_dir,
            iter_dir=iter_dir,
            impl_path=impl_path,
            workspace=cfg.workspace,
        )
        gate_tests = [r for r in results if r.test.severity == "gate"]
        log_tests = [r for r in results if r.test.severity == "log"]
        n_gate_pass = sum(1 for r in gate_tests if r.passed)
        n_log_fail = sum(1 for r in log_tests if not r.passed)
        print(f"  Gate: {n_gate_pass}/{len(gate_tests)} passed, "
              f"log: {n_log_fail}/{len(log_tests)} flagged")
        return gate_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    cfg = RunConfig.from_env()
    runner = EvolutionRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
