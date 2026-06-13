"""agent-pipeline/critique_audit.py -- Adversarial audit critique + test gate.

Two responsibilities:

1. AuditRunner: post-iteration agent that reads impl + output + evaluator source
   and writes NEW audit tests (C++ and/or Python) to runs/<run>/audit_checks/.
   Also appends to hack_log.md with findings.

2. AuditGate: pre-benchmark gate that discovers accumulated audit tests from
   prior iterations, compiles the C++ ones, runs them all, and reports
   pass/fail. Gate-severity failures block the benchmark; log-severity failures
   are recorded but do not block.

Both rely on a header-comment protocol for test metadata:

    // AUDIT: <one-line description>
    // HACK: <what reward hack this catches>
    // SEVERITY: gate              # or: log
    // STATUS: active              # or: deprecated
    // ADDED: iter <N>
    // SUPERSEDES: (none)          # or: <old_test_name>

Python tests use the same fields in a module docstring.

Enabled via SKYKV_CRITIQUE_MODE in {"off", "review", "audit", "full"}.
"""

import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime

# Reuse the shared extract helpers from critique.py
from critique import (
    _deterministic_session_id,
    _extract_text_codex,
    _extract_text_claude,
)

# ── Header-comment parsing ────────────────────────────────────────────────────

HEADER_KEYS = ("AUDIT", "HACK", "SEVERITY", "STATUS", "ADDED", "SUPERSEDES")

# Matches "KEY: value" anchored at the start of a line, allowing optional
# leading whitespace, C++ comment marker (`//`), and Python triple-quote
# docstring delimiters (`'''` or `"""`).
_HEADER_LINE_RE = re.compile(
    r"^[\s/'\"]*(AUDIT|HACK|SEVERITY|STATUS|ADDED|SUPERSEDES)\s*:\s*(.+?)\s*$"
)


@dataclass
class AuditTest:
    """One audit test discovered on disk."""
    path: str               # absolute path to the source file
    kind: str               # "cpp" or "python"
    audit: str              # description
    hack: str               # what it catches
    severity: str           # "gate" or "log"
    status: str             # "active" or "deprecated"
    added: str              # free-form, typically "iter N"
    supersedes: str         # free-form; "(none)" if fresh
    bin_path: str | None = None  # absolute path to compiled binary (cpp only)


def parse_test_header(path: str) -> AuditTest | None:
    """Read the file's header comment/docstring and extract metadata.

    Returns None if the file is not a valid audit test (missing required fields
    or wrong extension). Treating unparseable files as non-tests is deliberate
    so the gate is robust to stray files in audit_checks/.
    """
    if path.endswith(".cc") or path.endswith(".cpp"):
        kind = "cpp"
    elif path.endswith(".py"):
        kind = "python"
    else:
        return None

    try:
        with open(path) as f:
            # First ~60 lines is enough; headers are by convention at the top.
            head = "".join(f.readline() for _ in range(60))
    except OSError:
        return None

    fields: dict[str, str] = {}
    for line in head.splitlines():
        m = _HEADER_LINE_RE.match(line)
        if m:
            fields[m.group(1)] = m.group(2).strip()

    # All required keys must appear (SUPERSEDES may be "(none)" but must exist)
    required = ("AUDIT", "HACK", "SEVERITY", "STATUS", "ADDED")
    if not all(k in fields for k in required):
        return None

    severity = fields["SEVERITY"].lower()
    if severity not in ("gate", "log"):
        return None

    status = fields["STATUS"].lower()
    if status not in ("active", "deprecated"):
        return None

    return AuditTest(
        path=os.path.abspath(path),
        kind=kind,
        audit=fields["AUDIT"],
        hack=fields["HACK"],
        severity=severity,
        status=status,
        added=fields["ADDED"],
        supersedes=fields.get("SUPERSEDES", "(none)"),
    )


def discover_tests(audit_checks_dir: str) -> list[AuditTest]:
    """Scan audit_checks/ and return all parseable tests (active + deprecated).

    Callers filter by status/severity as needed.
    """
    if not os.path.isdir(audit_checks_dir):
        return []
    tests: list[AuditTest] = []
    for name in sorted(os.listdir(audit_checks_dir)):
        full = os.path.join(audit_checks_dir, name)
        if not os.path.isfile(full):
            continue
        t = parse_test_header(full)
        if t is not None:
            tests.append(t)
    return tests


# ── C++ test compilation ──────────────────────────────────────────────────────

def compile_cpp_test(test: AuditTest, workspace: str, bin_dir: str) -> str | None:
    """Compile a C++ audit test against the generator's current impl.

    Returns the absolute binary path on success, or None on failure.
    The binary is cached: if it already exists and is newer than both the test
    source and the impl source, compilation is skipped.
    """
    if test.kind != "cpp":
        return None

    impl_cc = os.path.join(workspace, "interface", "generated", "kvstore_impl.cc")
    iface_dir = os.path.join(workspace, "interface")

    os.makedirs(bin_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(test.path))[0]
    bin_path = os.path.join(bin_dir, f"audit_{base}")

    # Cache: skip if binary is newer than both sources
    try:
        if os.path.isfile(bin_path):
            bin_mtime = os.path.getmtime(bin_path)
            src_mtime = os.path.getmtime(test.path)
            impl_mtime = os.path.getmtime(impl_cc) if os.path.isfile(impl_cc) else 0
            if bin_mtime > src_mtime and bin_mtime > impl_mtime:
                return bin_path
    except OSError:
        pass

    if not os.path.isfile(impl_cc):
        # Can't compile without the impl; caller will treat as skipped
        return None

    # Mirror interface/CMakeLists.txt's FASTER integration: if the impl
    # uses FASTER (detected by presence of vendored source), add its
    # include path + 4 Linux .cc files + required system libs.
    # Priority: workspace-local generated/faster_src/ (agent-editable when
    # --seed faster was used), then repo setups/faster-single-machine/seeds_faster/faster_src/.
    faster_dir = None
    ws_fs = os.path.join(workspace, "interface", "generated", "faster_src")
    if os.path.isfile(os.path.join(ws_fs, "core", "faster.h")):
        faster_dir = ws_fs
    else:
        # Walk up from workspace looking for repo's setups/faster-single-machine/seeds_faster/faster_src
        scan = workspace
        for _ in range(6):
            cand = os.path.join(scan, "seeds", "faster", "faster_src", "core", "faster.h")
            if os.path.isfile(cand):
                faster_dir = os.path.dirname(os.path.dirname(cand))
                break
            scan = os.path.dirname(scan)

    extra_sources = []
    extra_libs = []
    extra_includes = [f"-I{iface_dir}"]
    extra_defs = []
    if faster_dir:
        extra_includes.append(f"-I{faster_dir}")
        extra_defs.append("-DFASTER_VALUE_PADDING=92")
        for src in ("core/lss_allocator.cc", "core/address.cc",
                    "core/thread.cc", "environment/file_linux.cc"):
            p = os.path.join(faster_dir, src)
            if os.path.isfile(p):
                extra_sources.append(p)
        extra_libs.extend(["-lstdc++fs", "-luuid", "-ltbb", "-laio"])

    cmd = [
        "g++", "-O2", "-std=c++17", "-pthread",
        *extra_includes, *extra_defs,
        test.path, impl_cc, *extra_sources,
        "-o", bin_path,
        *extra_libs,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        # Leave stderr on stdout so the gate log shows why
        sys.stderr.write(
            f"[audit] compile failed for {os.path.basename(test.path)}:\n"
            f"{proc.stderr}\n"
        )
        # Remove stale binary if any
        if os.path.isfile(bin_path):
            try:
                os.remove(bin_path)
            except OSError:
                pass
        return None
    return bin_path


# ── Gate: run all active tests and report ────────────────────────────────────

@dataclass
class TestResult:
    test: AuditTest
    passed: bool
    skipped: bool
    skip_reason: str | None
    duration_secs: float
    returncode: int | None
    stdout_tail: str


def run_test(test: AuditTest, impl_path: str, bench_dir: str, iter_dir: str,
             timeout_secs: int = 30) -> TestResult:
    """Run a single audit test and return its result."""
    t_start = time.monotonic()

    if test.kind == "cpp":
        if test.bin_path is None or not os.path.isfile(test.bin_path):
            return TestResult(
                test=test, passed=False, skipped=True,
                skip_reason="compile_failed_or_missing",
                duration_secs=0.0, returncode=None, stdout_tail="",
            )
        cmd = [test.bin_path]
    else:  # python
        cmd = [sys.executable, test.path, impl_path, bench_dir, iter_dir]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_secs)
    except subprocess.TimeoutExpired:
        return TestResult(
            test=test, passed=False, skipped=False,
            skip_reason=None,
            duration_secs=round(time.monotonic() - t_start, 2),
            returncode=None,
            stdout_tail="<TIMED OUT>",
        )
    except OSError as e:
        return TestResult(
            test=test, passed=False, skipped=True,
            skip_reason=f"exec_error: {e}",
            duration_secs=0.0, returncode=None, stdout_tail="",
        )

    duration = round(time.monotonic() - t_start, 2)
    combined = ((proc.stdout or "") + (proc.stderr or ""))
    # Keep last ~1 KB for the results log
    tail = combined[-1024:] if len(combined) > 1024 else combined
    return TestResult(
        test=test, passed=(proc.returncode == 0), skipped=False,
        skip_reason=None, duration_secs=duration,
        returncode=proc.returncode, stdout_tail=tail,
    )


def run_audit_gate(
    run_dir: str,
    iter_dir: str,
    impl_path: str,
    workspace: str,
) -> tuple[bool, list[TestResult]]:
    """Compile + run all active audit tests against the current impl.

    Gate semantics:
      - Only tests with status=active are executed.
      - Any gate-severity failure causes the gate to return False.
      - Log-severity failures never cause the gate to fail.

    Writes iter_dir/audit_results.txt with a human-readable summary.
    Returns (gate_passed, all_results) so callers can distinguish gate-blocking
    failures from informational log failures.
    """
    audit_checks_dir = os.path.join(run_dir, "audit_checks")
    bin_dir = os.path.join(run_dir, "audit_bin")

    all_tests = discover_tests(audit_checks_dir)
    active_tests = [t for t in all_tests if t.status == "active"]

    # Compile C++ tests up front so we can separately report compile failures.
    for t in active_tests:
        if t.kind == "cpp":
            t.bin_path = compile_cpp_test(t, workspace, bin_dir)

    bench_dir = os.path.join(iter_dir, "bench_output")
    results: list[TestResult] = []
    for t in active_tests:
        results.append(run_test(t, impl_path, bench_dir, iter_dir))

    gate_passed = all(
        r.passed or r.test.severity == "log"
        for r in results
    )

    # Write a human-readable results file per iteration.
    _write_audit_results_txt(iter_dir, all_tests, results, gate_passed)

    return gate_passed, results


def _write_audit_results_txt(
    iter_dir: str,
    all_tests: list[AuditTest],
    results: list[TestResult],
    gate_passed: bool,
) -> None:
    """Write iter_dir/audit_results.txt summarizing discovery + gate outcomes."""
    path = os.path.join(iter_dir, "audit_results.txt")
    total = len(all_tests)
    active = [t for t in all_tests if t.status == "active"]
    deprecated = [t for t in all_tests if t.status == "deprecated"]

    lines: list[str] = []
    lines.append("# Audit gate results")
    lines.append("")
    lines.append(f"Tests discovered:  {total}")
    lines.append(f"  active:          {len(active)}")
    lines.append(f"  deprecated:      {len(deprecated)} (not executed)")
    lines.append("")
    lines.append(f"Gate result: {'PASSED' if gate_passed else 'FAILED'}")
    lines.append("")
    lines.append("## Per-test results")
    lines.append("")

    for r in results:
        t = r.test
        status_str = "PASS" if r.passed else ("SKIP" if r.skipped else "FAIL")
        lines.append(
            f"[{status_str}] {os.path.basename(t.path)}  "
            f"(severity={t.severity}, added={t.added}, {r.duration_secs}s)"
        )
        lines.append(f"  AUDIT: {t.audit}")
        lines.append(f"  HACK:  {t.hack}")
        if r.skipped:
            lines.append(f"  SKIP_REASON: {r.skip_reason}")
        elif not r.passed:
            lines.append(f"  RETURNCODE: {r.returncode}")
            if r.stdout_tail:
                lines.append("  OUTPUT:")
                for ln in r.stdout_tail.splitlines()[-20:]:
                    lines.append(f"    {ln}")
        lines.append("")

    try:
        with open(path, "w") as f:
            f.write("\n".join(lines))
    except OSError as e:
        sys.stderr.write(f"[audit] failed to write {path}: {e}\n")


def format_gate_failure_feedback(results: list[TestResult]) -> str:
    """Format audit-gate failures as feedback text for the generator."""
    failures = [r for r in results
                if not r.passed and r.test.severity == "gate" and not r.skipped]
    if not failures:
        return ""
    parts = ["\n## Audit gate failed", ""]
    for r in failures:
        t = r.test
        parts.append(f"### {os.path.basename(t.path)} ({t.severity.upper()})")
        parts.append(f"Added: {t.added} | Catches: {t.hack}")
        parts.append("")
        parts.append("**AUDIT**: " + t.audit)
        parts.append("")
        parts.append("**Test output**:")
        parts.append("```")
        parts.append((r.stdout_tail or "(no output)").strip())
        parts.append("```")
        parts.append("")
        parts.append("See hack_log.md for the hack this test was added to catch.")
        parts.append("")
    return "\n".join(parts)


# ── AuditRunner: the post-iteration audit agent ──────────────────────────────

# Reuses the section loader from critique.py (same file format).
_AUDIT_PROMPT_CACHE: dict[str, str] | None = None
_SECTION_HEADER_RE = re.compile(r"^##\s+([A-Z][A-Z0-9_]*)\s*$")


def _load_audit_prompt_sections() -> dict[str, str]:
    """Parse critique_audit_prompt.txt into {SECTION_NAME: text} dict."""
    global _AUDIT_PROMPT_CACHE
    if _AUDIT_PROMPT_CACHE is not None:
        return _AUDIT_PROMPT_CACHE
    prompt_path = os.path.join(os.path.dirname(__file__),
                               "prompts", "critique_audit_prompt.txt")
    with open(prompt_path) as f:
        content = f.read()
    sections: dict[str, str] = {}
    current_section: str | None = None
    buf: list[str] = []
    for line in content.splitlines():
        m = _SECTION_HEADER_RE.match(line)
        if m:
            if current_section is not None:
                sections[current_section] = "\n".join(buf).strip() + "\n"
            current_section = m.group(1)
            buf = []
        elif current_section is not None:
            buf.append(line)
    if current_section is not None:
        sections[current_section] = "\n".join(buf).strip() + "\n"
    _AUDIT_PROMPT_CACHE = sections
    return sections


class AuditRunner:
    """Stateful adversarial auditor across iterations.

    Reads impl + output + evaluator source + hack_log, writes new tests to
    runs/<run>/audit_checks/ and updates hack_log.md. Like CritiqueRunner,
    uses a codex session resumed across iterations so the agent remembers its
    prior findings.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.iter_count = 0
        self.thread_id: str | None = None
        # Claude session UUID — distinct from the critic's and generator's
        # so the three agents don't collide.
        self.claude_session_id: str = _deterministic_session_id(
            cfg.experiments_dir, agent="auditor"
        )

    # ── Public entry point ─────────────────────────────────────────────────

    def run(self, iters: list[tuple[str, str, str]], run_dir: str) -> str:
        """Invoke the audit agent on a batch of iterations.

        `iters` is a chronological list of (iter_dir, status, impl_path). When
        `--audit-every N` is 1 (default) each call is a 1-element list, which
        matches the pre-batching behavior exactly. For N > 1 the agent sees the
        full batch and can reason across the evolution.

        Returns the audit's text output (empty if the batch is empty or the
        latest iter has no impl).
        """
        if not iters:
            return ""
        # The "latest" iter in the batch is what drives paths/logging/status —
        # artifacts land under its dir, session prompt treats it as "current".
        latest_iter_dir, latest_status, latest_impl_path = iters[-1]
        if not os.path.isfile(latest_impl_path):
            return ""

        self.iter_count += 1
        t_start = time.monotonic()
        ts_start = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        audit_checks_dir = os.path.join(run_dir, "audit_checks")
        os.makedirs(audit_checks_dir, exist_ok=True)

        prompt = self._build_prompt(iters, run_dir)
        # Downstream logging aliases (keep names readable below).
        iter_dir = latest_iter_dir
        iter_status = latest_status
        impl_path = latest_impl_path

        # Summary-rotation: one-shot the dying thread for a handoff memo
        # when session history grows past the token threshold, then start
        # a fresh thread with the memo prepended to the next prompt.
        if getattr(self.cfg, "summary_enabled", False) and self.cfg.backend == "codex":
            from memory import maybe_rotate_codex
            memo = maybe_rotate_codex(
                self, "thread_id", self.cfg.workspace, self.cfg.model,
                iter_dir, "auditor",
            )
            if memo is not None:
                prompt = (
                    "## Carryover summary (from prior session)\n\n"
                    f"{memo}\n\n---\n\n{prompt}"
                )

        cmd = self._build_cmd(prompt)

        log_file = os.path.join(iter_dir, "audit_log.txt")
        summary_path = os.path.join(iter_dir, "hack_audit.md")

        session_action = "resuming" if self.thread_id else "starting"
        print(f"▶ Audit critique ({session_action} session)...")

        with open(log_file, "w") as lf:
            proc = subprocess.run(cmd, stdin=subprocess.DEVNULL,
                                  stdout=lf, stderr=subprocess.STDOUT,
                                  check=False)
        exit_code = proc.returncode

        if self.cfg.backend == "codex" and self.thread_id is None:
            self._extract_thread_id(log_file)
            if self.thread_id is None:
                self._log_summary(iter_dir, {
                    "iter": self.iter_count,
                    "mode": "audit",
                    "error": "thread_id extraction failed",
                    "log_file": log_file,
                    "exit_code": exit_code,
                    "duration_secs": round(time.monotonic() - t_start, 2),
                })
                raise RuntimeError(
                    f"Failed to extract audit thread_id from {log_file}. "
                    "Aborting to avoid silent fallback to fresh-session-per-iteration."
                )

        audit_text = self._extract_agent_text(log_file)
        with open(summary_path, "w") as f:
            f.write(audit_text)

        # Record which tests exist AFTER this audit (so we can see what it wrote).
        tests_after = discover_tests(audit_checks_dir)
        active_after = [t for t in tests_after if t.status == "active"]
        deprecated_after = [t for t in tests_after if t.status == "deprecated"]

        duration = round(time.monotonic() - t_start, 2)
        ts_end = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        session_id = self.thread_id if self.cfg.backend == "codex" else self.claude_session_id
        self._log_summary(iter_dir, {
            "iter": self.iter_count,
            "mode": "audit",
            "backend": self.cfg.backend,
            "ts_start": ts_start,
            "ts_end": ts_end,
            "duration_secs": duration,
            "session": {
                "action": session_action,
                "session_id": session_id,
            },
            "subprocess": {
                "exit_code": exit_code,
                "log_file": os.path.relpath(log_file, iter_dir),
            },
            "output": {
                "hack_audit_path": os.path.relpath(summary_path, iter_dir),
                "hack_audit_chars": len(audit_text),
                "tests_after": {
                    "total": len(tests_after),
                    "active": len(active_after),
                    "deprecated": len(deprecated_after),
                    "active_names": [os.path.basename(t.path) for t in active_after],
                },
            },
        })
        self._append_trail(iter_dir, {
            "iter": self.iter_count,
            "mode": "audit",
            "ts": ts_end,
            "duration_secs": duration,
            "exit_code": exit_code,
            "hack_audit_chars": len(audit_text),
            "active_tests_count": len(active_after),
            "session_action": session_action,
            "iter_status": iter_status,
            "batch_size": len(iters),
            "batch_iters": [os.path.basename(d) for d, _, _ in iters],
        })

        print(f"  audit done: {len(active_after)} active tests, "
              f"hack_audit.md={len(audit_text)} chars, {duration}s")
        return audit_text

    # ── Command construction ───────────────────────────────────────────────

    def _build_cmd(self, prompt: str) -> list[str]:
        if self.cfg.backend == "claude":
            return self._build_cmd_claude(prompt)
        return self._build_cmd_codex(prompt)

    def _build_cmd_codex(self, prompt: str) -> list[str]:
        base = ["codex", "exec",
                "--model", self.cfg.model,
                "--sandbox", "danger-full-access",
                "-C", self.cfg.workspace,
                "--skip-git-repo-check",
                "--json"]
        if self.thread_id is not None:
            return base + ["resume", self.thread_id, prompt]
        return base + [prompt]

    def _build_cmd_claude(self, prompt: str) -> list[str]:
        session_flag = (
            ["--session-id", self.claude_session_id] if self.iter_count == 1
            else ["--resume", self.claude_session_id]
        )
        return ["claude", "-p", prompt,
                *session_flag,
                "--model", self.cfg.model,
                "--max-turns", str(self.cfg.max_turns),
                "--dangerously-skip-permissions",
                "--add-dir", self.cfg.workspace,
                "--verbose",
                "--output-format", "stream-json"]

    # ── JSONL parsing (mirrors CritiqueRunner) ─────────────────────────────

    def _extract_thread_id(self, log_file: str) -> None:
        try:
            with open(log_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        if event.get("type") == "thread.started" and "thread_id" in event:
                            self.thread_id = event["thread_id"]
                            print(f"  Audit session: {self.thread_id}")
                            return
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass

    def _extract_agent_text(self, log_file: str) -> str:
        if self.cfg.backend == "claude":
            text = _extract_text_claude(log_file)
        else:
            text = _extract_text_codex(log_file)
        return text if text and text != "(no critique produced)" else "(no audit output produced)"

    # ── Prompt construction ────────────────────────────────────────────────

    def _build_prompt(self, iters: list[tuple[str, str, str]],
                      run_dir: str) -> str:
        """Build the audit prompt for a batch of iterations.

        The last entry in `iters` is the "current" iter (paths block points at
        it, header status reflects it). Earlier entries are prior iters in the
        same batch window — the agent sees the evolution of impl + bench over
        those iters and can flag drift or regressions.

        For a 1-element batch (audit_every == 1, default) this produces the
        same prompt shape the pre-batching code produced.
        """
        cfg = self.cfg
        sections = _load_audit_prompt_sections()

        latest_iter_dir, latest_status, latest_impl_path = iters[-1]

        budgets_str = (
            "unlimited" if cfg.mem_budget_gb == [0]
            else ", ".join(str(b) for b in cfg.mem_budget_gb)
        )
        threads_str = ", ".join(str(t) for t in cfg.threads)
        header = sections["HEADER"].format(
            iter_count=self.iter_count,
            iter_status=latest_status,
            setup_name=cfg.setup_name,
            setup_desc=cfg.setup_desc,
            dist_desc=cfg.dist_desc,
            mode=cfg.mode,
            value_size=cfg.value_size,
            mem_budget_gb=budgets_str,
            threads=threads_str,
            baseline_name=cfg.baseline_name,
        )
        framing = sections.get("FRAMING", "")
        role = sections["ROLE_INITIAL"] if self.iter_count == 1 else sections["ROLE_SUBSEQUENT"]
        finding_shape = sections.get("FINDING_SHAPE", "")
        audit_advisory = sections.get("AUDIT_ADVISORY", "")
        null_finding = sections.get("NULL_FINDING", "")
        test_instructions = sections.get("TEST_INSTRUCTIONS", "")
        hack_log_format = sections.get("HACK_LOG_FORMAT", "")
        footer = sections.get("FOOTER", "")

        run_dir_abs = os.path.abspath(run_dir)
        audit_checks_abs = os.path.join(run_dir_abs, "audit_checks")
        hack_log_abs = os.path.join(run_dir_abs, "hack_log.md")
        latest_iter_abs = os.path.abspath(latest_iter_dir)
        latest_impl_abs = os.path.abspath(latest_impl_path)

        paths_block = (
            "\n## Paths (absolute — use verbatim)\n\n"
            f"- Run dir:          `{run_dir_abs}`\n"
            f"- Iteration dir:    `{latest_iter_abs}`  (current / most recent in batch)\n"
            f"- Impl source:      `{latest_impl_abs}`  (current / most recent in batch)\n"
            f"- Workspace:        `{os.path.abspath(cfg.workspace)}`  (your cwd)\n"
            f"- Audit tests dir:  `{audit_checks_abs}`  (write new tests here)\n"
            f"- Hack log:         `{hack_log_abs}`  (append/update; create if missing)\n"
        )

        # Batch section. For a 1-element batch this is one `### iter_NNN` block
        # that looks equivalent to the old single-iter prompt. For N > 1, each
        # iter contributes its own impl + bench + correctness; the latest impl
        # gets a bigger char budget since it's the "live" one.
        batch_blocks: list[str] = []
        n = len(iters)
        if n > 1:
            names = ", ".join(os.path.basename(d) for d, _, _ in iters)
            batch_blocks.append(
                f"\n## Iterations in this batch ({n})\n\n"
                f"You are auditing {n} consecutive iterations: {names}. They are\n"
                "presented in chronological order below (oldest → newest). The\n"
                "LAST one is the \"current\" iter that the paths block points at.\n"
                "Look for reward hacking introduced at any point in this window,\n"
                "and note regressions or drift across iters when relevant.\n"
            )
        # Per-impl char budget: latest gets the full 200K; older ones share a
        # smaller slice so the total prompt stays bounded.
        latest_budget = 200_000
        older_budget = 80_000 if n > 1 else 200_000
        for idx, (idir, status, ipath) in enumerate(iters):
            is_latest = idx == n - 1
            iter_tag = os.path.basename(idir)
            # "(current)" tag only disambiguates multi-iter batches; a
            # single-iter batch (audit_every == 1) is already unambiguous.
            label = iter_tag + ("  (current)" if (is_latest and n > 1) else "")
            impl_text = _read_file(
                ipath, max_chars=latest_budget if is_latest else older_budget
            )
            bench_section = _format_bench_dir(os.path.join(idir, "bench_output"))
            corr_section = _format_correctness(idir)
            batch_blocks.append(
                f"\n### {label} — status: {status}\n\n"
                f"**Implementation** (`{ipath}`):\n```cpp\n{impl_text}\n```\n"
                + bench_section
                + corr_section
            )

        return (
            header + "\n"
            + framing + "\n"
            + paths_block
            + role + "\n"
            + finding_shape + "\n"
            + audit_advisory + "\n"
            + null_finding + "\n"
            + test_instructions + "\n"
            + hack_log_format + "\n"
            + "".join(batch_blocks)
            + "\n" + footer
        )

    # ── Logging helpers ────────────────────────────────────────────────────

    @staticmethod
    def _log_summary(iter_dir: str, summary: dict) -> None:
        path = os.path.join(iter_dir, "audit_summary.json")
        try:
            with open(path, "w") as f:
                json.dump(summary, f, indent=2)
        except OSError as e:
            sys.stderr.write(f"[WARN] failed to write audit summary: {e}\n")

    @staticmethod
    def _append_trail(iter_dir: str, record: dict) -> None:
        run_dir = os.path.dirname(iter_dir)
        trail_path = os.path.join(run_dir, "audit_trail.jsonl")
        try:
            with open(trail_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as e:
            sys.stderr.write(f"[WARN] failed to append audit trail: {e}\n")


# ── File helpers (duplicated from critique.py to keep modules independent) ────

def _read_file(path: str, max_chars: int) -> str:
    try:
        with open(path, "r", errors="replace") as f:
            text = f.read()
    except OSError:
        return "(file not readable)"
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n... [truncated] ...\n\n" + text[-half:]


def _format_bench_dir(bench_dir: str) -> str:
    if not os.path.isdir(bench_dir):
        return ""
    chunks = []
    for fname in sorted(os.listdir(bench_dir)):
        fpath = os.path.join(bench_dir, fname)
        if not os.path.isfile(fpath):
            continue
        text = _read_file(fpath, max_chars=50_000)
        chunks.append(f"### {fname}\n```\n{text}\n```")
    if not chunks:
        return ""
    return "\n## Benchmark output\n\n" + "\n\n".join(chunks) + "\n"


def _format_correctness(iter_dir: str) -> str:
    chunks = []
    for tid in range(1, 7):
        cf = os.path.join(iter_dir, f"consistency_test{tid}.txt")
        if os.path.isfile(cf):
            text = _read_file(cf, max_chars=20_000)
            chunks.append(f"### consistency_test{tid}\n```\n{text}\n```")
    if not chunks:
        return ""
    return "\n## Correctness output\n\n" + "\n\n".join(chunks) + "\n"
