"""agent-pipeline/critique.py -- Stateful critique agent for iteration review.

Invoked after the benchmark/correctness steps of each iteration. Uses the same
`codex exec resume <thread_id>` mechanism as AgentRunner, so it accumulates
memory across iterations: it can see how the generator responded to its prior
feedback and push harder on what was ignored.

Enabled via SKYKV_CRITIQUE_ENABLED=1 (default off). When off, this module is
never instantiated and the orchestrator's behavior is unchanged.

Per iteration the critique reads:
  - iter_dir/kvstore_impl.cc                   (the generated impl)
  - iter_dir/status.txt                         (BUILD_FAILED / CORRECTNESS_FAILED / BENCH_FAILED / COMPLETED)
  - iter_dir/bench_output/*.txt                 (per-thread per-budget Mops + OpBreakdown + RSS)
  - iter_dir/consistency_test{1..6}.txt         (correctness outputs, whether passed or failed)

And writes:
  - iter_dir/critique_log.txt                   (full codex --json event stream, raw subprocess output)
  - iter_dir/critique.txt                       (concatenated agent_message text, read verbatim by the next iter's PromptBuilder)
  - iter_dir/critique_summary.json              (structured per-iteration summary: inputs, timings, outputs)
  - runs/<run>/critique_trail.jsonl      (cumulative one-line-per-iteration log, appended to)

The critique is wired into the generator's prompt via PromptBuilder._prev_critique_block.
"""

import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime


class CritiqueRunner:
    """Post-iteration reviewer, stateful across iterations.

    Lifecycle:
      - Iter 1: invokes `codex exec --json <prompt>`, captures thread_id.
      - Iter 2+: invokes `codex exec --json resume <thread_id> <prompt>`.

    Raises RuntimeError if iter-1 thread_id extraction fails, to avoid silent
    fallback to a fresh non-stateful session on every iter (the failure mode
    PR #14's --skip-git-repo-check fix already guards against for the generator).
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.iter_count = 0
        # Codex tracks session via thread_id parsed from the JSON stream.
        self.thread_id: str | None = None
        # Claude tracks session via a caller-provided UUID. We derive it
        # deterministically from the experiment + agent role so that it is
        # stable across orchestrator restarts.
        self.claude_session_id: str = _deterministic_session_id(
            cfg.experiments_dir, agent="critic"
        )

    # ── Public entry point ─────────────────────────────────────────────────────

    def run(self, iter_dir: str, iter_status: str, impl_path: str) -> str:
        """Produce a critique for this iter. Returns the critique text (possibly empty)."""
        if not os.path.isfile(impl_path):
            self._log_summary(iter_dir, {
                "iter": self.iter_count + 1,
                "mode": "review",
                "skipped": True,
                "reason": "impl file missing",
                "impl_path": impl_path,
            })
            return ""

        self.iter_count += 1
        t_start = time.monotonic()
        ts_start = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        # Capture input fingerprint for the log
        inputs = self._snapshot_inputs(iter_dir, iter_status, impl_path)

        prompt = self._build_prompt(iter_dir, iter_status, impl_path)

        # Summary-rotation: if --summary is on and this session's replay would
        # exceed the token threshold, one-shot the dying thread for a handoff
        # memo, clear thread_id, and prepend the memo to the next prompt.
        if getattr(self.cfg, "summary_enabled", False) and self.cfg.backend == "codex":
            from memory import maybe_rotate_codex
            memo = maybe_rotate_codex(
                self, "thread_id", self.cfg.workspace, self.cfg.model,
                iter_dir, "critic",
            )
            if memo is not None:
                prompt = (
                    "## Carryover summary (from prior session)\n\n"
                    f"{memo}\n\n---\n\n{prompt}"
                )

        cmd = self._build_cmd(prompt)

        log_file = os.path.join(iter_dir, "critique_log.txt")
        critique_path = os.path.join(iter_dir, "critique.txt")

        session_action = "resuming" if self.thread_id else "starting"
        print(f"▶ Step 7: critique ({session_action} session)...")

        with open(log_file, "w") as lf:
            proc = subprocess.run(cmd, stdin=subprocess.DEVNULL,
                                  stdout=lf, stderr=subprocess.STDOUT,
                                  check=False)
        exit_code = proc.returncode

        # First-iteration thread_id capture (codex only — for Claude we
        # generate and pass the session_id ourselves).
        if self.cfg.backend == "codex" and self.thread_id is None:
            self._extract_thread_id(log_file)
            if self.thread_id is None:
                # Log the failure before raising so debugging is easier
                self._log_summary(iter_dir, {
                    "iter": self.iter_count,
                    "mode": "review",
                    "error": "thread_id extraction failed",
                    "log_file": log_file,
                    "exit_code": exit_code,
                    "duration_secs": round(time.monotonic() - t_start, 2),
                })
                raise RuntimeError(
                    f"Failed to extract critique thread_id from {log_file}. "
                    "Aborting to avoid silent fallback to fresh-session-per-iteration."
                )

        critique_text = self._extract_critique_text(log_file)
        with open(critique_path, "w") as f:
            f.write(critique_text)

        duration = round(time.monotonic() - t_start, 2)
        ts_end = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        critique_chars = len(critique_text)
        critique_lines = critique_text.count("\n") + 1 if critique_text else 0

        # Structured per-iteration summary
        session_id = self.thread_id if self.cfg.backend == "codex" else self.claude_session_id
        self._log_summary(iter_dir, {
            "iter": self.iter_count,
            "mode": "review",
            "backend": self.cfg.backend,
            "ts_start": ts_start,
            "ts_end": ts_end,
            "duration_secs": duration,
            "session": {
                "action": session_action,
                "session_id": session_id,
            },
            "inputs": inputs,
            "subprocess": {
                "exit_code": exit_code,
                "log_file": os.path.relpath(log_file, iter_dir),
            },
            "output": {
                "critique_path": os.path.relpath(critique_path, iter_dir),
                "critique_chars": critique_chars,
                "critique_lines": critique_lines,
                "empty": critique_chars == 0 or critique_text == "(no critique produced)",
            },
        })

        # Append to cumulative trail for cross-iteration analysis
        self._append_trail(iter_dir, {
            "iter": self.iter_count,
            "mode": "review",
            "ts": ts_end,
            "duration_secs": duration,
            "exit_code": exit_code,
            "critique_chars": critique_chars,
            "session_action": session_action,
            "iter_status": iter_status,
        })

        print(f"  critique written: {critique_chars} chars "
              f"({critique_lines} lines, {duration}s)")
        return critique_text

    # ── Logging helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _snapshot_inputs(iter_dir: str, iter_status: str, impl_path: str) -> dict:
        """Record what the critique saw this iteration, so failures can be audited."""
        impl_lines = 0
        impl_bytes = 0
        try:
            with open(impl_path, "rb") as f:
                data = f.read()
                impl_bytes = len(data)
                # wc -l semantics: count newlines. Files ending in \n count
                # as N lines; files without trailing newline are N+1.
                impl_lines = data.count(b"\n")
                if data and not data.endswith(b"\n"):
                    impl_lines += 1
        except OSError:
            pass

        bench_dir = os.path.join(iter_dir, "bench_output")
        bench_files = []
        if os.path.isdir(bench_dir):
            bench_files = sorted(
                f for f in os.listdir(bench_dir)
                if os.path.isfile(os.path.join(bench_dir, f))
            )

        consistency_files = sorted(
            f for f in os.listdir(iter_dir)
            if f.startswith("consistency_test")
            and f.endswith(".txt")
            and os.path.isfile(os.path.join(iter_dir, f))
        ) if os.path.isdir(iter_dir) else []

        return {
            "iter_status": iter_status,
            "impl_path": impl_path,
            "impl_lines": impl_lines,
            "impl_bytes": impl_bytes,
            "bench_files": bench_files,
            "consistency_files": consistency_files,
        }

    @staticmethod
    def _log_summary(iter_dir: str, summary: dict) -> None:
        """Write per-iteration structured summary to critique_summary.json."""
        path = os.path.join(iter_dir, "critique_summary.json")
        try:
            with open(path, "w") as f:
                json.dump(summary, f, indent=2)
        except OSError as e:
            print(f"[WARN] failed to write critique summary to {path}: {e}",
                  file=sys.stderr)

    @staticmethod
    def _append_trail(iter_dir: str, record: dict) -> None:
        """Append one JSON-per-line record to runs/<run>/critique_trail.jsonl.

        The trail lives one directory up from iter_NNN/ (at runs/<run>/
        level), so it accumulates across all iterations for easy querying with
        `jq` or similar tools.
        """
        run_dir = os.path.dirname(iter_dir)
        trail_path = os.path.join(run_dir, "critique_trail.jsonl")
        try:
            with open(trail_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as e:
            print(f"[WARN] failed to append critique trail at {trail_path}: {e}",
                  file=sys.stderr)

    # ── Command construction ───────────────────────────────────────────────────

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
        # Claude's session model: --session-id <uuid> creates/attaches, then
        # --resume <uuid> continues it on subsequent calls. Each agent role
        # (generator/critic/auditor) needs a distinct UUID so they don't
        # collide when all three run in the same working directory.
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

    # ── JSONL parsing (mirrors AgentRunner._extract_thread_id) ────────────────

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
                            print(f"  Critique session: {self.thread_id}")
                            return
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass

    def _extract_critique_text(self, log_file: str) -> str:
        """Extract the agent's text output from the JSON event log.

        Dispatches by backend because codex and claude emit different event
        shapes.
        """
        if self.cfg.backend == "claude":
            return _extract_text_claude(log_file)
        return _extract_text_codex(log_file)

    # ── Prompt construction ────────────────────────────────────────────────────

    def _build_prompt(self, iter_dir: str, iter_status: str, impl_path: str) -> str:
        cfg = self.cfg
        impl_text = _read_file(impl_path, max_chars=200_000)
        bench_section = _format_bench_dir(os.path.join(iter_dir, "bench_output"))
        corr_section = _format_correctness(iter_dir)

        sections = _load_prompt_sections()
        # Render list-valued config fields as "8, 32" rather than "[8, 32]"
        # so the prompt reads naturally for the agent.
        budgets_str = (
            "unlimited" if cfg.mem_budget_gb == [0]
            else ", ".join(str(b) for b in cfg.mem_budget_gb)
        )
        threads_str = ", ".join(str(t) for t in cfg.threads)
        header = sections["HEADER"].format(
            iter_count=self.iter_count,
            iter_status=iter_status,
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

        # Audit advisory: accumulated exploit-indicator notes the auditor
        # wrote. Advisory-only per audit prompt — critic should factor
        # them into next iter's design.md but can also dispute if false
        # positive. Tail 6 KB so we surface recent entries not the whole
        # run-length history.
        audit_advisory_block = ""
        hack_audit_path = os.path.join(
            os.path.dirname(os.path.abspath(iter_dir)), "hack_audit.md")
        if os.path.isfile(hack_audit_path):
            try:
                # errors="replace": advisory may contain non-utf-8 scraps
                # (stack traces, binary hex) if auditor copy-pasted bench
                # output; don't crash prompt assembly on those.
                with open(hack_audit_path, "r", errors="replace") as f:
                    text = f.read()
                tail = text if len(text) <= 6000 else text[-6000:]
                audit_advisory_block = (
                    "\n## Audit advisory (from auditor — not a gate)\n\n"
                    "The auditor flagged these possible exploits. Factor "
                    "them into your review and into the next iter's "
                    f"planning/design.md. Path: `{hack_audit_path}`.\n\n"
                    f"```md\n{tail}\n```\n"
                )
            except OSError:
                pass

        review_output = sections.get("REVIEW_OUTPUT", "")
        # DESIGN_OUTPUT references {baseline_name} in the Learnings-update
        # section. No other placeholders in this section — keep it minimal.
        design_output = sections.get("DESIGN_OUTPUT", "")
        if design_output:
            design_output = design_output.format(baseline_name=cfg.baseline_name)
        footer = sections["FOOTER"]

        # Absolute paths for files outside the workspace, so the agent can
        # re-read them if the inlined content is truncated or needs verification.
        iter_dir_abs = os.path.abspath(iter_dir)
        run_dir_abs = os.path.dirname(iter_dir_abs)
        ws_abs = os.path.abspath(cfg.workspace)
        paths_block = (
            "\n## Paths (absolute)\n\n"
            f"- Iteration dir:    `{iter_dir_abs}`\n"
            f"- Run dir:          `{run_dir_abs}`\n"
            f"- Impl source:      `{os.path.abspath(impl_path)}`\n"
            f"- Workspace:        `{ws_abs}` (your cwd when reading spec files)\n"
            f"- Design file:      `{os.path.join(ws_abs, 'planning', 'design.md')}` "
            "(overwrite with next iter's spec per DESIGN_OUTPUT)\n"
            f"- Whiteboard:       `{os.path.join(ws_abs, 'planning', 'whiteboard.md')}` "
            "(append to ## Learnings per DESIGN_OUTPUT)\n"
        )

        return (
            header + "\n"
            + framing + "\n"
            + paths_block
            + role + "\n"
            + f"## Implementation (`{impl_path}`)\n```cpp\n{impl_text}\n```\n"
            + bench_section
            + corr_section
            + audit_advisory_block
            + "\n" + review_output + "\n"
            + "\n" + design_output + "\n"
            + "\n" + footer
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _deterministic_session_id(experiment_dir: str, agent: str) -> str:
    """Derive a stable UUIDv5 for a given experiment + agent role.

    Used for Claude's --session-id, which requires a UUID the caller controls.
    Stability across orchestrator restarts is desirable so a crashed run can
    be resumed without losing the critic's memory.
    """
    seed = f"{os.path.basename(experiment_dir.rstrip('/'))}::{agent}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))


def _extract_text_codex(log_file: str) -> str:
    """Concatenate all agent_message texts from codex JSON event stream."""
    parts = []
    try:
        with open(log_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") != "item.completed":
                    continue
                item = event.get("item") or {}
                if item.get("type") != "agent_message":
                    continue
                text = (item.get("text") or "").strip()
                if text:
                    parts.append(text)
    except OSError:
        pass
    return "\n\n".join(parts) if parts else "(no critique produced)"


def _extract_text_claude(log_file: str) -> str:
    """Concatenate assistant-message text blocks from claude stream-json."""
    parts = []
    try:
        with open(log_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") != "assistant":
                    continue
                msg = event.get("message") or {}
                for block in (msg.get("content") or []):
                    if block.get("type") == "text":
                        text = (block.get("text") or "").strip()
                        if text:
                            parts.append(text)
    except OSError:
        pass
    return "\n\n".join(parts) if parts else "(no critique produced)"


_PROMPT_SECTIONS_CACHE: dict[str, str] | None = None

import re

_SECTION_HEADER_RE = re.compile(r"^##\s+([A-Z][A-Z0-9_]*)\s*$")

def _load_prompt_sections() -> dict[str, str]:
    """Parse critique_prompt.txt into {SECTION_NAME: text} dict.

    Section delimiters must match `^##\\s+NAME$` where NAME is UPPER_SNAKE_CASE.
    This lets the body of each section use regular `##`-prefixed Markdown
    subheadings without being treated as new top-level sections.
    """
    global _PROMPT_SECTIONS_CACHE
    if _PROMPT_SECTIONS_CACHE is not None:
        return _PROMPT_SECTIONS_CACHE
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "critique_prompt.txt")
    with open(prompt_path) as f:
        content = f.read()
    sections: dict[str, str] = {}
    current_section: str | None = None
    lines: list[str] = []
    for line in content.splitlines():
        m = _SECTION_HEADER_RE.match(line)
        if m:
            if current_section is not None:
                sections[current_section] = "\n".join(lines).strip() + "\n"
            current_section = m.group(1)
            lines = []
        elif current_section is not None:
            lines.append(line)
    if current_section is not None:
        sections[current_section] = "\n".join(lines).strip() + "\n"
    _PROMPT_SECTIONS_CACHE = sections
    return sections


def _read_file(path: str, max_chars: int) -> str:
    """Read file, truncate from middle if oversized to keep prompt bounded."""
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
