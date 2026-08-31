"""run_command tool – execute a sandboxed shell command inside the codebase root.

The agent calls this tool with a shell command string.  The tool:
  1. Validates the command against a safety allowlist.
  2. Runs it as a subprocess with a hard wall-clock timeout.
  3. Captures stdout + stderr and returns them as text (truncated if huge).

Design goals
------------
* Useful for data exploration: ``python script.py``, ``head -n 50 data.csv``,
  ``wc -l *.py``, ``cat results.json``, etc.
* NOT a general shell: destructive or network commands are blocked.
* CWD is always pinned to ``codebase_root``; absolute paths outside it are
  also blocked.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safety configuration
# ---------------------------------------------------------------------------

# Maximum wall-clock seconds the subprocess may run (default; can be overridden by config/tool args).
DEFAULT_TIMEOUT = 100

# Maximum characters of combined stdout+stderr returned to the agent (default; can be overridden by config).
DEFAULT_MAX_OUTPUT_CHARS = 20_000

# Executables that are explicitly allowed.  Anything not in this set is
# rejected so the agent cannot run ``rm``, ``curl``, ``wget``, etc.
_ALLOWED_EXECUTABLES: frozenset[str] = frozenset(
    {
        # Python
        "python",
        "python3",
        "python3.10",
        "python3.11",
        "python3.12",
        # Data inspection
        "cat",
        "head",
        "tail",
        "less",
        "more",
        "wc",
        "grep",
        "egrep",
        "fgrep",
        "rg",          # ripgrep
        "find",
        "ls",
        "du",
        "df",
        "stat",
        # Text processing
        "sort",
        "uniq",
        "cut",
        "awk",
        "sed",
        "tr",
        "paste",
        "join",
        "diff",
        "comm",
        "xxd",
        "jq",
        # Computation / data science helpers
        "Rscript",
        "julia",
        "node",
        # Archive inspection (read-only)
        "tar",
        "unzip",
        "zipinfo",
        # Misc safe read-only tools
        "echo",
        "printf",
        "date",
        "whoami",
        "uname",
        "env",
        "pwd",
    }
)

# These substrings in the raw command string are unconditionally blocked
# regardless of the executable, to prevent shell injection via pipes /
# redirection that write outside the sandbox.
# Shell metacharacters — only dangerous in shell=True mode (unsafe).
# With shell=False these are harmless characters in argv strings.
_SHELL_PATTERNS: list[re.Pattern] = [
    re.compile(r"[|&;`]"),
    re.compile(r"\$[({]"),
    re.compile(r">{1,2}"),
    re.compile(r"<\("),
]

# Destructive / network executables — blocked in both modes.
_BLOCKED_EXECUTABLES: list[re.Pattern] = [
    re.compile(r"\brm\b"),
    re.compile(r"\bchmod\b"),
    re.compile(r"\bchown\b"),
    re.compile(r"\bsudo\b"),
    re.compile(r"\bsu\b"),
    re.compile(r"\bcurl\b"),
    re.compile(r"\bwget\b"),
    re.compile(r"\bscp\b"),
    re.compile(r"\brsync\b"),
    re.compile(r"\bssh\b"),
    re.compile(r"\bnc\b"),
    re.compile(r"\bkill\b"),
    re.compile(r"\bpkill\b"),
    re.compile(r"\bmv\b"),
    re.compile(r"\bcp\b"),
    re.compile(r"\bdd\b"),
    re.compile(r"\bmkdir\b"),
    re.compile(r"\brmdir\b"),
    re.compile(r"\btouch\b"),
    re.compile(r"\bln\b"),
    re.compile(r"\binstall\b"),
    re.compile(r"\bapt\b"),
    re.compile(r"\bpip\b"),
    re.compile(r"\bnpm\b"),
    re.compile(r"\bconda\b"),
]


def _check_command_safety(command: str, *, unsafe: bool = False) -> str | None:
    """Return an error string if *command* is unsafe, else None."""
    # Shell metacharacters only matter when running via shell=True.
    # With shell=False, subprocess passes argv directly — no redirects/pipes.
    if unsafe:
        for pat in _SHELL_PATTERNS:
            if pat.search(command):
                return (
                    f"Command blocked: contains forbidden pattern '{pat.pattern}'. "
                    "Only read-only, non-network commands without shell operators are allowed."
                )

    # Parse the first token as the executable name.
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        return f"Could not parse command: {exc}"

    if not tokens:
        return "Empty command."

    exe = os.path.basename(tokens[0])

    # Destructive / network executables are blocked regardless of mode.
    for pat in _BLOCKED_EXECUTABLES:
        if pat.fullmatch(exe):
            return (
                f"Command blocked: '{exe}' is not allowed. "
                "Only read-only, non-network commands are allowed."
            )

    if exe not in _ALLOWED_EXECUTABLES:
        allowed_sorted = ", ".join(sorted(_ALLOWED_EXECUTABLES))
        return (
            f"Executable '{exe}' is not in the allowed list. "
            f"Allowed executables: {allowed_sorted}"
        )

    return None


# ---------------------------------------------------------------------------
# Core execution logic
# ---------------------------------------------------------------------------


def _default_shell_executable() -> str | None:
    """Pick a reasonable shell executable for shell=True mode."""
    for candidate in ("/bin/bash", "/usr/bin/bash", "/bin/sh", "/usr/bin/sh"):
        if os.path.exists(candidate):
            return candidate
    return None


def run_command_sync(
    command: str,
    codebase_root: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    unsafe: bool = False,
) -> dict[str, Any]:
    """Run *command* synchronously.

    Returns a dict with keys:
      stdout      – captured stdout (str)
      stderr      – captured stderr (str)
      returncode  – process exit code (int)
      truncated   – True if output was trimmed (bool)
      error       – set only on OS-level failure (str)
    """
    try:
        if unsafe:
            result = subprocess.run(
                command,
                cwd=codebase_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=True,
                executable=_default_shell_executable(),
            )
        else:
            tokens = shlex.split(command)
            result = subprocess.run(
                tokens,
                cwd=codebase_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                # Intentionally NO shell=True – avoids shell injection.
            )
    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out after {timeout}s.", "returncode": -1}
    except FileNotFoundError as exc:
        return {"error": f"Executable not found: {exc}", "returncode": -1}
    except OSError as exc:
        return {"error": f"OS error running command: {exc}", "returncode": -1}

    combined = ""
    if result.stdout:
        combined += result.stdout
    if result.stderr:
        combined += "\n[stderr]\n" + result.stderr

    truncated = False
    if len(combined) > max_output_chars:
        half = max_output_chars // 2
        combined = (
            combined[:half]
            + f"\n\n... ({len(combined) - max_output_chars} chars truncated) ...\n\n"
            + combined[-half:]
        )
        truncated = True

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "combined": combined,
        "returncode": result.returncode,
        "truncated": truncated,
    }


# ---------------------------------------------------------------------------
# Async handler (called from agentic_generator._run_tool)
# ---------------------------------------------------------------------------


async def run_command_handler(
    arguments: dict[str, Any],
    codebase_root: str,
    run_command_default_timeout: int = DEFAULT_TIMEOUT,
    run_command_max_timeout: int = 120,
    run_command_max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    allow_unsafe_commands: bool = False,
    **_kw: Any,
) -> tuple[str, bool]:
    """Async handler called by the agentic loop."""
    command = arguments.get("command", "").strip()
    requested_timeout = arguments.get("timeout", run_command_default_timeout)
    timeout = int(requested_timeout)
    timeout = max(1, min(timeout, int(run_command_max_timeout)))  # clamp
    max_output_chars = int(run_command_max_output_chars)
    max_output_chars = max(1_000, min(max_output_chars, 500_000))
    unsafe = bool(arguments.get("unsafe", False))

    if not command:
        return "Error: run_command requires a 'command' argument.", False
    if not codebase_root:
        return "Error: codebase_root not configured.", False

    # Safety check (skipped only in explicit unsafe mode)
    if unsafe and not allow_unsafe_commands:
        return (
            "Error: unsafe=true was requested but agentic.allow_unsafe_commands is disabled.",
            False,
        )
    if not unsafe:
        safety_err = _check_command_safety(command)
        if safety_err:
            return f"Error: {safety_err}", False

    logger.info("run_command: executing %r in %s (timeout=%ds)", command, codebase_root, timeout)

    result = await asyncio.to_thread(
        run_command_sync,
        command,
        codebase_root,
        timeout=timeout,
        max_output_chars=max_output_chars,
        unsafe=unsafe,
    )

    if "error" in result:
        return f"Error running command: {result['error']}", False

    rc = result["returncode"]
    output = result["combined"] or "(no output)"
    trunc_note = "\n[Output was truncated]" if result.get("truncated") else ""

    mode = "shell" if unsafe else "exec"
    msg = (
        f"$ {command}\n"
        f"[mode: {mode}]\n"
        f"[exit code: {rc}]\n\n"
        f"{output}"
        f"{trunc_note}"
    )
    success = rc == 0
    return msg, success
