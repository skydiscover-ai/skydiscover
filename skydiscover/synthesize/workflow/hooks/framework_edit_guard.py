#!/usr/bin/env python3
"""PreToolUse guard: the agents may not edit SkyDiscover itself.

A run's agents own the run directory and the project they were started in, never the framework
that checks their work. A lead that hits a framework bug must report it (decision log, closing
message) rather than patch the installed skill: a patched checkout produces a result the release
never would, silently. Wired for the Edit / Write / MultiEdit / NotebookEdit tools; exit 2 blocks
the edit and returns the reason to the agent, exit 0 lets it through. Inside the SkyDiscover
repository itself (a developer working on the framework) nothing is blocked.

The payload arrives on stdin as JSON: {"tool_name": ..., "tool_input": {"file_path": ...}, "cwd": ...}.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}


def framework_root() -> Path:
    """The installed framework this hook belongs to: <...>/skydiscover/synthesize/."""
    return Path(__file__).resolve().parent.parent.parent


def repo_root() -> Path:
    """The source checkout the framework lives in (the parent of the `skydiscover` package)."""
    return framework_root().parent.parent


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def decide(payload: dict) -> tuple[bool, str]:
    """(block?, message)."""
    if payload.get("tool_name") not in EDIT_TOOLS:
        return False, ""
    tool_input = payload.get("tool_input") or {}
    raw = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
    if not raw:
        return False, ""
    cwd = Path(payload.get("cwd") or os.getcwd())
    target = Path(raw)
    if not target.is_absolute():
        target = cwd / target
    target = target.resolve()
    root = framework_root()
    if not _inside(target, root):
        return False, ""
    # A developer editing the framework in its own repository is not a run patching its checker.
    if _inside(cwd.resolve(), repo_root()):
        return False, ""
    rel = target.relative_to(root)
    return True, (
        f"framework-guard: {rel} is part of SkyDiscover, not of this run. The agents may not change "
        "the framework that checks their work; if it is wrong, record the problem in the decision "
        "log and say so in the closing message, and finish with what the released code produces."
    )


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            return 0
    except Exception:
        return 0  # unreadable payload -> never block
    try:
        block, msg = decide(payload)
    except Exception:
        return 0  # a bug in the guard must never block an edit
    if block:
        sys.stderr.write(msg + "\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
