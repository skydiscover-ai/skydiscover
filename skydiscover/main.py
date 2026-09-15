"""The skydiscover command: one entry point per module.

    skydiscover optimize <program> <evaluator> --search evox   # Optimize
    skydiscover init                                           # Synthesize: wire /skysynth into your coding agent
    skydiscover viewer <checkpoint>                            # replay a finished Optimize run

optimize and viewer forward everything after the subcommand to skydiscover.optimize.cli and
skydiscover.optimize.extras.monitor.viewer.

init installs the /skysynth skill, its agent roles, and the hook that runs the tests before anything
is delivered into the coding agents it finds on PATH (Claude Code, Cursor, Codex, pi), or the one named with --agent, and creates the
.skydiscover/ working directory. The wiring is symlinks into the installed package, so re-run init
after upgrading skydiscover. Wiring one agent leaves the others untouched.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from importlib import import_module, resources
from pathlib import Path
from typing import List


def _painter():
    """Return paint(code, text) that adds ANSI color only on a real TTY (honors NO_COLOR / TERM=dumb)."""
    use = sys.stdout.isatty() and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"
    return (lambda code, s: f"\033[{code}m{s}\033[0m") if use else (lambda code, s: s)


def _kit_root() -> Path:
    """The synthesize kit shipped inside the installed package (``skydiscover/synthesize``)."""
    return Path(str(resources.files("skydiscover"))) / "synthesize"


AGENTS = ("claude", "cursor", "codex", "pi")

# Executables that mean a coding agent is installed. Cursor ships two CLIs.
_AGENT_BINARIES = {
    "claude": ("claude",),
    "cursor": ("cursor-agent", "agent", "cursor"),
    "codex": ("codex",),
    "pi": ("pi",),
}


def detect_agents() -> List[str]:
    """The supported coding agents found on PATH, in canonical order."""
    return [a for a in AGENTS if any(shutil.which(b) for b in _AGENT_BINARIES[a])]


def _wire(agent: str, project: Path, no_hook: bool) -> None:
    """Wire the skill, the agent roles, and the hooks for one coding agent through install.sh."""
    installer = _kit_root() / "scripts" / "install.sh"
    workflow = _kit_root() / "workflow" / "SKILL.md"
    if not installer.is_file() or not workflow.is_file():
        raise FileNotFoundError(
            f"skydiscover init cannot find the skill at {workflow.parent}; "
            "reinstall skydiscover, or run init from a source checkout"
        )
    args = ["bash", str(installer), "--agent", agent]
    if no_hook:
        args.append("--no-hook")
    args.append(str(project))
    # The installer narrates each step on stdout; its warnings (a python3 that cannot import
    # skydiscover, a missing tool, a settings file it backed up) must reach the user, and a failed
    # install must read as one, not as a traceback.
    proc = subprocess.run(args, capture_output=True, text=True, errors="replace")
    for line in (proc.stdout or "").splitlines():
        if line.startswith(("WARNING", "backed up", "Could not write")):
            print(line, file=sys.stderr)
    if proc.returncode:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise SystemExit(
            f"skydiscover init: the installer failed (exit {proc.returncode})"
            + (f":\n{detail}" if detail else "")
        )


# Subcommands that own their own flag namespace. Everything after the verb is
# forwarded verbatim, so `skydiscover optimize --help` shows the Optimize help
# rather than this parser's. They are dispatched before argparse runs, because
# argparse.REMAINDER lets the parent parser swallow a leading option like --help.
_FORWARDED = {
    "optimize": (
        "skydiscover.optimize.cli",
        "run a discovery search against an evaluator (Optimize)",
    ),
    "viewer": (
        "skydiscover.optimize.extras.monitor.viewer",
        "replay a completed run in the monitor UI",
    ),
}


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="skydiscover",
        description="SkyDiscover: AI-driven scientific, algorithmic, and end-to-end systems discovery.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="<command>")

    for name, (_module, help_text) in _FORWARDED.items():
        sub.add_parser(name, help=help_text, add_help=False)

    p = sub.add_parser(
        "init",
        help="set up /skysynth in your coding agent: the skill, the agent roles, and the hook "
        "that runs the tests before anything is delivered",
    )
    p.add_argument("--path", default=".", help="project directory to set up (default: cwd)")
    p.add_argument(
        "--no-hook",
        action="store_true",
        help="don't install the hook that runs the tests before anything is delivered",
    )
    p.add_argument(
        "--agent",
        choices=[*AGENTS, "all", "auto"],
        default="auto",
        help="which coding agent to wire (default: auto, every supported agent found on PATH; "
        "'all' wires all four)",
    )
    return ap


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] in _FORWARDED:
        module_name, _help = _FORWARDED[argv[0]]
        entry = import_module(module_name).main
        return int(entry(argv[1:], prog=f"skydiscover {argv[0]}") or 0)

    return _run_init(_build_parser().parse_args(argv))


def _run_init(args: argparse.Namespace) -> int:
    project = Path(args.path).resolve()

    if args.agent == "all":
        agents = list(AGENTS)
    elif args.agent == "auto":
        agents = detect_agents()
        if not agents:
            print(
                "skydiscover init: no supported coding agent found on PATH (claude, cursor-agent, "
                "codex, pi).\nInstall one, or choose explicitly: skydiscover init --agent "
                + "|".join(AGENTS),
                file=sys.stderr,
            )
            return 2
    else:
        agents = [args.agent]
    # `init --path <new dir>` is expected to set that directory up.
    project.mkdir(parents=True, exist_ok=True)
    for agent in agents:
        _wire(agent, project, no_hook=args.no_hook)
    from skydiscover.synthesize.spec.paths import outputs, runs

    runs_dir = runs()  # .skydiscover unless config.toml or $SKYDISCOVER_RUNS says otherwise
    (runs_dir if runs_dir.is_absolute() else project / runs_dir).mkdir(parents=True, exist_ok=True)

    names = {"claude": "Claude Code", "cursor": "Cursor", "codex": "Codex", "pi": "pi"}
    wired = [names[a] for a in agents]
    where = (
        wired[0]
        if len(wired) == 1
        else ", ".join(wired[:-1]) + (" and " if len(wired) == 2 else ", and ") + wired[-1]
    )
    how = (
        "open it in this folder and run:"
        if len(wired) == 1
        else "open any of them in this folder and run:"
    )
    if agents == ["pi"]:
        how = "open pi in this folder (trust it once) and run:"

    paint = _painter()
    print()
    print(f"  {paint('32;1', '✓')}  {paint('1', f'skydiscover is ready in {where}')}")
    print(
        f"     {paint('2', 'build a system specialized for your workload, hardware, and requirements')}"
    )
    print()
    print(f"  Try it: {how}")
    print(f"     {paint('36;1', '/skysynth')}  a fast in-memory key-value store")
    print(f"  The result lands in {outputs()}/<name>_<timestamp>/best/; start with spec.md there.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
