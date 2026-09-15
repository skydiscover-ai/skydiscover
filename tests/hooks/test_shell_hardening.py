"""Tests for the shell entry points: the installer, the completion test hook, the keepalive watchdog.

Three failure modes are pinned here because each of them is silent in normal operation and expensive
when it fires:

* ``install.sh`` edits the user's GLOBAL ``~/.claude/settings.json``. It must never truncate it: the
  embedded writer takes a backup and does write-then-rename, so an interrupt leaves either the old
  file or the new one, never half of either.
* ``delivery_check.sh`` must fail CLOSED. Its "do not block" exit code is 1, which is also what
  ``set -e`` uses for an unexpected abort -- so any unhandled error once the task is known to be a
  checked delivery would wave an UNCHECKED impl through. It must exit 2 instead.
* ``keepalive.sh`` pastes its arguments into a systemd unit and a crontab line. A space or a shell
  metacharacter in either must be refused, not escaped-and-hoped-for.

Everything runs against a temp dir; no test touches the real ``$HOME``, crontab, or systemd.
"""

import json
import os
import pathlib
import shutil
import subprocess

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _ROOT / "skydiscover" / "synthesize" / "scripts"
_INSTALL = _SCRIPTS_DIR / "install.sh"
_HOOK = _ROOT / "skydiscover" / "synthesize" / "workflow" / "hooks" / "delivery_check.sh"
_KEEPALIVE = (
    _ROOT / "skydiscover" / "synthesize" / "workflow" / "scripts" / "unattended" / "keepalive.sh"
)
_SUPERVISOR = (
    _ROOT / "skydiscover" / "synthesize" / "workflow" / "scripts" / "unattended" / "supervisor.sh"
)

_SCRIPTS = [
    _INSTALL,
    _HOOK,
    _KEEPALIVE,
    _SUPERVISOR,
]


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


@pytest.mark.parametrize("script", _SCRIPTS, ids=lambda p: p.name)
def test_scripts_parse(script):
    """`bash -n` on every shell entry point -- a syntax error here breaks the install silently."""
    assert _run(["bash", "-n", str(script)]).returncode == 0


# install.sh settings writer


def _settings_writer() -> str:
    """The python heredoc embedded in install.sh that rewrites ~/.claude/settings.json.

    Extracted rather than reimplemented, and run against a temp path, so the test exercises the
    shipped code without symlinking into the repo or touching the real global config.
    """
    lines, out, inside = _INSTALL.read_text(encoding="utf-8").splitlines(), [], False
    for line in lines:
        if inside:
            if line == "PY":
                return "\n".join(out)
            out.append(line)
        elif line.endswith("<<'PY'"):
            inside = True
    raise AssertionError("no <<'PY' settings-writer heredoc found in install.sh")


def _write_settings(
    path,
    hook="bash /somewhere/delivery_check.sh",
    guard="python3 /somewhere/clone_reuse_guard.py",
    edit_guard="python3 /somewhere/framework_edit_guard.py",
):
    return _run(["python3", "-c", _settings_writer(), str(path), hook, guard, edit_guard])


def test_settings_writer_wires_the_framework_edit_guard_once(tmp_path):
    """Edit/Write/MultiEdit/NotebookEdit go through the framework-edit guard; a second run updates
    the command in place instead of adding a second entry."""
    path = tmp_path / "settings.json"
    assert _write_settings(path).returncode == 0
    cfg = json.loads(path.read_text())
    entries = [
        e
        for e in cfg["hooks"]["PreToolUse"]
        if any("framework_edit_guard" in h["command"] for h in e["hooks"])
    ]
    assert len(entries) == 1
    assert entries[0]["matcher"] == "Edit|Write|MultiEdit|NotebookEdit"
    assert (
        _write_settings(path, edit_guard="python3 /moved/framework_edit_guard.py").returncode == 0
    )
    cfg = json.loads(path.read_text())
    entries = [
        e
        for e in cfg["hooks"]["PreToolUse"]
        if any("framework_edit_guard" in h["command"] for h in e["hooks"])
    ]
    assert len(entries) == 1
    assert entries[0]["hooks"][0]["command"] == "python3 /moved/framework_edit_guard.py"


def test_settings_writer_syntax():
    compile(_settings_writer(), "install.sh:embedded", "exec")


def test_settings_writer_preserves_existing_config_and_backs_it_up(tmp_path):
    settings = tmp_path / "settings.json"
    original = {"theme": "dark", "env": {"MY_VAR": "keep-me"}, "hooks": {"Stop": [{"hooks": []}]}}
    settings.write_text(json.dumps(original, indent=2), encoding="utf-8")

    assert _write_settings(settings).returncode == 0

    cfg = json.loads(settings.read_text(encoding="utf-8"))
    assert cfg["theme"] == "dark"  # unrelated keys survive
    assert cfg["env"]["MY_VAR"] == "keep-me"  # unrelated env survives
    assert cfg["hooks"]["Stop"] == original["hooks"]["Stop"]  # unrelated hooks survive
    assert (
        "env" not in cfg or "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS" not in cfg["env"]
    )  # never flipped for the user
    assert "delivery_check.sh" in json.dumps(cfg["hooks"]["TaskCompleted"])
    assert all(
        h.get("timeout") == 3600
        for e in cfg["hooks"]["TaskCompleted"]
        for h in e["hooks"]
        if "delivery_check" in h["command"]
    )
    # the PreToolUse reuse guard is wired too, scoped to Bash
    pt = cfg["hooks"]["PreToolUse"]
    assert "clone_reuse_guard.py" in json.dumps(pt)
    assert any(e.get("matcher") == "Bash" for e in pt)

    baks = list(tmp_path.glob("settings.json.bak.*"))
    assert len(baks) == 1, "the user's global config must be backed up before it is modified"
    assert json.loads(baks[0].read_text(encoding="utf-8")) == original
    assert not list(tmp_path.glob("settings.json.tmp.*")), "temp file left behind"


def test_settings_writer_is_idempotent(tmp_path):
    """A second install must not rewrite the file or pile up backups."""
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    assert _write_settings(settings).returncode == 0
    first = settings.read_text(encoding="utf-8")

    assert _write_settings(settings).returncode == 0
    assert settings.read_text(encoding="utf-8") == first
    assert len(list(tmp_path.glob("settings.json.bak.*"))) == 1


def test_settings_writer_backs_up_an_unparseable_file(tmp_path):
    """A corrupt settings.json is replaced with a fresh config -- but the original is kept, since it
    is the only copy of whatever the user had."""
    settings = tmp_path / "settings.json"
    settings.write_text("{not json", encoding="utf-8")
    assert _write_settings(settings).returncode == 0
    assert json.loads(settings.read_text(encoding="utf-8"))["hooks"]
    (bak,) = tmp_path.glob("settings.json.bak.*")
    assert bak.read_text(encoding="utf-8") == "{not json"


# delivery_check.sh


def _gate(payload, env=None):
    clean = {k: v for k, v in os.environ.items() if not k.startswith("SKYDISCOVER_")}
    clean.update(env or {})
    return _run(["bash", str(_HOOK)], input=payload, env=clean)


def test_gate_ignores_an_ordinary_task():
    r = _gate(json.dumps({"title": "write the design note"}))
    assert r.returncode == 0


def test_gate_blocks_a_delivery_task_with_no_enforcement_configured():
    r = _gate(json.dumps({"title": "deliver the impl"}))
    assert r.returncode == 2
    assert "refusing to deliver unchecked" in r.stderr


def test_gate_blocks_when_enforcement_is_only_half_configured():
    """The env is the reliable delivery signal: any SKYDISCOVER_RUN / SKYDISCOVER_PRODREADY /
    SKYDISCOVER_IMPL set means "this is a checked delivery", and a missing run directory means we
    cannot check -- block, whatever the title says."""
    r = _gate(json.dumps({"title": "some unrelated task"}), {"SKYDISCOVER_IMPL": "/tmp/impl.h"})
    assert r.returncode == 2
    assert "refusing to deliver unchecked" in r.stderr


def test_gate_blocks_on_an_unparseable_payload():
    r = _gate("not json at all", {"SKYDISCOVER_PRODREADY": "1"})
    assert r.returncode == 2


def test_gate_exits_2_not_1_on_an_unexpected_abort(tmp_path):
    """The fail-closed guarantee. 1 means "do not block", so an abort AFTER the task is known to be a
    checked delivery must be converted to 2 -- otherwise a crash mid-check completes the delivery clean.

    The abort is induced without editing the script: BASH_ENV is sourced by every non-interactive
    bash, so a `pwd` that fails makes the hook's `repo_root="$(cd ... && pwd)"` fail the way a real
    unexpected error would. Without the ERR trap this exits 1.
    """
    ovr = tmp_path / "ovr.sh"
    ovr.write_text("pwd(){ return 1; }\n", encoding="utf-8")
    r = _gate(
        json.dumps({"title": "deliver the impl"}),
        {"BASH_ENV": str(ovr), "SKYDISCOVER_RUN": str(tmp_path)},
    )
    assert r.returncode == 2, f"fail-open: exit {r.returncode}\n{r.stderr}"
    assert "aborted unexpectedly" in r.stderr


def test_gate_reads_claude_codes_task_subject_and_the_phase_3_title():
    """Claude Code's TaskCompleted payload names the task in `task_subject`, and the skill's own
    Phase 3 task is titled "Phase 3: Final Deliverables". Both must be recognised as a delivery
    (with no run directory anywhere, recognition shows as the fail-closed block)."""
    for subject in ("Deliver the implementation to best/", "Phase 3: Final Deliverables"):
        r = _gate(
            json.dumps(
                {"hook_event_name": "TaskCompleted", "task_id": "3", "task_subject": subject}
            ),
        )
        assert r.returncode == 2, f"{subject!r}: exit {r.returncode}\n{r.stderr}"
        assert "refusing to deliver unchecked" in r.stderr


@pytest.mark.skipif(shutil.which("jq") is None, reason="needs jq")
def test_gate_reads_codex_subagent_stop_and_answers_it_in_json(tmp_path):
    """Codex's SubagentStop payload has no title: the delivery shows in last_assistant_message. On
    that event Codex rejects plain text on stdout, so a passing check is reported as JSON."""
    payload = json.dumps(
        {
            "hook_event_name": "SubagentStop",
            "agent_type": "coding-agent",
            "last_assistant_message": "Delivered the implementation to synthesis/impl/.",
        }
    )
    r = _gate(payload)
    assert r.returncode == 2 and "refusing to deliver unchecked" in r.stderr
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "python3").write_text("#!/bin/sh\necho 'TESTS PASSED (2 of 2)'\n", encoding="utf-8")
    (fake / "python3").chmod(0o755)
    r = _gate(
        payload,
        {"SKYDISCOVER_RUN": str(tmp_path), "PATH": f"{fake}:{os.environ['PATH']}"},
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout) == {"systemMessage": "TESTS PASSED (2 of 2)"}


def test_gate_finds_the_run_under_the_payloads_cwd_when_the_env_is_unset(tmp_path):
    """The lead's `export SKYDISCOVER_RUN` lives in its own shell; a hook process never sees it.
    The hook derives the run directory from the payload's cwd (<project>/.skydiscover/<run>/) and
    says which run it is checking. The run here is empty, so the check itself fails and blocks;
    what this proves is that the right directory was found."""
    run = tmp_path / ".skydiscover" / "demo"
    (run / "synthesis" / "impl").mkdir(parents=True)
    r = _gate(
        json.dumps(
            {
                "hook_event_name": "TaskCompleted",
                "task_subject": "Phase 3: Final Deliverables",
                "cwd": str(tmp_path),
            }
        )
    )
    assert f"checking the run at {run}" in r.stderr, r.stderr
    assert r.returncode == 2


def test_install_hook_commands_for_an_external_project_survive_a_moved_checkout(tmp_path):
    """Outside the repo, install.sh writes absolute hook commands. After the checkout moves they
    must not block: PreToolUse exit 2 would refuse every Bash call, and a failing TaskCompleted
    command would fail every task. Both must exit 0, the delivery hook saying why."""
    target = tmp_path / "proj"
    target.mkdir()
    r = _run(["bash", str(_INSTALL), str(target)])
    assert r.returncode == 0, r.stdout + r.stderr
    cfg = json.loads((target / ".claude" / "settings.local.json").read_text())
    hook = cfg["hooks"]["TaskCompleted"][0]["hooks"][0]["command"]
    guard = next(
        h["command"]
        for e in cfg["hooks"]["PreToolUse"]
        for h in e["hooks"]
        if "clone_reuse_guard" in h["command"]
    )
    moved = str(_ROOT)
    gone = str(tmp_path / "moved-away")
    g = _run(
        ["bash", "-c", guard.replace(moved, gone)],
        input='{"tool_name":"Bash","tool_input":{"command":"ls"}}',
    )
    assert g.returncode == 0, g.stderr
    h = _run(
        ["bash", "-c", hook.replace(moved, gone)],
        input='{"task_subject":"Phase 3: Final Deliverables"}',
    )
    assert h.returncode == 0, h.stderr
    assert "is missing" in h.stderr


# keepalive.sh validation


@pytest.mark.parametrize(
    "session,rundir",
    [
        ("sky", "/tmp/a run dir"),  # a space breaks the systemd unit and the cron line
        ("sky", "/tmp/run;touch /tmp/pwn"),  # command separator
        ("sky", "/tmp/run$(id)"),  # command substitution
        ("a b", "/tmp/run"),  # a space in the tmux session name
        ("sky;id", "/tmp/run"),  # ...which also names the systemd unit file
        ("sky\nid", "/tmp/run"),  # a newline injects a second line into the unit
    ],
)
def test_keepalive_refuses_unsafe_arguments(session, rundir, tmp_path):
    """These values are pasted into a systemd unit and a crontab line; refuse them outright.

    HOME is redirected at the temp dir so that a regression which installs anyway cannot write a unit
    file into the real ~/.config/systemd/user.
    """
    env = dict(os.environ, HOME=str(tmp_path))
    r = _run(["bash", str(_KEEPALIVE), session, rundir], env=env)
    assert r.returncode == 2, f"accepted an unsafe argument: {r.stdout}{r.stderr}"
    assert "refusing to install" in r.stderr
    assert not (tmp_path / ".config").exists(), "installed a systemd unit despite bad input"


# supervisor.sh termination


def _supervise(rundir, **overrides):
    """Run the supervisor against a run dir with no tmux session behind it. Both terminal paths are
    reached without ever talking to tmux, so this needs no session and cannot touch a real run."""
    env = dict(os.environ, SKYDISCOVER_SUP_POLL="1", **overrides)
    r = _run(
        ["bash", str(_SUPERVISOR), "sky", str(rundir), str(rundir / "nosuch.sock")],
        env=env,
        timeout=60,
    )
    return r, (rundir / ".skydiscover" / "supervisor.sky.log").read_text(encoding="utf-8")


def test_supervisor_exits_when_the_run_signals_completion(tmp_path):
    """A finished run looks exactly like a wedged one -- no new artifacts, an idle prompt. Without
    the .done marker `run finish` leaves, the supervisor would /compact it and SIGKILL-restart the
    lead forever."""
    (tmp_path / ".skydiscover").mkdir(parents=True)
    start = tmp_path / ".skydiscover" / "supervisor.sky.start"
    start.touch()
    os.utime(start, (0, 0))  # the session began long ago; the marker below postdates it
    (tmp_path / ".skydiscover" / "demo.done").write_text("outputs/synthesize/demo_1\n")

    r, log = _supervise(tmp_path)
    assert r.returncode == 0, "a nonzero exit would make systemd Restart= relaunch it"
    assert "run finished" in log
    assert (tmp_path / ".skydiscover" / "supervisor.sky.stop").is_file()
    assert not start.exists(), "the start stamp is cleared when the session ends"


def test_supervisor_ignores_an_earlier_runs_done_marker(tmp_path):
    """A project that already finished one run keeps its <slug>.done; a new session must not read
    it as its own completion."""
    (tmp_path / ".skydiscover").mkdir(parents=True)
    stale = tmp_path / ".skydiscover" / "old.done"
    stale.write_text("outputs/synthesize/old_1\n")
    os.utime(stale, (0, 0))

    r, log = _supervise(tmp_path, SKYDISCOVER_SUP_MAX_RESTARTS="0")
    assert r.returncode == 0
    assert "GIVING UP" in log and "run finished" not in log


def test_supervisor_gives_up_after_fruitless_restarts(tmp_path):
    """No completion signal and no progress: the loop must still terminate, with a diagnostic."""
    (tmp_path / ".skydiscover").mkdir()

    r, log = _supervise(tmp_path, SKYDISCOVER_SUP_MAX_RESTARTS="0")
    assert r.returncode == 0
    assert "GIVING UP" in log
    assert (tmp_path / ".skydiscover" / "supervisor.sky.stop").is_file()


def test_keepalive_refuses_a_missing_run_dir(tmp_path):
    env = dict(os.environ, HOME=str(tmp_path))
    r = _run(["bash", str(_KEEPALIVE), "sky", str(tmp_path / "nope")], env=env)
    assert r.returncode == 2
    assert "no such run dir" in r.stderr


def test_keepalive_stands_down_once_the_run_is_finished(tmp_path):
    """The supervisor's terminal stop marker must stop the watchdog from relaunching it forever."""
    (tmp_path / ".skydiscover").mkdir()
    (tmp_path / ".skydiscover" / "supervisor.sky.stop").touch()
    env = dict(os.environ, HOME=str(tmp_path))
    r = _run(["bash", str(_KEEPALIVE), "sky", str(tmp_path)], env=env)
    assert r.returncode == 1
    assert "already marked finished" in r.stdout


def test_settings_writer_migrates_a_stale_gate_hook_path_in_place(tmp_path):
    """An upgrade rewrites an existing hook entry whose command points at an older install path, in
    place: not left stale (it would run a missing script and block nothing) and not duplicated."""
    settings = tmp_path / "settings.json"
    old = "bash /OLD/skydiscover/synthesize/hooks/delivery_check.sh"
    new = "bash /NEW/skydiscover/synthesize/workflow/hooks/delivery_check.sh"
    settings.write_text(
        json.dumps(
            {"hooks": {"TaskCompleted": [{"hooks": [{"type": "command", "command": old}]}]}}
        ),
        encoding="utf-8",
    )

    assert _write_settings(settings, hook=new).returncode == 0

    tc = json.loads(settings.read_text(encoding="utf-8"))["hooks"]["TaskCompleted"]
    cmds = [h["command"] for e in tc for h in e.get("hooks", [])]
    assert cmds == [
        new
    ], f"stale path must be rewritten in place to exactly one current entry, got {cmds}"
    assert old not in json.dumps(tc), "the stale old path must be gone"


def test_settings_writer_leaves_a_current_gate_hook_untouched(tmp_path):
    """When the existing entry already points at the current path, a re-install rewrites nothing and
    adds no backup (no spurious change)."""
    settings = tmp_path / "settings.json"
    cur = "bash /NEW/skydiscover/synthesize/workflow/hooks/delivery_check.sh"
    curg = "python3 /NEW/skydiscover/synthesize/workflow/hooks/clone_reuse_guard.py"
    cure = "python3 /NEW/skydiscover/synthesize/workflow/hooks/framework_edit_guard.py"
    # A fully-current file: all current hooks already present with their timeouts, so nothing changes.
    settings.write_text(
        json.dumps(
            {
                "hooks": {
                    "TaskCompleted": [
                        {"hooks": [{"type": "command", "command": cur, "timeout": 3600}]}
                    ],
                    "PreToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [{"type": "command", "command": curg, "timeout": 60}],
                        },
                        {
                            "matcher": "Edit|Write|MultiEdit|NotebookEdit",
                            "hooks": [{"type": "command", "command": cure, "timeout": 60}],
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    assert _write_settings(settings, hook=cur, guard=curg, edit_guard=cure).returncode == 0
    tc = json.loads(settings.read_text(encoding="utf-8"))["hooks"]["TaskCompleted"]
    cmds = [h["command"] for e in tc for h in e.get("hooks", [])]
    assert cmds == [cur], "no duplicate entry on re-install"
    assert not list(tmp_path.glob("settings.json.bak.*")), "no backup when nothing changed"


def test_settings_writer_migrates_a_stale_guard_hook_path_in_place(tmp_path):
    """The same for the PreToolUse reuse guard: an existing entry at an older script path is
    rewritten, never left stale and never duplicated."""
    settings = tmp_path / "settings.json"
    old = "python3 /OLD/skydiscover/synthesize/hooks/clone_reuse_guard.py"
    new = "python3 /NEW/skydiscover/synthesize/workflow/hooks/clone_reuse_guard.py"
    settings.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {"matcher": "Bash", "hooks": [{"type": "command", "command": old}]}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    assert _write_settings(settings, guard=new).returncode == 0

    pt = json.loads(settings.read_text(encoding="utf-8"))["hooks"]["PreToolUse"]
    cmds = [
        h["command"]
        for e in pt
        for h in e.get("hooks", [])
        if "clone_reuse_guard.py" in h["command"]
    ]
    assert cmds == [
        new
    ], f"stale guard path must be rewritten in place to one current entry, got {cmds}"
    assert old not in json.dumps(pt), "the stale old guard path must be gone"
