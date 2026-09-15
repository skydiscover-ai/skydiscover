"""skydiscover init wires the coding agents it finds; with none found it says so and touches nothing."""

from __future__ import annotations

import skydiscover.main as entry


def test_detect_agents_follows_path(monkeypatch):
    present = {"claude", "codex"}
    monkeypatch.setattr(entry.shutil, "which", lambda b: f"/bin/{b}" if b in present else None)
    assert entry.detect_agents() == ["claude", "codex"]


def test_detect_agents_accepts_any_cursor_binary(monkeypatch):
    monkeypatch.setattr(entry.shutil, "which", lambda b: "/bin/agent" if b == "agent" else None)
    assert entry.detect_agents() == ["cursor"]


def test_auto_with_no_agents_exits_2_and_creates_nothing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(entry.shutil, "which", lambda b: None)
    target = tmp_path / "proj"
    assert entry.main(["init", "--path", str(target)]) == 2
    assert not target.exists()
    assert "--agent claude|cursor|codex|pi" in capsys.readouterr().err


def test_auto_wires_each_detected_agent(monkeypatch, tmp_path):
    monkeypatch.setattr(
        entry.shutil, "which", lambda b: "/bin/x" if b in ("claude", "pi") else None
    )
    wired = []
    monkeypatch.setattr(entry, "_wire", lambda agent, project, no_hook: wired.append(agent))
    assert entry.main(["init", "--path", str(tmp_path / "p")]) == 0
    assert wired == ["claude", "pi"]
    assert (tmp_path / "p" / ".skydiscover").is_dir()


def test_explicit_agent_ignores_path(monkeypatch, tmp_path):
    monkeypatch.setattr(entry.shutil, "which", lambda b: None)
    wired = []
    monkeypatch.setattr(entry, "_wire", lambda agent, project, no_hook: wired.append(agent))
    assert entry.main(["init", "--agent", "codex", "--path", str(tmp_path)]) == 0
    assert wired == ["codex"]
