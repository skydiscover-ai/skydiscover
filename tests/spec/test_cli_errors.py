"""Every `python3 -m skydiscover.synthesize.spec.*` tool answers a predictable input error with one
line and exit code 2, never a traceback. Tests elsewhere call main() directly and still see the
exception; this is the boundary a user or an agent hits."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "SKYDISCOVER_HOME": str(tmp_path / "home"), "PYTHONPATH": str(REPO)}
    return subprocess.run(
        [sys.executable, "-m", *args], cwd=tmp_path, env=env, capture_output=True, text=True
    )


@pytest.mark.parametrize(
    "args",
    [
        ("skydiscover.synthesize.spec.paths", "domain", "shared"),
        ("skydiscover.synthesize.spec.checkpoint", "snapshot", "nowhere"),
        (
            "skydiscover.synthesize.spec.build",
            "nowhere/spec.json",
            "--answers",
            "a.json",
            "-d",
            "cards",
        ),
        ("skydiscover.synthesize.spec.requirements", "check", "nowhere.json"),
        ("skydiscover.synthesize.spec.run", "check", "nowhere"),
    ],
)
def test_a_missing_or_reserved_input_is_one_line_and_exit_2(tmp_path, args):
    r = _run(tmp_path, *args)
    assert r.returncode == 2, r.stderr
    assert "Traceback" not in r.stderr and "Traceback" not in r.stdout
    assert r.stderr.strip().count("\n") == 0, r.stderr


def test_a_corrupt_decision_log_is_reported_not_raised(tmp_path):
    run = tmp_path / "r"
    (run / "specification").mkdir(parents=True)
    (run / "decision_log.json").write_text("{bad", encoding="utf-8")
    for tool, *rest in (
        (
            "skydiscover.synthesize.spec.findings",
            "add",
            str(run),
            "--title",
            "t",
            "--kind",
            "spec",
            "--detail",
            "d",
        ),
        ("skydiscover.synthesize.spec.decisions", str(run), "list"),
    ):
        r = _run(tmp_path, tool, *rest)
        assert r.returncode == 2 and "Traceback" not in r.stderr, r.stderr
        assert "not valid JSON" in r.stderr


def test_an_unknown_kind_is_rejected_by_argparse(tmp_path):
    r = _run(
        tmp_path,
        "skydiscover.synthesize.spec.findings",
        "add",
        "r",
        "--title",
        "t",
        "--kind",
        "bogus",
        "--detail",
        "d",
    )
    assert r.returncode == 2
    assert "invalid choice: 'bogus'" in r.stderr
