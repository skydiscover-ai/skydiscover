"""A plugin install copies workflow/ on its own; the scripts and hooks then find spec/ through the
installed skydiscover package instead of the sibling folder of a source checkout."""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
WORKFLOW = REPO / "skydiscover" / "synthesize" / "workflow"


@pytest.fixture(scope="module")
def copied_workflow(tmp_path_factory):
    dest = tmp_path_factory.mktemp("plugin") / "workflow"
    shutil.copytree(
        WORKFLOW, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), symlinks=True
    )
    return dest


def _run(script: pathlib.Path, *args: str) -> subprocess.CompletedProcess:
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO), "HOME": str(script.parents[3])}
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(script.parents[3]),
    )


@pytest.mark.parametrize(
    "rel",
    [
        "scripts/run_tests.py",
        "scripts/validate_test.py",
        "scripts/check_release.py",
        "scripts/kb/kbtool.py",
    ],
)
def test_every_script_starts_without_a_sibling_spec_folder(copied_workflow, rel):
    r = _run(copied_workflow / rel, "--help")
    assert "Traceback" not in r.stderr and "No module named" not in r.stderr, r.stderr
    assert r.returncode == 0


def test_the_clone_guard_resolves_the_knowledge_base_root_without_a_sibling_spec(copied_workflow):
    r = _run(copied_workflow / "hooks" / "clone_reuse_guard.py")
    assert r.returncode == 0, r.stderr  # an empty payload is allowed, never a traceback
