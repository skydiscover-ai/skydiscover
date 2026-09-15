"""CONTRIBUTING must list the commands CI actually runs.

It promises "a green local run means a green CI run". That only holds if the two
stay in step -- and they had already drifted: CI gained a `mypy` step and widened
black/isort to cover benchmarks/, while CONTRIBUTING still showed the old pair.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _ci_lint_commands() -> list[str]:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    return [
        step["run"]
        for step in workflow["jobs"]["lint"]["steps"]
        if "run" in step and "uv sync" not in step["run"]
    ]


def test_contributing_shows_every_ci_lint_command():
    contributing = (ROOT / "CONTRIBUTING.md").read_text()
    missing = []
    for command in _ci_lint_commands():
        # CONTRIBUTING shows the fixing form (`black X`), CI the checking form
        # (`black --check X`); compare on tool plus paths.
        spoken = command.replace(" --check", "")
        if spoken not in contributing:
            missing.append(f"{command}  (expected to find: {spoken!r})")
    assert not missing, "CONTRIBUTING is out of step with the CI lint job:\n  " + "\n  ".join(
        missing
    )


def test_contributing_shows_the_ci_test_command():
    contributing = (ROOT / "CONTRIBUTING.md").read_text()
    assert 'pytest tests/ -m "not slow and not integration"' in contributing
