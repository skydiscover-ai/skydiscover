"""The pre-commit hooks must run the same lint tools as CI.

The config exists so a clean commit is a green CI run. That only holds while the
two stay in step -- add a tool to `ci.yml` and forget the hook, and pre-commit
starts quietly passing things CI will reject.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ("black", "isort", "mypy")


def _ci_lint_tools() -> dict[str, str]:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    found = {}
    for step in workflow["jobs"]["lint"]["steps"]:
        command = step.get("run", "")
        for tool in TOOLS:
            if re.search(rf"\b{tool}\b", command):
                found[tool] = command
    return found


def _pre_commit_hooks() -> dict[str, dict]:
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    return {
        hook["id"]: hook
        for repo in config["repos"]
        if repo["repo"] == "local"
        for hook in repo["hooks"]
    }


def test_every_ci_lint_tool_has_a_hook():
    missing = sorted(set(_ci_lint_tools()) - set(_pre_commit_hooks()))
    assert not missing, f"lint tools in CI with no pre-commit hook: {missing}"


def test_no_hook_runs_a_tool_ci_does_not():
    extra = sorted(set(_pre_commit_hooks()) - set(_ci_lint_tools()))
    assert not extra, f"pre-commit hooks with no CI counterpart: {extra}"


def test_hooks_use_the_projects_pinned_versions():
    """`uv run` means uv.lock is the single source of truth for tool versions."""
    for hook_id, hook in _pre_commit_hooks().items():
        assert hook["entry"].startswith("uv run "), f"{hook_id} bypasses uv.lock: {hook['entry']}"


def test_formatter_scopes_match_ci():
    """black/isort must cover the same directories CI checks."""
    ci = _ci_lint_tools()
    hooks = _pre_commit_hooks()
    for tool in ("black", "isort"):
        ci_dirs = set(re.findall(r"(\w+)/", ci[tool].split("--check")[-1]))
        hook_dirs = set(re.findall(r"(\w+)", hooks[tool]["files"]))
        assert ci_dirs <= hook_dirs, f"{tool}: CI checks {ci_dirs}, hook covers {hook_dirs}"


def test_editorconfig_agrees_with_black():
    text = (ROOT / ".editorconfig").read_text()
    assert "max_line_length = 100" in text, "must match black's line-length in pyproject.toml"
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert "line-length = 100" in pyproject
