"""Every docker call in the evaluators is time-bounded.

An unattended search runs for hours. A stalled `docker build` or a wedged
daemon must fail the run with a diagnosable error rather than hang it
forever, so no `subprocess` call in the evaluators may omit `timeout=`.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

from skydiscover.optimize.config import EvaluatorConfig

EVALUATORS = [
    Path("skydiscover/optimize/evaluation/container_evaluator.py"),
    Path("skydiscover/optimize/evaluation/harbor_evaluator.py"),
]
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("rel", EVALUATORS, ids=lambda p: p.name)
def test_no_unbounded_subprocess_call(rel):
    tree = ast.parse((ROOT / rel).read_text())
    unbounded = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"run", "check_output", "call"}
        and getattr(node.func.value, "id", None) == "subprocess"
        and not any(kw.arg == "timeout" for kw in node.keywords)
    ]
    assert not unbounded, f"{rel} has subprocess calls without timeout= at lines {unbounded}"


def test_defaults_separate_builds_from_admin_calls():
    cfg = EvaluatorConfig()
    # A build may pull and compile layers; an exec/cp should never take minutes.
    assert cfg.container_build_timeout > cfg.container_setup_timeout
    assert cfg.container_setup_timeout > 0


def test_build_timeout_raises_a_diagnosable_error(monkeypatch, tmp_path):
    """A hung `docker build` surfaces as RuntimeError naming the knob to raise."""
    from skydiscover.optimize.evaluation import container_evaluator as ce

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="docker build", timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr(ce.subprocess, "run", fake_run)
    cfg = EvaluatorConfig()
    ev = ce.ContainerizedEvaluator.__new__(ce.ContainerizedEvaluator)
    ev.benchmark_dir = str(tmp_path)
    ev.config = cfg

    with pytest.raises(RuntimeError) as exc:
        ev._build_image()
    msg = str(exc.value)
    assert "timed out" in msg
    assert "container_build_timeout" in msg
