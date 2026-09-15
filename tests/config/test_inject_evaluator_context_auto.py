"""inject_evaluator_context defaults to auto, resolved by the Runner.

Without a seed program the model has nothing but the evaluator to learn the
task from, and a Harbor task's instruction.md exists precisely to be shown to
the model — so both must resolve to on. Seeded non-Harbor runs stay off, and
an explicit true/false is never overridden.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from skydiscover.optimize.config import Config
from skydiscover.optimize.runner import Runner


@pytest.fixture
def evaluator_file(tmp_path: Path) -> str:
    path = tmp_path / "evaluator.py"
    path.write_text("def evaluate(program_path):\n    return {'combined_score': 0.0}\n")
    return str(path)


@pytest.fixture
def seed_file(tmp_path: Path) -> str:
    path = tmp_path / "initial_program.py"
    path.write_text("def solve(x):\n    return x\n")
    return str(path)


@pytest.fixture
def harbor_dir(tmp_path: Path) -> str:
    task = tmp_path / "harbor_task"
    (task / "tests").mkdir(parents=True)
    (task / "environment").mkdir()
    (task / "instruction.md").write_text("Implement solve().")
    (task / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    return str(task)


def _runner(tmp_path: Path, evaluation_file: str, **kwargs) -> Runner:
    return Runner(
        evaluation_file=evaluation_file,
        output_dir=str(tmp_path / "out"),
        **kwargs,
    )


def test_default_is_auto():
    assert Config().evaluator.inject_evaluator_context is None


def test_from_scratch_resolves_on(tmp_path, evaluator_file):
    runner = _runner(tmp_path, evaluator_file)
    assert runner.config.evaluator.inject_evaluator_context is True


def test_seeded_resolves_off(tmp_path, evaluator_file, seed_file):
    runner = _runner(tmp_path, evaluator_file, initial_program_path=seed_file)
    assert runner.config.evaluator.inject_evaluator_context is False


def test_harbor_resolves_on_even_when_seeded(tmp_path, harbor_dir, seed_file):
    runner = _runner(tmp_path, harbor_dir, initial_program_path=seed_file)
    assert runner.config.evaluator.inject_evaluator_context is True


@pytest.mark.parametrize("explicit", [True, False])
def test_explicit_setting_is_never_overridden(tmp_path, evaluator_file, explicit):
    config = Config()
    config.evaluator.inject_evaluator_context = explicit
    runner = _runner(tmp_path, evaluator_file, config=config)
    assert runner.config.evaluator.inject_evaluator_context is explicit
