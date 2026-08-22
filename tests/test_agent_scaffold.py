"""Regression tests for agent-scaffold ground-truth isolation."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "agent_scaffold"
sys.path.insert(0, str(ROOT))

from evaluator import evaluate  # noqa: E402

IMPORT_TASKS_CHEAT = textwrap.dedent(
    """\
    def run_agent(question, tools):
        import tasks
        for task in tasks.TASKS:
            if task.question == question:
                return task.answer
        return ""
    """
)


def test_evaluate_baseline_program():
    metrics = evaluate(str(ROOT / "initial_program.py"))
    assert "combined_score" in metrics
    assert "error" not in metrics
    assert 0.0 < metrics["combined_score"] < 1.0


def test_import_tasks_cheat_does_not_score_perfect(tmp_path):
    program = tmp_path / "import_tasks.py"
    program.write_text(IMPORT_TASKS_CHEAT)
    metrics = evaluate(str(program))
    assert metrics["combined_score"] < 1.0
    # Without TASKS, this scaffold answers empty / errors; must not be perfect.
    assert metrics.get("accuracy", 0.0) < 1.0
    assert metrics["combined_score"] < 0.5
