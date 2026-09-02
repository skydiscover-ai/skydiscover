"""Regression tests for agent-scaffold ground-truth isolation."""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "agent_scaffold"


def _load_evaluate():
    path = ROOT / "evaluator.py"
    name = "agent_scaffold_evaluator"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.evaluate


evaluate = _load_evaluate()

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


DISK_READ_CHEAT = textwrap.dedent(
    """\
    def run_agent(question, tools):
        from pathlib import Path
        import sys

        paths = [Path.cwd() / "tasks.py"]
        main = sys.modules.get("__main__")
        if main is not None and getattr(main, "__file__", None):
            paths.append(Path(main.__file__).resolve().parent / "tasks.py")
        for path in paths:
            if not path.is_file():
                continue
            ns = {"__name__": "tasks"}
            exec(compile(path.read_text(), str(path), "exec"), ns)
            for task in ns.get("TASKS", []):
                if getattr(task, "question", None) == question:
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


def test_disk_read_of_tasks_py_does_not_score_perfect(tmp_path):
    program = tmp_path / "disk_read.py"
    program.write_text(DISK_READ_CHEAT)
    metrics = evaluate(str(program))
    assert metrics["combined_score"] < 0.5
    assert metrics.get("accuracy", 0.0) < 1.0

