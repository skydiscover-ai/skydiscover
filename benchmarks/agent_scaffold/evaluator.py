"""Evaluate an agent scaffold on the synthetic multi-hop tool suite (#1)."""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import traceback
from typing import Any

# Local imports (evaluator runs with this directory on sys.path / cwd).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from tasks import TASKS, normalize_answer  # noqa: E402
from tools import ToolBudget, make_tools  # noqa: E402

MAX_TOOL_CALLS = 12
# Hold out the last 3 tasks so evolution cannot overfit the full suite blindly.
TRAIN_TASKS = TASKS[:-3]
TEST_TASKS = TASKS[-3:]


def _load_run_agent(program_path: str):
    spec = importlib.util.spec_from_file_location("candidate_scaffold", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run_agent"):
        raise AttributeError("program must define run_agent(question, tools)")
    return module.run_agent


def _score_tasks(run_agent, tasks) -> dict[str, Any]:
    correct = 0
    total_calls = 0
    failures = 0
    details = []

    for task in tasks:
        budget = ToolBudget(max_calls=MAX_TOOL_CALLS)
        tools = make_tools(budget)
        error = None
        answer = ""
        try:
            answer = str(run_agent(task.question, tools))
        except Exception as exc:  # noqa: BLE001 — surface scaffold crashes
            failures += 1
            error = f"{type(exc).__name__}: {exc}"
            answer = ""

        ok = normalize_answer(answer) == normalize_answer(task.answer)
        if ok:
            correct += 1
        total_calls += budget.calls
        details.append(
            {
                "task_id": task.task_id,
                "correct": ok,
                "answer": answer,
                "expected": task.answer,
                "tool_calls": budget.calls,
                "error": error,
            }
        )

    n = max(1, len(tasks))
    accuracy = correct / n
    avg_calls = total_calls / n
    # Efficiency in [0, 1]: fewer tool calls is better (cap at MAX_TOOL_CALLS).
    efficiency = max(0.0, 1.0 - (avg_calls / float(MAX_TOOL_CALLS)))
    crash_rate = failures / n
    return {
        "accuracy": accuracy,
        "avg_tool_calls": avg_calls,
        "efficiency": efficiency,
        "crash_rate": crash_rate,
        "n_correct": correct,
        "n_tasks": len(tasks),
        "details": details,
    }


def evaluate(program_path: str) -> dict[str, Any]:
    """SkyDiscover evaluator entrypoint."""
    t0 = time.time()
    try:
        run_agent = _load_run_agent(program_path)
        train = _score_tasks(run_agent, TRAIN_TASKS)
        test = _score_tasks(run_agent, TEST_TASKS)

        # Primary signal is held-out accuracy; efficiency is secondary.
        # Mild train term keeps the cascade responsive early in search.
        combined = (
            0.55 * test["accuracy"]
            + 0.25 * train["accuracy"]
            + 0.15 * test["efficiency"]
            + 0.05 * (1.0 - test["crash_rate"])
        )

        return {
            "combined_score": float(combined),
            "accuracy": float(test["accuracy"]),
            "train_accuracy": float(train["accuracy"]),
            "efficiency": float(test["efficiency"]),
            "avg_tool_calls": float(test["avg_tool_calls"]),
            "crash_rate": float(test["crash_rate"]),
            "latency_s": float(time.time() - t0),
            "artifacts": {
                "train_details": train["details"],
                "test_details": test["details"],
            },
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "combined_score": 0.0,
            "accuracy": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "latency_s": float(time.time() - t0),
        }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "initial_program.py")
    print(evaluate(path))
