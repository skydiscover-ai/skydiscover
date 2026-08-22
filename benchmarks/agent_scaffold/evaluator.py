"""Evaluate an agent scaffold on the synthetic multi-hop tool suite (#1)."""

from __future__ import annotations

import importlib
import importlib.util
import os
import signal
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from typing import Any, Iterator

_HERE = os.path.dirname(os.path.abspath(__file__))
MAX_TOOL_CALLS = 12
_TASK_TIMEOUT_S = 8.0


def _strip_here_from_path() -> None:
    here = os.path.abspath(_HERE)
    sys.path[:] = [p for p in sys.path if os.path.abspath(p) != here]


def _hide_ground_truth_modules() -> None:
    sys.modules.pop("tasks", None)
    sys.modules.pop("tools", None)


def _load_env() -> dict[str, Any]:
    """Import tasks/tools, then hide them so candidates cannot `import tasks`."""
    added = False
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
        added = True
    tasks_mod = importlib.import_module("tasks")
    tools_mod = importlib.import_module("tools")
    env = {
        "train_tasks": tasks_mod.TASKS[:-3],
        "test_tasks": tasks_mod.TASKS[-3:],
        "normalize_answer": tasks_mod.normalize_answer,
        "ToolBudget": tools_mod.ToolBudget,
        "make_tools": tools_mod.make_tools,
    }
    _hide_ground_truth_modules()
    if added:
        try:
            sys.path.remove(_HERE)
        except ValueError:
            pass
    _strip_here_from_path()
    return env


@contextmanager
def _sandbox_candidate() -> Iterator[None]:
    saved_path = list(sys.path)
    _strip_here_from_path()
    _hide_ground_truth_modules()
    try:
        yield
    finally:
        _hide_ground_truth_modules()
        sys.path[:] = saved_path
        _strip_here_from_path()
        _hide_ground_truth_modules()


class _TaskTimeout(Exception):
    pass


def _run_with_timeout(fn, timeout_s: float, *args):
    if (
        timeout_s <= 0
        or not hasattr(signal, "SIGALRM")
        or threading.current_thread() is not threading.main_thread()
    ):
        return fn(*args)

    def _handle(_signum, _frame):
        raise _TaskTimeout("per-task timeout")

    old = signal.signal(signal.SIGALRM, _handle)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        return fn(*args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old)


def _load_run_agent(program_path: str):
    spec = importlib.util.spec_from_file_location("candidate_scaffold", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run_agent"):
        raise AttributeError("program must define run_agent(question, tools)")
    return module.run_agent


def _score_tasks(
    run_agent,
    tasks,
    *,
    make_tools,
    tool_budget_cls,
    normalize_answer,
    max_calls: int,
) -> dict[str, Any]:
    correct = 0
    total_calls = 0
    failures = 0
    details = []

    for task in tasks:
        budget = tool_budget_cls(max_calls=max_calls)
        tools = make_tools(budget)
        error = None
        answer = ""
        try:
            with _sandbox_candidate():
                answer = str(
                    _run_with_timeout(run_agent, _TASK_TIMEOUT_S, task.question, tools)
                )
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
    raw_efficiency = max(0.0, 1.0 - (avg_calls / float(max_calls)))
    # Do not reward a no-op scaffold for burning zero tools.
    efficiency = raw_efficiency * accuracy
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


def _make_evaluate():
    env = _load_env()
    make_tools = env["make_tools"]
    tool_budget_cls = env["ToolBudget"]
    normalize_answer = env["normalize_answer"]
    train_tasks = env["train_tasks"]
    test_tasks = env["test_tasks"]

    def evaluate(program_path: str) -> dict[str, Any]:
        """SkyDiscover evaluator entrypoint."""
        t0 = time.time()
        try:
            with _sandbox_candidate():
                run_agent = _load_run_agent(program_path)
            train = _score_tasks(
                run_agent,
                train_tasks,
                make_tools=make_tools,
                tool_budget_cls=tool_budget_cls,
                normalize_answer=normalize_answer,
                max_calls=MAX_TOOL_CALLS,
            )
            test = _score_tasks(
                run_agent,
                test_tasks,
                make_tools=make_tools,
                tool_budget_cls=tool_budget_cls,
                normalize_answer=normalize_answer,
                max_calls=MAX_TOOL_CALLS,
            )

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

    return evaluate


evaluate = _make_evaluate()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "initial_program.py")
    print(evaluate(path))
