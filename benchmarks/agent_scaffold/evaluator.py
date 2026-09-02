"""Evaluate an agent scaffold on the synthetic multi-hop tool suite (#1).

``run_agent`` executes in a subprocess whose working directory contains only
the candidate and ``tools.py``. ``tasks.py`` (held-out answers) and this
evaluator stay in the parent, so cwd / ``__main__.__file__`` filesystem reads
cannot recover ground truth. Per-task timeouts use ``subprocess`` (works off
the main thread and on platforms without SIGALRM).
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
MAX_TOOL_CALLS = 12
_TASK_TIMEOUT_S = 8.0

_WORKER_SOURCE = r"""
import importlib.util
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path[:] = [HERE] + [
    p for p in sys.path if p not in ("", ".") and os.path.abspath(p) != HERE
]
sys.stdout = sys.stderr


def _send(obj):
    sys.__stdout__.write(json.dumps(obj) + "\n")
    sys.__stdout__.flush()


def _recv():
    line = sys.__stdin__.readline()
    if not line:
        return None
    return json.loads(line)


def _load_run_agent():
    spec = importlib.util.spec_from_file_location(
        "candidate_scaffold", os.path.join(HERE, "candidate.py")
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load candidate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run_agent"):
        raise AttributeError("program must define run_agent(question, tools)")
    return module.run_agent


def main():
    from tools import ToolBudget, make_tools

    run_agent = _load_run_agent()
    while True:
        req = _recv()
        if req is None:
            return
        budget = ToolBudget(max_calls=int(req["max_calls"]))
        tools = make_tools(budget)
        error = None
        answer = ""
        try:
            answer = str(run_agent(req["question"], tools))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            answer = ""
        _send({"answer": answer, "calls": int(budget.calls), "error": error})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        _send({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        sys.exit(1)
"""


def _load_tasks():
    path = os.path.join(_HERE, "tasks.py")
    spec = importlib.util.spec_from_file_location("_agent_scaffold_tasks", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses look up cls.__module__ in sys.modules while decorating.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _clean_env(workdir: str) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PWD"] = workdir
    return env


class _IsolatedAgent:
    """One worker process: candidate + tools, no benchmark directory."""

    def __init__(self, program_path: str):
        self._program_src = Path(program_path).read_text(encoding="utf-8")
        self._tools_src = Path(_HERE, "tools.py").read_text(encoding="utf-8")
        self._td = tempfile.mkdtemp(prefix="skydiscover_agent_")
        Path(self._td, "candidate.py").write_text(self._program_src, encoding="utf-8")
        Path(self._td, "tools.py").write_text(self._tools_src, encoding="utf-8")
        Path(self._td, "worker.py").write_text(_WORKER_SOURCE, encoding="utf-8")
        self._proc: subprocess.Popen | None = None
        self._spawn()

    def _spawn(self) -> None:
        self._close_proc()
        self._proc = subprocess.Popen(
            [sys.executable, "-u", os.path.join(self._td, "worker.py")],
            cwd=self._td,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=_clean_env(self._td),
        )

    def _close_proc(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.kill()
            proc.wait(timeout=2)
        except Exception:
            pass

    def ask(self, question: str, max_calls: int, timeout_s: float) -> tuple[str, int, str | None]:
        import threading

        if self._proc is None or self._proc.poll() is not None:
            self._spawn()
        proc = self._proc
        assert proc is not None and proc.stdin is not None and proc.stdout is not None
        timed_out = threading.Event()

        def _kill() -> None:
            timed_out.set()
            try:
                proc.kill()
            except Exception:
                pass

        timer = threading.Timer(timeout_s, _kill)
        timer.start()
        try:
            proc.stdin.write(json.dumps({"question": question, "max_calls": max_calls}) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            if timed_out.is_set() or not line:
                self._spawn()
                return "", 0, "timeout"
            msg = json.loads(line)
            if msg.get("error") and "answer" not in msg:
                self._spawn()
                return "", 0, str(msg["error"])
            return str(msg.get("answer") or ""), int(msg.get("calls") or 0), msg.get("error")
        except Exception as exc:  # noqa: BLE001
            self._spawn()
            if timed_out.is_set():
                return "", 0, "timeout"
            return "", 0, f"{type(exc).__name__}: {exc}"
        finally:
            timer.cancel()

    def close(self) -> None:
        self._close_proc()
        shutil.rmtree(self._td, ignore_errors=True)


def _score_tasks(agent: _IsolatedAgent, tasks, *, normalize_answer, max_calls: int) -> dict[str, Any]:
    correct = 0
    total_calls = 0
    failures = 0
    details = []

    for task in tasks:
        answer, calls, error = agent.ask(task.question, max_calls, _TASK_TIMEOUT_S)
        if error == "timeout":
            failures += 1
        elif error:
            failures += 1
        ok = normalize_answer(answer) == normalize_answer(task.answer)
        if ok:
            correct += 1
        total_calls += calls
        details.append(
            {
                "task_id": task.task_id,
                "correct": ok,
                "answer": answer,
                "expected": task.answer,
                "tool_calls": calls,
                "error": error,
            }
        )

    n = max(1, len(tasks))
    accuracy = correct / n
    avg_calls = total_calls / n
    raw_efficiency = max(0.0, 1.0 - (avg_calls / float(max_calls)))
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
    tasks_mod = _load_tasks()
    normalize_answer = tasks_mod.normalize_answer
    train_tasks = tasks_mod.TASKS[:-3]
    test_tasks = tasks_mod.TASKS[-3:]

    def evaluate(program_path: str) -> dict[str, Any]:
        t0 = time.time()
        agent: _IsolatedAgent | None = None
        try:
            agent = _IsolatedAgent(program_path)
            train = _score_tasks(
                agent, train_tasks, normalize_answer=normalize_answer, max_calls=MAX_TOOL_CALLS
            )
            test = _score_tasks(
                agent, test_tasks, normalize_answer=normalize_answer, max_calls=MAX_TOOL_CALLS
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
        finally:
            if agent is not None:
                agent.close()

    return evaluate


evaluate = _make_evaluate()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "initial_program.py")
    print(evaluate(path))
