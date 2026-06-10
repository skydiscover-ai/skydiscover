"""Tests for SubprocessEvaluator — process isolation and result parsing."""

import asyncio
import os
import tempfile

import pytest

from skydiscover.config import EvaluatorConfig
from skydiscover.evaluation.subprocess_evaluator import SubprocessEvaluator

# --- Mock evaluator scripts written to temp files ---

EVALUATOR_SUCCESS = """\
import json

def evaluate(program_path):
    with open(program_path) as f:
        code = f.read()
    return {"combined_score": 0.85, "lines": len(code.splitlines())}
"""

EVALUATOR_CRASH = """\
import os, signal

def evaluate(program_path):
    # Simulate a crash (e.g. CUDA illegal memory access / segfault)
    os.kill(os.getpid(), signal.SIGTERM)
"""

EVALUATOR_RETURNS_ERROR = """\
def evaluate(program_path):
    raise RuntimeError("CUDA error: illegal memory access")
"""

EVALUATOR_NOISY_STDOUT = """\
import json

print("WARNING: some library noise on stdout")
print("Another warning line")

def evaluate(program_path):
    return {"combined_score": 0.5}
"""


def _write_temp_evaluator(source: str) -> str:
    """Write evaluator source to a temp .py file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    f.write(source)
    f.close()
    return f.name


def _make_config(eval_file: str, timeout: int = 30) -> EvaluatorConfig:
    config = EvaluatorConfig()
    config.evaluation_file = eval_file
    config.timeout = timeout
    config.max_retries = 0
    config.subprocess_isolation = True
    return config


class TestSubprocessEvaluatorSuccess:
    def test_returns_metrics(self, tmp_path):
        eval_file = _write_temp_evaluator(EVALUATOR_SUCCESS)
        try:
            evaluator = SubprocessEvaluator(_make_config(eval_file))
            result = asyncio.run(evaluator.evaluate_program("x = 1\ny = 2\n"))
            assert result.metrics["combined_score"] == 0.85
            assert result.metrics["lines"] == 2
        finally:
            os.unlink(eval_file)

    def test_noisy_stdout_still_parses(self):
        eval_file = _write_temp_evaluator(EVALUATOR_NOISY_STDOUT)
        try:
            evaluator = SubprocessEvaluator(_make_config(eval_file))
            result = asyncio.run(evaluator.evaluate_program("pass"))
            assert result.metrics["combined_score"] == 0.5
        finally:
            os.unlink(eval_file)


class TestSubprocessEvaluatorIsolation:
    def test_crash_does_not_affect_parent(self):
        eval_file = _write_temp_evaluator(EVALUATOR_CRASH)
        try:
            evaluator = SubprocessEvaluator(_make_config(eval_file))
            result = asyncio.run(evaluator.evaluate_program("x = 1"))
            # Parent is still alive — got an error result, not a crash
            assert result.metrics.get("error") == 0.0
        finally:
            os.unlink(eval_file)

    def test_exception_in_evaluate_returns_error(self):
        eval_file = _write_temp_evaluator(EVALUATOR_RETURNS_ERROR)
        try:
            evaluator = SubprocessEvaluator(_make_config(eval_file))
            result = asyncio.run(evaluator.evaluate_program("x = 1"))
            assert result.metrics.get("error") == 0.0
        finally:
            os.unlink(eval_file)

    def test_crash_then_success(self):
        """After a crash, subsequent evaluations still work (new process each time)."""
        crash_file = _write_temp_evaluator(EVALUATOR_CRASH)
        success_file = _write_temp_evaluator(EVALUATOR_SUCCESS)
        try:
            # First: crash
            crash_eval = SubprocessEvaluator(_make_config(crash_file))
            result1 = asyncio.run(crash_eval.evaluate_program("x = 1"))
            assert result1.metrics.get("error") == 0.0

            # Second: success (proves parent process is fine)
            success_eval = SubprocessEvaluator(_make_config(success_file))
            result2 = asyncio.run(success_eval.evaluate_program("a = 1\nb = 2\n"))
            assert result2.metrics["combined_score"] == 0.85
        finally:
            os.unlink(crash_file)
            os.unlink(success_file)


class TestSubprocessEvaluatorTimeout:
    def test_timeout_returns_timeout_metric(self):
        slow_evaluator = """\
import time

def evaluate(program_path):
    time.sleep(60)
    return {"combined_score": 1.0}
"""
        eval_file = _write_temp_evaluator(slow_evaluator)
        try:
            evaluator = SubprocessEvaluator(_make_config(eval_file, timeout=2))
            result = asyncio.run(evaluator.evaluate_program("x = 1"))
            assert result.metrics.get("timeout") is True or result.metrics.get("error") == 0.0
        finally:
            os.unlink(eval_file)
