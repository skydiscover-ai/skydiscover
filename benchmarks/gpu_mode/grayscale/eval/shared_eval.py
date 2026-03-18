"""
Shared evaluator for GPU Mode Triton kernel optimization.

No @triton.jit requirement — pure PyTorch submissions are allowed.
Supports local GPU and Modal cloud GPU evaluation.
Set GPUMODE_USE_MODAL=true and GPUMODE_MODAL_GPU=H100 for Modal.

Scoring: combined_score = SCORE_SCALE / geom_mean_us (higher is better).
The geom_mean_us metric is also reported for absolute runtime tracking.

Each problem provides a reference.py module with:
  - ref_kernel(data)
  - generate_input(**kwargs)
  - check_implementation(data, output) -> (bool, str)
  - TEST_CASES: list of dicts
  - BENCHMARK_CASES: list of dicts
  - SCORE_SCALE: float

Optional benchmark configuration in reference.py:
  - BENCH_USE_CUDA_EVENTS: bool (default True)
  - BENCH_REL_ERROR: float (default 0.001)
  - BENCH_WALL_TIMEOUT_NS: float or None (default 120e9)
  - BENCH_NO_GRAD: bool (default False)
  - BENCH_MAX_REPEATS: int (default 100)
  - BENCH_MAX_TIME_NS: float (default 10e9)
  - BENCH_WARMUP_STYLE: str ('tiny_benchmark' or 'timed_calls', default 'tiny_benchmark')
"""

import os
import sys
import copy
import time
import math
import contextlib
import dataclasses
import traceback
import importlib.util

import torch
import torch.cuda

from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class EvaluationResult:
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)

# Import problem-specific reference (the problem dir is already on sys.path
# because SkyDiscover adds the evaluator file's directory before loading it).
import reference

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

USE_MODAL = os.environ.get("GPUMODE_USE_MODAL", "false").lower() == "true"
MODAL_GPU = os.environ.get("GPUMODE_MODAL_GPU", "H100")

# Read benchmark configuration from reference module with defaults
SCORE_SCALE = getattr(reference, 'SCORE_SCALE', 3000.0)
BENCH_USE_CUDA_EVENTS = getattr(reference, 'BENCH_USE_CUDA_EVENTS', True)
BENCH_REL_ERROR = getattr(reference, 'BENCH_REL_ERROR', 0.001)
BENCH_WALL_TIMEOUT_NS = getattr(reference, 'BENCH_WALL_TIMEOUT_NS', 120e9)
BENCH_NO_GRAD = getattr(reference, 'BENCH_NO_GRAD', False)
BENCH_MAX_REPEATS = getattr(reference, 'BENCH_MAX_REPEATS', 100)
BENCH_MAX_TIME_NS = getattr(reference, 'BENCH_MAX_TIME_NS', 10e9)
BENCH_WARMUP_STYLE = getattr(reference, 'BENCH_WARMUP_STYLE', 'tiny_benchmark')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clone(data):
    """Recursively clone data, handling tensors, dataclasses, and nn.Modules."""
    if isinstance(data, tuple):
        return tuple(_clone(x) for x in data)
    if isinstance(data, list):
        return [_clone(x) for x in data]
    if isinstance(data, dict):
        return {k: _clone(v) for k, v in data.items()}
    if isinstance(data, torch.Tensor):
        return data.clone()
    if dataclasses.is_dataclass(data) and not isinstance(data, type):
        fields = {f.name: _clone(getattr(data, f.name)) for f in dataclasses.fields(data)}
        return type(data)(**fields)
    if isinstance(data, torch.nn.Module):
        cloned = copy.deepcopy(data)
        if hasattr(data, 'seq_len'):
            cloned.seq_len = data.seq_len
        return cloned
    return data


def _stats(durations):
    """Compute statistics from a list of durations (in nanoseconds)."""
    n = len(durations)
    avg = sum(durations) / n
    if n > 1:
        var = sum((x - avg) ** 2 for x in durations) / (n - 1)
        std = math.sqrt(var)
        err = std / math.sqrt(n)
    else:
        std, err = 0.0, 0.0
    return {"runs": n, "mean": avg, "std": std, "err": err}


def _warmup(kernel_fn, bench_args):
    """Warmup the kernel to trigger Triton compilation."""
    if BENCH_WARMUP_STYLE == 'timed_calls':
        # MLA-style: run repeatedly for 200ms
        data = reference.generate_input(**bench_args)
        start = time.perf_counter()
        while time.perf_counter() - start < 0.2:
            kernel_fn(data)
            torch.cuda.synchronize()
    else:
        # trimul-style: run first benchmark with tiny time budget (10ms)
        _bench_single(kernel_fn, bench_args, max_time_ns=10e7)


def _bench_single(kernel_fn, bench_args, max_time_ns=None):
    """Benchmark a kernel on a single case.

    Returns (stats_dict_or_None, error_str_or_None).
    Stats dict has durations in nanoseconds.
    """
    if max_time_ns is None:
        max_time_ns = BENCH_MAX_TIME_NS

    data = reference.generate_input(**bench_args)
    data_copy = _clone(data)

    # Correctness check first
    ctx = torch.no_grad() if BENCH_NO_GRAD else contextlib.nullcontext()
    with ctx:
        output = kernel_fn(data)
        torch.cuda.synchronize()
        passed, msg = reference.check_implementation(data_copy, output)
    if not passed:
        return None, f"Benchmark correctness: {msg}"
    del output

    # Timed runs — durations in nanoseconds
    durations_ns = []
    bm_start = time.perf_counter_ns()

    with ctx:
        for i in range(BENCH_MAX_REPEATS):
            torch.cuda.synchronize()

            if BENCH_USE_CUDA_EVENTS:
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                output = kernel_fn(data)
                e.record()
                torch.cuda.synchronize()
                duration_ns = s.elapsed_time(e) * 1e6  # ms -> ns
            else:
                start_ns = time.perf_counter_ns()
                output = kernel_fn(data)
                torch.cuda.synchronize()
                duration_ns = time.perf_counter_ns() - start_ns

            del output
            durations_ns.append(duration_ns)

            if i > 1:
                st = _stats(durations_ns)
                if st["mean"] > 0 and st["err"] / st["mean"] < BENCH_REL_ERROR:
                    break
                if st["mean"] * st["runs"] > max_time_ns:
                    break
                if BENCH_WALL_TIMEOUT_NS is not None and \
                   (time.perf_counter_ns() - bm_start) > BENCH_WALL_TIMEOUT_NS:
                    break

    return _stats(durations_ns), None


# ---------------------------------------------------------------------------
# Modal path
# ---------------------------------------------------------------------------


def _evaluate_modal(submission_code):
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from modal_eval import (
        eval_triton_h100, eval_triton_a100, eval_triton_l40s, eval_triton_t4,
        eval_triton_h200, app as modal_app,
    )

    gpu_fns = {
        "H100": eval_triton_h100,
        "A100": eval_triton_a100,
        "L40S": eval_triton_l40s,
        "T4": eval_triton_t4,
        "H200": eval_triton_h200,
    }
    eval_fn = gpu_fns.get(MODAL_GPU, eval_triton_h100)

    ref_code = getattr(reference, 'MODAL_REFERENCE_CODE', None)
    if ref_code is None:
        return EvaluationResult(
            metrics={"combined_score": 0.0, "correctness": 0.0},
            artifacts={"error": "MODAL_REFERENCE_CODE not defined in reference.py",
                       "failure_stage": "modal_setup"},
        )

    with modal_app.run():
        result = eval_fn.remote(
            submission_code=submission_code,
            reference_code=ref_code,
            test_cases=reference.TEST_CASES,
            benchmark_cases=reference.BENCHMARK_CASES,
            score_scale=SCORE_SCALE,
            bench_use_cuda_events=BENCH_USE_CUDA_EVENTS,
            bench_rel_error=BENCH_REL_ERROR,
            bench_wall_timeout_ns=BENCH_WALL_TIMEOUT_NS,
            bench_no_grad=BENCH_NO_GRAD,
            bench_max_repeats=BENCH_MAX_REPEATS,
            bench_max_time_ns=BENCH_MAX_TIME_NS,
            bench_warmup_style=BENCH_WARMUP_STYLE,
        )

    if isinstance(result, dict):
        error = result.get("error")
        score = float(result.get("combined_score", 0.0))
        metrics = {"combined_score": score, "correctness": float(result.get("correctness", 0.0))}
        if "geom_mean_us" in result:
            metrics["geom_mean_us"] = float(result["geom_mean_us"])
        artifacts = {}
        if error:
            artifacts["error"] = str(error)
            artifacts["failure_stage"] = "modal_eval"
        if "bench_means_us" in result:
            for i, us in enumerate(result["bench_means_us"]):
                artifacts[f"bench_{i}_mean_us"] = f"{us:.2f}"
        artifacts["hardware"] = MODAL_GPU
        return EvaluationResult(metrics=metrics, artifacts=artifacts)

    return EvaluationResult(
        metrics={"combined_score": 0.0, "correctness": 0.0},
        artifacts={"error": "Modal returned unexpected type", "failure_stage": "modal_eval"},
    )


# ---------------------------------------------------------------------------
# Local path
# ---------------------------------------------------------------------------


def _evaluate_local(program_path):
    try:
        spec = importlib.util.spec_from_file_location("submission", program_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["submission"] = mod
        spec.loader.exec_module(mod)
        custom_kernel = mod.custom_kernel
    except Exception as exc:
        return EvaluationResult(
            metrics={"combined_score": 0.0, "correctness": 0.0},
            artifacts={
                "error": f"Failed to load submission: {exc}",
                "traceback": traceback.format_exc(),
                "failure_stage": "import",
            },
        )

    # Correctness
    for i, tc in enumerate(reference.TEST_CASES):
        try:
            data = reference.generate_input(**tc)
            data_copy = _clone(data)
            torch.cuda.synchronize()
            output = custom_kernel(data)
            torch.cuda.synchronize()
            passed, msg = reference.check_implementation(data_copy, output)
            if not passed:
                return EvaluationResult(
                    metrics={"combined_score": 0.0, "correctness": 0.0},
                    artifacts={
                        "error": f"Test {i} failed: {msg}",
                        "failure_stage": "correctness",
                        "test_index": str(i),
                    },
                )
        except Exception as exc:
            return EvaluationResult(
                metrics={"combined_score": 0.0, "correctness": 0.0},
                artifacts={
                    "error": f"Test {i} error: {exc}",
                    "traceback": traceback.format_exc(),
                    "failure_stage": "correctness",
                    "test_index": str(i),
                },
            )

    # Warmup
    _warmup(custom_kernel, reference.BENCHMARK_CASES[0])

    # Benchmarks — collect mean runtimes in nanoseconds
    bench_means_ns = []
    for bench_args in reference.BENCHMARK_CASES:
        st, err = _bench_single(custom_kernel, bench_args)
        if err:
            return EvaluationResult(
                metrics={"combined_score": 0.0, "correctness": 1.0},
                artifacts={"error": err, "failure_stage": "benchmark"},
            )
        bench_means_ns.append(st["mean"])

    # Scoring: geometric mean of benchmark means → microseconds → score
    means_seconds = [ns / 1e9 for ns in bench_means_ns]
    geom_mean_s = math.pow(math.prod(means_seconds), 1.0 / len(means_seconds))
    geom_mean_us = geom_mean_s * 1e6
    score = SCORE_SCALE / geom_mean_us

    metrics = {
        "combined_score": score,
        "correctness": 1.0,
        "geom_mean_us": geom_mean_us,
    }
    artifacts = {
        "hardware": "local",
    }
    for i, ns in enumerate(bench_means_ns):
        artifacts[f"bench_{i}_mean_us"] = f"{ns / 1e3:.2f}"

    return EvaluationResult(
        metrics=metrics,
        artifacts=artifacts,
    )


# ---------------------------------------------------------------------------
# Public API (used by SkyDiscover)
# ---------------------------------------------------------------------------


def evaluate(program_path):
    try:
        with open(program_path, "r") as f:
            code = f.read()
    except Exception as exc:
        return EvaluationResult(
            metrics={"combined_score": 0.0, "correctness": 0.0},
            artifacts={"error": f"Failed to read file: {exc}", "failure_stage": "file_read"},
        )

    if USE_MODAL:
        try:
            return _evaluate_modal(code)
        except Exception as exc:
            return EvaluationResult(
                metrics={"combined_score": 0.0, "correctness": 0.0},
                artifacts={
                    "error": f"Modal evaluation failed: {exc}",
                    "traceback": traceback.format_exc(),
                    "failure_stage": "modal_eval",
                },
            )

    return _evaluate_local(program_path)


def evaluate_stage1(program_path):
    try:
        with open(program_path, "r") as f:
            code = f.read()
    except Exception as exc:
        return EvaluationResult(
            metrics={"combined_score": 0.0, "stage1_passed": 0.0},
            artifacts={"error": f"Failed to read file: {exc}", "failure_stage": "file_read"},
        )

    if "custom_kernel" not in code:
        return EvaluationResult(
            metrics={"combined_score": 0.0, "stage1_passed": 0.0},
            artifacts={"error": "Missing custom_kernel function", "failure_stage": "validation"},
        )

    try:
        compile(code, program_path, "exec")
    except SyntaxError as exc:
        return EvaluationResult(
            metrics={"combined_score": 0.0, "stage1_passed": 0.0},
            artifacts={
                "error": f"Syntax error at line {exc.lineno}: {exc.msg}",
                "failure_stage": "syntax_check",
            },
        )

    # When using Modal, skip local import check (triton may not be installed locally).
    if not USE_MODAL:
        try:
            spec = importlib.util.spec_from_file_location("submission_check", program_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if not hasattr(mod, "custom_kernel"):
                return EvaluationResult(
                    metrics={"combined_score": 0.0, "stage1_passed": 0.0},
                    artifacts={"error": "custom_kernel not found after import", "failure_stage": "import"},
                )
        except Exception as exc:
            return EvaluationResult(
                metrics={"combined_score": 0.0, "stage1_passed": 0.0},
                artifacts={
                    "error": f"Import failed: {exc}",
                    "traceback": traceback.format_exc(),
                    "failure_stage": "import",
                },
            )

    return EvaluationResult(
        metrics={"combined_score": 0.5, "stage1_passed": 1.0},
        artifacts={},
    )


def evaluate_stage2(program_path):
    return evaluate(program_path)
