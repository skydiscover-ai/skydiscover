"""
Shared Modal app for evaluating Triton kernels on cloud GPUs.
Scoring: score = score_scale / geom_mean_runtime_us.

Usage:
    Set GPUMODE_USE_MODAL=true and GPUMODE_MODAL_GPU=H100 (or A100, L40S, T4, H200)
    in environment variables, then call eval functions from evaluators.
"""

import modal

app = modal.App("gpu-mode-triton-eval")

cuda_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0",
    "triton>=3.0.0",
    "numpy",
)


def _eval_triton_impl(
    submission_code: str,
    reference_code: str,
    test_cases: list,
    benchmark_cases: list,
    score_scale: float = 3000.0,
    bench_use_cuda_events: bool = True,
    bench_rel_error: float = 0.001,
    bench_wall_timeout_ns: float = 120e9,
    bench_no_grad: bool = False,
    bench_max_repeats: int = 100,
    bench_max_time_ns: float = 10e9,
    bench_warmup_style: str = "tiny_benchmark",
) -> dict:
    """
    Core evaluation logic that runs inside a Modal GPU container.

    Returns dict with: combined_score, correctness, geom_mean_us, error
    """
    import contextlib
    import copy
    import dataclasses
    import gc
    import math
    import os
    import sys
    import tempfile
    import time

    # Help with memory fragmentation for large models (MLA bs=128)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import importlib.util
    import traceback

    import torch
    import torch.cuda

    def clone_data(data):
        if isinstance(data, tuple):
            return tuple(clone_data(x) for x in data)
        elif isinstance(data, list):
            return [clone_data(x) for x in data]
        elif isinstance(data, dict):
            return {k: clone_data(v) for k, v in data.items()}
        elif isinstance(data, torch.Tensor):
            return data.clone()
        elif dataclasses.is_dataclass(data) and not isinstance(data, type):
            fields = {f.name: clone_data(getattr(data, f.name)) for f in dataclasses.fields(data)}
            return type(data)(**fields)
        elif isinstance(data, torch.nn.Module):
            cloned = copy.deepcopy(data)
            if hasattr(data, "seq_len"):
                cloned.seq_len = data.seq_len
            return cloned
        return data

    def stats(durations):
        n = len(durations)
        avg = sum(durations) / n
        if n > 1:
            var = sum((x - avg) ** 2 for x in durations) / (n - 1)
            std = math.sqrt(var)
            err = std / math.sqrt(n)
        else:
            std, err = 0.0, 0.0
        return {"runs": n, "mean": avg, "std": std, "err": err}

    tmpdir = tempfile.mkdtemp()

    try:
        ref_path = os.path.join(tmpdir, "reference.py")
        sub_path = os.path.join(tmpdir, "submission.py")

        with open(ref_path, "w") as f:
            f.write(reference_code)
        with open(sub_path, "w") as f:
            f.write(submission_code)

        sys.path.insert(0, tmpdir)

        spec = importlib.util.spec_from_file_location("reference", ref_path)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)

        generate_input = reference.generate_input
        check_implementation = reference.check_implementation

        spec = importlib.util.spec_from_file_location("submission", sub_path)
        submission = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(submission)
        custom_kernel = submission.custom_kernel

        # Correctness tests (use no_grad to reduce memory from autograd)
        for i, test_args in enumerate(test_cases):
            data = generate_input(**test_args)
            data_copy = clone_data(data)
            torch.cuda.synchronize()
            with torch.no_grad():
                output = custom_kernel(data)
            torch.cuda.synchronize()
            # Aggressively free GPU memory before ref kernel runs
            del data
            gc.collect()
            torch.cuda.empty_cache()
            passed, msg = check_implementation(data_copy, output)
            del data_copy, output
            gc.collect()
            torch.cuda.empty_cache()
            if not passed:
                return {
                    "combined_score": 0.0,
                    "correctness": 0.0,
                    "error": f"Test {i} failed: {msg}",
                }

        # Warmup
        wb = benchmark_cases[0]
        if bench_warmup_style == "timed_calls":
            wdata = generate_input(**wb)
            start = time.perf_counter()
            while time.perf_counter() - start < 0.2:
                custom_kernel(wdata)
                torch.cuda.synchronize()
        else:
            # tiny_benchmark: quick run to trigger compilation
            wdata = generate_input(**wb)
            for _ in range(3):
                custom_kernel(wdata)
                torch.cuda.synchronize()

        # Benchmarks — collect mean runtimes in nanoseconds
        ctx = torch.no_grad() if bench_no_grad else contextlib.nullcontext()
        bench_means_ns = []

        for bench_args in benchmark_cases:
            data = generate_input(**bench_args)
            data_copy = clone_data(data)

            # Correctness check
            with ctx:
                output = custom_kernel(data)
                torch.cuda.synchronize()
                # Aggressively free GPU memory before ref kernel runs
                del data
                gc.collect()
                torch.cuda.empty_cache()
                passed, msg = check_implementation(data_copy, output)
                del data_copy, output
                gc.collect()
                torch.cuda.empty_cache()
            if not passed:
                return {
                    "combined_score": 0.0,
                    "correctness": 1.0,
                    "error": f"Benchmark correctness: {msg}",
                }

            # Regenerate data for timed runs (was freed during correctness check)
            data = generate_input(**bench_args)

            # Timed runs
            durations_ns = []
            bm_start = time.perf_counter_ns()

            with ctx:
                for t in range(bench_max_repeats):
                    torch.cuda.synchronize()

                    if bench_use_cuda_events:
                        s = torch.cuda.Event(enable_timing=True)
                        e = torch.cuda.Event(enable_timing=True)
                        s.record()
                        output = custom_kernel(data)
                        e.record()
                        torch.cuda.synchronize()
                        duration_ns = s.elapsed_time(e) * 1e6  # ms -> ns
                    else:
                        start_ns = time.perf_counter_ns()
                        output = custom_kernel(data)
                        torch.cuda.synchronize()
                        duration_ns = time.perf_counter_ns() - start_ns

                    del output
                    durations_ns.append(duration_ns)

                    if t > 1:
                        st = stats(durations_ns)
                        if st["mean"] > 0 and st["err"] / st["mean"] < bench_rel_error:
                            break
                        if st["mean"] * st["runs"] > bench_max_time_ns:
                            break
                        if (
                            bench_wall_timeout_ns is not None
                            and (time.perf_counter_ns() - bm_start) > bench_wall_timeout_ns
                        ):
                            break

            bench_means_ns.append(stats(durations_ns)["mean"])

        # Scoring: geometric mean → microseconds → score
        means_seconds = [ns / 1e9 for ns in bench_means_ns]
        geom_mean_s = math.pow(math.prod(means_seconds), 1.0 / len(means_seconds))
        geom_mean_us = geom_mean_s * 1e6
        score = score_scale / geom_mean_us

        bench_means_us = [ns / 1e3 for ns in bench_means_ns]
        return {
            "combined_score": score,
            "correctness": 1.0,
            "geom_mean_us": geom_mean_us,
            "bench_means_us": bench_means_us,
        }
    except Exception as e:
        return {
            "combined_score": 0.0,
            "correctness": 0.0,
            "error": f"{e}\n{traceback.format_exc()}",
        }
    finally:
        sys.path.remove(tmpdir)
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


@app.function(image=cuda_image, gpu="H100", timeout=600)
def eval_triton_h100(**kwargs) -> dict:
    return _eval_triton_impl(**kwargs)


@app.function(image=cuda_image, gpu="A100", timeout=600)
def eval_triton_a100(**kwargs) -> dict:
    return _eval_triton_impl(**kwargs)


@app.function(image=cuda_image, gpu="L40S", timeout=600)
def eval_triton_l40s(**kwargs) -> dict:
    return _eval_triton_impl(**kwargs)


@app.function(image=cuda_image, gpu="T4", timeout=600)
def eval_triton_t4(**kwargs) -> dict:
    return _eval_triton_impl(**kwargs)


@app.function(image=cuda_image, gpu="H200", timeout=600)
def eval_triton_h200(**kwargs) -> dict:
    return _eval_triton_impl(**kwargs)
