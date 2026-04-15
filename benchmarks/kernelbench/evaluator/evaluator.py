"""
Evaluator for KernelBench problems using kernelbench evaluation logic.

This evaluator can run inside a Docker container or as a native Python script,
and evaluates candidate kernel programs against KernelBench reference implementations.
"""

import os
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path


def ensure_run_and_check(evaluator_dir: Path):
    """Download run_and_check.py if not present.

    This allows the evaluator to work in native Python mode without Docker,
    automatically fetching the KernelBench evaluation script on first use.

    Args:
        evaluator_dir: Directory where the evaluator is located

    Returns:
        Path to run_and_check.py
    """
    run_and_check_path = evaluator_dir / "run_and_check.py"

    if not run_and_check_path.exists():
        import urllib.request

        commit = "423217d"
        url = f"https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/{commit}/scripts/run_and_check.py"

        print(
            f"[INFO] Downloading run_and_check.py from KernelBench (commit {commit})...",
            file=sys.stderr,
        )
        try:
            urllib.request.urlretrieve(url, run_and_check_path)
            print(f"[INFO] Downloaded to {run_and_check_path}", file=sys.stderr)
        except Exception as e:
            raise RuntimeError(f"Failed to download run_and_check.py: {e}")

    return run_and_check_path


def evaluate(program_path: str):
    """
    Evaluate a candidate kernel program against the reference using run_and_check.py.

    Args:
        program_path: Path to the candidate program file

    Returns:
        Dictionary with combined_score (higher is better) and optional artifacts
    """
    try:
        # Read configuration from environment variables
        # These are injected by the benchmark setup
        level = int(os.environ.get("KERNELBENCH_LEVEL", "1"))
        problem_id = int(os.environ.get("KERNELBENCH_PROBLEM_ID", "1"))
        eval_mode = os.environ.get("KERNELBENCH_EVAL_MODE", "local")
        gpu = os.environ.get("KERNELBENCH_GPU", "H100")
        num_correct_trials = int(os.environ.get("KERNELBENCH_NUM_CORRECT_TRIALS", "5"))
        num_perf_trials = int(os.environ.get("KERNELBENCH_NUM_PERF_TRIALS", "100"))
        timeout = int(os.environ.get("KERNELBENCH_TIMEOUT", "300"))

        # Read the program and wrap it in ModelNew class for KernelBench format
        with open(program_path, "r") as f:
            program_content = f.read()

        is_triton = bool(
            re.search(r"^(import triton|from triton)", program_content, flags=re.MULTILINE)
        )

        # Create a temporary file with ModelNew wrapper
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
            # Replace class Model with class ModelNew (if not already ModelNew)
            converted_content = program_content
            if "class ModelNew" not in converted_content:
                converted_content = re.sub(
                    r"^class Model(?=[(:])", "class ModelNew", converted_content, flags=re.MULTILINE
                )
                # Fix super() calls - use modern Python 3 super() without arguments
                converted_content = re.sub(r"super\(Model,\s*self\)", "super()", converted_content)
                converted_content = re.sub(r"super\(Model,\s*cls\)", "super()", converted_content)

            tmp_file.write(converted_content)
            kernel_src_path = tmp_file.name

        try:
            # Ensure run_and_check.py is available (downloads if needed)
            evaluator_dir = Path(__file__).parent
            run_and_check_path = ensure_run_and_check(evaluator_dir)

            # Build command to run run_and_check.py
            cmd = [
                sys.executable,
                str(run_and_check_path),
                "ref_origin=kernelbench",
                f"level={level}",
                f"problem_id={problem_id}",
                f"kernel_src_path={kernel_src_path}",
                f"eval_mode={eval_mode}",
                f"gpu={gpu}",
                f"num_correct_trials={num_correct_trials}",
                f"num_perf_trials={num_perf_trials}",
                f"timeout={timeout}",
                "check_kernel=False",  # Disable static checker to allow reference code
            ]

            # Setting the backend is important for KernelBench triton evaluation to work
            if is_triton:
                cmd.append("backend=triton")

            # Set up environment
            env = os.environ.copy()

            # Run the evaluation from the evaluator directory
            print(f"[INFO] Running evaluation command: {' '.join(cmd)}", file=sys.stderr)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(evaluator_dir),
                env=env,
            )
        finally:
            # Clean up temporary file
            try:
                os.unlink(kernel_src_path)
            except Exception:
                pass

        # Parse the output to extract speedup
        stdout = result.stdout
        stderr = result.stderr

        if result.returncode != 0:
            print(
                f"[ERROR] Evaluation failed with return code {result.returncode}", file=sys.stderr
            )
            print(f"[ERROR] stdout: {stdout}", file=sys.stderr)
            print(f"[ERROR] stderr: {stderr}", file=sys.stderr)
            return {
                "combined_score": -100.0,
                "error": f"Evaluation subprocess failed: {stderr[:500]}",
                "return_code": result.returncode,
            }

        # Extract speedup from output
        speedup_eager = None
        speedup_compile = None
        kernel_time = None
        ref_eager_time = None

        for line in stdout.split("\n"):
            if "Speedup over eager:" in line:
                match = re.search(r"([0-9.]+)x", line)
                if match:
                    speedup_eager = float(match.group(1))
            elif "Speedup over torch.compile:" in line:
                match = re.search(r"([0-9.]+)x", line)
                if match:
                    speedup_compile = float(match.group(1))
            elif "Custom Kernel exec time:" in line:
                match = re.search(r"([0-9.]+) ms", line)
                if match:
                    kernel_time = float(match.group(1))
            elif "PyTorch Reference Eager exec time:" in line:
                match = re.search(r"([0-9.]+) ms", line)
                if match:
                    ref_eager_time = float(match.group(1))

        # If we found speedup, use it as the score
        if speedup_eager is not None and speedup_eager > 0:
            return {
                "combined_score": float(speedup_eager),
                "speedup_over_eager": speedup_eager,
                "speedup_over_compile": speedup_compile,
                "kernel_time_ms": kernel_time,
                "ref_eager_time_ms": ref_eager_time,
                "eval_mode": eval_mode,
                "gpu": gpu,
            }
        else:
            # Kernel failed correctness or didn't compile
            # Extract only relevant output starting from [Eval]
            stdout_excerpt = stdout
            if "[Eval]" in stdout:
                eval_start = stdout.find("[Eval]")
                stdout_excerpt = stdout[eval_start:]

            # Take last 5000 chars if too long
            if len(stdout_excerpt) > 5000:
                stdout_excerpt = stdout_excerpt[-5000:]

            return {
                "combined_score": -100.0,
                "error": "Kernel failed correctness check or did not compile",
                "stdout_excerpt": stdout_excerpt,
            }

    except subprocess.TimeoutExpired:
        return {
            "combined_score": -1.0,
            "error": f"Evaluation timed out after {timeout} seconds",
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "combined_score": -100.0,
            "error": f"Error during evaluation: {str(e)}",
            "error_type": type(e).__name__,
        }


if __name__ == "__main__":
    # Backwards-compat: bridges old evaluate() -> dict to the container JSON
    # protocol.  wrapper.py is copied from skydiscover/evaluation/wrapper.py.
    from wrapper import run

    run(evaluate)
