#!/usr/bin/env python3
"""
Evaluate a circle-packing candidate program (n=26 circles in a unit square).

Usage: run.py <program_path>

The candidate must define: run_packing() -> (centers, radii, sum_radii)

Writes a single JSON object to stdout following the SkyDiscover evaluator schema.
"""

import importlib.util
import json
import signal
import sys
import time
import traceback

import numpy as np

N = 26
TIMEOUT_SECONDS = 300


def _alarm_handler(signum, frame):
    raise TimeoutError(f"Program timed out after {TIMEOUT_SECONDS}s")


def run_program(program_path):
    """Import and call run_packing() from the candidate program."""
    spec = importlib.util.spec_from_file_location("program", program_path)
    prog = importlib.util.module_from_spec(spec)
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(TIMEOUT_SECONDS)
    try:
        spec.loader.exec_module(prog)
        return prog.run_packing()
    finally:
        signal.alarm(0)


def validate_packing(centers, radii):
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False, "NaN values in output"
    for i in range(len(radii)):
        if radii[i] < 0:
            return False, f"Circle {i} has negative radius"
    for i in range(len(radii)):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-6 or x + r > 1 + 1e-6 or y - r < -1e-6 or y + r > 1 + 1e-6:
            return False, f"Circle {i} outside unit square"
    for i in range(len(radii)):
        for j in range(i + 1, len(radii)):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-6:
                return False, f"Circles {i} and {j} overlap"
    return True, ""


def fail(status, reason, elapsed=0.0):
    print(json.dumps({
        "status": status,
        "combined_score": 0.0,
        "metrics": {"combined_score": 0.0, "sum_radii": 0.0, "validity": 0.0,
                    "eval_time": elapsed},
        "artifacts": {"error": reason},
    }))


def main():
    if len(sys.argv) != 2:
        print("Usage: run.py <program_path>", file=sys.stderr)
        sys.exit(1)

    program_path = sys.argv[1]

    def log(msg):
        with open("/tmp/eval.log", "a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

    start = time.time()
    try:
        centers, radii, _ = run_program(program_path)
    except TimeoutError as e:
        log(f"timeout: {e}")
        fail("timeout", str(e))
        return
    except Exception as e:
        log(f"error: {e}")
        fail("error", f"{e}\n{traceback.format_exc()}")
        return
    elapsed = time.time() - start

    centers = np.asarray(centers)
    radii = np.asarray(radii)

    if centers.shape != (N, 2) or radii.shape != (N,):
        log(f"bad shapes: centers={centers.shape}, radii={radii.shape}")
        fail("success", f"Wrong shapes: centers={centers.shape}, radii={radii.shape}", elapsed)
        return

    valid, reason = validate_packing(centers, radii)
    sum_radii = float(np.sum(radii)) if valid else 0.0
    log(f"done in {elapsed:.3f}s — sum_radii={sum_radii:.6f} valid={valid}" +
        (f" ({reason})" if not valid else ""))

    print(json.dumps({
        "status": "success",
        "combined_score": sum_radii,
        "metrics": {
            "combined_score": sum_radii,
            "sum_radii": sum_radii,
            "validity": 1.0 if valid else 0.0,
            "eval_time": elapsed,
        },
        "artifacts": ({} if valid else {"error": reason}),
    }))


if __name__ == "__main__":
    main()
