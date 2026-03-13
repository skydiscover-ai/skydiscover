#!/usr/bin/env python3
"""Generic wrapper: bridges evaluate(program_path) -> dict to the container protocol.

Drop this file alongside any Python evaluator.py that defines evaluate(program_path).
The wrapper imports evaluator.py, calls evaluate(), and prints a JSON result to stdout
in the format expected by SkyDiscover's ContainerizedEvaluator.
"""

import importlib.util
import json
import os
import sys
import traceback

# Add the benchmark directory to sys.path so the evaluator's sibling
# imports (e.g. "from utils import *") resolve correctly.
BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCHMARK_DIR)

# Load the evaluator module
_spec = importlib.util.spec_from_file_location(
    "evaluator", os.path.join(BENCHMARK_DIR, "evaluator.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def main():
    if len(sys.argv) < 2:
        print("Usage: wrapper.py <program_path>", file=sys.stderr)
        sys.exit(1)

    program_path = sys.argv[1]

    # Redirect stdout → stderr during evaluation so debug prints from
    # the evaluator don't contaminate the JSON output on stdout.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        result = _mod.evaluate(program_path)
    except Exception as e:
        sys.stdout = real_stdout
        print(json.dumps({
            "status": "error",
            "combined_score": 0.0,
            "metrics": {"combined_score": 0.0},
            "artifacts": {"error": str(e), "traceback": traceback.format_exc()},
        }))
        return
    sys.stdout = real_stdout

    if not isinstance(result, dict):
        print(json.dumps({
            "status": "error",
            "combined_score": 0.0,
            "metrics": {"combined_score": 0.0},
            "artifacts": {"error": f"evaluate() returned {type(result).__name__}, expected dict"},
        }))
        return

    # Separate numeric metrics from non-numeric artifacts.
    metrics = {}
    artifacts = {}
    for k, v in result.items():
        if isinstance(v, bool):
            metrics[k] = float(v)
        elif isinstance(v, (int, float)):
            metrics[k] = float(v)
        elif isinstance(v, str):
            artifacts[k] = v
        elif isinstance(v, (list, dict)):
            artifacts[k] = json.dumps(v)

    if "combined_score" not in metrics:
        metrics["combined_score"] = 0.0

    status = "error" if "error" in artifacts else "success"
    output = {
        "status": status,
        "combined_score": metrics["combined_score"],
        "metrics": metrics,
    }
    if artifacts:
        output["artifacts"] = artifacts

    print(json.dumps(output))


if __name__ == "__main__":
    main()
