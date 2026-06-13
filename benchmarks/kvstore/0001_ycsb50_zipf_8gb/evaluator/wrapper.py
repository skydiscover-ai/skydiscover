"""Backwards-compat wrapper for old Python-based evaluators.

Old-style evaluators define ``evaluate(program_path) -> dict``.  This module
bridges that interface to the container JSON protocol expected by
ContainerizedEvaluator.

Usage — add this to the bottom of your evaluator.py::

    if __name__ == "__main__":
        from wrapper import run
        run(evaluate)
"""

import json
import sys
import traceback


def run(evaluate_fn):
    """Call *evaluate_fn*, format the result as container-protocol JSON on stdout.

    * Reads ``sys.argv[1]`` as the program path.
    * Redirects stdout → stderr while *evaluate_fn* runs so that debug prints
      don't contaminate the JSON output.
    * Separates numeric metrics from non-numeric artifacts.
    * Guarantees ``combined_score`` is always present in metrics.
    """
    if len(sys.argv) < 2:
        print("Usage: evaluator.py <program_path>", file=sys.stderr)
        sys.exit(1)

    program_path = sys.argv[1]

    # Redirect stdout → stderr during evaluation so debug prints from
    # the evaluator don't contaminate the JSON output on stdout.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        result = evaluate_fn(program_path)
    except Exception as e:
        sys.stdout = real_stdout
        print(
            json.dumps(
                {
                    "status": "error",
                    "combined_score": 0.0,
                    "metrics": {"combined_score": 0.0},
                    "artifacts": {
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    },
                }
            )
        )
        return
    sys.stdout = real_stdout

    if not isinstance(result, dict):
        print(
            json.dumps(
                {
                    "status": "error",
                    "combined_score": 0.0,
                    "metrics": {"combined_score": 0.0},
                    "artifacts": {
                        "error": f"evaluate() returned {type(result).__name__}, expected dict"
                    },
                }
            )
        )
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
