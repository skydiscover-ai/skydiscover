"""Container evaluator for GPU Mode -- bridges shared_eval to JSON protocol."""
import json
import sys
import traceback

# shared_eval.py and reference.py are in the same directory
from shared_eval import evaluate


def main():
    if len(sys.argv) < 2:
        print("Usage: evaluator.py <program_path>", file=sys.stderr)
        sys.exit(1)

    program_path = sys.argv[1]

    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        result = evaluate(program_path)
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

    # EvaluationResult has .metrics and .artifacts
    metrics = {k: float(v) for k, v in result.metrics.items() if isinstance(v, (int, float, bool))}
    artifacts = {k: str(v) for k, v in result.artifacts.items()}

    if "combined_score" not in metrics:
        metrics["combined_score"] = 0.0

    status = "error" if "error" in artifacts else "success"
    output = {
        "status": status,
        "combined_score": metrics.get("combined_score", 0.0),
        "metrics": metrics,
    }
    if artifacts:
        output["artifacts"] = artifacts
    print(json.dumps(output))


if __name__ == "__main__":
    main()
