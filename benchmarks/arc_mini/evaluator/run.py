#!/usr/bin/env python3
"""
Evaluate a candidate program on the arc_mini color-swap task.

Usage: run.py <program_path> <data_file>

The candidate program must define:
    def solve(grid: list[list[int]]) -> list[list[int]]: ...

Each example in the data file is {"input": grid, "output": grid}.
Score = fraction of examples solved exactly correctly.

Writes a single JSON object to stdout following the SkyDiscover evaluator schema.
"""

import importlib.util
import json
import sys
import traceback


def load_solver(program_path):
    spec = importlib.util.spec_from_file_location("program", program_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "solve"):
        raise AttributeError("Candidate program must define solve(grid)")
    return mod.solve


def grids_equal(a, b):
    if len(a) != len(b):
        return False
    return all(
        len(row_a) == len(row_b) and all(x == y for x, y in zip(row_a, row_b))
        for row_a, row_b in zip(a, b)
    )


def main():
    if len(sys.argv) != 3:
        print("Usage: run.py <program_path> <data_file>", file=sys.stderr)
        sys.exit(1)

    program_path, data_file = sys.argv[1], sys.argv[2]

    with open(data_file) as f:
        examples = json.load(f)

    try:
        solve = load_solver(program_path)
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "combined_score": 0.0,
            "metrics": {"combined_score": 0.0, "accuracy": 0.0},
            "artifacts": {"error": str(e), "traceback": traceback.format_exc()},
        }))
        return

    n_correct = 0
    per_example = []
    for i, ex in enumerate(examples):
        try:
            predicted = solve(ex["input"])
            correct = grids_equal(predicted, ex["output"])
        except Exception as e:
            correct = False
            per_example.append({"index": i, "error": str(e)})
            continue
        if not correct:
            per_example.append({
                "index": i,
                "expected": ex["output"],
                "got": predicted,
            })
        n_correct += int(correct)

    accuracy = n_correct / len(examples) if examples else 0.0

    result = {
        "status": "success",
        "combined_score": accuracy,
        "metrics": {
            "combined_score": accuracy,
            "accuracy": accuracy,
            "n_correct": float(n_correct),
            "n_total": float(len(examples)),
        },
    }
    if per_example:
        result["artifacts"] = {"failures": json.dumps(per_example)}

    print(json.dumps(result))


if __name__ == "__main__":
    main()
