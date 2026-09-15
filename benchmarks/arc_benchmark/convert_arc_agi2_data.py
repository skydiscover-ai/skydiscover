#!/usr/bin/env python3
"""
Convert ARC-AGI-2-style data (data/training/*.json, data/evaluation/*.json)
into the format expected by this benchmark:
  - arc-agi_{split}_challenges.json  (task_id -> { train, test with inputs only })
  - arc-agi_{split}_solutions.json   (task_id -> list of test output grids)

Usage (from benchmarks/arc_benchmark, with data already in ./data/training and ./data/evaluation):
  OUT_DIR=./data python3 convert_arc_agi2_data.py .

Or with an external ARC-AGI-2 clone:
  python3 convert_arc_agi2_data.py /path/to/ARC-AGI-2
  # Writes into that path by default; set OUT_DIR to write elsewhere.
"""

import json
import os
import sys


def convert_split(repo_root: str, split: str, out_dir: str) -> None:
    """Convert data/{split}/*.json into challenges + solutions JSON."""
    split_dir = os.path.join(repo_root, "data", split)
    if not os.path.isdir(split_dir):
        print(f"Skip {split}: no directory {split_dir}")
        return

    challenges = {}
    solutions = {}

    for name in sorted(os.listdir(split_dir)):
        if not name.endswith(".json"):
            continue
        task_id = name[:-5]  # strip .json
        path = os.path.join(split_dir, name)
        with open(path, "r") as f:
            task = json.load(f)
        # Challenge: train as-is; test with only "input" (no output)
        challenges[task_id] = {
            "train": task["train"],
            "test": [{"input": p["input"]} for p in task["test"]],
        }
        # Solutions: list of test output grids
        solutions[task_id] = [p["output"] for p in task["test"]]

    challenges_path = os.path.join(out_dir, f"arc-agi_{split}_challenges.json")
    solutions_path = os.path.join(out_dir, f"arc-agi_{split}_solutions.json")
    with open(challenges_path, "w") as f:
        json.dump(challenges, f)
    with open(solutions_path, "w") as f:
        json.dump(solutions, f)
    print(f"Wrote {challenges_path} ({len(challenges)} tasks)")
    print(f"Wrote {solutions_path} ({len(solutions)} tasks)")


def main():
    repo_root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
    out_dir = os.getenv("OUT_DIR", repo_root)
    for split in ("training", "evaluation"):
        convert_split(repo_root, split, out_dir)


if __name__ == "__main__":
    main()
