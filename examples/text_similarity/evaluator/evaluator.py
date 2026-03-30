#!/usr/bin/env python3
"""Score a candidate text-similarity function against human judgments."""

import importlib.util
import json
import random
import sys

from scipy.stats import spearmanr

PAIRS = json.load(open("/benchmark/pairs.json"))


def main():
    program_path = sys.argv[1]

    # Load the candidate's similarity() function
    spec = importlib.util.spec_from_file_location("candidate", program_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Score every pair
    predicted = []
    for a, b, _ in PAIRS:
        try:
            score = max(0.0, min(1.0, float(mod.similarity(a, b))))
        except Exception:
            score = 0.0
        predicted.append(score)

    human = [h for _, _, h in PAIRS]
    correlation = spearmanr(predicted, human).statistic

    samples = random.sample(range(len(PAIRS)), 3)
    lines = [f"Spearman correlation: {correlation:.4f}", ""]
    for i in samples:
        a, b, h = PAIRS[i]
        lines.append(f"  '{a}' vs '{b}': predicted={predicted[i]:.2f}, human={h:.2f}")

    print(json.dumps({
        "status": "success",
        "combined_score": round(max(0.0, correlation), 4),
        "artifacts": {"feedback": "\n".join(lines)},
    }))


if __name__ == "__main__":
    main()
