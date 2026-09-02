"""Local non-LLM topology search for QNN benchmark smoke measurement.

Mutates gate topologies for a fixed budget and reports best test accuracy.
Useful when no LLM API key is available; SkyDiscover LLM search should beat this.
"""

from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from evaluator import evaluate  # noqa: E402


GATES = [
    {"gate": "RY", "qubit": 0},
    {"gate": "RY", "qubit": 1},
    {"gate": "RX", "qubit": 0},
    {"gate": "RX", "qubit": 1},
    {"gate": "RZ", "qubit": 0},
    {"gate": "RZ", "qubit": 1},
    {"gate": "CNOT", "control": 0, "target": 1},
    {"gate": "CNOT", "control": 1, "target": 0},
]


def _write_program(path: Path, topology: list[dict]) -> None:
    body = "def build_topology():\n    return " + repr(topology) + "\n"
    path.write_text(body)


def _mutate(topology: list[dict], rng: random.Random) -> list[dict]:
    t = copy.deepcopy(topology)
    op = rng.choice(["add", "remove", "replace", "swap"])
    if op == "add" and len(t) < 16:
        t.insert(rng.randrange(len(t) + 1), copy.deepcopy(rng.choice(GATES)))
    elif op == "remove" and len(t) > 3:
        del t[rng.randrange(len(t))]
    elif op == "replace":
        t[rng.randrange(len(t))] = copy.deepcopy(rng.choice(GATES))
    elif op == "swap" and len(t) > 1:
        i, j = rng.sample(range(len(t)), 2)
        t[i], t[j] = t[j], t[i]
    return t


def main(iterations: int = 12, seed: int = 0) -> dict:
    rng = random.Random(seed)
    work = ROOT / "_search_candidate.py"
    best_topo = [
        {"gate": "RY", "qubit": 0},
        {"gate": "RY", "qubit": 1},
        {"gate": "CNOT", "control": 0, "target": 1},
        {"gate": "RZ", "qubit": 0},
        {"gate": "RY", "qubit": 1},
    ]
    _write_program(work, best_topo)
    best = evaluate(str(work))
    history = [best["accuracy"]]
    print(f"iter 0 accuracy={best['accuracy']:.4f} gates={best.get('n_gates')}")

    for i in range(1, iterations + 1):
        cand = _mutate(best_topo, rng)
        _write_program(work, cand)
        metrics = evaluate(str(work))
        history.append(metrics["accuracy"])
        print(f"iter {i} accuracy={metrics['accuracy']:.4f} gates={metrics.get('n_gates')}")
        if metrics["accuracy"] > best["accuracy"]:
            best = metrics
            best_topo = cand
            print(f"  new best: {best['accuracy']:.4f}")

    if work.exists():
        work.unlink()
    result = {
        "iterations": iterations,
        "best_accuracy": best["accuracy"],
        "best_topology": best_topo,
        "history": history,
    }
    print(
        f"\nbest after {iterations} iterations: "
        f"accuracy={result['best_accuracy']:.4%} topology={best_topo}"
    )
    return result


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    main(iters)
