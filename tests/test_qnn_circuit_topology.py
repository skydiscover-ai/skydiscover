"""Unit smoke tests for the QNN circuit topology evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "qnn_circuit_topology"
sys.path.insert(0, str(ROOT))

from evaluator import _validate_topology, evaluate  # noqa: E402


def test_validate_topology_accepts_baseline():
    topo = [
        {"gate": "RY", "qubit": 0},
        {"gate": "RY", "qubit": 1},
        {"gate": "CNOT", "control": 0, "target": 1},
    ]
    _validate_topology(topo)


def test_validate_topology_rejects_bad_cnot():
    with pytest.raises(ValueError):
        _validate_topology([{"gate": "CNOT", "control": 0, "target": 0}])


def test_evaluate_baseline_program():
    metrics = evaluate(str(ROOT / "initial_program.py"))
    assert "combined_score" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["n_gates"] >= 1
