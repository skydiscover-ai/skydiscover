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
    # Baseline must stay in CI-smoke range (was ~20s; must not approach the 600s cap).
    assert metrics["eval_time"] < 30.0


def test_validate_topology_cap_is_24():
    from evaluator import MAX_GATES

    assert MAX_GATES == 24
    ok = [{"gate": "RY", "qubit": i % 2} for i in range(24)]
    _validate_topology(ok)
    with pytest.raises(ValueError, match="24"):
        _validate_topology(ok + [{"gate": "H", "qubit": 0}])


def test_sixteen_gate_fit_finishes_well_under_old_timeout(tmp_path):
    """A 16-gate / many-param ansatz must not hang past the old 120s cap."""
    gates = []
    for i in range(8):
        gates.append({"gate": "RY", "qubit": 0})
        gates.append({"gate": "RX", "qubit": 1})
        if i % 2 == 0:
            gates.append({"gate": "CNOT", "control": 0, "target": 1})
    topology = gates[:16]
    program = tmp_path / "sixteen.py"
    program.write_text("def build_topology():\n    return " + repr(topology) + "\n")
    metrics = evaluate(str(program))
    assert "error" not in metrics
    assert metrics["n_gates"] == 16
    assert metrics["eval_time"] < 45.0
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_max_gate_cap_fit_finishes_under_timeout(tmp_path):
    """A 24-gate circuit (validator max) must finish far below the 600s timeout."""
    topology = [{"gate": "RY", "qubit": i % 2} for i in range(20)]
    topology += [
        {"gate": "CNOT", "control": 0, "target": 1},
        {"gate": "CNOT", "control": 1, "target": 0},
        {"gate": "H", "qubit": 0},
        {"gate": "RZ", "qubit": 1},
    ]
    assert len(topology) == 24
    program = tmp_path / "twentyfour.py"
    program.write_text("def build_topology():\n    return " + repr(topology) + "\n")
    metrics = evaluate(str(program))
    assert "error" not in metrics
    assert metrics["n_gates"] == 24
    assert metrics["eval_time"] < 45.0
    assert 0.0 <= metrics["accuracy"] <= 1.0
