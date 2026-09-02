# QNN Circuit Topology Optimization

Discover a **quantum neural network (QNN) circuit topology** (ansatz gate sequence) that maximizes binary classification accuracy on a 2D synthetic dataset, using a pure-NumPy 2-qubit statevector simulator.

## Task

Evolve `build_topology()` in `initial_program.py` so the returned gate list yields high test accuracy after a short parameter fit.

Allowed gates (strings):

| Gate | Meaning |
|------|---------|
| `RX`, `RY`, `RZ` | Parameterized single-qubit rotation on qubit 0 or 1 (qubit encoded in gate dict) |
| `CNOT` | Entangling CNOT(control → target) |
| `H` | Hadamard (fixed, no parameter) |

Each topology entry is a dict, e.g. `{"gate": "RY", "qubit": 0}` or `{"gate": "CNOT", "control": 0, "target": 1}`.

## Run

```bash
uv run skydiscover-run \
  benchmarks/qnn_circuit_topology/initial_program.py \
  benchmarks/qnn_circuit_topology/evaluator.py \
  -c benchmarks/qnn_circuit_topology/config.yaml \
  -s best_of_n -i 12
```

Containerized:

```bash
uv run skydiscover-run \
  benchmarks/qnn_circuit_topology/initial_program.py \
  benchmarks/qnn_circuit_topology/evaluator \
  -c benchmarks/qnn_circuit_topology/config.yaml \
  -s adaevolve -i 50
```

## Scoring

- `accuracy`: held-out classification accuracy in `[0, 1]`
- `combined_score`: same as accuracy (primary)
- Invalid topologies / runtime errors → score `0`
- Validator cap: **24 gates**. Prefer compact circuits (`<= 16`); deeper is not always better.
- Evaluator timeout is **600s**. The fit is batched/vectorized (cached static gates + parameter-shift) so a 24-gate circuit finishes in seconds, not minutes. CI still runs the 5-gate baseline.

## Measured smoke results (no LLM)

| Method | Iterations | Test accuracy |
|--------|------------|---------------|
| Baseline `initial_program.py` | 0 | ~0.70 |
| Local mutation search (`local_search_smoke.py`) | 12 | **0.833** |
| Strong hand-written topology | — | ~0.917 |

LLM-guided SkyDiscover search (`best_of_n` / `adaevolve`) should be run for the final reported number.

## Design notes

- Feature map: angle-encode 2D inputs as `RY(x0)`, `RY(x1)` on two qubits before the ansatz.
- Readout: measure ⟨Z⟩ on qubit 0; label = sign.
- Parameters are fit inside the evaluator (not evolved); only **topology** is discovered.
