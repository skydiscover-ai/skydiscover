"""Baseline QNN circuit topology for 2-qubit binary classification.

Evolve ``build_topology`` to discover a better ansatz. The evaluator angle-encodes
2D inputs, applies this topology with trainable parameters, and scores accuracy.
"""

# EVOLVE-BLOCK-START
def build_topology():
    """Return a list of gate dicts defining the variational ansatz topology.

    Supported gates:
      - {"gate": "RX"|"RY"|"RZ", "qubit": 0|1}
      - {"gate": "H", "qubit": 0|1}
      - {"gate": "CNOT", "control": 0|1, "target": 0|1}  # control != target

    Keep the circuit within the evaluator cap (24 gates). Compact ansatze
    (<= 16 gates) usually fit better on this toy task.
    """
    return [
        {"gate": "RY", "qubit": 0},
        {"gate": "RY", "qubit": 1},
        {"gate": "CNOT", "control": 0, "target": 1},
        {"gate": "RZ", "qubit": 0},
        {"gate": "RY", "qubit": 1},
    ]


# EVOLVE-BLOCK-END


def run():
    """Optional local smoke entrypoint."""
    topo = build_topology()
    return topo, len(topo)


if __name__ == "__main__":
    topology, n = run()
    print(f"baseline topology ({n} gates): {topology}")
