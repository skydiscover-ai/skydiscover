"""Baseline MLP architecture for Post-NAS microbenchmark (#2).

Evolve ``build_architecture`` — a list of layer dicts. The evaluator trains
the resulting feed-forward net with NumPy SGD on a tiny synthetic task.
"""

from __future__ import annotations

from typing import Any


# EVOLVE-BLOCK-START
def build_architecture(input_dim: int = 8, num_classes: int = 3) -> list[dict[str, Any]]:
    """Return an ordered list of layer specs for a feed-forward classifier.

    Supported layers:
      - {"type": "linear", "out_features": int}
      - {"type": "relu"} | {"type": "tanh"} | {"type": "sigmoid"}
      - {"type": "dropout", "p": float}   # train-time only; ignored at eval

    Constraints enforced by the evaluator:
      - first effective linear layer must accept ``input_dim``
      - final linear layer must output ``num_classes``
      - at most 8 layers total; hidden width ≤ 64
    """
    return [
        {"type": "linear", "out_features": 16},
        {"type": "relu"},
        {"type": "linear", "out_features": num_classes},
    ]


# EVOLVE-BLOCK-END
