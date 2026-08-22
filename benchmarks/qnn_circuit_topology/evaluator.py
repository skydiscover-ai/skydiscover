"""Evaluator for QNN circuit topology discovery (2-qubit binary classification)."""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import time
from typing import Any

import numpy as np

N_QUBITS = 2
DIM = 1 << N_QUBITS
MAX_GATES = 24
TRAIN_SIZE = 120
TEST_SIZE = 60
STEPS = 50
LR = 0.15
RESTARTS = 2
BATCH = 48
SEED = 0
_PARAM_SHIFT = 0.5 * math.pi


def _ry(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rx(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def _rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
        dtype=np.complex128,
    )


def _h() -> np.ndarray:
    return (1 / math.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)


def _embed_1q(gate: np.ndarray, qubit: int) -> np.ndarray:
    ops = [np.eye(2, dtype=np.complex128)] * N_QUBITS
    ops[qubit] = gate
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def _cnot(control: int, target: int) -> np.ndarray:
    u = np.zeros((DIM, DIM), dtype=np.complex128)
    for i in range(DIM):
        bits = [(i >> q) & 1 for q in range(N_QUBITS)]
        if bits[control] == 1:
            bits[target] ^= 1
        j = sum(bits[q] << q for q in range(N_QUBITS))
        u[j, i] = 1
    return u


_ROT = {"RX": _rx, "RY": _ry, "RZ": _rz}
_H_MAT = _h()
_H_EMBED = tuple(_embed_1q(_H_MAT, q) for q in range(N_QUBITS))
_CNOT_CACHE = {
    (0, 1): _cnot(0, 1),
    (1, 0): _cnot(1, 0),
}
# ⟨Z on qubit 0⟩ with q0 as LSB.
_Z0 = np.array([1.0 if (i & 1) == 0 else -1.0 for i in range(DIM)], dtype=np.float64)


def _validate_topology(topology: list[dict[str, Any]]) -> None:
    if not isinstance(topology, list) or not topology:
        raise ValueError("topology must be a non-empty list of gate dicts")
    if len(topology) > MAX_GATES:
        raise ValueError(f"topology too long (>{MAX_GATES} gates); keep circuits compact")
    for g in topology:
        if not isinstance(g, dict) or "gate" not in g:
            raise ValueError(f"invalid gate entry: {g}")
        name = str(g["gate"]).upper()
        if name in {"RX", "RY", "RZ", "H"}:
            q = int(g["qubit"])
            if q not in (0, 1):
                raise ValueError(f"qubit out of range in {g}")
        elif name == "CNOT":
            c, t = int(g["control"]), int(g["target"])
            if c not in (0, 1) or t not in (0, 1) or c == t:
                raise ValueError(f"invalid CNOT: {g}")
        else:
            raise ValueError(f"unsupported gate: {name}")


def _n_params(topology: list[dict[str, Any]]) -> int:
    return sum(1 for g in topology if str(g["gate"]).upper() in {"RX", "RY", "RZ"})


def _compile(topology: list[dict[str, Any]]) -> list[tuple]:
    """Cache static 4x4 gate matrices; parameterized rotations stay symbolic."""
    layers: list[tuple] = []
    for g in topology:
        name = str(g["gate"]).upper()
        if name == "H":
            layers.append(("static", _H_EMBED[int(g["qubit"])]))
        elif name == "CNOT":
            layers.append(("static", _CNOT_CACHE[(int(g["control"]), int(g["target"]))]))
        else:
            layers.append(("param", name, int(g["qubit"])))
    return layers


def _encode_features(x: np.ndarray) -> np.ndarray:
    """Angle-encode 2D inputs as RY(x0)⊗RY(x1) on |00>, vector of shape (N, DIM)."""
    n = x.shape[0]
    states = np.zeros((n, DIM), dtype=np.complex128)
    states[:, 0] = 1.0
    for i in range(n):
        states[i] = _embed_1q(_ry(float(x[i, 0])), 0) @ states[i]
        states[i] = _embed_1q(_ry(float(x[i, 1])), 1) @ states[i]
    return states


def _apply_compiled(states: np.ndarray, compiled: list[tuple], params: np.ndarray) -> np.ndarray:
    """Apply compiled ansatz to a batch of encoded states, shape (B, DIM)."""
    out = states
    p_idx = 0
    for layer in compiled:
        if layer[0] == "static":
            u = layer[1]
        else:
            _, name, qubit = layer
            u = _embed_1q(_ROT[name](float(params[p_idx])), qubit)
            p_idx += 1
        out = out @ u.T
    return out


def _expectation_z0_batch(states: np.ndarray) -> np.ndarray:
    probs = np.abs(states) ** 2
    return probs.real @ _Z0


def _apply_circuit(x: np.ndarray, topology: list[dict[str, Any]], params: np.ndarray) -> np.ndarray:
    compiled = _compile(topology)
    encoded = _encode_features(np.asarray(x, dtype=np.float64).reshape(1, 2))
    return _apply_compiled(encoded, compiled, params)[0]


def _expectation_z0(state: np.ndarray) -> float:
    return float(_expectation_z0_batch(np.asarray(state, dtype=np.complex128).reshape(1, -1))[0])


def _make_dataset(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # Angle-encoded 2D inputs; labels from a nonlinear but learnable decision rule.
    x = rng.uniform(-math.pi / 2, math.pi / 2, size=(n, 2))
    # XOR-like rule in the sign pattern of the two features — needs entanglement to fit well.
    y = (np.sign(x[:, 0]) * np.sign(x[:, 1]) > 0).astype(np.float64)
    y = 2 * y - 1  # ±1
    # Avoid zeros at axes
    y[y == 0] = 1
    return x, y


def _accuracy_encoded(
    encoded: np.ndarray,
    y: np.ndarray,
    compiled: list[tuple],
    params: np.ndarray,
) -> float:
    z = _expectation_z0_batch(_apply_compiled(encoded, compiled, params))
    preds = np.where(z >= 0.0, 1.0, -1.0)
    return float(np.mean(preds == y))


def _predict(x: np.ndarray, topology: list[dict[str, Any]], params: np.ndarray) -> np.ndarray:
    compiled = _compile(topology)
    encoded = _encode_features(np.asarray(x, dtype=np.float64))
    z = _expectation_z0_batch(_apply_compiled(encoded, compiled, params))
    return np.where(z >= 0.0, 1.0, -1.0)


def _accuracy(x: np.ndarray, y: np.ndarray, topology: list[dict[str, Any]], params: np.ndarray) -> float:
    preds = _predict(x, topology, params)
    return float(np.mean(preds == y))


def _fit_once(
    compiled: list[tuple],
    encoded: np.ndarray,
    y: np.ndarray,
    n_params: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_params == 0:
        return np.zeros(0)
    params = rng.normal(0, 0.4, size=n_params)
    for _ in range(STEPS):
        idx = rng.choice(len(encoded), size=min(BATCH, len(encoded)), replace=False)
        enc_b = encoded[idx]
        y_b = y[idx]
        z = _expectation_z0_batch(_apply_compiled(enc_b, compiled, params))
        active = (1.0 - y_b * z) > 0.0
        grads = np.zeros(n_params, dtype=np.float64)
        for j in range(n_params):
            plus = params.copy()
            minus = params.copy()
            plus[j] += _PARAM_SHIFT
            minus[j] -= _PARAM_SHIFT
            zp = _expectation_z0_batch(_apply_compiled(enc_b, compiled, plus))
            zm = _expectation_z0_batch(_apply_compiled(enc_b, compiled, minus))
            dz = 0.5 * (zp - zm)
            grads[j] = float(np.sum((-y_b * dz) * active))
        params -= LR * grads
    return params


def _fit(topology: list[dict[str, Any]], x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    compiled = _compile(topology)
    encoded = _encode_features(x)
    n_params = _n_params(topology)
    best_params = None
    best_acc = -1.0
    for _ in range(RESTARTS):
        params = _fit_once(compiled, encoded, y, n_params, rng)
        acc = _accuracy_encoded(encoded, y, compiled, params)
        if acc > best_acc:
            best_acc = acc
            best_params = params
    assert best_params is not None
    return best_params


def _load_topology(program_path: str) -> list[dict[str, Any]]:
    abs_path = os.path.abspath(program_path)
    module_name = "qnn_candidate"
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load program: {program_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "build_topology"):
        raise AttributeError("program must define build_topology()")
    return module.build_topology()


def evaluate(program_path: str) -> dict[str, Any]:
    start = time.time()
    try:
        topology = _load_topology(program_path)
        _validate_topology(topology)
        rng = np.random.default_rng(SEED)
        x_train, y_train = _make_dataset(TRAIN_SIZE, rng)
        x_test, y_test = _make_dataset(TEST_SIZE, rng)
        params = _fit(topology, x_train, y_train, rng)
        acc = _accuracy(x_test, y_test, topology, params)
        train_acc = _accuracy(x_train, y_train, topology, params)
        return {
            "combined_score": acc,
            "accuracy": acc,
            "train_accuracy": train_acc,
            "n_gates": float(len(topology)),
            "n_params": float(_n_params(topology)),
            "eval_time": time.time() - start,
        }
    except Exception as exc:
        return {
            "combined_score": 0.0,
            "accuracy": 0.0,
            "error": str(exc),
            "eval_time": time.time() - start,
        }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "initial_program.py")
    result = evaluate(path)
    print(result)
