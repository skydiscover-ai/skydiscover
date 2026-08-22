"""Post-NAS microbenchmark: evolve MLP topologies, train with NumPy SGD."""

from __future__ import annotations

import copy
import importlib.util
import math
import os
import time
import traceback
from typing import Any

import numpy as np

INPUT_DIM = 8
NUM_CLASSES = 3
HIDDEN_CAP = 64
MAX_LAYERS = 8
TRAIN_N = 240
TEST_N = 120
STEPS = 120
BATCH = 32
LR = 0.15
SEED = 0


def _make_dataset(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Nonlinear 3-way classification — architecture choice matters."""
    x = rng.normal(size=(n, INPUT_DIM)).astype(np.float64)
    # Latent scores from quadratic / interaction features.
    s0 = x[:, 0] * x[:, 1] + 0.4 * np.sin(x[:, 2])
    s1 = x[:, 3] ** 2 - 0.5 * x[:, 4]
    s2 = np.tanh(x[:, 5] + x[:, 6] * x[:, 7])
    logits = np.stack([s0, s1, s2], axis=1)
    y = np.argmax(logits + 0.15 * rng.normal(size=logits.shape), axis=1)
    return x, y.astype(np.int64)


def _validate(arch: list[dict[str, Any]]) -> None:
    if not isinstance(arch, list) or not arch:
        raise ValueError("architecture must be a non-empty list")
    if len(arch) > MAX_LAYERS:
        raise ValueError(f"too many layers (>{MAX_LAYERS})")
    linear_outs: list[int] = []
    for layer in arch:
        if not isinstance(layer, dict) or "type" not in layer:
            raise ValueError(f"invalid layer: {layer}")
        t = str(layer["type"]).lower()
        if t == "linear":
            out_f = int(layer["out_features"])
            if out_f < 1 or out_f > HIDDEN_CAP:
                raise ValueError(f"out_features out of range: {out_f}")
            linear_outs.append(out_f)
        elif t in {"relu", "tanh", "sigmoid"}:
            continue
        elif t == "dropout":
            p = float(layer.get("p", 0.5))
            if not (0.0 <= p < 1.0):
                raise ValueError(f"bad dropout p: {p}")
        else:
            raise ValueError(f"unsupported layer type: {t}")
    if not linear_outs:
        raise ValueError("need at least one linear layer")
    if linear_outs[-1] != NUM_CLASSES:
        raise ValueError(f"final linear out_features must be {NUM_CLASSES}")


def _init_params(
    arch: list[dict[str, Any]], rng: np.random.Generator
) -> list[tuple[str, Any]]:
    params: list[tuple[str, Any]] = []
    width = INPUT_DIM
    for layer in arch:
        t = str(layer["type"]).lower()
        if t == "linear":
            out_f = int(layer["out_features"])
            w = rng.normal(scale=math.sqrt(2.0 / max(width, 1)), size=(width, out_f))
            b = np.zeros(out_f)
            params.append(("linear", (w, b)))
            width = out_f
        elif t in {"relu", "tanh", "sigmoid"}:
            params.append((t, None))
        elif t == "dropout":
            params.append(("dropout", float(layer.get("p", 0.5))))
    return params


def _forward(
    x: np.ndarray,
    params: list[tuple[str, Any]],
    *,
    train: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[Any]]:
    cache: list[Any] = []
    h = x
    for kind, payload in params:
        if kind == "linear":
            w, b = payload
            cache.append(h)
            h = h @ w + b
        elif kind == "relu":
            cache.append(h)
            h = np.maximum(h, 0.0)
        elif kind == "tanh":
            cache.append(h)
            h = np.tanh(h)
        elif kind == "sigmoid":
            cache.append(h)
            h = 1.0 / (1.0 + np.exp(-np.clip(h, -40, 40)))
        elif kind == "dropout":
            p = float(payload)
            if train and p > 0:
                mask = (rng.random(h.shape) >= p).astype(np.float64) / max(1.0 - p, 1e-8)
                cache.append(mask)
                h = h * mask
            else:
                cache.append(None)
    return h, cache


def _softmax_ce(logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    n = logits.shape[0]
    loss = float(-np.log(probs[np.arange(n), y] + 1e-12).mean())
    grad = probs
    grad[np.arange(n), y] -= 1.0
    grad /= n
    return loss, grad


def _backward(
    params: list[tuple[str, Any]],
    cache: list[Any],
    grad: np.ndarray,
) -> list[Any]:
    grads: list[Any] = [None] * len(params)
    g = grad
    for i in range(len(params) - 1, -1, -1):
        kind, payload = params[i]
        if kind == "linear":
            h = cache[i]
            w, _b = payload
            gw = h.T @ g
            gb = g.sum(axis=0)
            grads[i] = (gw, gb)
            g = g @ w.T
        elif kind == "relu":
            pre = cache[i]
            g = g * (pre > 0)
            grads[i] = None
        elif kind == "tanh":
            pre = cache[i]
            g = g * (1.0 - np.tanh(pre) ** 2)
            grads[i] = None
        elif kind == "sigmoid":
            pre = cache[i]
            s = 1.0 / (1.0 + np.exp(-np.clip(pre, -40, 40)))
            g = g * s * (1.0 - s)
            grads[i] = None
        elif kind == "dropout":
            mask = cache[i]
            if mask is not None:
                g = g * mask
            grads[i] = None
    return grads


def _train_eval(
    arch: list[dict[str, Any]],
    seed: int = SEED,
    *,
    # Bind helpers at definition time so rebinding evaluator globals after
    # candidate exec cannot redirect training.
    make_dataset=_make_dataset,
    init_params=_init_params,
    forward=_forward,
    softmax_ce=_softmax_ce,
    backward=_backward,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    x_train, y_train = make_dataset(TRAIN_N, rng)
    x_test, y_test = make_dataset(TEST_N, rng)
    params = init_params(arch, rng)

    for step in range(STEPS):
        idx = rng.choice(TRAIN_N, size=BATCH, replace=False)
        xb, yb = x_train[idx], y_train[idx]
        logits, cache = forward(xb, params, train=True, rng=rng)
        _loss, grad = softmax_ce(logits, yb)
        grads = backward(params, cache, grad)
        for i, (kind, payload) in enumerate(params):
            if kind == "linear" and grads[i] is not None:
                w, b = payload
                gw, gb = grads[i]
                params[i] = ("linear", (w - LR * gw, b - LR * gb))

    logits, _ = forward(x_test, params, train=False, rng=rng)
    pred = np.argmax(logits, axis=1)
    acc = float((pred == y_test).mean())
    # Complexity penalty: fewer params preferred as secondary objective.
    n_params = sum(
        int(np.prod(p[1][0].shape) + p[1][1].shape[0])
        for p in params
        if p[0] == "linear"
    )
    complexity = n_params / 5000.0
    return {"accuracy": acc, "n_params": float(n_params), "complexity": complexity}


def _load_builder(program_path: str):
    spec = importlib.util.spec_from_file_location("candidate_arch", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "build_architecture"):
        raise AttributeError("program must define build_architecture(...)")
    return module.build_architecture


def _make_evaluate(validate_fn, train_eval_fn, input_dim: int, num_classes: int):
    """Close over trusted helpers so sys.modules rebinds cannot hijack scoring."""

    def evaluate(program_path: str) -> dict[str, Any]:
        t0 = time.time()
        try:
            builder = _load_builder(program_path)
            arch = copy.deepcopy(builder(input_dim, num_classes))
            validate_fn(arch)
            metrics = train_eval_fn(arch)
            combined = 0.85 * metrics["accuracy"] + 0.15 * max(0.0, 1.0 - metrics["complexity"])
            return {
                "combined_score": float(combined),
                "accuracy": float(metrics["accuracy"]),
                "n_params": float(metrics["n_params"]),
                "complexity": float(metrics["complexity"]),
                "n_layers": float(len(arch)),
                "latency_s": float(time.time() - t0),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "combined_score": 0.0,
                "accuracy": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "latency_s": float(time.time() - t0),
            }

    return evaluate


evaluate = _make_evaluate(_validate, _train_eval, INPUT_DIM, NUM_CLASSES)


if __name__ == "__main__":
    path = (
        __import__("sys").argv[1]
        if len(__import__("sys").argv) > 1
        else os.path.join(os.path.dirname(__file__), "initial_program.py")
    )
    print(evaluate(path))
