"""Evaluate hypothesis→experiment discovery loops on a hidden nonlinear target.

Candidate code (design / fit / predict) runs in a subprocess whose working
directory is a throwaway tree that does not contain ``world.py`` or this
evaluator. The hidden target stays in the parent, so frame walks, closure
rewrites, and filesystem reads of the benchmark dir cannot recover it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# Undisclosed evaluation entropy. Candidate code never sees this generator, and
# the held-out test RNG is spawned independently so snapshotting the RNG handed
# to design_experiments cannot reconstruct x_test.
_EVAL_ENTROPY = 0xA5C1D15C07E57
_CANDIDATE_TIMEOUT_S = 30.0

# Runs in an isolated temp dir. Candidate prints go to stderr; JSON lines on stdout.
_WORKER_SOURCE = r"""
import importlib.util
import json
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path[:] = [HERE] + [
    p for p in sys.path if p not in ("", ".") and os.path.abspath(p) != HERE
]
sys.stdout = sys.stderr


def _send(obj):
    sys.__stdout__.write(json.dumps(obj) + "\n")
    sys.__stdout__.flush()


def _recv():
    line = sys.__stdin__.readline()
    if not line:
        raise EOFError("parent closed the protocol pipe")
    return json.loads(line)


def _load_candidate():
    path = os.path.join(HERE, "candidate.py")
    spec = importlib.util.spec_from_file_location("candidate_science", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("design_experiments", "fit_hypothesis"):
        if not hasattr(module, name):
            raise AttributeError(f"program must define {name}")
    return module.design_experiments, module.fit_hypothesis


def main():
    design_experiments, fit_hypothesis = _load_candidate()
    req = _recv()
    rng = np.random.default_rng(int(req["rng_seed"]))
    queries = np.asarray(
        design_experiments(int(req["budget"]), int(req["n_features"]), rng),
        dtype=np.float64,
    )
    _send({"queries": queries.tolist()})
    fit_req = _recv()
    x = np.asarray(fit_req["x"], dtype=np.float64)
    y = np.asarray(fit_req["y"], dtype=np.float64)
    predict = fit_hypothesis(x, y)
    pred_req = _recv()
    x_test = np.asarray(pred_req["x"], dtype=np.float64)
    y_hat = np.asarray(predict(x_test), dtype=np.float64).reshape(-1)
    _send({"y_hat": y_hat.tolist()})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        _send({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        sys.exit(1)
"""


def _hidden_world() -> dict[str, Any]:
    """Load world.py into a private dict — never registered as sys.modules['world']."""
    path = os.path.join(_HERE, "world.py")
    ns: dict[str, Any] = {"__name__": "_skydiscover_hidden_world", "__file__": path}
    with open(path, encoding="utf-8") as f:
        exec(compile(f.read(), path, "exec"), ns)
    return ns


def _independent_rngs() -> tuple[np.random.Generator, np.random.Generator, np.random.Generator]:
    """Return (candidate, oracle-noise, test) generators that do not share state."""
    master = np.random.default_rng(_EVAL_ENTROPY)
    spawn = getattr(master, "spawn", None)
    if callable(spawn):
        cand_rng, noise_rng, test_rng = spawn(3)
        return cand_rng, noise_rng, test_rng
    ss = np.random.SeedSequence(_EVAL_ENTROPY)
    cand_ss, noise_ss, test_ss = ss.spawn(3)
    return (
        np.random.default_rng(cand_ss),
        np.random.default_rng(noise_ss),
        np.random.default_rng(test_ss),
    )


def _clean_env(workdir: str) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PWD"] = workdir
    return env


def _send(proc: subprocess.Popen, obj: dict[str, Any]) -> None:
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(obj) + "\n")
    proc.stdin.flush()


def _recv(proc: subprocess.Popen) -> dict[str, Any]:
    assert proc.stdout is not None
    line = proc.stdout.readline()
    if not line:
        stderr = ""
        if proc.stderr is not None:
            stderr = proc.stderr.read()
        raise RuntimeError(f"candidate worker exited without a reply: {stderr[-2000:]}")
    msg = json.loads(line)
    if not isinstance(msg, dict):
        raise TypeError("candidate worker returned a non-object")
    if "error" in msg:
        raise RuntimeError(msg["error"])
    return msg


def _run_candidate_subprocess(
    program_path: str,
    *,
    budget: int,
    n_features: int,
    rng_seed: int,
    y_obs_for,
    x_test: np.ndarray,
    timeout_s: float = _CANDIDATE_TIMEOUT_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Exec the candidate in a process that cannot see this evaluator or world.py."""
    source = Path(program_path).read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="skydiscover_hyp_") as td:
        Path(td, "candidate.py").write_text(source, encoding="utf-8")
        Path(td, "worker.py").write_text(_WORKER_SOURCE, encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-u", os.path.join(td, "worker.py")],
            cwd=td,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_clean_env(td),
        )
        timed_out = threading.Event()

        def _kill() -> None:
            timed_out.set()
            proc.kill()

        timer = threading.Timer(timeout_s, _kill)
        timer.start()
        try:
            _send(proc, {"budget": budget, "n_features": n_features, "rng_seed": int(rng_seed)})
            msg = _recv(proc)
            queries = np.asarray(msg["queries"], dtype=np.float64)
            if queries.ndim != 2 or queries.shape[1] != n_features:
                raise ValueError(
                    f"design_experiments must return (budget, {n_features}), got {queries.shape}"
                )
            if len(queries) > budget:
                queries = queries[:budget]
            queries = np.clip(queries, -2.0, 2.0)
            y_obs = y_obs_for(queries)
            _send(proc, {"x": queries.tolist(), "y": np.asarray(y_obs, dtype=np.float64).tolist()})
            _send(proc, {"x": np.asarray(x_test, dtype=np.float64).tolist()})
            pred = _recv(proc)
            if timed_out.is_set():
                raise TimeoutError(f"candidate worker exceeded {timeout_s}s")
            y_hat = np.asarray(pred["y_hat"], dtype=np.float64).reshape(-1)
            return queries, y_hat
        except Exception:
            proc.kill()
            if timed_out.is_set():
                raise TimeoutError(f"candidate worker exceeded {timeout_s}s") from None
            raise
        finally:
            timer.cancel()
            try:
                proc.stdin.close()  # type: ignore[union-attr]
            except Exception:
                pass
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def evaluate(program_path: str) -> dict[str, Any]:
    t0 = time.time()
    try:
        world = _hidden_world()
        n_features = world["N_FEATURES"]
        query_budget = world["QUERY_BUDGET"]
        test_n = world["TEST_N"]
        true_function = world["true_function"]
        oracle_observe = world["oracle_observe"]
        sample_inputs = world["sample_inputs"]

        cand_rng, noise_rng, test_rng = _independent_rngs()
        rng_seed = int(cand_rng.integers(0, 2**31 - 1))
        x_test = sample_inputs(test_n, test_rng)
        y_test = true_function(x_test)  # noiseless held-out truth; never sent to the child

        queries, y_hat = _run_candidate_subprocess(
            program_path,
            budget=query_budget,
            n_features=n_features,
            rng_seed=rng_seed,
            y_obs_for=lambda q: oracle_observe(q, noise_rng),
            x_test=x_test,
            timeout_s=_CANDIDATE_TIMEOUT_S,
        )
        if y_hat.shape != y_test.shape:
            raise ValueError("predict() must return a 1-D array matching y")

        r2 = _r2(y_test, y_hat)
        coverage = min(1.0, len(queries) / float(query_budget))
        mse = float(np.mean((y_test - y_hat) ** 2))
        combined = 0.9 * max(0.0, r2) + 0.1 * coverage

        return {
            "combined_score": float(combined),
            "r2": float(r2),
            "mse": mse,
            "n_queries": float(len(queries)),
            "coverage": float(coverage),
            "latency_s": float(time.time() - t0),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "combined_score": 0.0,
            "r2": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "latency_s": float(time.time() - t0),
        }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "initial_program.py")
    print(evaluate(path))
