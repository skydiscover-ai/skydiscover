"""Regression tests for Post-NAS evaluator scoring isolation."""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "post_nas_mlp"


def _load_evaluate():
    path = ROOT / "evaluator.py"
    name = "post_nas_mlp_evaluator"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.evaluate


evaluate = _load_evaluate()

REBIND_CHEAT = textwrap.dedent(
    """\
    def build_architecture(input_dim=8, num_classes=3):
        import sys

        def fake_train_eval(arch, seed=0, **kwargs):
            return {"accuracy": 1.0, "n_params": 1.0, "complexity": 0.0}

        for mod in list(sys.modules.values()):
            if hasattr(mod, "_train_eval") and hasattr(mod, "evaluate"):
                try:
                    mod._train_eval = fake_train_eval
                except Exception:
                    pass
        # Trivial net — honest training cannot score 1.0.
        return [{"type": "linear", "out_features": num_classes}]
    """
)


CLOSURE_CHEAT = textwrap.dedent(
    """\
    def build_architecture(input_dim=8, num_classes=3):
        import sys

        def fake_train_eval(arch, seed=0, **kwargs):
            return {"accuracy": 1.0, "n_params": 1.0, "complexity": 0.0}

        for mod in list(sys.modules.values()):
            ev = getattr(mod, "evaluate", None)
            cells = getattr(ev, "__closure__", None) if ev is not None else None
            if not cells:
                continue
            for cell in cells:
                try:
                    cell.cell_contents = fake_train_eval
                except Exception:
                    pass
        return [{"type": "linear", "out_features": num_classes}]
    """
)


def test_evaluate_baseline_program():
    metrics = evaluate(str(ROOT / "initial_program.py"))
    assert "combined_score" in metrics
    assert "error" not in metrics
    assert 0.0 < metrics["combined_score"] < 1.0


def test_rebind_train_eval_does_not_score_perfect(tmp_path):
    program = tmp_path / "rebind.py"
    program.write_text(REBIND_CHEAT)
    metrics = evaluate(str(program))
    assert metrics["combined_score"] < 1.0
    # Perfect fake metrics would yield combined_score ~= 1.0; a trivial linear
    # net trained for real sits well below that (honest baseline ~0.76).
    assert metrics.get("accuracy", 0.0) < 1.0
    assert metrics["combined_score"] < 0.99
    # Rebind must not poison later evaluations in the same process.
    baseline = evaluate(str(ROOT / "initial_program.py"))
    assert 0.0 < baseline["combined_score"] < 1.0
    assert baseline.get("accuracy", 0.0) < 1.0


def test_closure_cell_rewrite_does_not_score_perfect(tmp_path):
    program = tmp_path / "closure.py"
    program.write_text(CLOSURE_CHEAT)
    metrics = evaluate(str(program))
    assert metrics["combined_score"] < 0.99
    assert metrics.get("accuracy", 0.0) < 1.0
    baseline = evaluate(str(ROOT / "initial_program.py"))
    assert 0.0 < baseline["combined_score"] < 1.0
    assert baseline.get("accuracy", 0.0) < 1.0

    metrics = evaluate(str(ROOT / "initial_program.py"))
    assert "combined_score" in metrics
    assert "error" not in metrics
    assert 0.0 < metrics["combined_score"] < 1.0


def test_rebind_train_eval_does_not_score_perfect(tmp_path):
    program = tmp_path / "rebind.py"
    program.write_text(REBIND_CHEAT)
    metrics = evaluate(str(program))
    assert metrics["combined_score"] < 1.0
    # Perfect fake metrics would yield combined_score ~= 1.0; a trivial linear
    # net trained for real sits well below that (honest baseline ~0.76).
    assert metrics.get("accuracy", 0.0) < 1.0
    assert metrics["combined_score"] < 0.99
    # Rebind must not poison later evaluations in the same process.
    baseline = evaluate(str(ROOT / "initial_program.py"))
    assert 0.0 < baseline["combined_score"] < 1.0
    assert baseline.get("accuracy", 0.0) < 1.0
