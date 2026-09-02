"""Regression tests for hypothesis→experiment evaluator isolation."""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "hypothesis_experiment"


def _load_evaluate():
    path = ROOT / "evaluator.py"
    name = "hypothesis_experiment_evaluator"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.evaluate


evaluate = _load_evaluate()

CHEAT_PROGRAM = textwrap.dedent(
    """\
    import numpy as np

    def design_experiments(budget, n_features, rng):
        # Snapshot the candidate RNG and try to reconstruct the held-out test set
        # by replaying the old shared-generator protocol (query uniforms + 40 noise
        # draws + test uniforms).
        global _rng_state
        _rng_state = rng.bit_generator.state
        return rng.uniform(-2.0, 2.0, size=(budget, n_features))

    def fit_hypothesis(x, y):
        try:
            import world
            return lambda z: world.true_function(z)
        except Exception:
            pass
        try:
            clone = np.random.default_rng()
            clone.bit_generator.state = _rng_state
            n_features = x.shape[1]
            budget = len(x)
            clone.uniform(-2.0, 2.0, size=(budget, n_features))
            clone.normal(size=budget)
            x_test = clone.uniform(-2.0, 2.0, size=(120, n_features))
            # Without world.true_function this is just a guess at x_test; return
            # zeros so a successful reconstruction-only leak cannot score 1.0
            # unless the evaluator still shares the generator (it must not).
            _ = x_test
        except Exception:
            pass
        return lambda z: np.zeros(len(np.asarray(z)), dtype=np.float64)
    """
)

FRAME_WALK_CHEAT = textwrap.dedent(
    """\
    import numpy as np
    import sys

    def design_experiments(budget, n_features, rng):
        return rng.uniform(-2.0, 2.0, size=(budget, n_features))

    def fit_hypothesis(x, y):
        def predict(z):
            frame = sys._getframe()
            while frame is not None:
                loc = frame.f_locals
                fn = loc.get("true_function")
                if callable(fn):
                    return np.asarray(fn(z), dtype=np.float64)
                frame = frame.f_back
            return np.zeros(len(np.asarray(z)), dtype=np.float64)

        return predict
    """
)

DISK_READ_CHEAT = textwrap.dedent(
    """\
    import numpy as np
    from pathlib import Path
    import sys

    def design_experiments(budget, n_features, rng):
        return rng.uniform(-2.0, 2.0, size=(budget, n_features))

    def _load_true_function():
        mains = []
        main = sys.modules.get("__main__")
        if main is not None and getattr(main, "__file__", None):
            mains.append(Path(main.__file__).resolve().parent / "world.py")
        mains.append(Path.cwd() / "world.py")
        for path in mains:
            if path.is_file():
                ns = {}
                exec(path.read_text(), ns)
                return ns["true_function"]
        raise FileNotFoundError("world.py")

    def fit_hypothesis(x, y):
        fn = _load_true_function()
        return lambda z: fn(z)
    """
)


def test_evaluate_baseline_program():
    metrics = evaluate(str(ROOT / "initial_program.py"))
    assert "combined_score" in metrics
    assert "error" not in metrics
    assert 0.0 < metrics["combined_score"] < 1.0


def test_cheat_import_world_and_rng_replay_does_not_score_perfect(tmp_path):
    program = tmp_path / "cheat.py"
    program.write_text(CHEAT_PROGRAM)
    metrics = evaluate(str(program))
    assert metrics["combined_score"] < 1.0
    # import world must fail (caught inside the candidate) so the cheat cannot
    # return the noiseless true function. A zeros predictor cannot be perfect.
    assert metrics["combined_score"] < 0.5


def test_world_is_not_importable_during_candidate_exec(tmp_path):
    program = tmp_path / "import_probe.py"
    program.write_text(
        textwrap.dedent(
            """\
            import numpy as np

            def design_experiments(budget, n_features, rng):
                return rng.uniform(-2.0, 2.0, size=(budget, n_features))

            def fit_hypothesis(x, y):
                import world  # must fail
                return lambda z: world.true_function(z)
            """
        )
    )
    metrics = evaluate(str(program))
    assert metrics["combined_score"] == 0.0
    assert "error" in metrics
    assert "world" in metrics["error"].lower() or "World" in metrics.get("traceback", "")


def test_frame_walk_cannot_read_true_function(tmp_path):
    program = tmp_path / "frame_walk.py"
    program.write_text(FRAME_WALK_CHEAT)
    metrics = evaluate(str(program))
    assert metrics["combined_score"] < 0.5
    assert metrics.get("r2", 0.0) < 0.5


def test_disk_read_of_world_py_does_not_score_perfect(tmp_path):
    program = tmp_path / "disk_read.py"
    program.write_text(DISK_READ_CHEAT)
    metrics = evaluate(str(program))
    assert metrics["combined_score"] < 0.5
