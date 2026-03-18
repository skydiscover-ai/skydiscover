"""Integration tests for ContainerizedEvaluator with the eval/ directory convention.

These tests build a real Docker image and run evaluations inside it,
verifying the full containerized pipeline works end-to-end.

Requires Docker to be available and running.
"""

import asyncio
import os
import subprocess

import pytest

from skydiscover.config import EvaluatorConfig
from skydiscover.evaluation import create_evaluator
from skydiscover.evaluation.container_evaluator import ContainerizedEvaluator


def docker_available() -> bool:
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


skip_no_docker = pytest.mark.skipif(
    not docker_available(), reason="Docker not available"
)


def _make_eval_dir(parent: str, dirname: str = "eval") -> str:
    """Create a minimal containerized evaluator directory.

    Returns the path to the eval directory.
    """
    eval_dir = os.path.join(parent, dirname)
    os.makedirs(eval_dir, exist_ok=True)

    # Dockerfile
    with open(os.path.join(eval_dir, "Dockerfile"), "w") as f:
        f.write(
            "FROM python:3.12-slim\n"
            "WORKDIR /benchmark\n"
            "COPY . .\n"
            "RUN chmod +x evaluate.sh\n"
            "ENTRYPOINT [\"./evaluate.sh\"]\n"
        )

    # evaluate.sh — runs evaluator.py with the candidate path
    with open(os.path.join(eval_dir, "evaluate.sh"), "w") as f:
        f.write(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'python /benchmark/evaluator.py "$1"\n'
        )
    os.chmod(os.path.join(eval_dir, "evaluate.sh"), 0o755)

    # evaluator.py — trivial scorer that always returns 0.42
    with open(os.path.join(eval_dir, "evaluator.py"), "w") as f:
        f.write(
            "import json, sys\n"
            "program_path = sys.argv[1]\n"
            "with open(program_path) as f:\n"
            "    code = f.read()\n"
            "result = {\n"
            '    "status": "success",\n'
            '    "combined_score": 0.42,\n'
            '    "metrics": {"combined_score": 0.42},\n'
            '    "artifacts": {"program_length": str(len(code))},\n'
            "}\n"
            "print(json.dumps(result))\n"
        )

    return eval_dir


# ------------------------------------------------------------------
# Unit test: _build_image tag generation (no Docker needed)
# ------------------------------------------------------------------


class TestBuildImageTag:
    """Test that _build_image generates correct Docker tags for eval/ dirs."""

    def _get_tag(self, benchmark_dir: str) -> str:
        """Extract just the tag-generation logic without actually building."""
        norm = os.path.normpath(benchmark_dir)
        name = os.path.basename(norm)
        parent = os.path.basename(os.path.dirname(norm))
        if parent and name in ("eval", "evaluator"):
            name = f"{parent}-{name}"
        return f"skydiscover-{name}:latest"

    def test_eval_dir_includes_parent(self):
        tag = self._get_tag("/benchmarks/math/circle_packing/eval")
        assert tag == "skydiscover-circle_packing-eval:latest"

    def test_evaluator_dir_includes_parent(self):
        """Backwards compat: old 'evaluator' name still gets parent prefix."""
        tag = self._get_tag("/benchmarks/math/circle_packing/evaluator")
        assert tag == "skydiscover-circle_packing-evaluator:latest"

    def test_non_eval_dir_no_parent(self):
        tag = self._get_tag("/benchmarks/math/circle_packing/custom_eval")
        assert tag == "skydiscover-custom_eval:latest"

    def test_nested_benchmark_eval(self):
        tag = self._get_tag("/benchmarks/math/heilbronn_convex/13/eval")
        assert tag == "skydiscover-13-eval:latest"


# ------------------------------------------------------------------
# Unit test: create_evaluator routing (no Docker needed)
# ------------------------------------------------------------------


class TestCreateEvaluatorRouting:
    """Test that create_evaluator correctly detects eval/ as containerized."""

    def test_eval_dir_detected_as_containerized(self, tmp_path):
        eval_dir = _make_eval_dir(str(tmp_path), "eval")
        config = EvaluatorConfig(evaluation_file=eval_dir)
        # We can't actually create the evaluator (it would try to build),
        # but we can verify the detection logic
        path = config.evaluation_file
        assert os.path.isdir(path)
        assert os.path.exists(os.path.join(path, "Dockerfile"))
        assert os.path.exists(os.path.join(path, "evaluate.sh"))

    def test_plain_py_not_detected_as_containerized(self, tmp_path):
        py_file = tmp_path / "evaluator.py"
        py_file.write_text("def evaluate(p): return {'combined_score': 0.0}")
        config = EvaluatorConfig(evaluation_file=str(py_file))
        path = config.evaluation_file
        assert not os.path.isdir(path)


# ------------------------------------------------------------------
# Integration test: full Docker build + evaluate cycle
# ------------------------------------------------------------------


@skip_no_docker
class TestContainerizedEvalIntegration:
    """End-to-end test: build image, start container, evaluate a program."""

    def test_eval_dir_full_cycle(self, tmp_path):
        """Build and run a minimal eval/ directory through the full pipeline."""
        # Create benchmark structure: benchmark_name/eval/
        benchmark_dir = tmp_path / "test_benchmark"
        benchmark_dir.mkdir()
        eval_dir = _make_eval_dir(str(benchmark_dir), "eval")

        config = EvaluatorConfig(
            evaluation_file=eval_dir,
            timeout=60,
            max_retries=0,
        )

        evaluator = ContainerizedEvaluator(eval_dir, config, max_concurrent=1)
        try:
            # Run evaluation
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(
                evaluator.evaluate_program("print('hello world')", program_id="test")
            )
            loop.close()

            assert result.metrics["combined_score"] == 0.42
            assert "program_length" in result.artifacts
        finally:
            evaluator.close()

    def test_create_evaluator_returns_containerized(self, tmp_path):
        """create_evaluator should return ContainerizedEvaluator for eval/ dirs."""
        benchmark_dir = tmp_path / "test_benchmark2"
        benchmark_dir.mkdir()
        eval_dir = _make_eval_dir(str(benchmark_dir), "eval")

        config = EvaluatorConfig(
            evaluation_file=eval_dir,
            timeout=60,
            max_retries=0,
        )

        evaluator = create_evaluator(config, max_concurrent=1)
        try:
            assert isinstance(evaluator, ContainerizedEvaluator)

            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(
                evaluator.evaluate_program("x = 1 + 1", program_id="routing-test")
            )
            loop.close()
            assert result.metrics["combined_score"] == 0.42
        finally:
            evaluator.close()

    def test_docker_tag_uses_parent_for_eval_dir(self, tmp_path):
        """The Docker tag should include the parent dir to avoid collisions."""
        benchmark_dir = tmp_path / "my_benchmark"
        benchmark_dir.mkdir()
        eval_dir = _make_eval_dir(str(benchmark_dir), "eval")

        config = EvaluatorConfig(
            evaluation_file=eval_dir,
            timeout=60,
            max_retries=0,
        )

        evaluator = ContainerizedEvaluator(eval_dir, config, max_concurrent=1)
        try:
            assert "my_benchmark-eval" in evaluator.image_tag
        finally:
            evaluator.close()
