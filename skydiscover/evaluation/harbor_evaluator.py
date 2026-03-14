"""Harbor evaluator: runs Harbor-format tasks inside a persistent Docker container.

Harbor tasks use a different container protocol from the standard
ContainerizedEvaluator:

  - Solution is injected at a task-specific path (e.g. ``/app/solver.py``)
    extracted from ``instruction.md``.
  - Evaluation runs ``tests/test.sh`` instead of ``evaluate.sh``.
  - The reward is read from ``/logs/verifier/reward.txt`` (float) or
    ``/logs/verifier/reward.json`` (dict) instead of JSON on stdout.

A Harbor task directory has this structure::

    task_dir/
    ├── task.toml              # metadata, timeouts
    ├── instruction.md         # problem description (shown to LLM)
    ├── environment/
    │   └── Dockerfile
    ├── tests/
    │   ├── test.sh            # verification entrypoint
    │   └── ...                # supporting test files
    └── solution/              # reference solution (optional, never shown to LLM)
        └── solve.sh

See https://harborframework.com/docs for the full specification.
"""

import json
import logging
import os
import re
import subprocess

from skydiscover.evaluation.container_evaluator import ContainerizedEvaluator
from skydiscover.evaluation.evaluation_result import EvaluationResult

logger = logging.getLogger(__name__)

# Common solution paths in Harbor tasks — used as fallback.
_DEFAULT_SOLUTION_PATH = "/app/solver.py"


class HarborEvaluator(ContainerizedEvaluator):
    """Evaluates programs using the Harbor container protocol.

    Extends ContainerizedEvaluator, overriding only the container interaction
    methods: image building, solution injection, test execution, and reward
    reading.
    """

    def __init__(self, benchmark_dir, config, max_concurrent=4):
        self.task_dir = os.path.abspath(benchmark_dir)
        self.solution_path = self._extract_solution_path()
        self._tests_uploaded = False
        super().__init__(benchmark_dir, config, max_concurrent)
        self._init_container()

    # ------------------------------------------------------------------
    # Override: image building
    # ------------------------------------------------------------------

    def _build_image(self) -> str:
        """Build from environment/Dockerfile."""
        name = os.path.basename(os.path.normpath(self.task_dir))
        tag = f"skydiscover-harbor-{name}:latest"
        dockerfile_dir = os.path.join(self.task_dir, "environment")

        logger.info(f"Building Harbor image: {tag} (from {dockerfile_dir})")
        result = subprocess.run(
            ["docker", "build", "-t", tag, dockerfile_dir],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Docker build failed for {dockerfile_dir}:\n{result.stderr}")
        return tag

    # ------------------------------------------------------------------
    # Override: container interaction
    # ------------------------------------------------------------------

    def _run_container(self, program_solution: str, mode: str) -> EvaluationResult:
        """Inject solution, run tests, read reward."""
        # Clear stale reward files from previous evaluations.
        self._exec("rm -f /logs/verifier/reward.txt /logs/verifier/reward.json")

        # Ensure parent directory exists and inject solution.
        parent_dir = os.path.dirname(self.solution_path)
        if parent_dir:
            self._exec(f"mkdir -p {parent_dir}")
        inject = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                self.container_id,
                "/bin/sh",
                "-c",
                f"cat > {self.solution_path}",
            ],
            input=program_solution.encode(),
            capture_output=True,
        )
        if inject.returncode != 0:
            logger.error(f"Failed to inject solution: {inject.stderr.decode()}")
            return EvaluationResult(
                metrics={"combined_score": 0.0},
                artifacts={"error": f"injection failed: {inject.stderr.decode()}"},
            )

        try:
            # Run tests.
            proc = subprocess.run(
                [
                    "docker",
                    "exec",
                    self.container_id,
                    "bash",
                    "-c",
                    "chmod +x /tests/test.sh && /tests/test.sh",
                ],
                capture_output=True,
                text=True,
            )

            # Read reward regardless of exit code — test.sh may exit non-zero
            # but still write a reward (e.g. partial credit).
            result = self._read_reward(proc.stdout, proc.stderr)

            if proc.returncode != 0:
                result.artifacts.setdefault("test_exit_code", str(proc.returncode))
            if proc.stderr.strip():
                result.artifacts.setdefault("stderr", proc.stderr)
            if proc.stdout.strip():
                result.artifacts.setdefault("stdout", proc.stdout)

            return result

        finally:
            # Clean up solution so the container is fresh for next evaluation.
            self._exec(f"rm -f {self.solution_path}")

    # ------------------------------------------------------------------
    # Harbor-specific helpers
    # ------------------------------------------------------------------

    def _init_container(self):
        """Create log directories and upload test files into the container."""
        self._exec("mkdir -p /logs/verifier /logs/agent /logs/artifacts")

        # Upload the tests/ directory.
        tests_dir = os.path.join(self.task_dir, "tests")
        if os.path.isdir(tests_dir):
            self._exec("rm -rf /tests")
            subprocess.run(
                ["docker", "cp", tests_dir, f"{self.container_id}:/tests"],
                capture_output=True,
                check=True,
            )
            self._tests_uploaded = True
            logger.debug("Uploaded tests/ to container")
        else:
            raise RuntimeError(f"No tests/ directory found in {self.task_dir}")

    def _read_reward(self, test_stdout: str = "", test_stderr: str = "") -> EvaluationResult:
        """Read the reward from /logs/verifier/reward.txt or reward.json."""
        for path, is_json in [
            ("/logs/verifier/reward.json", True),
            ("/logs/verifier/reward.txt", False),
        ]:
            proc = subprocess.run(
                ["docker", "exec", self.container_id, "cat", path],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0 or not proc.stdout.strip():
                continue

            try:
                if is_json:
                    data = json.loads(proc.stdout.strip())
                    reward = float(
                        data.get(
                            "reward",
                            data.get("score", next(iter(data.values()))),
                        )
                    )
                    metrics = {"combined_score": reward}
                    for k, v in data.items():
                        if isinstance(v, (int, float)) and k not in (
                            "reward",
                            "score",
                        ):
                            metrics[k] = float(v)
                    return EvaluationResult(metrics=metrics)
                else:
                    reward = float(proc.stdout.strip())
                    return EvaluationResult(metrics={"combined_score": reward})
            except (ValueError, json.JSONDecodeError, StopIteration) as e:
                logger.warning(f"Failed to parse reward from {path}: {e}")
                continue

        logger.error("No reward file found in /logs/verifier/")
        return EvaluationResult(
            metrics={"combined_score": 0.0},
            artifacts={
                "error": "no reward file written by test.sh",
                "test_stdout": test_stdout,
                "test_stderr": test_stderr,
            },
        )

    def _extract_solution_path(self) -> str:
        """Extract the expected solution file path from instruction.md.

        Looks for patterns like ``in `/app/solver.py` `` or
        ``at /app/solution.py``.  Falls back to /app/solver.py.
        
        TODO(akrentsel): Consider using a smarter heuristic, or ask an LLM to extract.
        """
        instruction_path = os.path.join(self.task_dir, "instruction.md")
        if not os.path.exists(instruction_path):
            logger.warning(
                f"No instruction.md in {self.task_dir}, "
                f"using default solution path: {_DEFAULT_SOLUTION_PATH}"
            )
            return _DEFAULT_SOLUTION_PATH

        with open(instruction_path) as f:
            text = f.read()

        patterns = [
            r'[`"\'](/\S+\.(?:py|sh|js|ts|cpp|c|rs|go|java))[`"\']',
            r"(?:in|at|to|into)\s+(/\S+\.(?:py|sh|js|ts|cpp|c|rs|go|java))",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                path = match.group(1)
                logger.info(f"Extracted solution path from instruction.md: {path}")
                return path

        logger.warning(
            f"Could not extract solution path from instruction.md, "
            f"using default: {_DEFAULT_SOLUTION_PATH}"
        )
        return _DEFAULT_SOLUTION_PATH

    def _exec(self, cmd: str) -> subprocess.CompletedProcess:
        """Run a shell command inside the container."""
        return subprocess.run(
            ["docker", "exec", self.container_id, "/bin/sh", "-c", cmd],
            capture_output=True,
            text=True,
        )
