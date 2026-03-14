"""Containerized evaluator: runs evaluate.sh inside a persistent Docker container."""

import asyncio
import json
import logging
import os
import subprocess
import time
import uuid
from typing import List, Tuple

from skydiscover.config import EvaluatorConfig
from skydiscover.evaluation.evaluation_result import EvaluationResult
from skydiscover.utils.async_utils import TaskPool
from skydiscover.utils.metrics import format_metrics

logger = logging.getLogger(__name__)


class ContainerizedEvaluator:
    """Evaluates programs by running them inside a persistent Docker container.

    The benchmark directory must contain:
      - Dockerfile
      - evaluate.sh  (called as: evaluate.sh <solution_path> <mode>)

    Any data files or other resources needed by evaluate.sh, such as a
    requirements.txt or data files, are the benchmark's own concern — the
    framework imposes no structure on them.

    evaluate.sh receives two arguments:
      1. ``<solution_path>`` — absolute path to the candidate program inside
         the container (e.g. ``/tmp/candidate_abc123.py``).
      2. ``<mode>`` — either ``"train"`` or ``"test"``.

         - **train**: called during the optimization loop in the process
           of iterating towards a single solution. This may be called multiple
           times per program, thus should be relatively fast.
         - **test**: called at publish time (e.g. end-of-run best program).
           Should be the authoritative, full evaluation, which will be used
           for reporting and leaderboard ranking.

         Evaluators that don't need the distinction can ignore the mode.

    evaluate.sh writes a single JSON object to stdout::

        {
          "status": "success" | "error" | "timeout",
          "combined_score": <float>,
          "metrics": {<str>: <float>},
          "artifacts": {<str>: <str>}   // optional
        }

    Exit codes:
      0 — evaluation completed (score may still reflect failure)
      1 — evaluator itself crashed (infrastructure problem)

    The image is built once at init time (Docker's layer cache makes
    subsequent builds near-instant when nothing changed).

    A single container is started at init time and reused across evaluations.
    Each evaluation injects its candidate file via stdin (no host filesystem
    dependency) and runs evaluate.sh with docker exec.  Concurrent evaluations
    are safe because each uses a unique path inside the container's /tmp.

    Design note: ``_run_single_in_container`` is intentionally a plain method
    (not async) so it can be overridden by adapters targeting other container
    interfaces (e.g. Harbor's /solution + /logs/verifier/reward.json).
    """

    def __init__(
        self,
        benchmark_dir: str,
        config: EvaluatorConfig,
        max_concurrent: int = 4,
    ):
        self.benchmark_dir = os.path.abspath(benchmark_dir)
        self.config = config
        self.program_suffix = config.file_suffix
        self.task_pool = TaskPool(max_concurrency=max_concurrent)
        self.image_tag = self._build_image()
        self.container_id = self._start_container()
        logger.info(f"ContainerizedEvaluator ready: container={self.container_id[:12]}")

    def close(self):
        """Stop and remove the persistent container."""
        if getattr(self, "container_id", None):
            subprocess.run(
                ["docker", "stop", self.container_id],
                capture_output=True,
            )
            self.container_id = None

    # ------------------------------------------------------------------
    # Public API — mirrors Evaluator's interface
    # ------------------------------------------------------------------

    async def evaluate_program(
        self,
        program_solution: str,
        program_id: str = "",
        mode: str = "train",
    ) -> EvaluationResult:
        """Evaluate one candidate program and return scores.

        Args:
            program_solution: Source code (or path, for image mode) of the candidate.
            program_id: Optional identifier for logging.
            mode: ``"train"`` for hot-loop evaluation, ``"test"`` for
                  authoritative/publish evaluation.
        """
        start_time = time.time()
        label = f" {program_id}" if program_id else ""

        last_exception = None
        for attempt in range(self.config.max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(
                        None, self._run_container, program_solution, mode
                    ),
                    timeout=self.config.timeout,
                )
                elapsed = time.time() - start_time
                logger.info(
                    f"Evaluated program{label} [{mode}] in {elapsed:.2f}s:"
                    f" {format_metrics(result.metrics)}"
                )
                return result

            except asyncio.TimeoutError:
                logger.error(f"Container timed out after {self.config.timeout}s{label}")
                return EvaluationResult(metrics={"error": 0.0, "timeout": True})

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Attempt {attempt + 1}/{self.config.max_retries + 1} failed{label}: {e}"
                )
                if attempt < self.config.max_retries:
                    await asyncio.sleep(1.0)

        logger.error(f"All attempts failed{label}: {last_exception}")
        return EvaluationResult(metrics={"error": 0.0})

    async def evaluate_batch(
        self,
        programs: List[Tuple[str, str]],
    ) -> List[EvaluationResult]:
        """Evaluate multiple programs concurrently.

        Args:
            programs: List of (solution, program_id) tuples.

        Returns:
            Results in the same order as *programs*.
        """
        return await self.task_pool.gather(
            coros=[self.evaluate_program] * len(programs),
            args_list=list(programs),
        )

    # ------------------------------------------------------------------
    # Container interaction — override for alternative interfaces
    # ------------------------------------------------------------------

    def _run_container(self, program_solution: str, mode: str) -> EvaluationResult:
        """Inject the candidate program and run evaluate.sh inside the container.

        Uses a unique /tmp path per call so concurrent evaluations don't collide.

        Override this method to target a different container interface
        (e.g. Harbor: cp to /solution/, read reward from /logs/verifier/reward.json).
        """
        candidate_path = self._inject_file(program_solution, self.program_suffix)
        try:
            return self._run_single_in_container(candidate_path, mode)
        finally:
            self._remove_file(candidate_path)

    def _run_single_in_container(self, candidate_path: str, mode: str) -> EvaluationResult:
        """Execute evaluate.sh inside the container and parse its JSON output."""
        try:
            proc = subprocess.run(
                [
                    "docker",
                    "exec",
                    self.container_id,
                    "/benchmark/evaluate.sh",
                    candidate_path,
                    mode,
                ],
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
            )
        except subprocess.TimeoutExpired:
            logger.error(f"docker exec timed out after {self.config.timeout}s")
            return EvaluationResult(
                metrics={"error": 0.0, "timeout": True},
                artifacts={"error": f"docker exec timed out after {self.config.timeout}s"},
            )
        if proc.returncode != 0:
            logger.error(f"Evaluator exited with code {proc.returncode}:\n{proc.stderr}")
            return EvaluationResult(
                metrics={"error": 0.0},
                artifacts={"stderr": proc.stderr, "exit_code": str(proc.returncode)},
            )

        result = self._parse_output(proc.stdout)
        # Always surface stderr (e.g. warnings, partial tracebacks) even on
        # successful exit — the evaluator may have caught the error internally
        # and returned valid JSON, but stderr still has useful context.
        if proc.stderr.strip():
            result.artifacts.setdefault("stderr", proc.stderr)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _inject_file(self, content: str, suffix: str) -> str:
        """Write content to a unique temp file inside the container via stdin."""
        path = f"/tmp/{uuid.uuid4().hex}{suffix}"
        inject = subprocess.run(
            ["docker", "exec", "-i", self.container_id, "tee", path],
            input=content.encode(),
            capture_output=True,
        )
        if inject.returncode != 0:
            raise RuntimeError(f"Failed to inject file into container: {inject.stderr.decode()}")
        return path

    def _remove_file(self, path: str) -> None:
        """Remove a file inside the container."""
        subprocess.run(
            ["docker", "exec", self.container_id, "rm", "-f", path],
            capture_output=True,
        )

    def _parse_output(self, stdout: str) -> EvaluationResult:
        try:
            data = json.loads(stdout.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse evaluator JSON: {e}\nOutput: {stdout!r}")
            return EvaluationResult(
                metrics={"error": 0.0},
                artifacts={"raw_output": stdout},
            )

        status = data.get("status", "error")
        combined_score = float(data.get("combined_score", 0.0))
        metrics = {
            k: float(v) for k, v in data.get("metrics", {}).items() if isinstance(v, (int, float))
        }
        if "combined_score" not in metrics:
            metrics["combined_score"] = combined_score

        artifacts = {k: str(v) for k, v in data.get("artifacts", {}).items()}
        if status != "success":
            artifacts.setdefault("status", status)

        return EvaluationResult(metrics=metrics, artifacts=artifacts)

    def _start_container(self) -> str:
        """Start a persistent container and return its ID."""
        result = subprocess.run(
            ["docker", "run", "-d", "--rm", "--entrypoint", "sleep", self.image_tag, "infinity"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    def _build_image(self) -> str:
        norm = os.path.normpath(self.benchmark_dir)
        name = os.path.basename(norm)
        # Include parent dir to avoid tag collisions when multiple benchmarks
        # share the same leaf directory name (e.g. "evaluator").
        parent = os.path.basename(os.path.dirname(norm))
        if parent and name == "evaluator":
            name = f"{parent}-{name}"
        tag = f"skydiscover-{name}:latest"

        logger.info(f"Building Docker image: {tag} (from {self.benchmark_dir})")
        result = subprocess.run(
            ["docker", "build", "-t", tag, self.benchmark_dir],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Docker build failed for {self.benchmark_dir}:\n{result.stderr}")
        return tag
