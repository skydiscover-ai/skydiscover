"""Containerized evaluator: runs evaluate.sh inside a persistent Docker container."""

import asyncio
import hashlib
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
      - evaluate.sh  (called as: evaluate.sh <mode> <program_path>)

    Any data files or other resources needed by evaluate.sh are the
    benchmark's own concern — the framework imposes no structure on them.

    evaluate.sh writes a single JSON object to stdout:
      {
        "status": "success" | "error" | "timeout",
        "combined_score": <float in [0,1]>,
        "metrics": {<str>: <float>},
        "artifacts": {<str>: <str>}   # optional
      }

    Exit codes:
      0 — evaluation completed (score may still reflect failure)
      1 — evaluator itself crashed (infrastructure problem)

    The image is built once and cached by a hash of all files in the benchmark
    directory; rebuilds happen automatically when anything changes.

    A single container is started at init time and reused across evaluations.
    Each evaluation injects its candidate file via stdin (no host filesystem
    dependency) and runs evaluate.sh with docker exec.  Concurrent evaluations
    are safe because each uses a unique path inside the container's /tmp.

    Design note: _run_container is intentionally a plain method (not async)
    so it can be overridden by adapters targeting other interfaces (e.g.
    Harbor's /solution + /logs/verifier/reward.json convention).
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

    def __del__(self):
        self.close()

    # ------------------------------------------------------------------
    # Public API — mirrors Evaluator's interface
    # ------------------------------------------------------------------

    async def evaluate_program(
        self,
        program_solution: str,
        program_id: str = "",
    ) -> EvaluationResult:
        """Evaluate one candidate program (train mode) and return scores."""
        start_time = time.time()
        label = f" {program_id}" if program_id else ""

        last_exception = None
        for attempt in range(self.config.max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(
                        None, self._run_container, program_solution, "train"
                    ),
                    timeout=self.config.timeout,
                )
                elapsed = time.time() - start_time
                logger.info(
                    f"Evaluated program{label} in {elapsed:.2f}s:"
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
        """Evaluate multiple programs concurrently (train mode).

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
    # Docker helpers — override _run_container for alternative interfaces
    # ------------------------------------------------------------------

    def _run_container(self, program_solution: str, mode: str) -> EvaluationResult:
        """Inject the candidate program and run evaluate.sh inside the container.

        Uses a unique /tmp path per call so concurrent evaluations don't collide.

        Override this method to target a different container interface
        (e.g. Harbor: cp to /solution/, read reward from /logs/verifier/reward.json).
        """
        candidate_path = f"/tmp/candidate_{uuid.uuid4().hex}{self.program_suffix}"

        # Inject solution into the container via stdin — no host temp file needed.
        inject = subprocess.run(
            ["docker", "exec", "-i", self.container_id, "/bin/sh", "-c", f"cat > {candidate_path}"],
            input=program_solution.encode(),
            capture_output=True,
        )
        if inject.returncode != 0:
            logger.error(f"Failed to inject candidate: {inject.stderr.decode()}")
            return EvaluationResult(
                metrics={"error": 0.0},
                artifacts={"stderr": inject.stderr.decode()},
            )

        try:
            proc = subprocess.run(
                [
                    "docker",
                    "exec",
                    self.container_id,
                    "/benchmark/evaluate.sh",
                    mode,
                    candidate_path,
                ],
                capture_output=True,
                text=True,
            )
            if proc.returncode == 1:
                logger.error(f"Evaluator crashed (exit 1):\n{proc.stderr}")
                return EvaluationResult(
                    metrics={"error": 0.0},
                    artifacts={"stderr": proc.stderr},
                )
            return self._parse_output(proc.stdout)
        finally:
            subprocess.run(
                ["docker", "exec", self.container_id, "rm", "-f", candidate_path],
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
        name = os.path.basename(self.benchmark_dir)
        tag = f"skydiscover-{name}:{self._content_hash()}"

        check = subprocess.run(
            ["docker", "image", "inspect", tag],
            capture_output=True,
        )
        if check.returncode == 0:
            logger.info(f"Reusing cached Docker image: {tag}")
            return tag

        logger.info(f"Building Docker image: {tag} (from {self.benchmark_dir})")
        result = subprocess.run(
            ["docker", "build", "-t", tag, self.benchmark_dir],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Docker build failed for {self.benchmark_dir}:\n{result.stderr}")
        return tag

    def _content_hash(self) -> str:
        """12-char hash of all files in the benchmark directory for image cache invalidation."""
        h = hashlib.sha256()
        for root, dirs, files in os.walk(self.benchmark_dir):
            dirs.sort()  # deterministic traversal order
            for fname in sorted(files):
                path = os.path.join(root, fname)
                # Include the relative path so renames also invalidate the cache
                h.update(os.path.relpath(path, self.benchmark_dir).encode())
                with open(path, "rb") as f:
                    h.update(f.read())
        return h.hexdigest()[:12]
