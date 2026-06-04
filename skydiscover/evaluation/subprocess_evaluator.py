"""
Subprocess-isolated evaluator.

Runs each candidate evaluation in a separate Python process so that crashes
(e.g. CUDA illegal memory access, segfaults, memory corruption) in one
candidate cannot affect subsequent evaluations.

Sits between the in-process Evaluator (fast, no isolation) and the
ContainerizedEvaluator (full Docker, high overhead).
"""

import asyncio
import errno
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

from skydiscover.config import EvaluatorConfig
from skydiscover.evaluation.evaluation_result import EvaluationResult
from skydiscover.evaluation.llm_judge import LLMJudge
from skydiscover.utils.async_utils import TaskPool
from skydiscover.utils.metrics import format_metrics

logger = logging.getLogger(__name__)

_WRAPPER_TEMPLATE = """\
import json
import sys
import importlib.util

spec = importlib.util.spec_from_file_location("_eval_mod", {evaluator_path!r})
mod = importlib.util.module_from_spec(spec)
sys.modules["_eval_mod"] = mod
spec.loader.exec_module(mod)

result = mod.evaluate(sys.argv[1])
print(json.dumps(result))
"""


class SubprocessEvaluator:
    """
    Runs the user-provided evaluate() function in a child process.

    Each call to evaluate_program() spawns a new Python subprocess that:
      1. Imports the evaluation module fresh
      2. Calls evaluate(program_path)
      3. Prints the result dict as JSON to stdout
      4. Exits

    This gives full process isolation: if the candidate program corrupts
    GPU state, segfaults, or leaks memory, only the child dies.
    """

    def __init__(
        self,
        config: EvaluatorConfig,
        llm_judge: Optional[LLMJudge] = None,
        max_concurrent: int = 4,
        env_vars: Optional[Dict[str, str]] = None,
    ):
        if not config.evaluation_file:
            raise ValueError("EvaluatorConfig.evaluation_file must be set")
        if not os.path.exists(config.evaluation_file):
            raise ValueError(f"Evaluation file not found: {config.evaluation_file}")

        self.config = config
        self.evaluation_file = os.path.abspath(config.evaluation_file)
        self.program_suffix = config.file_suffix
        self.is_image_mode = config.is_image_mode
        self.llm_judge = llm_judge
        self.task_pool = TaskPool(max_concurrency=max_concurrent)
        self.env_vars = dict(env_vars or {})

        eval_dir = os.path.dirname(self.evaluation_file)
        if eval_dir not in sys.path:
            sys.path.insert(0, eval_dir)

        self._wrapper_script = _WRAPPER_TEMPLATE.format(evaluator_path=self.evaluation_file)
        logger.info(
            f"Initialized SubprocessEvaluator with {self.evaluation_file} "
            f"(timeout={config.timeout}s, max_concurrent={max_concurrent})"
        )

    async def evaluate_program(
        self,
        program_solution: str,
        program_id: str = "",
        mode: str = "train",
    ) -> EvaluationResult:
        """Evaluate a candidate program in an isolated subprocess."""
        start_time = time.time()
        label = f" {program_id}" if program_id else ""

        last_exception = None
        for attempt in range(self.config.max_retries + 1):
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=self.program_suffix, delete=False, mode="w", encoding="utf-8"
                ) as f:
                    f.write(program_solution)
                    temp_path = f.name
            except OSError as e:
                if e.errno == errno.ENOSPC:
                    logger.error("Disk full — cannot create temp file")
                    return EvaluationResult(metrics={"error": 0.0, "disk_space_error": True})
                raise

            try:
                result = await self._run_subprocess(temp_path)
                eval_result = self._normalize_result(result)

                if self.llm_judge:
                    llm_result = await self.llm_judge.evaluate(program_solution, program_id)
                    if llm_result:
                        for name, value in llm_result.metrics.items():
                            eval_result.metrics[f"llm_{name}"] = value
                        eval_result.artifacts.update(llm_result.artifacts)

                elapsed = time.time() - start_time
                logger.info(
                    f"Evaluated program{label} in {elapsed:.2f}s: "
                    f"{format_metrics(eval_result.metrics)}"
                )
                return eval_result

            except asyncio.TimeoutError:
                logger.error(
                    f"Program{label} timed out after {time.time() - start_time:.0f}s "
                    f"(limit: {self.config.timeout}s)"
                )
                return EvaluationResult(metrics={"error": 0.0, "timeout": True})

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Attempt {attempt + 1}/{self.config.max_retries + 1} failed{label}: {e}"
                )
                if attempt < self.config.max_retries:
                    await asyncio.sleep(1.0)

            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        logger.error(f"All attempts failed{label}: {last_exception}")
        return EvaluationResult(metrics={"error": 0.0})

    async def evaluate_batch(
        self,
        programs: List[Tuple[str, str]],
    ) -> List[EvaluationResult]:
        """Evaluate multiple programs concurrently (each in its own subprocess)."""
        return await self.task_pool.gather(
            coros=[self.evaluate_program] * len(programs),
            args_list=list(programs),
        )

    def close(self) -> None:
        """No persistent resources to clean up."""
        pass

    async def _run_subprocess(self, program_path: str) -> Dict[str, Any]:
        """Spawn a child process, run the evaluation, parse JSON result."""
        env = os.environ.copy()
        env.update(self.env_vars)

        eval_dir = os.path.dirname(self.evaluation_file)
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = eval_dir + os.pathsep + env["PYTHONPATH"]
        else:
            env["PYTHONPATH"] = eval_dir

        loop = asyncio.get_running_loop()
        proc_result = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                self._execute_subprocess,
                program_path,
                env,
            ),
            timeout=self.config.timeout,
        )
        return proc_result

    def _execute_subprocess(self, program_path: str, env: Dict[str, str]) -> Dict[str, Any]:
        """Run the subprocess synchronously (called from executor thread)."""
        try:
            result = subprocess.run(
                [sys.executable, "-c", self._wrapper_script, program_path],
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                env=env,
                cwd=os.path.dirname(self.evaluation_file),
            )
        except subprocess.TimeoutExpired:
            raise asyncio.TimeoutError()

        if result.returncode != 0:
            stderr_tail = result.stderr[-1000:] if result.stderr else ""
            raise RuntimeError(
                f"Evaluation subprocess failed (exit {result.returncode}): {stderr_tail}"
            )

        stdout = result.stdout.strip()
        # Libraries may print warnings to stdout before the JSON.
        # Find the last JSON object in the output.
        json_start = stdout.rfind("\n{")
        if json_start != -1:
            stdout = stdout[json_start + 1:]
        elif not stdout.startswith("{"):
            raise RuntimeError(f"No JSON in subprocess output: {stdout[-500:]}")

        return json.loads(stdout)

    def _normalize_result(self, result: Any) -> EvaluationResult:
        if isinstance(result, EvaluationResult):
            return result
        if isinstance(result, dict):
            return EvaluationResult.from_dict(result)
        logger.warning(f"Unexpected result type: {type(result)}")
        return EvaluationResult(metrics={"error": 0.0})
