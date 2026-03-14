import os
from typing import Optional, Union

from skydiscover.evaluation.container_evaluator import ContainerizedEvaluator
from skydiscover.evaluation.evaluation_result import EvaluationResult
from skydiscover.evaluation.evaluator import Evaluator
from skydiscover.evaluation.harbor_evaluator import HarborEvaluator
from skydiscover.evaluation.llm_judge import LLMJudge

__all__ = [
    "EvaluationResult",
    "Evaluator",
    "ContainerizedEvaluator",
    "HarborEvaluator",
    "LLMJudge",
    "create_evaluator",
]


def _is_harbor_task(path: str) -> bool:
    """Detect a Harbor task directory (instruction.md + tests/test.sh + environment/Dockerfile)."""
    return (
        os.path.isdir(path)
        and os.path.exists(os.path.join(path, "instruction.md"))
        and os.path.isdir(os.path.join(path, "tests"))
        and os.path.isdir(os.path.join(path, "environment"))
        and os.path.exists(os.path.join(path, "environment", "Dockerfile"))
    )


def _is_containerized(path: str) -> bool:
    """Detect a standard containerized benchmark (Dockerfile + evaluate.sh)."""
    return (
        os.path.isdir(path)
        and os.path.exists(os.path.join(path, "Dockerfile"))
        and os.path.exists(os.path.join(path, "evaluate.sh"))
    )


def create_evaluator(
    config,
    llm_judge: Optional[LLMJudge] = None,
    max_concurrent: int = 4,
) -> Union[Evaluator, ContainerizedEvaluator, HarborEvaluator]:
    """Return the right evaluator for the given config.

    Detection order (most specific first):
      1. Harbor task — instruction.md + tests/ + environment/Dockerfile
      2. Containerized — Dockerfile + evaluate.sh
      3. Python evaluator — fallback
    """
    path = config.evaluation_file or ""
    if _is_harbor_task(path):
        return HarborEvaluator(path, config, max_concurrent=max_concurrent)
    if _is_containerized(path):
        return ContainerizedEvaluator(path, config, max_concurrent=max_concurrent)
    return Evaluator(config, llm_judge=llm_judge, max_concurrent=max_concurrent)
