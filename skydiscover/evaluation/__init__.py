"""Program evaluation framework.

Evaluator hierarchy (pick one per benchmark)::

    Evaluator                  Python function: evaluate(program_path) -> dict
    SubprocessEvaluator        Same as Evaluator but each call runs in a child process
    ContainerizedEvaluator     Docker: Dockerfile + evaluate.sh  ->  JSON on stdout
    └── HarborEvaluator        Harbor protocol: instruction.md + tests/test.sh + environment/

``create_evaluator()`` auto-detects which one to use based on the benchmark
directory contents.  Detection order: Harbor > Containerized > Python.
Set ``evaluator.subprocess_isolation: true`` in config to use SubprocessEvaluator
instead of the in-process Evaluator.

Supporting modules:

- ``evaluation_result``  — ``EvaluationResult`` dataclass (metrics + artifacts)
- ``llm_judge``          — optional LLM-as-a-judge scorer (adds ``llm_*`` metrics)
- ``wrapper``            — CLI bridge so old ``evaluate(path)->dict`` functions
                           can emit container-protocol JSON on stdout
"""

import os
from typing import Dict, Optional, Union

from skydiscover.evaluation.container_evaluator import ContainerizedEvaluator
from skydiscover.evaluation.evaluation_result import EvaluationResult
from skydiscover.evaluation.evaluator import Evaluator
from skydiscover.evaluation.harbor_evaluator import HarborEvaluator
from skydiscover.evaluation.llm_judge import LLMJudge
from skydiscover.evaluation.subprocess_evaluator import SubprocessEvaluator

__all__ = [
    "EvaluationResult",
    "Evaluator",
    "SubprocessEvaluator",
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
        and os.path.exists(os.path.join(path, "tests", "test.sh"))
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
    env_vars: Optional[Dict[str, str]] = None,
) -> Union[Evaluator, SubprocessEvaluator, ContainerizedEvaluator, HarborEvaluator]:
    """Return the right evaluator for the given config.

    Detection order (most specific first):
      1. Harbor task — instruction.md + tests/ + environment/Dockerfile
      2. Containerized — Dockerfile + evaluate.sh
      3. subprocess_isolation=True — SubprocessEvaluator (process-per-evaluation)
      4. Python evaluator — fallback (in-process)
    """
    path = config.evaluation_file or ""
    if _is_harbor_task(path):
        return HarborEvaluator(path, config, max_concurrent=max_concurrent, env_vars=env_vars)
    if _is_containerized(path):
        return ContainerizedEvaluator(
            path, config, max_concurrent=max_concurrent, env_vars=env_vars
        )
    if getattr(config, "subprocess_isolation", False):
        return SubprocessEvaluator(
            config, llm_judge=llm_judge, max_concurrent=max_concurrent, env_vars=env_vars
        )
    return Evaluator(config, llm_judge=llm_judge, max_concurrent=max_concurrent, env_vars=env_vars)
