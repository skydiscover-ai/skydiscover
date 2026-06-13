"""KV-store task resolver for SkyDiscover.

Unlike the KernelBench resolver (which downloads problems from a dataset), every
kvstore task is **checked in** under ``benchmarks/kvstore/<id>/``. This resolver
just maps a ``task`` id to that directory's ``initial_program.cc`` + ``evaluator/``
so you can run a task path-free::

    skydiscover-run -c benchmarks/kvstore/0001_ycsb50_zipf_8gb/config.yaml -s jitskit

It also forwards the human-facing spec (``benchmark.params``) to the evaluator as
``SKYKV_*`` environment variables, so the same task is scorable by *any* strategy
(``jitskit`` reads the knobs from ``search.database`` directly; ``claude_code`` /
evolutionary strategies score the candidate through the task ``evaluator/``).
"""

import logging
from pathlib import Path
from typing import Any, Dict

from skydiscover.benchmarks.base import BenchmarkResolution, BenchmarkResolver

logger = logging.getLogger(__name__)

_FAMILY_DIR = Path(__file__).parent

# Spec keys forwarded to the evaluator as SKYKV_<UPPER> env vars.
_SPEC_ENV_KEYS = ("workload", "distribution", "value_size", "mem_budget_gb", "baseline")


class KVStoreResolver(BenchmarkResolver):
    """Resolve a checked-in kvstore task by its directory id.

    Required ``benchmark.params``:
        task: the task directory id, e.g. ``"0001_ycsb50_zipf_8gb"``.
    """

    def resolve(self, config: Dict[str, Any], output_dir: Path) -> BenchmarkResolution:
        task_id = config.get("task")
        if not task_id:
            available = sorted(p.name for p in _FAMILY_DIR.iterdir() if _is_task_dir(p))
            raise ValueError(
                "kvstore resolver requires benchmark.params.task (a task id). "
                f"Available tasks: {available}"
            )

        task_dir = _FAMILY_DIR / task_id
        initial_program = task_dir / "initial_program.cc"
        # Return the evaluator .py file (not the dir): create_evaluator routes a .py to the
        # host Evaluator. A bare dir only works for a Dockerfile-based ContainerizedEvaluator.
        evaluator = task_dir / "evaluator" / "evaluator.py"
        if not initial_program.exists():
            raise FileNotFoundError(
                f"No initial_program.cc for task {task_id!r} at {initial_program}"
            )
        if not evaluator.exists():
            raise FileNotFoundError(
                f"No evaluator/evaluator.py for task {task_id!r} at {evaluator}"
            )

        env_vars = {
            f"SKYKV_{key.upper()}": str(config[key])
            for key in _SPEC_ENV_KEYS
            if config.get(key) is not None
        }
        logger.info("Resolved kvstore task %s -> %s", task_id, task_dir)
        return BenchmarkResolution(
            initial_program_path=str(initial_program),
            evaluator_path=str(evaluator),
            evaluator_env_vars=env_vars,
        )


def _is_task_dir(path: Path) -> bool:
    """A task dir is ``NNNN_<slug>/`` (not a shared ``_harness``/``_baselines`` dir)."""
    return path.is_dir() and not path.name.startswith(("_", ".")) and "__" not in path.name


# The loader (skydiscover.benchmarks.resolution) imports ``resolver`` from this module.
resolver = KVStoreResolver()
