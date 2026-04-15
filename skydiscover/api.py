"""
Public library API for SkyDiscover.

This module exposes the two main entry points for programmatic use:

* `run_discovery`: accept file paths or inline strings for the initial program and evaluator,
  wires up configuration, and returns a `DiscoveryResult`.
* `discover_solution`: convenience wrapper when the initial solution is a plain string and
  the evaluator is a Python callable.

Quick-start::

    from skydiscover import run_discovery

    result = run_discovery(
        evaluator="examples/my_problem/eval.py",
        initial_program="examples/my_problem/init.py",  # optional
        model="gpt-5",
        iterations=50,
    )
    print(result.best_score, result.best_solution)
"""

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from skydiscover.benchmarks.resolution import resolve_benchmark_problem
from skydiscover.config import Config, apply_overrides, load_config
from skydiscover.runner import Runner
from skydiscover.search.base_database import Program
from skydiscover.utils.metrics import get_score
from skydiscover.utils.prepare import cleanup_temp, prepare_evaluator, prepare_program

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryResult:
    """Result of a single discovery run."""

    best_program: Optional[Program]
    best_score: float
    best_solution: str
    metrics: Dict[str, Any]
    output_dir: Optional[str]
    initial_score: Optional[float] = None

    def __repr__(self) -> str:
        init = f"{self.initial_score:.4f}" if self.initial_score is not None else "N/A"
        return f"DiscoveryResult(best_score={self.best_score:.4f}, initial_score={init})"


def run_discovery(
    evaluator: Union[str, Path, Callable],
    initial_program: Optional[Union[str, Path, List[str]]] = None,
    model: Optional[str] = None,
    iterations: Optional[int] = None,
    search: Optional[str] = None,
    config: Union[str, Path, Config, None] = None,
    agentic: bool = False,
    output_dir: Optional[str] = None,
    system_prompt: Optional[str] = None,
    api_base: Optional[str] = None,
    cleanup: bool = True,
) -> DiscoveryResult:
    """Run a discovery process and return the best result.

    Args:
        evaluator: File path or callable (program_path) -> metrics_dict.
        initial_program: File path or inline source code (string / list of lines).
            Optional — when omitted the LLM generates a solution from scratch.
        model: Model name(s), comma-separated. e.g. "gpt-5" or "gpt-5,gemini/gemini-3-pro".
        iterations: Max iterations (overrides config).
        search: Algorithm name ("topk", "adaevolve", "evox", "openevolve_native", etc.).
        config: YAML path, Config object, or None for defaults.
        agentic: Enable agentic mode (codebase root derived from initial_program).
        output_dir: Where to write results (temp dir if None).
        system_prompt: Domain-specific context for the LLM.
        api_base: Base URL for an OpenAI-compatible API.
        cleanup: Remove temp files after the run.

    Returns:
        DiscoveryResult with best program, score, solution, metrics, and output directory.
    """
    return asyncio.run(
        _run_discovery_async(
            initial_program,
            evaluator,
            config,
            iterations=iterations,
            output_dir=output_dir,
            cleanup=cleanup,
            agentic=agentic,
            model=model,
            search=search,
            system_prompt=system_prompt,
            api_base=api_base,
        )
    )


async def _run_discovery_async(
    initial_program: Optional[Union[str, Path, List[str]]],
    evaluator: Union[str, Path, Callable],
    config: Union[str, Path, Config, None],
    *,
    model: Optional[str] = None,
    iterations: Optional[int] = None,
    search: Optional[str] = None,
    agentic: bool = False,
    output_dir: Optional[str] = None,
    system_prompt: Optional[str] = None,
    api_base: Optional[str] = None,
    cleanup: bool = True,
) -> DiscoveryResult:
    """Async implementation of run_discovery."""

    temp_dir: Optional[str] = None
    temp_files: List[str] = []
    evaluator_env_vars: Dict[str, str] = {}

    try:
        if isinstance(config, Config):
            config_obj = config
        else:
            config_obj = load_config(str(config) if config else None)

        apply_overrides(
            config_obj,
            model=model,
            api_base=api_base,
            agentic=agentic,
            search=search,
            system_prompt=system_prompt,
        )

        # Resolve benchmark problem if configured and no initial_program provided
        if initial_program is None and config_obj.benchmark and config_obj.benchmark.enabled:
            try:
                resolution = resolve_benchmark_problem(config_obj.benchmark)
                initial_program = resolution.initial_program_path
                evaluator = resolution.evaluator_path
                evaluator_env_vars = resolution.evaluator_env_vars
                logger.info(
                    f"[Benchmark Loader] Benchmark: {config_obj.benchmark.name}, Initial program: {initial_program}, Evaluator: {evaluator}"
                )
            except Exception as exc:
                raise ValueError(f"Failed to load benchmark problem: {exc}") from exc

        # Prepare the program (optional — None means "from scratch")
        program_path = (
            prepare_program(initial_program, temp_dir, temp_files)
            if initial_program is not None
            else None
        )

        if program_path and config_obj.agentic.enabled and not config_obj.agentic.codebase_root:
            config_obj.agentic.codebase_root = os.path.dirname(os.path.abspath(program_path))

        # Prepare the evaluator
        evaluator_path = prepare_evaluator(evaluator, temp_dir, temp_files)

        # Prepare the output directory
        search_type = (
            getattr(config_obj.search, "type", None) if hasattr(config_obj, "search") else None
        )
        if output_dir is None and cleanup:
            temp_dir = tempfile.mkdtemp(prefix="skydiscover_")
            actual_output_dir = temp_dir
        else:
            from skydiscover.config import build_output_dir

            actual_output_dir = output_dir or build_output_dir(
                search_type or "default", program_path or "scratch"
            )
            os.makedirs(actual_output_dir, exist_ok=True)

        # External backends (openevolve, shinkaevolve, gepa)
        if search_type:
            from skydiscover.extras.external import KNOWN_EXTERNAL, get_runner, is_external

            if is_external(search_type):
                if evaluator_env_vars:
                    env_var_names = ", ".join(sorted(evaluator_env_vars))
                    raise ValueError(
                        "Passing evaluator environment variables to external backends is not yet supported. "
                        f"External backend '{search_type}' cannot be used with evaluator env vars: "
                        f"{env_var_names}"
                    )

                from skydiscover.extras.monitor import start_monitor, stop_monitor

                monitor_server, monitor_callback, feedback_reader = start_monitor(
                    config_obj, actual_output_dir
                )
                try:
                    result = await get_runner(search_type)(
                        program_path=program_path,
                        evaluator_path=evaluator_path,
                        config_obj=config_obj,
                        iterations=iterations or config_obj.max_iterations,
                        output_dir=actual_output_dir,
                        monitor_callback=monitor_callback,
                        feedback_reader=feedback_reader,
                    )
                except ModuleNotFoundError as exc:
                    from skydiscover.extras.external import get_package_name

                    pkg = get_package_name(search_type)
                    raise ImportError(
                        f"{exc}\n\nThe '{search_type}' backend requires its package. "
                        f"Install with: pip install {pkg}"
                    ) from exc
                finally:
                    stop_monitor(monitor_server)
                result.output_dir = actual_output_dir if not cleanup else None
                return result

            if search_type in KNOWN_EXTERNAL:
                from skydiscover.extras.external import get_package_name

                pkg = get_package_name(search_type)
                raise ImportError(
                    f"Search type '{search_type}' requires the '{pkg}' package. "
                    f"Install with: pip install {pkg}"
                )

        if not config_obj.llm.models:
            raise ValueError(
                "No LLM models configured. Provide a config with models or "
                "pass model= directly:\n\n"
                "  result = run_discovery(evaluator, model='gpt-5')"
            )

        # Initialize the runner
        controller = Runner(
            initial_program_path=program_path,
            evaluation_file=evaluator_path,
            config=config_obj,
            output_dir=actual_output_dir,
            evaluator_env_vars=evaluator_env_vars,
        )

        best_program = await controller.run(iterations=iterations)

        best_score = 0.0
        best_solution = ""
        metrics: Dict[str, Any] = {}

        if best_program:
            best_solution = best_program.solution
            metrics = best_program.metrics or {}
            best_score = get_score(metrics)

        initial_score = controller.initial_score

        # Return the result
        return DiscoveryResult(
            best_program=best_program,
            best_score=best_score,
            best_solution=best_solution,
            metrics=metrics,
            output_dir=actual_output_dir if not cleanup else None,
            initial_score=initial_score,
        )

    finally:
        if cleanup:
            cleanup_temp(temp_files, temp_dir)


def discover_solution(
    evaluator: Callable[[str], Dict[str, Any]],
    initial_solution: Optional[str] = None,
    iterations: int = 100,
    search: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs: Any,
) -> DiscoveryResult:
    """Convenience wrapper: evolve a string solution with a callable evaluator.

    Same as run_discovery but defaults to string input + callable evaluator.
    """
    return run_discovery(
        evaluator=evaluator,
        initial_program=initial_solution,
        iterations=iterations,
        search=search,
        model=model,
        **kwargs,
    )
