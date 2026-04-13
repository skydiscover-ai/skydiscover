"""KernelBench problem resolver for SkyDiscover.

This resolver fetches GPU kernel optimization problems from the KernelBench
dataset and generates the necessary files for SkyDiscover to run optimization.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

from skydiscover.benchmarks.base import BenchmarkResolution, BenchmarkResolver
from skydiscover.utils.prepare import prepare_program

logger = logging.getLogger(__name__)


class KernelBenchResolver(BenchmarkResolver):
    """Resolves KernelBench problems by fetching from dataset and generating files.

    The resolver:
    1. Fetches the reference implementation from KernelBench dataset
    2. Generates initial_program.py with EVOLVE-BLOCK markers
    3. Sets environment variables for the evaluator
    4. Returns paths to the generated initial program and existing evaluator

    Required config parameters:
        - level: Problem difficulty level (1, 2, or 3)
        - problem_id: Specific problem ID within the level

    Optional config parameters:
        - dataset_src: 'huggingface' (default) or 'local'
        - dataset_name: HuggingFace dataset name (default: 'ScalingIntelligence/KernelBench')
        - eval_mode: 'local' (default) or 'modal'
        - gpu: GPU type for evaluation (default: 'H100')
        - num_correct_trials: Number of correctness validation runs (default: 5)
        - num_perf_trials: Number of performance measurement runs (default: 100)
    """

    def resolve(self, config: Dict[str, Any], output_dir: Path) -> BenchmarkResolution:
        """Fetch KernelBench problem and generate initial_program + configure evaluator.

        Args:
            config: Configuration dictionary with 'level', 'problem_id', and optional params
            output_dir: Directory where generated files will be placed

        Returns:
            BenchmarkResolution with initial program, evaluator path, and evaluator env vars
        """
        # Validate required parameters
        level = config.get("level")
        problem_id = config.get("problem_id")

        if level is None or problem_id is None:
            raise ValueError(
                "KernelBench resolver requires 'level' and 'problem_id' in config. "
                f"Got: level={level}, problem_id={problem_id}"
            )

        # Extract optional parameters with defaults
        dataset_src = config.get("dataset_src", "huggingface")
        dataset_name = config.get("dataset_name", "ScalingIntelligence/KernelBench")
        eval_mode = config.get("eval_mode", "local")
        gpu = config.get("gpu", "H100")
        num_correct_trials = config.get("num_correct_trials", 5)
        num_perf_trials = config.get("num_perf_trials", 100)

        logger.info(f"Resolving KernelBench problem: level={level}, problem_id={problem_id}")
        logger.info(f"Eval mode: {eval_mode}, GPU: {gpu}")

        # Import KernelBench dataset utilities
        try:
            from kernelbench.dataset import construct_kernelbench_dataset
        except ImportError as e:
            raise ImportError(
                "KernelBench package not found. Install with: "
                "uv pip install 'kernelbench @ git+https://github.com/ScalingIntelligence/KernelBench.git'"
            ) from e

        # Fetch the problem from KernelBench dataset
        try:
            dataset = construct_kernelbench_dataset(
                level=level,
                source=dataset_src,
                dataset_name=dataset_name,
            )
            problem = dataset.get_problem_by_id(problem_id)
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch KernelBench problem (level={level}, id={problem_id}): {e}"
            ) from e

        logger.info(f"Fetched problem: {problem.name} (ID: {problem.problem_id})")

        # Generate initial_program.py with EVOLVE-BLOCK markers using prepare_program
        output_dir.mkdir(parents=True, exist_ok=True)
        initial_program_path = prepare_program(
            initial_program=problem.code, temp_dir=str(output_dir), temp_files=[]
        )
        logger.info(f"Generated initial program: {initial_program_path}")

        use_docker = config.get("use_docker", True)

        # Use evaluator.py file for native mode, directory for container mode
        if use_docker:
            evaluator_path = Path(__file__).parent / "evaluator"
            logger.info("Using containerized evaluator (Docker required)")
        else:
            evaluator_path = Path(__file__).parent / "evaluator" / "evaluator.py"
            logger.info("Using native Python evaluator (no Docker required)")

        evaluator_env_vars = {
            "KERNELBENCH_LEVEL": str(level),
            "KERNELBENCH_PROBLEM_ID": str(problem_id),
            "KERNELBENCH_EVAL_MODE": eval_mode,
            "KERNELBENCH_GPU": gpu,
            "KERNELBENCH_NUM_CORRECT_TRIALS": str(num_correct_trials),
            "KERNELBENCH_NUM_PERF_TRIALS": str(num_perf_trials),
            "KERNELBENCH_TIMEOUT": str(config.get("timeout", 300)),
        }

        mode_desc = "container" if use_docker else "native evaluator"
        logger.info(f"Prepared evaluator environment for {mode_desc}:")
        logger.info(f"  KERNELBENCH_LEVEL={level}")
        logger.info(f"  KERNELBENCH_PROBLEM_ID={problem_id}")
        logger.info(f"  KERNELBENCH_EVAL_MODE={eval_mode}")
        logger.info(f"  KERNELBENCH_GPU={gpu}")

        return BenchmarkResolution(
            initial_program_path=str(initial_program_path),
            evaluator_path=str(evaluator_path),
            evaluator_env_vars=evaluator_env_vars,
        )


# Module-level resolver instance
resolver = KernelBenchResolver()
