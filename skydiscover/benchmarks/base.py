"""Base interface for benchmark resolvers.

Benchmark resolvers fetch problems from external sources (e.g., datasets, APIs)
and generate the necessary files (initial_program, evaluator configuration) for
SkyDiscover to run optimization on them.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple


class BenchmarkResolver(ABC):
    """Base class for benchmark-specific problem resolvers.

    Resolvers are responsible for:
    1. Fetching problem specifications from external sources
    2. Generating initial_program files with appropriate structure
    3. Configuring evaluators (via environment variables or generated files)

    Example usage:
        resolver = KernelBenchResolver()
        initial_program, evaluator = resolver.resolve(
            config={'level': 1, 'problem_id': 3},
            output_dir=Path('/tmp/skydiscover_kernelbench_123')
        )
    """

    @abstractmethod
    def resolve(self, config: Dict[str, Any], output_dir: Path) -> Tuple[str, str]:
        """Resolve a benchmark problem to concrete file paths.

        Args:
            config: Benchmark configuration dictionary containing benchmark-specific
                   problem specifications and parameters.
                   The exact keys depend on the benchmark implementation.
            output_dir: Directory where generated files should be placed.

        Returns:
            Tuple of (initial_program_path, evaluator_path):
                - initial_program_path: Path to the generated initial program file
                - evaluator_path: Path to the evaluator (file or directory)

        Note:
            Resolvers can set environment variables with the CONTAINER_ENV_PREFIX
            (from skydiscover.evaluation.container_evaluator) to pass configuration
            to containerized evaluators. The prefix will be stripped before passing
            to the container.

        """
        pass
