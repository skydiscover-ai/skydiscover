"""Benchmark resolution helpers."""

import importlib
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class BenchmarkResolution:
    """Resolved benchmark assets and evaluator-scoped configuration."""

    initial_program_path: str
    evaluator_path: str
    evaluator_env_vars: Dict[str, str] = field(default_factory=dict)


def resolve_benchmark_problem(benchmark_config: Any) -> BenchmarkResolution:
    """Load benchmark problem from external dataset using the configured resolver."""
    resolver_path = getattr(benchmark_config, "resolver", None)
    if not resolver_path:
        raise ValueError("BenchmarkConfig.resolver must be set to use benchmark loading")

    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    resolver_module = importlib.import_module(resolver_path)
    resolver = resolver_module.resolver

    benchmark_name = getattr(benchmark_config, "name", None) or "benchmark"
    output_dir = Path(tempfile.mkdtemp(prefix=f"skydiscover_{benchmark_name}_"))

    params = getattr(benchmark_config, "params", {})
    return resolver.resolve(config=params, output_dir=output_dir)
