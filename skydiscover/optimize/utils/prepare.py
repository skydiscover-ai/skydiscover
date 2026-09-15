"""
Input preparation utilities for discovery runs.

Handles materializing user-provided programs and evaluators (file paths,
inline strings, callables) into concrete file paths on disk, plus cleanup
of any temporary files created in the process.
"""

import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def prepare_program(
    initial_program: Union[str, Path, List[str]],
    temp_dir: Optional[str],
    temp_files: List[str],
) -> str:
    """Resolve initial_program to a file path, writing a temp file if needed."""
    if isinstance(initial_program, (str, Path)) and os.path.exists(str(initial_program)):
        return str(initial_program)

    solution = (
        "\n".join(initial_program) if isinstance(initial_program, list) else str(initial_program)
    )

    if "EVOLVE-BLOCK-START" not in solution:
        solution = f"# EVOLVE-BLOCK-START\n{solution}\n# EVOLVE-BLOCK-END"

    if temp_dir is None:
        temp_dir = tempfile.gettempdir()

    program_file = os.path.join(temp_dir, f"program_{uuid.uuid4().hex[:8]}.py")
    with open(program_file, "w") as fh:
        fh.write(solution)
    temp_files.append(program_file)
    return program_file


def prepare_evaluator(
    evaluator: Union[str, Path, Callable],
    temp_dir: Optional[str],
    temp_files: List[str],
    caller_module_name: str = "skydiscover.optimize.api",
) -> str:
    """Resolve evaluator to a file path, writing a temp file if needed.

    When *evaluator* is a callable, it is registered in the caller module's
    globals so the generated wrapper script can import it at runtime.
    ``caller_module_name`` must match the module whose globals hold the callable.
    """
    if isinstance(evaluator, (str, Path)) and os.path.exists(str(evaluator)):
        return str(evaluator)

    if callable(evaluator):
        import sys

        caller_module = sys.modules.get(caller_module_name)
        evaluator_id = f"_skydiscover_evaluator_{uuid.uuid4().hex[:8]}"
        if caller_module is not None:
            setattr(caller_module, evaluator_id, evaluator)
        evaluator_code = (
            f"import {caller_module_name} as _api\n\n"
            f"def evaluate(program_path):\n"
            f"    return getattr(_api, '{evaluator_id}')(program_path)\n"
        )
    else:
        evaluator_code = str(evaluator)
        if "def evaluate" not in evaluator_code:
            raise ValueError("Evaluator code must contain a 'def evaluate(program_path)' function")

    if temp_dir is None:
        temp_dir = tempfile.gettempdir()

    eval_file = os.path.join(temp_dir, f"evaluator_{uuid.uuid4().hex[:8]}.py")
    with open(eval_file, "w") as fh:
        fh.write(evaluator_code)
    temp_files.append(eval_file)
    return eval_file


def cleanup_temp(temp_files: List[str], temp_dir: Optional[str]) -> None:
    """Best-effort removal of temporary files and directories."""
    for path in temp_files:
        try:
            os.unlink(path)
        except OSError as exc:
            logger.warning("Failed to delete temp file %s: %s", path, exc)
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except OSError as exc:
            logger.warning("Failed to delete temp directory %s: %s", temp_dir, exc)
