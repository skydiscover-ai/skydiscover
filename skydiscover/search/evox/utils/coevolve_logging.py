"""
Utility functions for co-evolution logging and JSON serialization.

All search-algorithm logging lives here as plain async functions so that
CoEvolutionController stays focused on orchestration.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from skydiscover.search.utils.discovery_utils import SerializableResult

logger = logging.getLogger(__name__)


def make_json_serializable(obj: Any) -> Any:
    """Recursively convert objects to JSON-serializable types."""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (str, int, float)):
        return obj
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted([make_json_serializable(item) for item in obj], key=lambda x: str(x))
    if hasattr(obj, "to_dict"):
        return make_json_serializable(obj.to_dict())
    return str(obj)


async def log_search_algorithm_generated(
    outputs_dir: str,
    result: SerializableResult,
    iteration: int,
    diverge_label: str = "",
    refine_label: str = "",
) -> None:
    """Save newly generated search algorithm details to files (before scoring)."""
    if result.error:
        logger.warning(f"Search strategy generation failed (iteration {iteration}): {result.error}")
        return

    child_dict = result.child_program_dict or {}

    await save_search_algorithm(
        outputs_dir=outputs_dir,
        iteration=iteration,
        program_id=child_dict.get("id", "unknown"),
        solution=child_dict.get("solution", ""),
        score=None,
        metrics={},
        diverge_label=diverge_label,
        refine_label=refine_label,
        pending_score=True,
    )

    iteration_dir = os.path.join(outputs_dir, f"iteration_{iteration}")
    logger.debug(
        f"New search algorithm generated (iteration {iteration}) - "
        f"saved to {os.path.abspath(iteration_dir)} (score pending)"
    )


async def save_search_algorithm(
    outputs_dir: str,
    iteration: int,
    program_id: str,
    solution: str,
    score: Optional[float],
    metrics: Dict[str, Any],
    system_prompt: str = "",
    user_prompt: str = "",
    llm_response: str = "",
    diverge_label: str = "",
    refine_label: str = "",
    pending_score: bool = False,
) -> None:
    """Save search algorithm details to files in search/iteration_x/."""
    iteration_dir = os.path.join(outputs_dir, f"iteration_{iteration}")
    os.makedirs(iteration_dir, exist_ok=True)

    metadata = {
        "iteration": iteration,
        "program_id": program_id,
        "score": score,
        "metrics": metrics,
        "pending_score": pending_score,
    }

    with open(os.path.join(iteration_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    if solution:
        with open(os.path.join(iteration_dir, "code.py"), "w") as f:
            f.write(solution)

    prompts = {
        k: v
        for k, v in [
            ("system_prompt", system_prompt),
            ("user_prompt", user_prompt),
            ("llm_response", llm_response),
        ]
        if v
    }
    if prompts:
        with open(os.path.join(iteration_dir, "prompts.json"), "w") as f:
            json.dump(prompts, f, indent=2)

    if iteration == 1:
        labels_path = os.path.join(iteration_dir, "labels.yaml")
        try:
            labels_text = (
                f"iteration: {iteration}\n"
                f"diverge_label_length: {len(diverge_label)}\n"
                f"refine_label_length: {len(refine_label)}\n\n"
                f"diverge_label: |\n"
            )
            for line in diverge_label.splitlines():
                labels_text += f"  {line}\n"
            labels_text += "\nrefine_label: |\n"
            for line in refine_label.splitlines():
                labels_text += f"  {line}\n"
            with open(labels_path, "w") as f:
                f.write(labels_text)
            logger.debug(
                f"Saved labels to {labels_path} ({len(diverge_label)}/{len(refine_label)} chars)"
            )
        except Exception as e:
            logger.warning(f"Failed to save labels.yaml: {e}")


async def update_saved_search_algorithm_score(
    outputs_dir: str,
    iteration: int,
    result: SerializableResult,
    is_new_best: bool,
    db_stats: Dict[str, Any],
) -> None:
    """Update the saved metadata file with the newly assigned score."""
    metadata_path = os.path.join(outputs_dir, f"iteration_{iteration}", "metadata.json")

    if not os.path.exists(metadata_path):
        logger.warning(
            f"Metadata file not found for iteration {iteration}, cannot update search strategy score"
        )
        return

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    child_dict = result.child_program_dict or {}
    metrics = child_dict.get("metrics", {})
    metadata["combined_score"] = metrics.get("combined_score")
    metadata["metrics"] = make_json_serializable(metrics)
    metadata["pending_score"] = False
    metadata["is_new_best"] = bool(is_new_best)

    start_db_stats = child_dict.get("metadata", {}).get("start_db_stats")
    if start_db_stats:
        metadata["start_db_stats"] = make_json_serializable(start_db_stats)

    metadata["end_db_stats"] = make_json_serializable(db_stats)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


async def handle_generation_failure(
    outputs_dir: str,
    active_search_code: str,
    iteration: int,
    result: Optional[SerializableResult],
    solution_iter: int,
    stage: str = "generation",
) -> None:
    """Handle failed search algorithm generation or validation."""
    error_msg = result.error if result else "Unknown error"
    await log_failed_attempt(
        outputs_dir,
        iteration,
        result,
        error_msg if stage == "generation" else "Failed to load/validate",
        stage,
        solution_iter=solution_iter,
    )
    if stage == "generation":
        logger.warning(f"Failed to generate search algorithm: {error_msg}")
    await log_active_algorithm(outputs_dir, active_search_code, iteration)


async def log_failed_attempt(
    outputs_dir: str,
    iteration: int,
    result: Optional[SerializableResult],
    error: str,
    stage: str,
    solution_iter: Optional[int] = None,
) -> None:
    """Log a single failed attempt."""
    iteration_dir = os.path.join(outputs_dir, f"iteration_{iteration}")
    os.makedirs(iteration_dir, exist_ok=True)

    child_dict = (result.child_program_dict or {}) if result else {}
    prompt = (result.prompt or {}) if result else {}
    llm_response = (result.llm_response or "") if result else ""

    failed_file = os.path.join(iteration_dir, "failed_attempts.json")
    failed_attempts: List[Dict[str, Any]] = []
    if os.path.exists(failed_file):
        with open(failed_file, "r") as f:
            failed_attempts = json.load(f).get("failed_attempts", [])

    attempt_number = len(failed_attempts) + 1
    attempt_data: Dict[str, Any] = {
        "attempt_number": attempt_number,
        "solution_iter": solution_iter,
        "error": error,
        "stage": stage,
        "solution": child_dict.get("solution", ""),
        "program_id": child_dict.get("id", "unknown"),
    }

    if llm_response:
        llm_filename = f"failed_attempt_{attempt_number}_llm_response.txt"
        with open(os.path.join(iteration_dir, llm_filename), "w") as f:
            f.write(llm_response)
        attempt_data["llm_response_file"] = llm_filename
        attempt_data["llm_response_preview"] = llm_response[:2000]

    if isinstance(prompt, dict) and (prompt.get("system") or prompt.get("user")):
        prompt_filename = f"failed_attempt_{attempt_number}_prompt.json"
        with open(os.path.join(iteration_dir, prompt_filename), "w") as f:
            json.dump(
                {"system": prompt.get("system", ""), "user": prompt.get("user", "")}, f, indent=2
            )
        attempt_data["prompt_file"] = prompt_filename

    failed_attempts.append(attempt_data)
    with open(failed_file, "w") as f:
        json.dump({"iteration": iteration, "failed_attempts": failed_attempts}, f, indent=2)

    if attempt_data["solution"]:
        code_file = os.path.join(iteration_dir, f"failed_attempt_{attempt_number}.py")
        with open(code_file, "w") as f:
            f.write(attempt_data["solution"])


async def log_active_algorithm(
    outputs_dir: str,
    active_search_code: str,
    iteration: int,
) -> None:
    """Log the active algorithm on the solution side (fallback if generation/validation failed)."""
    active_code = active_search_code or ""

    iteration_dir = os.path.join(outputs_dir, f"iteration_{iteration}")
    os.makedirs(iteration_dir, exist_ok=True)

    candidate_code_path = os.path.join(iteration_dir, "code.py")
    active_code_filename = "active_code.py" if os.path.exists(candidate_code_path) else "code.py"
    with open(os.path.join(iteration_dir, active_code_filename), "w") as f:
        f.write(active_code)

    metadata_path = os.path.join(iteration_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {"iteration": iteration}

    metadata["is_fallback"] = True
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
