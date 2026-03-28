"""
Evaluator for Frontier-CS algorithmic problems (containerized version).

This evaluator integrates with SkyDiscover to evaluate generated C++ solutions
against Frontier-CS benchmark problems using the local judge server.
"""

import traceback
from pathlib import Path
import logging
import os
import random

logger = logging.getLogger(__name__)

# Support multiple judge servers for load balancing
DEFAULT_JUDGE_URL = "http://localhost:8081"
JUDGE_URLS = os.environ.get("JUDGE_URLS", DEFAULT_JUDGE_URL).split(",")
JUDGE_URLS = [url.strip() for url in JUDGE_URLS if url.strip()]

def get_judge_url() -> str:
    """Get a judge URL using random selection for load balancing."""
    return random.choice(JUDGE_URLS)

try:
    from frontier_cs.single_evaluator import SingleEvaluator as FrontierCSEvaluator
    from frontier_cs.runner.base import EvaluationStatus
except ImportError as e:
    logger.error(f"Failed to import Frontier-CS: {e}")
    logger.error("Please ensure Frontier-CS is installed in the container")
    raise

def evaluate(program_path: str, problem_id: str = None, **kwargs) -> dict:
    """
    Evaluate a C++ solution for a Frontier-CS algorithmic problem.

    Args:
        program_path: Path to the C++ solution file
        problem_id: Frontier-CS problem ID (e.g., "0", "1", "2", etc.)
                    If None, will be read from FRONTIER_CS_PROBLEM env var or config

    Returns:
        dict with evaluation results:
            - combined_score: The score from the judge (higher is better)
            - runs_successfully: 1.0 if evaluation succeeded, 0.0 otherwise
            - status: Evaluation status string
            - message: Any error or status messages
            - problem_id: The problem ID
            - program_path: Path to the evaluated program
            - score_unbounded: Unbounded score if available
            - metadata: Additional evaluation metadata
    """
    # Get problem_id from parameter, environment, or kwargs
    if problem_id is None:
        problem_id = os.environ.get('FRONTIER_CS_PROBLEM')
        if problem_id is None:
            problem_id = kwargs.get('frontier_cs_problem', '0')

    logger.info(f"Evaluating program {program_path} for Frontier-CS problem {problem_id}")

    try:
        # Initialize evaluator with judge server (load balanced if multiple configured)
        judge_url = get_judge_url()
        logger.info(f"Using judge server: {judge_url}")
        evaluator = FrontierCSEvaluator(
            backend="docker",
            judge_url=judge_url,
            register_cleanup=False,
        )

        # Read the solution code
        solution_path = Path(program_path)
        if not solution_path.exists():
            error_msg = f"Solution file not found: {program_path}"
            logger.error(error_msg)
            return {
                "combined_score": 0.0,
                "runs_successfully": 0.0,
                "status": "error",
                "message": error_msg,
                "problem_id": problem_id,
                "program_path": program_path,
            }

        # Extract code and remove any EVOLVE-BLOCK markers
        code = solution_path.read_text().replace(
            "// EVOLVE-BLOCK-START", ""
        ).replace(
            "// EVOLVE-BLOCK-END", ""
        ).strip()

        logger.info(f"Code extracted from {program_path}")

        # Evaluate the solution
        result = evaluator.evaluate(
            track="algorithmic",
            problem_id=problem_id,
            code=code,
            backend="docker",
        )

        logger.info(f"Evaluation completed with status: {result.status}")

        # Process result
        if result.status == EvaluationStatus.SUCCESS:
            print(result)
            score = result.score
            # Use unbounded score for optimization (allows >100 if beating reference)
            score_unbounded = result.metadata.get('scoreUnbounded', score) if result.metadata else score
            print(f"score={score}, score_unbounded={score_unbounded}")

            # Extract only essential metadata (exclude large test case outputs)
            essential_metadata = {}
            if result.metadata:
                essential_metadata = {
                    "status": result.metadata.get("status"),
                    "passed": result.metadata.get("passed"),
                    "result": result.metadata.get("result"),
                    "score": result.metadata.get("score"),
                    "scoreUnbounded": result.metadata.get("scoreUnbounded"),
                }

            return {
                "combined_score": float(score),  # Ensure it's a float
                "score_unbounded": score_unbounded,
                "runs_successfully": 1.0,
                "status": "success",
                "message": result.message or "Evaluation successful",
                "problem_id": problem_id,
                "program_path": program_path,
                "duration_seconds": result.duration_seconds,
                "metadata": essential_metadata,
            }
        elif result.status == EvaluationStatus.TIMEOUT:
            logger.warning(f"Evaluation timed out: {result.message}")
            return {
                "combined_score": 0.0,
                "runs_successfully": 0.0,
                "status": "timeout",
                "message": result.message or "Evaluation timed out",
                "problem_id": problem_id,
                "program_path": program_path,
            }
        else:  # ERROR status
            logger.error(f"Evaluation error: {result.message}")
            return {
                "combined_score": 0.0,
                "runs_successfully": 0.0,
                "status": "error",
                "message": result.message or "Evaluation failed",
                "problem_id": problem_id,
                "program_path": program_path,
                "logs": result.logs,
            }

    except Exception as e:
        logger.error(f"Evaluation failed completely: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "combined_score": 0.0,
            "runs_successfully": 0.0,
            "status": "error",
            "message": str(e),
            "problem_id": problem_id,
            "program_path": program_path,
            "error": str(e),
        }


if __name__ == "__main__":
    from wrapper import run
    run(evaluate)
