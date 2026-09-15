import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add Frontier-CS to path
frontier_cs_path = Path(__file__).resolve().parent / "Frontier-CS" / "src"
if str(frontier_cs_path) not in sys.path:
    sys.path.insert(0, str(frontier_cs_path))

try:
    from frontier_cs.evaluator import FrontierCSEvaluator
    from frontier_cs.runner.base import EvaluationStatus
except ImportError as e:
    logger.error(f"Failed to import Frontier-CS: {e}")
    logger.error(
        "Please ensure Frontier-CS is installed as a submodule in benchmarks/frontier-cs-eval/Frontier-CS"
    )
    sys.exit(1)


class BestProgramEvaluator:
    """Evaluates all best_program.cpp files in the outputs directory."""

    def __init__(
        self, outputs_dir: str, judge_url: str = "http://localhost:8081", num_workers: int = 8
    ):
        """
        Initialize the evaluator.

        Args:
            outputs_dir: Path to the outputs directory containing problem folders
            judge_url: URL of the judge server
            num_workers: Number of parallel workers for evaluation
        """
        self.outputs_dir = Path(outputs_dir)
        self.judge_url = judge_url
        self.num_workers = num_workers

        # Use thread-local storage for evaluator instances (avoid race condition)
        self._evaluator_local = threading.local()

        self.results = []

        # Create results directory in the script's directory
        self.results_dir = Path(__file__).resolve().parent / "evaluation_results"
        self.results_dir.mkdir(exist_ok=True)
        logger.info(f"Results will be saved to {self.results_dir}")
        logger.info(f"Using {self.num_workers} parallel workers with thread-local evaluators")

    def _get_evaluator(self) -> "FrontierCSEvaluator":
        """
        Get the evaluator for the current thread.
        Creates a new instance if this thread hasn't created one yet.
        This avoids race conditions from sharing a single evaluator across threads.
        """
        if not hasattr(self._evaluator_local, "evaluator"):
            self._evaluator_local.evaluator = FrontierCSEvaluator(
                backend="docker",
                judge_url=self.judge_url,
            )
            logger.debug(f"Created new evaluator for thread {threading.current_thread().name}")
        return self._evaluator_local.evaluator

    def find_best_programs(self) -> Dict[str, Path]:
        """
        Find all best_program.cpp files in the outputs directory.

        Returns:
            Dict mapping problem_id to best_program.cpp path
        """
        best_programs = {}

        # Look for frontier_cs subdirectory
        frontier_cs_dir = self.outputs_dir / "frontier_cs"
        if not frontier_cs_dir.exists():
            logger.error(f"frontier_cs directory not found at {frontier_cs_dir}")
            return best_programs

        # Iterate through problem directories
        for problem_dir in sorted(frontier_cs_dir.iterdir()):
            if not problem_dir.is_dir() or not problem_dir.name.startswith("problem_"):
                continue

            # Extract problem ID
            problem_id = problem_dir.name.replace("problem_", "")

            # Look for best_program.cpp
            best_program_path = problem_dir / "best" / "best_program.cpp"
            if best_program_path.exists():
                best_programs[problem_id] = best_program_path
                logger.info(f"Found best_program.cpp for problem {problem_id}")
            else:
                logger.warning(
                    f"best_program.cpp not found for problem {problem_id} at {best_program_path}"
                )

        return best_programs

    def evaluate_program(self, problem_id: str, program_path: Path) -> Dict:
        """
        Evaluate a single best_program.cpp file.

        Args:
            problem_id: The Frontier-CS problem ID
            program_path: Path to the best_program.cpp file

        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating problem {problem_id}: {program_path}")

        try:
            # Read the solution code
            if not program_path.exists():
                error_msg = f"Solution file not found: {program_path}"
                logger.error(error_msg)
                return {
                    "problem_id": problem_id,
                    "program_path": str(program_path),
                    "combined_score": 0.0,
                    "runs_successfully": 0.0,
                    "status": "error",
                    "message": error_msg,
                }

            # Read the code
            code = (
                program_path.read_text()
                .replace("// EVOLVE-BLOCK-START", "")
                .replace("// EVOLVE-BLOCK-END", "")
                .strip()
            )

            logger.info(f"Code extracted from {program_path}, length: {len(code)} characters")

            # Evaluate the solution (use thread-local evaluator)
            evaluator = self._get_evaluator()
            result = evaluator.evaluate(
                track="algorithmic",
                problem_id=problem_id,
                code=code,
                backend="docker",
            )

            logger.info(
                f"Evaluation completed for problem {problem_id} with status: {result.status}"
            )

            # Log the result object and its properties
            logger.info(f"Judger output for problem {problem_id}:")
            logger.info(f"  Status: {result.status}")
            logger.info(f"  Message: {result.message}")
            if hasattr(result, "score"):
                logger.info(f"  Score: {result.score}")
            if hasattr(result, "duration_seconds"):
                logger.info(f"  Duration: {result.duration_seconds}s")
            if hasattr(result, "metadata"):
                logger.info(f"  Metadata: {result.metadata}")
            logger.info(f"  Full result object: {result}")

            # Process result
            if result.status == EvaluationStatus.SUCCESS:
                score = result.score
                logger.info(f"Problem {problem_id}: Score = {score}")

                return {
                    "problem_id": problem_id,
                    "program_path": str(program_path),
                    "combined_score": float(score),
                    "runs_successfully": 1.0,
                    "status": "success",
                    "message": result.message or "Evaluation successful",
                    "duration_seconds": result.duration_seconds,
                    "judger_output": str(result),
                    "metadata": result.metadata if hasattr(result, "metadata") else None,
                }
            elif result.status == EvaluationStatus.TIMEOUT:
                logger.warning(f"Problem {problem_id}: Evaluation timed out")
                return {
                    "problem_id": problem_id,
                    "program_path": str(program_path),
                    "combined_score": 0.0,
                    "runs_successfully": 0.0,
                    "status": "timeout",
                    "message": f"Evaluation timed out: {result.message}",
                    "duration_seconds": result.duration_seconds,
                    "judger_output": str(result),
                }
            elif result.status == EvaluationStatus.COMPILATION_ERROR:
                logger.warning(f"Problem {problem_id}: Compilation error")
                return {
                    "problem_id": problem_id,
                    "program_path": str(program_path),
                    "combined_score": 0.0,
                    "runs_successfully": 0.0,
                    "status": "compilation_error",
                    "message": f"Compilation error: {result.message}",
                    "duration_seconds": result.duration_seconds,
                    "judger_output": str(result),
                }
            else:
                logger.error(f"Problem {problem_id}: Evaluation failed with status {result.status}")
                return {
                    "problem_id": problem_id,
                    "program_path": str(program_path),
                    "combined_score": 0.0,
                    "runs_successfully": 0.0,
                    "status": str(result.status),
                    "message": f"Evaluation failed: {result.message}",
                    "duration_seconds": result.duration_seconds,
                    "judger_output": str(result),
                }

        except Exception as e:
            logger.error(f"Exception while evaluating problem {problem_id}: {str(e)}")
            logger.error(f"Exception traceback: {type(e).__name__}")
            import traceback

            logger.error(traceback.format_exc())

            return {
                "problem_id": problem_id,
                "program_path": str(program_path),
                "combined_score": 0.0,
                "runs_successfully": 0.0,
                "status": "exception",
                "message": str(e),
            }

    def run_all_evaluations(self) -> List[Dict]:
        """
        Run evaluations for all best_program.cpp files sequentially (one at a time).

        Returns:
            List of evaluation results
        """
        logger.info(f"Starting evaluation of all best programs in {self.outputs_dir}")

        best_programs = self.find_best_programs()
        logger.info(f"Found {len(best_programs)} best_program.cpp files")

        if not best_programs:
            logger.warning("No best_program.cpp files found!")
            return []

        # Sort problems by ID for consistent ordering
        sorted_problems = sorted(best_programs.items(), key=lambda x: int(x[0]))

        # Evaluate each program sequentially (no parallelization)
        results = []
        total = len(sorted_problems)
        for idx, (problem_id, program_path) in enumerate(sorted_problems, 1):
            logger.info(f"[SEQ] Evaluating problem {problem_id} ({idx}/{total})")
            try:
                result = self.evaluate_program(problem_id, program_path)

                # CRITICAL: Ensure problem_id matches
                if result.get("problem_id") != problem_id:
                    logger.error(
                        f"[CRITICAL] Problem ID MISMATCH! Expected {problem_id}, got {result.get('problem_id')}"
                    )
                    result["problem_id"] = problem_id  # Force correct problem_id

                results.append(result)
                self.results.append(result)

                logger.info(f"[SAVE] Saving problem {problem_id} result to file")
                # Save result immediately after evaluation
                self.save_problem_result(result)

            except Exception as e:
                logger.error(f"Exception evaluating problem {problem_id}: {str(e)}")
                import traceback

                logger.error(traceback.format_exc())

                error_result = {
                    "problem_id": problem_id,
                    "combined_score": 0.0,
                    "runs_successfully": 0.0,
                    "status": "exception",
                    "message": str(e),
                }
                results.append(error_result)
                self.results.append(error_result)
                self.save_problem_result(error_result)

        return results

    def save_results(self, output_file: str = "evaluation_results.json"):
        """
        Save evaluation results to a JSON file.

        Args:
            output_file: Path to save the results
        """
        output_path = Path(output_file)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    def save_problem_result(self, result: Dict):
        """
        Save individual problem result to a separate file.

        Args:
            result: The evaluation result for a single problem
        """
        problem_id = result.get("problem_id", "unknown")
        result_file = self.results_dir / f"problem_{problem_id}.json"

        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Problem {problem_id} result saved to {result_file}")

    def print_summary(self):
        """Print a summary of the evaluation results."""
        if not self.results:
            logger.info("No results to summarize")
            return

        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 80)

        successful = [r for r in self.results if r.get("status") == "success"]
        timeout = [r for r in self.results if r.get("status") == "timeout"]
        compilation_error = [r for r in self.results if r.get("status") == "compilation_error"]
        other_error = [
            r
            for r in self.results
            if r.get("status") not in ["success", "timeout", "compilation_error"]
        ]

        logger.info(f"Total problems evaluated: {len(self.results)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Timeouts: {len(timeout)}")
        logger.info(f"Compilation errors: {len(compilation_error)}")
        logger.info(f"Other errors: {len(other_error)}")

        if successful:
            scores = [r["combined_score"] for r in successful]
            logger.info(f"\nSuccessful evaluation scores:")
            logger.info(f"  Average score: {sum(scores) / len(scores):.2f}")
            logger.info(f"  Min score: {min(scores):.2f}")
            logger.info(f"  Max score: {max(scores):.2f}")

            logger.info(f"\nTop 5 problems by score:")
            top_5 = sorted(successful, key=lambda r: r["combined_score"], reverse=True)[:5]
            for i, result in enumerate(top_5, 1):
                logger.info(
                    f"  {i}. Problem {result['problem_id']}: {result['combined_score']:.2f}"
                )

        logger.info("=" * 80 + "\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate all best_program.cpp files in the outputs directory"
    )

    # Default outputs directory is two levels up from this script
    default_outputs_dir = Path(__file__).resolve().parent.parent.parent / "outputs"

    parser.add_argument(
        "--outputs-dir",
        type=str,
        default=str(default_outputs_dir),
        help="Path to the outputs directory (default: ../../outputs from script location)",
    )
    parser.add_argument(
        "--judge-url",
        type=str,
        default="http://localhost:8081",
        help="URL of the judge server (default: http://localhost:8081)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="evaluation_results.json",
        help="Path to save the evaluation results (default: evaluation_results.json)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers for evaluation (default: 8)",
    )

    args = parser.parse_args()

    # Run evaluations
    evaluator = BestProgramEvaluator(
        outputs_dir=args.outputs_dir, judge_url=args.judge_url, num_workers=args.workers
    )

    results = evaluator.run_all_evaluations()
    evaluator.save_results(args.output_file)
    evaluator.print_summary()

    logger.info(f"Evaluation complete. Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
