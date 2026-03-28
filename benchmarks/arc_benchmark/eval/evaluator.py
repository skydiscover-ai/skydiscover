import numpy as np
from typing import List, Tuple, Dict, Any
import json
import os

try:
    from skydiscover.evaluation.evaluation_result import EvaluationResult
except ImportError:
    from dataclasses import dataclass, field
    from typing import Union

    @dataclass
    class EvaluationResult:
        metrics: Dict[str, float]
        artifacts: Dict[str, Union[str, bytes]] = field(default_factory=dict)
import importlib.util

TASK_FILE = os.getenv("ARC_TASK_FILE", "training")
TASK_NUM = os.getenv("TASK_NUM", 0)
DATA_ROOT = os.getenv("DATA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
INCLUDE_TEST = os.getenv("ARC_EVAL_INCLUDE_TEST", "0").lower() in ("1", "true", "yes")
USE_TEST_IN_SCORE = os.getenv("ARC_EVAL_USE_TEST_FOR_SCORE", "0").lower() in ("1", "true", "yes")


def cell_accuracy_single(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute continuous cell-level accuracy between prediction and ground truth.
    Returns a float in [0, 1]. Handles shape mismatches gracefully.
    """
    if pred.shape != gt.shape:
        # Partial credit for getting shape partially right
        shape_score = 0.0
        if len(pred.shape) == len(gt.shape) == 2:
            row_match = 1.0 if pred.shape[0] == gt.shape[0] else 0.0
            col_match = 1.0 if pred.shape[1] == gt.shape[1] else 0.0
            shape_score = (row_match + col_match) * 0.1  # up to 0.2 for correct dimensions
        return shape_score
    # Cell-level accuracy
    total_cells = gt.size
    if total_cells == 0:
        return 1.0
    correct_cells = int(np.sum(pred == gt))
    return correct_cells / total_cells


def best_attempt_cell_accuracy(attempts: List[np.ndarray], gt: np.ndarray) -> float:
    """Return the best cell accuracy across all attempts for one example."""
    return max(cell_accuracy_single(a, gt) for a in attempts)


def pass_at_2_accuracy_single(
    attempts: List[np.ndarray],
    gt: np.ndarray
) -> Tuple[int, Dict[int, Any]]:
    """
    Compute pass@2 accuracy for a single ARC test case.

    Args:
        attempts: List of 2 numpy arrays representing model attempts.
        gt: Ground-truth output as a 2D numpy array.

    Returns:
        pass_at_2: int (1 if any attempt is perfectly correct, else 0)
        diagnostics: dict mapping attempt index -> diagnostic info.
                     If sizes match, includes indices of incorrect cells.
    """
    assert len(attempts) == 2, "Expected exactly 2 attempts for pass@2 evaluation."

    diagnostics = {}
    passed = False

    for i, pred in enumerate(attempts):
        attempt_info = {}

        # Size check
        if pred.shape != gt.shape:
            attempt_info["size_match"] = False
            attempt_info["pred_shape"] = list(pred.shape)
            attempt_info["gt_shape"] = list(gt.shape)
            attempt_info["incorrect_indices"] = None
            attempt_info["cell_accuracy"] = 0.0
            attempt_passed = False
        else:
            attempt_info["size_match"] = True

            # Find incorrect cells
            incorrect_mask = pred != gt
            incorrect_indices = np.argwhere(incorrect_mask)

            attempt_info["incorrect_indices"] = incorrect_indices.tolist()
            attempt_info["num_incorrect"] = int(incorrect_mask.sum())
            attempt_info["num_total"] = int(gt.size)
            attempt_info["cell_accuracy"] = float(np.sum(~incorrect_mask)) / gt.size

            # Perfect match
            if incorrect_mask.sum() == 0:
                attempt_passed = True
            else:
                attempt_passed = False

        attempt_info["perfect_match"] = attempt_passed
        passed = attempt_passed or passed

        diagnostics[i] = attempt_info

    pass_at_2 = 1 if passed else 0

    return pass_at_2, diagnostics

def pass_at_2_accuracy_multi_test(
    all_attempts: List[List[np.ndarray]],
    all_gt: List[np.ndarray]
) -> Tuple[List[int], List[Dict[int, Any]]]:
    """
    Compute pass@2 accuracy across multiple ARC test cases.

    Args:
        all_attempts: List of lists of 2 numpy arrays for each test case.
        all_gt: List of ground-truth outputs as 2D numpy arrays.
    """
    assert len(all_attempts) == len(all_gt), "Mismatched number of test cases."

    all_diagnostics = []
    all_pass = []

    for attempts, gt in zip(all_attempts, all_gt):
        pass_at_2, diagnostics = pass_at_2_accuracy_single(attempts, gt)
        all_pass.append(pass_at_2)
        all_diagnostics.append(diagnostics)

    return all_pass, all_diagnostics

def extract_failure_artifacts(diagnostics, pred=None, gt=None):
    """
    Extract failure artifacts from diagnostics for a given example.
    Includes actual vs expected output snippets for better LLM feedback.
    """
    artifacts = {}
    if not diagnostics["size_match"]:
        artifacts["error_type"] = "SizeMismatch"
        artifacts["error_message"] = (
            f"Output shape {diagnostics['pred_shape']} does not match "
            f"expected shape {diagnostics['gt_shape']}."
        )
        artifacts["suggestion"] = (
            f"Your output has shape {diagnostics['pred_shape']} but the correct output "
            f"has shape {diagnostics['gt_shape']}. Review how you determine output dimensions."
        )
    else:
        num_incorrect = diagnostics['num_incorrect']
        num_total = diagnostics['num_total']
        accuracy = diagnostics['cell_accuracy']
        artifacts["error_type"] = "IncorrectCells"
        artifacts["error_message"] = (
            f"{num_incorrect}/{num_total} cells incorrect "
            f"(cell accuracy: {accuracy:.1%})."
        )
        # Show a compact diff of expected vs actual for first few wrong cells
        if diagnostics['incorrect_indices'] and pred is not None and gt is not None:
            wrong = diagnostics['incorrect_indices'][:8]  # first 8 wrong cells
            diff_lines = []
            for r, c in wrong:
                diff_lines.append(f"  [{r},{c}]: got {int(pred[r,c])}, expected {int(gt[r,c])}")
            artifacts["cell_diffs"] = "\n".join(diff_lines)
            if len(diagnostics['incorrect_indices']) > 8:
                artifacts["cell_diffs"] += f"\n  ... and {len(diagnostics['incorrect_indices'])-8} more"
        artifacts["suggestion"] = (
            f"Your solution gets {accuracy:.1%} of cells correct. "
            f"Review the transformation logic for the failing cells."
        )

    return artifacts

def evaluate(program_path):
    """
    Evaluate the program on ARC task training (and optionally test) examples.

    Returns a combined_score that blends:
      - pass@2 (binary perfect-match, weighted 0.6)
      - cell accuracy (continuous partial credit, weighted 0.4)
    This gives evolution gradient signal even when no example is solved perfectly.
    """
    spec = importlib.util.spec_from_file_location("program_module", program_path)
    program_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program_module)

    if not hasattr(program_module, 'transform_grid_attempt_1') or not hasattr(program_module, 'transform_grid_attempt_2'):
        print(f"Stage 1 validation failed: Program must define 'transform_grid_attempt_1' and 'transform_grid_attempt_2' functions.")

        error_artifacts = {
                "error_type": "MissingFunction",
                "error_message": "Stage 1: Program is missing required 'transform_grid_attempt_1' and 'transform_grid_attempt_2' functions.",
                "suggestion": "Make sure your program includes a functions named 'transform_grid_attempt_1' and 'transform_grid_attempt_2' that take as an argument a 2D numpy array and return a 2D numpy array."
            }

        return EvaluationResult(
                metrics={
                    "runs_successfully": 0.0,
                    "combined_score": 0.0,
                    "error": "Missing transform_grid_attempt_1 and transform_grid_attempt_2 functions"
                },
                artifacts=error_artifacts
            )

    # Load ARC tasks
    challenge_path = os.path.join(DATA_ROOT, f"arc-agi_{TASK_FILE}_challenges.json")

    with open(challenge_path, 'r') as f:
        tasks = json.load(f)

    task_id = list(tasks.keys())[int(TASK_NUM)]
    task = tasks[task_id]

    train_inputs = [np.array(inp["input"]) for inp in task['train']]
    train_gts = [np.array(gt["output"]) for gt in task['train']]

    train_attempts = []

    # Generate attempts for training data
    for inp in train_inputs:
        attempt_1 = program_module.transform_grid_attempt_1(inp)
        if not isinstance(attempt_1, np.ndarray):
            print(f"transform_grid_attempt_1 did not return a numpy array")

            error_artifacts = {
                "error_type": "InvalidReturnType",
                "error_message": "Stage 1: transform_grid_attempt_1 did not return a numpy array.",
                "suggestion": "Make sure your transform_grid_attempt_1 function returns a 2D numpy array."
            }

            return EvaluationResult(
                metrics={
                    "runs_successfully": 0.0,
                    "combined_score": 0.0,
                    "error": "transform_grid_attempt_1 did not return a numpy array"
                },
                artifacts=error_artifacts
            )

        attempt_2 = program_module.transform_grid_attempt_2(inp)
        if not isinstance(attempt_2, np.ndarray):
            print(f"transform_grid_attempt_2 did not return a numpy array")

            error_artifacts = {
                "error_type": "InvalidReturnType",
                "error_message": "Stage 1: transform_grid_attempt_2 did not return a numpy array.",
                "suggestion": "Make sure your transform_grid_attempt_2 function returns a 2D numpy array."
            }

            return EvaluationResult(
                metrics={
                    "runs_successfully": 0.0,
                    "combined_score": 0.0,
                    "error": "transform_grid_attempt_2 did not return a numpy array"
                },
                artifacts=error_artifacts
            )
        train_attempts.append([attempt_1, attempt_2])

    pass_at_2_train, train_diagnostics_list = pass_at_2_accuracy_multi_test(train_attempts, train_gts)

    # Compute both binary pass@2 and continuous cell accuracy
    train_pass_score = sum(pass_at_2_train) / len(pass_at_2_train)
    train_cell_acc = sum(
        best_attempt_cell_accuracy(attempts, gt)
        for attempts, gt in zip(train_attempts, train_gts)
    ) / len(train_gts)

    # Blended score: pass@2 (60%) + cell accuracy (40%) gives gradient signal
    train_score = 0.6 * train_pass_score + 0.4 * train_cell_acc

    metrics = {
        "runs_successfully": 1.0,
        "combined_score": train_score,
        "train_combined_score": train_score,
        "train_pass_at_2_score": train_pass_score,
        "train_cell_accuracy": round(train_cell_acc, 4),
    }
    error_artifacts = {}
    for i, (train_pass, train_diagnostics) in enumerate(zip(pass_at_2_train, train_diagnostics_list)):
        example_name = f"train_example_{i}"
        metrics[f"{example_name}_pass_at_2"] = train_pass
        best_acc = best_attempt_cell_accuracy(train_attempts[i], train_gts[i])
        metrics[f"{example_name}_cell_accuracy"] = round(best_acc, 4)
        for attempt in train_diagnostics:
            attempt_pass = train_diagnostics[attempt]["perfect_match"]
            metrics[f"{example_name}_attempt_{attempt}"] = attempt_pass
            if not attempt_pass:
                pred = train_attempts[i][attempt]
                gt = train_gts[i]
                error_artifacts[f"{example_name}_attempt_{attempt}_diagnostics"] = extract_failure_artifacts(
                    train_diagnostics[attempt], pred=pred, gt=gt
                )

    # Optional: include test feedback (uses solutions if available)
    if INCLUDE_TEST:
        solution_path = os.path.join(DATA_ROOT, f"arc-agi_{TASK_FILE}_solutions.json")
        if os.path.isfile(solution_path):
            with open(solution_path, 'r') as f:
                solutions = json.load(f)
            task_id = list(tasks.keys())[int(TASK_NUM)]
            solution = solutions.get(task_id)
            if solution is not None and "test" in task:
                if len(task["test"]) != len(solution):
                    raise ValueError(
                        f"Train/test data mismatch: task {task_id} has {len(task['test'])} test inputs "
                        f"but {len(solution)} solution outputs. Check that arc-agi_{TASK_FILE}_challenges.json "
                        f"and arc-agi_{TASK_FILE}_solutions.json were generated together."
                    )
                test_inputs = [np.array(inp["input"]) for inp in task['test']]
                test_gts = [np.array(gt) for gt in solution]

                test_attempts = []
                for inp in test_inputs:
                    attempt_1 = program_module.transform_grid_attempt_1(inp)
                    if not isinstance(attempt_1, np.ndarray):
                        print(f"transform_grid_attempt_1 did not return a numpy array (test)")
                        return EvaluationResult(
                            metrics={
                                "runs_successfully": 0.0,
                                "combined_score": 0.0,
                                "error": "transform_grid_attempt_1 did not return a numpy array (test)"
                            },
                            artifacts={
                                "error_type": "InvalidReturnType",
                                "error_message": "Stage 1: transform_grid_attempt_1 did not return a numpy array (test).",
                                "suggestion": "Make sure transform_grid_attempt_1 returns a 2D numpy array."
                            }
                        )

                    attempt_2 = program_module.transform_grid_attempt_2(inp)
                    if not isinstance(attempt_2, np.ndarray):
                        print(f"transform_grid_attempt_2 did not return a numpy array (test)")
                        return EvaluationResult(
                            metrics={
                                "runs_successfully": 0.0,
                                "combined_score": 0.0,
                                "error": "transform_grid_attempt_2 did not return a numpy array (test)"
                            },
                            artifacts={
                                "error_type": "InvalidReturnType",
                                "error_message": "Stage 1: transform_grid_attempt_2 did not return a numpy array (test).",
                                "suggestion": "Make sure transform_grid_attempt_2 returns a 2D numpy array."
                            }
                        )
                    test_attempts.append([attempt_1, attempt_2])

                pass_at_2_test, test_diagnostics_list = pass_at_2_accuracy_multi_test(test_attempts, test_gts)
                test_pass_score = sum(pass_at_2_test) / len(pass_at_2_test)
                test_cell_acc = sum(
                    best_attempt_cell_accuracy(attempts, gt)
                    for attempts, gt in zip(test_attempts, test_gts)
                ) / len(test_gts)
                test_score = 0.6 * test_pass_score + 0.4 * test_cell_acc

                metrics["test_combined_score"] = test_score
                metrics["test_pass_at_2_score"] = test_pass_score
                metrics["test_cell_accuracy"] = round(test_cell_acc, 4)
                metrics["test_included"] = 1

                for i, (test_pass, test_diagnostics) in enumerate(zip(pass_at_2_test, test_diagnostics_list)):
                    example_name = f"test_example_{i}"
                    metrics[f"{example_name}_pass_at_2"] = test_pass
                    best_acc = best_attempt_cell_accuracy(test_attempts[i], test_gts[i])
                    metrics[f"{example_name}_cell_accuracy"] = round(best_acc, 4)
                    for attempt in test_diagnostics:
                        metrics[f"{example_name}_attempt_{attempt}"] = test_diagnostics[attempt]["perfect_match"]
                    if test_pass == 0:
                        first_failing_idx = next(
                            (a for a in test_diagnostics if not test_diagnostics[a]["perfect_match"]),
                            0,
                        )
                        pred = test_attempts[i][first_failing_idx]
                        gt = test_gts[i]
                        error_artifacts[f"{example_name}"] = extract_failure_artifacts(
                            test_diagnostics[first_failing_idx], pred=pred, gt=gt
                        )

                if USE_TEST_IN_SCORE:
                    metrics["combined_score"] = (train_score + test_score) / 2.0
            else:
                metrics["test_included"] = 0
        else:
            metrics["test_included"] = 0

    return EvaluationResult(
        metrics=metrics,
        artifacts=error_artifacts
    )


def _evaluate_as_dict(program_path):
    """Adapter: calls evaluate() and converts EvaluationResult to a plain dict."""
    result = evaluate(program_path)
    d = dict(result.metrics)
    for k, v in result.artifacts.items():
        d[k] = v
    return d


if __name__ == "__main__":
    # Backwards-compat: bridges old evaluate() -> EvaluationResult to the
    # container JSON protocol.  wrapper.py is copied from
    # skydiscover/evaluation/wrapper.py.
    from wrapper import run

    run(_evaluate_as_dict)
