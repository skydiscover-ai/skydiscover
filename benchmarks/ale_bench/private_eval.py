import argparse
import json
import os
import traceback
from pathlib import Path

import ale_bench
from ale_bench.result import CaseResult, JudgeResult, Result


def result_feedback(result: Result) -> CaseResult:
    if result.overall_judge_result == JudgeResult.ACCEPTED:
        return result.case_results[0]
    else:
        selected_case_idx = 0
        for idx, case_result in enumerate(result.case_results):
            if case_result.judge_result == result.overall_judge_result:
                selected_case_idx = idx
                break
        return result.case_results[selected_case_idx]


def main(program_path: str, problem_id: str) -> dict:
    """Runs the evaluation using the shinka.eval utility."""
    print(f"Problem ID: {problem_id}")
    print(f"Evaluating program: {program_path}")

    try:

        session = ale_bench.start(
            problem_id=problem_id,
            lite_version=False,
            num_workers=13,
        )

        code = (
            Path(program_path)
            .read_text()
            .replace("# EVOLVE-BLOCK-START", "")
            .replace("# EVOLVE-BLOCK-END", "")
            .strip()
        )

        private_result, final_rank, final_performance = session.private_eval(
            code,
            code_language="cpp20",
        )
        # Store the private_result as JSON in the results directory

        private_json_str = private_result.model_dump_json(indent=4)
        private_json = json.loads(private_json_str)

        private_passed_cases, private_failed_cases = 0, 0
        num_private_cases = len(private_json["case_results"])
        for case in private_json["case_results"]:
            if case["judge_result"] == "ACCEPTED":
                private_passed_cases += 1
            else:
                private_failed_cases += 1
        print(
            f"Passed {private_passed_cases} cases, failed {private_failed_cases} cases out of {num_private_cases}"
        )

        print(
            f"Final Private Score: {private_result.overall_absolute_score} - Mean Score: {private_result.overall_absolute_score / num_private_cases}"
        )
        print(f"Rank: {final_rank}, Performance: {final_performance}")
        metrics = {}
        private_metrics = {
            "private_rank": final_rank,
            "private_performance": final_performance,
            "private_score": private_result.overall_absolute_score,
            "num_private_passed_cases": private_passed_cases,
            "num_private_failed_cases": private_failed_cases,
        }
        metrics["private"] = private_metrics

        # Monitor resource consumption
        print(f"Current Resource Usage: {session.current_resource_usage}")
        print(f"Remaining Resources: {session.remaining_resource_usage}")

        return metrics
    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        print(traceback.format_exc())
        metrics = {
            "combined_score": 0.0,
            "public": {"judge_result": "REJECTED"},
            "private": {
                "private_rank": 0,
                "private_performance": 0,
                "private_score": 0,
                "num_private_passed_cases": 0,
                "num_private_failed_cases": 0,
            },
        }
        return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent evaluation script using shinka.eval")
    parser.add_argument(
        "--program-path",
        type=str,
        default="program.cpp",
        help="Path to the program to evaluate",
    )

    parser.add_argument(
        "--problem-id",
        type=str,
        default="ahc025",
        help="Problem ID",
    )
    parsed_args = parser.parse_args()

    # Collect results from 3 runs
    all_results = []
    for i in range(3):
        print(f"\n{'='*60}")
        print(f"Running evaluation {i+1} of 3")
        print("=" * 60)
        result = main(
            parsed_args.program_path,
            parsed_args.problem_id,
        )
        all_results.append(result)
        print("=" * 60)

    # Compute averages
    print(f"\n{'='*60}")
    print("FINAL AVERAGED RESULTS ACROSS 3 RUNS")
    print("=" * 60)

    private_scores = [r["private"]["private_score"] for r in all_results]
    private_performances = [r["private"]["private_performance"] for r in all_results]
    private_ranks = [r["private"]["private_rank"] for r in all_results]
    passed_cases = [r["private"]["num_private_passed_cases"] for r in all_results]
    failed_cases = [r["private"]["num_private_failed_cases"] for r in all_results]

    avg_private_score = sum(private_scores) / len(private_scores)
    avg_private_performance = sum(private_performances) / len(private_performances)
    avg_private_rank = sum(private_ranks) / len(private_ranks)
    avg_passed_cases = sum(passed_cases) / len(passed_cases)
    avg_failed_cases = sum(failed_cases) / len(failed_cases)

    print(f"\nAverage Private Score: {avg_private_score:.2f}")
    print(f"  Individual scores: {private_scores}")
    print(f"\nAverage Private Performance: {avg_private_performance:.4f}")
    print(f"  Individual performances: {private_performances}")
    print(f"\nAverage Private Rank: {avg_private_rank:.2f}")
    print(f"  Individual ranks: {private_ranks}")
    print(f"\nAverage Passed Cases: {avg_passed_cases:.2f}")
    print(f"Average Failed Cases: {avg_failed_cases:.2f}")
    print("=" * 60)

    # Return summary
    summary = {
        "avg_private_score": avg_private_score,
        "avg_private_performance": avg_private_performance,
        "avg_private_rank": avg_private_rank,
        "all_results": all_results,
    }

    print(f"\nFinal Summary:")
    print(json.dumps(summary, indent=2))
