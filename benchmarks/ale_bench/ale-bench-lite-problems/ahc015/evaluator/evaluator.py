import traceback
from pathlib import Path
from ale_bench.result import CaseResult, JudgeResult, Result
from ale_bench_eval.safe_ale_session import start_ale_bench_session
import logging
import sys
logger = logging.getLogger(__name__ + "_" + "ALE_BENCH_EVALUATOR")

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

def evaluate(program_path):
    problem_id = "ahc015"
    logger.info(f"Evaluating program {program_path} for problem {problem_id} in ale bench evaluator")
    try:
        session = None
        logger.info("Starting ALE-Bench session")
        session = start_ale_bench_session(
            problem_id=problem_id,
            lite_version=True,
            num_workers=13,
        )
        logger.info("ALE-Bench session started")
        if not session:
            raise RuntimeError("Failed to start or restart the session.")
        optim_factor = 1 if session.problem.metadata.score_type == "maximize" else -1
        code = Path(program_path).read_text().replace("# EVOLVE-BLOCK-START", "").replace("# EVOLVE-BLOCK-END", "").strip()
        logger.info("Code extracted")
        num_public_cases = 50
        cases = session.case_gen(list(range(num_public_cases)))
        public_result = session.case_eval(
            cases, code, code_language="cpp20", skip_local_visualization=True
        )
        logger.info("Public evaluation completed")
        extracted_case = result_feedback(public_result)
        logger.info("Result feedback completed")
        logger.info("ALE-Bench session closed")
        combined_score = public_result.overall_absolute_score * optim_factor / num_public_cases
        if public_result.overall_judge_result != JudgeResult.ACCEPTED and optim_factor == -1:
            combined_score = -sys.maxsize - 1
        session.close()
        return {
            "judge_result": public_result.overall_judge_result.value,
            "overall_score": public_result.overall_absolute_score,
            "max_execution_time_sec": max([case_result.execution_time for case_result in public_result.case_results]),
            "max_memory_usage_mib": max([case_result.memory_usage for case_result in public_result.case_results]) // 1024 // 1024,
            "standard_error": extracted_case.error_str,
            "message": extracted_case.message,
            "combined_score": combined_score,
        }
    except Exception as e:
        logger.error(f"Evaluation failed completely: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "overall_score": 0.0,
            "error": str(e),
        }
if __name__ == "__main__":
    from wrapper import run
    run(evaluate)
