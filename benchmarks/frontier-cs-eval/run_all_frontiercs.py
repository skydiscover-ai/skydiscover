import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent

frontier_cs_path = SCRIPT_DIR / "Frontier-CS" / "src"
if str(frontier_cs_path) not in sys.path:
    sys.path.insert(0, str(frontier_cs_path))

from frontier_cs.runner.algorithmic_local import AlgorithmicLocalRunner


def run_single_problem(args):
    p_id, search, iterations, env = args
    print(f"\n[START] Problem ID: {p_id}")
    command = [
        "uv",
        "run",
        "skydiscover",
        "optimize",
        "initial_program.cpp",
        "evaluator.py",
        "-c",
        "config.yaml",
        "-s",
        search,
        "-i",
        str(iterations),
        "-o",
        f"outputs/frontier_cs/problem_{p_id}",
    ]
    env = {**env, "FRONTIER_CS_PROBLEM": str(p_id)}
    try:
        subprocess.run(command, check=True, env=env, cwd=str(SCRIPT_DIR))
        return f"✅ Problem {p_id} completed."
    except subprocess.CalledProcessError as e:
        return f"❌ Problem {p_id} failed: {e}"


def main():
    parser = argparse.ArgumentParser(description="Run SkyDiscover on all Frontier-CS problems")
    parser.add_argument(
        "--search", "-s", default="adaevolve", help="Search algorithm (default: adaevolve)"
    )
    parser.add_argument(
        "--iterations", "-i", type=int, default=50, help="Iterations per problem (default: 50)"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=6, help="Parallel workers (default: 6)"
    )
    args = parser.parse_args()

    runner = AlgorithmicLocalRunner()
    problems_data = runner.list_problems()
    problem_ids = sorted([p["id"] for p in problems_data["problems"]], key=int)

    print(
        f"Running {len(problem_ids)} problems with {args.workers} workers "
        f"(search={args.search}, iterations={args.iterations})..."
    )

    env = os.environ.copy()
    task_args = [(p_id, args.search, args.iterations, env) for p_id in problem_ids]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(run_single_problem, task_args))

    print("\n" + "=" * 30)
    print("ALL RUNS COMPLETE")
    print("=" * 30)
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
