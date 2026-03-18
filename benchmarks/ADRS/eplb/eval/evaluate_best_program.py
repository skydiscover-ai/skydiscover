#!/usr/bin/env python3
"""
Evaluate a best_program.py file using the eplb evaluator.
Runs multiple times and averages the results.
"""
import sys
import json
from pathlib import Path
from evaluator import evaluate

def main():
    if len(sys.argv) < 2:
        print("Usage: evaluate_best_program.py <path_to_best_program.py> [num_runs]")
        sys.exit(1)
    
    program_path = Path(sys.argv[1])
    if not program_path.exists():
        print(f"Error: File not found: {program_path}")
        sys.exit(1)
    
    num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    print(f"Evaluating: {program_path}")
    print(f"Running {num_runs} times and averaging results...")
    print("=" * 60)
    
    results = []
    for run in range(1, num_runs + 1):
        print(f"\n--- Run {run}/{num_runs} ---")
        result = evaluate(str(program_path))
        
        if "error" in result:
            print(f"❌ Error in run {run}: {result['error']}")
            sys.exit(1)
        
        results.append(result)
        print(f"Run {run} - Combined Score: {result.get('combined_score', 0.0):.6f}")
    
    # Compute averages
    avg_result = {
        "balancedness_score_gpu": sum(r.get("balancedness_score_gpu", 0.0) for r in results) / len(results),
        "balancedness_score_expert": sum(r.get("balancedness_score_expert", 0.0) for r in results) / len(results),
        "times_algorithm": sum(r.get("times_algorithm", 0.0) for r in results) / len(results),
        "times_inference": sum(r.get("times_inference", 0.0) for r in results) / len(results),
        "speed_score": sum(r.get("speed_score", 0.0) for r in results) / len(results),
        "combined_score": sum(r.get("combined_score", 0.0) for r in results) / len(results),
    }
    
    print("\n" + "=" * 60)
    print("AVERAGED RESULTS (over {} runs):".format(num_runs))
    print("=" * 60)
    print(json.dumps(avg_result, indent=2))
    
    print("\n" + "-" * 60)
    print("Summary:")
    print(f"✅ Combined Score: {avg_result['combined_score']:.6f}")
    print(f"   Balancedness (GPU): {avg_result['balancedness_score_gpu']:.6f}")
    print(f"   Balancedness (Expert): {avg_result['balancedness_score_expert']:.6f}")
    print(f"   Speed Score: {avg_result['speed_score']:.6f}")
    print(f"   Avg Algorithm Time: {avg_result['times_algorithm']:.6f}s")
    print(f"   Avg Inference Time: {avg_result['times_inference']:.6f}s")
    print("-" * 60)

if __name__ == "__main__":
    main()

