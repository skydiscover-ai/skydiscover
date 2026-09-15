import csv
import json
import os
from pathlib import Path

# Define paths
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent.parent
training_dir = str(_repo_root / "outputs" / "frontier_cs")
testing_dir = str(_script_dir / "evaluation_results")
output_csv = str(_script_dir / "combined_results.csv")

# Collect all problems
results = []

# Get all problem directories from training data
training_problems = sorted([d for d in os.listdir(training_dir) if d.startswith("problem_")])

print(f"Found {len(training_problems)} training problems")

for problem_dir in training_problems:
    problem_id = problem_dir.replace("problem_", "")

    # Get training score from best_program_info.json
    training_score = None
    training_info_path = os.path.join(training_dir, problem_dir, "best", "best_program_info.json")

    if os.path.exists(training_info_path):
        try:
            with open(training_info_path, "r") as f:
                training_data = json.load(f)
                training_score = training_data.get("metrics", {}).get("combined_score")
        except Exception as e:
            print(f"Error reading training data for problem {problem_id}: {e}")

    # Get testing score from evaluation_results json
    testing_score = None
    testing_json_path = os.path.join(testing_dir, f"problem_{problem_id}.json")

    if os.path.exists(testing_json_path):
        try:
            with open(testing_json_path, "r") as f:
                testing_data = json.load(f)
                testing_score = testing_data.get("combined_score")
        except Exception as e:
            print(f"Error reading testing data for problem {problem_id}: {e}")

    results.append(
        {"problem_id": problem_id, "training_score": training_score, "testing_score": testing_score}
    )

# Write to CSV
with open(output_csv, "w", newline="") as csvfile:
    fieldnames = ["problem_id", "training_score", "testing_score"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(results)

print(f"\nResults written to {output_csv}")
print(f"Total problems: {len(results)}")
print(
    f"Problems with both scores: {sum(1 for r in results if r['training_score'] is not None and r['testing_score'] is not None)}"
)
print(f"Problems missing training score: {sum(1 for r in results if r['training_score'] is None)}")
print(f"Problems missing testing score: {sum(1 for r in results if r['testing_score'] is None)}")
