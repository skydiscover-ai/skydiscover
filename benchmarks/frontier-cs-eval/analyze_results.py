from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define paths
_script_dir = str(Path(__file__).resolve().parent)
input_csv = str(Path(_script_dir) / "combined_results.csv")
output_dir = _script_dir

# Read the CSV file
df = pd.read_csv(input_csv)

# Calculate average of training and testing scores
df["average_score"] = (df["training_score"] + df["testing_score"]) / 2

# Remove rows where either score is None (NaN)
df_complete = df.dropna(subset=["training_score", "testing_score"])

print(f"\n=== Analysis Results ===")
print(f"Total problems: {len(df)}")
print(f"Problems with complete data: {len(df_complete)}")
print(f"\nTraining Scores:")
print(f"  Mean: {df_complete['training_score'].mean():.4f}")
print(f"  Median: {df_complete['training_score'].median():.4f}")
print(f"  Std Dev: {df_complete['training_score'].std():.4f}")
print(f"  Min: {df_complete['training_score'].min():.4f}")
print(f"  Max: {df_complete['training_score'].max():.4f}")

print(f"\nTesting Scores:")
print(f"  Mean: {df_complete['testing_score'].mean():.4f}")
print(f"  Median: {df_complete['testing_score'].median():.4f}")
print(f"  Std Dev: {df_complete['testing_score'].std():.4f}")
print(f"  Min: {df_complete['testing_score'].min():.4f}")
print(f"  Max: {df_complete['testing_score'].max():.4f}")

print(f"\nAverage Scores:")
print(f"  Mean: {df_complete['average_score'].mean():.4f}")
print(f"  Median: {df_complete['average_score'].median():.4f}")
print(f"  Std Dev: {df_complete['average_score'].std():.4f}")

# Save the updated CSV with averages
output_csv = Path(output_dir) / "combined_results_with_averages.csv"
df.to_csv(output_csv, index=False)
print(f"\nUpdated CSV with averages saved to {output_csv}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Scatter plot: Training vs Testing scores
ax = axes[0, 0]
ax.scatter(df_complete["training_score"], df_complete["testing_score"], alpha=0.6, s=50)
# Add diagonal line for reference (where training == testing)
lim = [
    min(df_complete["training_score"].min(), df_complete["testing_score"].min()),
    max(df_complete["training_score"].max(), df_complete["testing_score"].max()),
]
ax.plot(lim, lim, "r--", alpha=0.5, label="Training = Testing")
ax.set_xlabel("Training Score")
ax.set_ylabel("Testing Score")
ax.set_title("Training vs Testing Scores")
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Distribution comparison - histograms
ax = axes[0, 1]
ax.hist(df_complete["training_score"], bins=20, alpha=0.6, label="Training", edgecolor="black")
ax.hist(df_complete["testing_score"], bins=20, alpha=0.6, label="Testing", edgecolor="black")
ax.set_xlabel("Score")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Training vs Testing Scores")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

# 3. Box plot comparison
ax = axes[1, 0]
box_data = [
    df_complete["training_score"],
    df_complete["testing_score"],
    df_complete["average_score"],
]
bp = ax.boxplot(box_data, labels=["Training", "Testing", "Average"])
ax.set_ylabel("Score")
ax.set_title("Score Comparison (Box Plot)")
ax.grid(True, alpha=0.3, axis="y")

# 4. Difference plot: Training - Testing
ax = axes[1, 1]
difference = df_complete["training_score"] - df_complete["testing_score"]
ax.scatter(df_complete["problem_id"].astype(int), difference, alpha=0.6, s=50)
ax.axhline(y=0, color="r", linestyle="--", alpha=0.5, label="No Difference")
ax.set_xlabel("Problem ID")
ax.set_ylabel("Training Score - Testing Score")
ax.set_title("Score Difference (Training - Testing)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = Path(output_dir) / "results_analysis.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"Plot saved to {plot_path}")

# Additional statistics about differences
print(f"\nScore Differences (Training - Testing):")
print(f"  Mean Difference: {difference.mean():.4f}")
print(f"  Median Difference: {difference.median():.4f}")
print(f"  Std Dev: {difference.std():.4f}")
print(f"  Problems where training > testing: {(difference > 0).sum()}")
print(f"  Problems where testing > training: {(difference < 0).sum()}")

plt.show()
