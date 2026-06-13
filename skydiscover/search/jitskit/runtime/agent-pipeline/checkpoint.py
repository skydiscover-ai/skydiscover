#!/usr/bin/env python3
"""
agent-pipeline/checkpoint.py -- Produce a leaderboard ranking over iter_* dirs.

Scans results_dir for iter_* subdirectories, collects each iteration's
status + analysis metrics, and writes a single `leaderboard.json` at the
top of results_dir. Does NOT duplicate iter_*/kvstore_impl.cc — leaderboard
entries reference the original iter_NNN/kvstore_impl.cc by relative path.

This replaces an earlier version that created a parallel `checkpoints/iter_*/`
directory tree with copies of each iter's impl, which made the experiments
layout confusing (two mirror trees of iter_*).

Usage:
  python3 agent-pipeline/checkpoint.py [results_dir] [output_dir]

Defaults:
  results_dir: ./results
  output_dir:  same as results_dir (leaderboard.json lands next to iter_*)

The `output_dir` argument is accepted for back-compat with the orchestrator's
existing call site; pass the results_dir for both to get the flat layout.
"""

import json
import sys
from pathlib import Path


def collect_leaderboard(results_dir: Path, output_dir: Path | None = None) -> None:
    results_dir = Path(results_dir)
    if output_dir is None:
        output_dir = results_dir
    else:
        output_dir = Path(output_dir)

    if not results_dir.exists():
        print(f"ERROR: {results_dir} does not exist")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    leaderboard = []
    for iter_dir in sorted(results_dir.glob("iter_*")):
        if not iter_dir.is_dir():
            continue

        iter_name = iter_dir.name
        impl_file = iter_dir / "kvstore_impl.cc"
        if not impl_file.exists():
            continue

        status_file = iter_dir / "status.txt"
        status = status_file.read_text().strip() if status_file.exists() else "UNKNOWN"

        analysis_file = iter_dir / "analysis.json"
        analysis = None
        if analysis_file.exists():
            try:
                analysis = json.loads(analysis_file.read_text())
            except json.JSONDecodeError:
                pass

        summary = analysis.get("summary", {}) if analysis else {}
        workload_peaks = {}
        if analysis:
            for wl_name, wl_data in analysis.get("workloads", {}).items():
                workload_peaks[wl_name] = {
                    "peak_mops": wl_data.get("peak_mops", 0),
                    "peak_threads": wl_data.get("peak_threads", 0),
                }

        leaderboard.append({
            "iteration": iter_name,
            "status": status,
            "avg_pct_of_faster": summary.get("avg_pct_of_faster", 0),
            "best_pct_of_faster": summary.get("best_pct_of_faster", 0),
            "all_validation_passed": summary.get("all_validation_passed", False),
            "num_failed_runs": summary.get("num_failed_runs", 0),
            "num_invalid_runs": summary.get("num_invalid_runs", 0),
            "workload_peaks": workload_peaks,
            # Relative paths for easy navigation — no file duplication.
            "impl_path": f"{iter_name}/kvstore_impl.cc",
            "iter_dir": iter_name,
        })

    # Sort: valid COMPLETED solutions first, then by avg throughput.
    leaderboard.sort(
        key=lambda e: (
            e["status"] == "COMPLETED" and e["all_validation_passed"],
            e["avg_pct_of_faster"],
        ),
        reverse=True,
    )

    (output_dir / "leaderboard.json").write_text(
        json.dumps(leaderboard, indent=2) + "\n"
    )

    # Summary table
    print(f"{'='*72}")
    print(f"  Leaderboard: {output_dir}/leaderboard.json")
    print(f"  Solutions:   {len(leaderboard)}")
    print(f"{'='*72}")
    print(f"  {'Rank':>4}  {'Iteration':<12}  {'Avg %':>8}  {'Best %':>8}  {'Status':<20}")
    print(f"  {'─'*4}  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*20}")
    for rank, entry in enumerate(leaderboard, 1):
        status_str = entry["status"]
        if not entry["all_validation_passed"]:
            status_str += " (invalid)"
        print(f"  {rank:>4}  {entry['iteration']:<12}  "
              f"{entry['avg_pct_of_faster']:>7.1f}%  "
              f"{entry['best_pct_of_faster']:>7.1f}%  "
              f"{status_str:<20}")
    print(f"{'='*72}")

    if leaderboard and leaderboard[0]["status"] == "COMPLETED":
        best = leaderboard[0]
        print(f"\n  Best solution: {best['iteration']} "
              f"({best['avg_pct_of_faster']:.1f}% avg of FASTER)")
        print(f"  Source: {results_dir / best['impl_path']}")


# Keep the old entry-point name as a thin shim so any external callers keep
# working, but the new behavior is the single-flat-leaderboard one.
def collect_checkpoints(results_dir: Path, checkpoints_dir: Path) -> None:
    collect_leaderboard(results_dir, output_dir=checkpoints_dir)


if __name__ == "__main__":
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else results
    collect_leaderboard(results, output)
