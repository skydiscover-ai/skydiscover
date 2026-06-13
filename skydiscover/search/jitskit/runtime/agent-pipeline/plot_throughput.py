#!/usr/bin/env python3
"""Plot throughput per setup (in-mem / LTM), with FASTER scaling comparison."""

import json
import re
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "runs"

# Classify experiment by setup type based on folder name
def classify_setup(name):
    if "inmem" in name:
        return "in-memory"
    elif "ltm" in name:
        return "LTM"
    return "unknown"


def load_experiment(exp_dir):
    """Load iteration data from an experiment directory."""
    iterations = []
    faster_baseline = {}

    for iter_dir in sorted(exp_dir.glob("iter_*")):
        match = re.match(r"iter_(\d+)", iter_dir.name)
        if not match:
            continue
        iter_num = int(match.group(1))
        analysis_file = iter_dir / "analysis.json"
        if not analysis_file.exists():
            continue
        try:
            with open(analysis_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        wl_keys = list(data.get("workloads", {}).keys())
        if not wl_keys:
            continue
        wl = data["workloads"][wl_keys[0]]

        if not faster_baseline:
            for tc, td in wl.get("threads", {}).items():
                ref = td.get("local_faster_mops") or td.get("reference_mops")
                if ref is not None:
                    faster_baseline[int(tc)] = ref

        iterations.append((iter_num, wl))

    iterations.sort(key=lambda x: x[0])
    setup = classify_setup(exp_dir.name)

    return {
        "name": exp_dir.name,
        "setup": setup,
        "iterations": iterations,
        "faster_baseline": faster_baseline if faster_baseline else None,
    }


def short_label(name):
    # Strip backend prefix and timestamp suffix for cleaner labels
    name = re.sub(r'^(codex|claude)_', '', name)
    name = re.sub(r'_\d{8}_\d{6}$', '', name)
    return name


def plot_scaling_curve(exp, output_dir):
    """Scaling curve: throughput vs thread count — best iteration vs FASTER.
    Also overlay a few milestone iterations to show progression."""
    if not exp["iterations"]:
        return

    # Collect all thread counts
    all_tcs = set()
    for _, wl in exp["iterations"]:
        all_tcs.update(int(t) for t in wl.get("threads", {}).keys())
    all_tcs = sorted(all_tcs)

    if len(all_tcs) < 2:
        return  # Can't show scaling with one thread count

    # Best throughput per thread count across all iterations
    best_per_tc = {}
    best_iter_per_tc = {}
    for iter_num, wl in exp["iterations"]:
        for tc_str, td in wl.get("threads", {}).items():
            tc = int(tc_str)
            if tc not in best_per_tc or td["total_mops"] > best_per_tc[tc]:
                best_per_tc[tc] = td["total_mops"]
                best_iter_per_tc[tc] = iter_num

    fig, ax = plt.subplots(figsize=(8, 6))

    # FASTER scaling curve
    if exp["faster_baseline"]:
        faster_tcs = sorted(exp["faster_baseline"].keys())
        faster_vals = [exp["faster_baseline"][tc] for tc in faster_tcs]
        ax.plot(faster_tcs, faster_vals, 'r--o', linewidth=2.5, markersize=8,
                label='FASTER', zorder=3)
        for tc, val in zip(faster_tcs, faster_vals):
            ax.annotate(f'{val:.1f}', (tc, val), textcoords="offset points",
                        xytext=(0, 12), ha='center', fontsize=9, color='red', fontweight='bold')

    # Our best scaling curve
    our_tcs = sorted(best_per_tc.keys())
    our_vals = [best_per_tc[tc] for tc in our_tcs]
    ax.plot(our_tcs, our_vals, 'b-s', linewidth=2.5, markersize=8,
            label=f'{short_label(exp["name"])} (best iter)', zorder=3)
    for tc, val in zip(our_tcs, our_vals):
        iter_label = f'{val:.1f}\n(iter {best_iter_per_tc[tc]})'
        ax.annotate(iter_label, (tc, val), textcoords="offset points",
                    xytext=(0, -22), ha='center', fontsize=8, color='blue')

    # Also show first and last iteration scaling to show progression
    first_iter_num, first_wl = exp["iterations"][0]
    last_iter_num, last_wl = exp["iterations"][-1]

    for iter_num, wl, style, alpha, color in [
        (first_iter_num, first_wl, ':', 0.5, 'gray'),
        (last_iter_num, last_wl, '-.', 0.6, 'green'),
    ]:
        tcs = sorted(int(t) for t in wl.get("threads", {}).keys())
        vals = [wl["threads"][str(tc)]["total_mops"] for tc in tcs]
        ax.plot(tcs, vals, linestyle=style, marker='o', linewidth=1.5, markersize=5,
                alpha=alpha, color=color, label=f'iter {iter_num}')

    setup_label = exp["setup"].upper()
    ax.set_xlabel("Thread Count", fontsize=13)
    ax.set_ylabel("Throughput (Mops/s)", fontsize=13)
    ax.set_title(f"[{setup_label}] Thread Scaling: Best Iteration vs. FASTER", fontsize=14, fontweight='bold')
    ax.set_xticks(all_tcs)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = output_dir / f"{exp['setup']}_scaling_curve.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


def plot_throughput_over_iterations(exp, output_dir):
    """Per-thread throughput across iterations with FASTER baselines. One subplot per thread count."""
    if not exp["iterations"]:
        return

    all_tcs = set()
    for _, wl in exp["iterations"]:
        all_tcs.update(int(t) for t in wl.get("threads", {}).keys())
    all_tcs = sorted(all_tcs)

    n = len(all_tcs)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(max(6, 5 * n), 5), squeeze=False)

    for tc_idx, tc in enumerate(all_tcs):
        ax = axes[0][tc_idx]
        iters, mops = [], []
        for iter_num, wl in exp["iterations"]:
            if str(tc) in wl.get("threads", {}):
                iters.append(iter_num)
                mops.append(wl["threads"][str(tc)]["total_mops"])

        ax.plot(iters, mops, 'b-o', linewidth=2, markersize=6, label=short_label(exp["name"]))

        # FASTER baseline
        if exp["faster_baseline"] and tc in exp["faster_baseline"]:
            fval = exp["faster_baseline"][tc]
            ax.axhline(y=fval, color='red', linestyle='--', linewidth=2, alpha=0.8,
                        label=f'FASTER ({fval:.1f})')

        ax.set_title(f"{tc} Threads", fontsize=12, fontweight='bold')
        ax.set_xlabel("Iteration", fontsize=11)
        if tc_idx == 0:
            ax.set_ylabel("Throughput (Mops/s)", fontsize=11)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_xlim(left=0.5)
        ax.set_ylim(bottom=0)

    setup_label = exp["setup"].upper()
    fig.suptitle(f"[{setup_label}] Throughput Across Iterations by Thread Count",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = output_dir / f"{exp['setup']}_throughput_over_iterations.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


def plot_best_vs_faster_bar(exp, output_dir):
    """Bar chart: best throughput per thread count vs FASTER, for one experiment."""
    if not exp["iterations"]:
        return

    all_tcs = set()
    for _, wl in exp["iterations"]:
        all_tcs.update(int(t) for t in wl.get("threads", {}).keys())
    all_tcs = sorted(all_tcs)

    # Best throughput per thread count
    best_per_tc = {}
    for _, wl in exp["iterations"]:
        for tc_str, td in wl.get("threads", {}).items():
            tc = int(tc_str)
            if tc not in best_per_tc or td["total_mops"] > best_per_tc[tc]:
                best_per_tc[tc] = td["total_mops"]

    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(all_tcs)), 6))
    x = np.arange(len(all_tcs))
    bar_width = 0.35

    # FASTER bars
    has_faster = exp["faster_baseline"] is not None
    if has_faster:
        faster_vals = [exp["faster_baseline"].get(tc, 0) for tc in all_tcs]
        bars_f = ax.bar(x - bar_width / 2, faster_vals, bar_width,
                        label='FASTER', color='#d62728', alpha=0.85, edgecolor='white')
        for bar, val in zip(bars_f, faster_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        offset = bar_width / 2
    else:
        offset = 0

    our_vals = [best_per_tc.get(tc, 0) for tc in all_tcs]
    bars_o = ax.bar(x + offset, our_vals, bar_width,
                    label=short_label(exp["name"]), color='#1f77b4', alpha=0.85, edgecolor='white')
    for bar, val in zip(bars_o, our_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    setup_label = exp["setup"].upper()
    ax.set_xlabel("Thread Count", fontsize=13)
    ax.set_ylabel("Throughput (Mops/s)", fontsize=13)
    ax.set_title(f"[{setup_label}] Best Throughput per Thread Count vs. FASTER",
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(tc) for tc in all_tcs])
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = output_dir / f"{exp['setup']}_best_vs_faster_bar.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


def main():
    exp_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else EXPERIMENTS_DIR
    output_dir = exp_dir / "plots"
    output_dir.mkdir(exist_ok=True)

    # If exp_dir itself contains iter_* folders, treat it as a single experiment
    if any(exp_dir.glob("iter_*")):
        exp_folders = [exp_dir]
    else:
        exp_folders = sorted([
            d for d in exp_dir.iterdir()
            if d.is_dir() and any(d.glob("iter_*"))
        ])

    if not exp_folders:
        print(f"No experiment folders found in {exp_dir}")
        sys.exit(1)

    experiments = [load_experiment(f) for f in exp_folders]
    experiments = [e for e in experiments if e["iterations"]]

    # Group by setup
    by_setup = {}
    for exp in experiments:
        by_setup.setdefault(exp["setup"], []).append(exp)

    print(f"Found {len(experiments)} experiment(s) across {len(by_setup)} setup(s):")
    for setup, exps in by_setup.items():
        print(f"\n  [{setup}]")
        for e in exps:
            print(f"    - {e['name']} ({len(e['iterations'])} iterations)")

    # Pick the best (most iterations) experiment per setup for main plots
    # If there are multiple experiments for the same setup, pick the one with most data
    for setup, exps in by_setup.items():
        # Sort by number of iterations descending, use the richest one
        exps.sort(key=lambda e: len(e["iterations"]), reverse=True)
        exp = exps[0]
        print(f"\n--- Plotting [{setup}]: {exp['name']} ---")

        # 1. Scaling curve (only if multiple thread counts)
        plot_scaling_curve(exp, output_dir)

        # 2. Throughput across iterations per thread count
        plot_throughput_over_iterations(exp, output_dir)

        # 3. Bar chart: best vs FASTER
        plot_best_vs_faster_bar(exp, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
