#!/usr/bin/env python3
"""Belady optimal cache hit rate for any trace + memory constraint.

Builds belady_sim if needed, computes cache capacity from the memory
budget and value size, and runs the simulation.

Usage:
  # Simple: trace path + memory budget (all you need)
  python3 traces/belady/run.py /mnt/ssd/ycsb_data \
      --mem-budget-gb 8 --value-size 100

  # Specify workload (default: 5050)
  python3 traces/belady/run.py /mnt/ssd/ycsb_data \
      --mem-budget-gb 8 --value-size 100 --workload rmw

  # Explicit load/run files
  python3 traces/belady/run.py \
      --load /mnt/ssd/ycsb_data/load_zipf_t099_250M_raw.dat \
      --run  /mnt/ssd/ycsb_data/run_zipf_t099_250M_1000M_raw.dat \
      --mem-budget-gb 8 --value-size 100

  # Sweep multiple budgets
  python3 traces/belady/run.py /mnt/ssd/ycsb_data \
      --mem-budget-gb 4,8,16,24 --value-size 100

  # Raw cache-items (bypass budget math)
  python3 traces/belady/run.py /mnt/ssd/ycsb_data \
      --cache-items 50000000

Workloads: 5050 (default), rmw, read_only, upsert_only
"""

import argparse
import glob
import os
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY = os.path.join(SCRIPT_DIR, "belady_sim")

# FASTER record layout for computing cache items from memory budget.
# Hash table overhead: 32M primary buckets (2GB) + ~1GB overflow = ~3GB.
_RECORD_HEADER = 8
_KEY_BYTES = 8
_HASH_OVERHEAD_GB = 3.0


def _record_size(value_size: int) -> int:
    raw = _RECORD_HEADER + _KEY_BYTES + value_size
    return (raw + 7) & ~7


def _cache_items(mem_budget_gb: float, value_size: int) -> int:
    """How many records fit in (budget - hash overhead)."""
    usable = max(0, mem_budget_gb - _HASH_OVERHEAD_GB) * (1 << 30)
    return int(usable) // _record_size(value_size)


def _find_traces(trace_dir: str) -> tuple[str, str] | None:
    """Find a load/run .dat pair in a directory. Picks the first match."""
    loads = sorted(glob.glob(os.path.join(trace_dir, "load_*_raw.dat")))
    runs = sorted(glob.glob(os.path.join(trace_dir, "run_*_raw.dat")))
    if loads and runs:
        return loads[0], runs[0]
    return None


def _build():
    src = os.path.join(SCRIPT_DIR, "belady_sim.cc")
    if os.path.exists(BINARY) and os.path.getmtime(BINARY) > os.path.getmtime(src):
        return
    print("Building belady_sim ...")
    subprocess.check_call(["make", "-C", SCRIPT_DIR, "belady_sim"])


def main():
    parser = argparse.ArgumentParser(
        description="Compute Belady optimal cache hit rate for a trace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python3 traces/belady/run.py /mnt/ssd/ycsb_data --mem-budget-gb 8 --value-size 100\n"
               "  python3 traces/belady/run.py /mnt/ssd/ycsb_data --mem-budget-gb 4,8,16,24 --value-size 100\n",
    )
    parser.add_argument("trace_dir", nargs="?", default=None,
                        help="Directory containing load_*_raw.dat + run_*_raw.dat")
    parser.add_argument("--load", default=None, help="Explicit load trace path")
    parser.add_argument("--run", default=None, help="Explicit run trace path")
    parser.add_argument("--mem-budget-gb", type=str, default=None,
                        help="Memory budget in GB (comma-separated for sweep)")
    parser.add_argument("--value-size", type=int, default=100,
                        help="Value size in bytes (default: 100)")
    parser.add_argument("--cache-items", type=str, default=None,
                        help="Direct cache capacity in items (bypass budget math)")
    parser.add_argument("--workload", default="5050",
                        choices=["5050", "rmw", "read_only", "upsert_only"],
                        help="Workload type (default: 5050)")
    parser.add_argument("--max-ops", type=int, default=0,
                        help="Cap run-phase ops (0 = all)")
    args = parser.parse_args()

    # Resolve trace files
    load_file = args.load
    run_file = args.run
    if not load_file or not run_file:
        if not args.trace_dir:
            parser.error("Provide trace_dir or --load/--run")
        pair = _find_traces(args.trace_dir)
        if not pair:
            parser.error(f"No load_*_raw.dat + run_*_raw.dat found in {args.trace_dir}")
        load_file, run_file = pair
        print(f"Traces: {os.path.basename(load_file)}, {os.path.basename(run_file)}")

    # Resolve cache sizes
    cache_sizes: list[tuple[int, str]] = []  # (items, label)
    if args.cache_items:
        for v in args.cache_items.split(","):
            items = int(v.strip())
            cache_sizes.append((items, f"{items:,} items"))
    elif args.mem_budget_gb:
        for v in args.mem_budget_gb.split(","):
            gb = float(v.strip())
            items = _cache_items(gb, args.value_size)
            rec = _record_size(args.value_size)
            cache_sizes.append((items, f"{gb}GB budget → {items:,} items "
                                       f"(rec={rec}B, hash={_HASH_OVERHEAD_GB}GB)"))
    else:
        parser.error("Provide --mem-budget-gb or --cache-items")

    _build()

    for items, label in cache_sizes:
        print(f"\n{'='*64}")
        print(f"  Belady optimal: {label}")
        print(f"  workload={args.workload}, value={args.value_size}B")
        print(f"{'='*64}\n")

        cmd = [BINARY, args.workload, str(items), load_file, run_file]
        if args.max_ops > 0:
            cmd.append(str(args.max_ops))
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
