#!/usr/bin/env python3
"""Unified trace generation CLI.

Generate binary key traces for the SkyKV benchmark harness.  Each trace is
a pair of files (load_*.dat + run_*.dat) containing flat uint64_t keys.

Usage:
  python3 traces/generate.py zipf --theta 0.99 --outdir /mnt/ssd/ycsb_data
  python3 traces/generate.py zipf --theta 0.5 --load-count 50000000
  python3 traces/generate.py uniform --outdir /mnt/ssd/ycsb_data
  python3 traces/generate.py scan --outdir /mnt/ssd/ycsb_data
  python3 traces/generate.py belady --mem-budget-gb 8 --value-size 100
  python3 traces/generate.py stride --mem-budget-gb 8 --value-size 100
  python3 traces/generate.py metakv --outdir /mnt/ssd/ycsb_data
  python3 traces/generate.py twitter --cluster 18

List available traces:
  python3 traces/list.py /mnt/ssd/ycsb_data
"""

import argparse
import sys
import os

# Allow running as `python3 traces/generate.py` from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generators import GENERATORS


def main():
    parser = argparse.ArgumentParser(
        description="Generate binary key traces for SkyKV benchmarks.",
    )
    sub = parser.add_subparsers(dest="generator")
    sub.required = True

    # Register each generator as a subcommand
    for name, gen in GENERATORS.items():
        sp = sub.add_parser(name, help=gen.description,
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        gen.add_args(sp)

    args = parser.parse_args()
    gen = GENERATORS[args.generator]
    outdir = getattr(args, "outdir", "/mnt/ssd/ycsb_data")

    print(f"Generator: {gen.name} ({gen.description})")
    print(f"Output:    {outdir}")
    meta = gen.generate(outdir, args)
    print(f"\nDone. Metadata: {meta.name}.meta.json")
    print(f"  load: {meta.load_file} ({meta.load_count:,} keys)")
    print(f"  run:  {meta.run_file} ({meta.run_count:,} keys)")


if __name__ == "__main__":
    main()
