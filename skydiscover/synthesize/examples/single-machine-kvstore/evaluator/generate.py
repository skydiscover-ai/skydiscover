#!/usr/bin/env python3
"""Generate binary key traces (load_*.dat + run_*.dat of uint64 keys) for the benchmark harness.

python3 evaluator/generate.py zipf --theta 0.99 --outdir /tmp/ycsb_data
python3 evaluator/generate.py --help    # all generators
"""

import argparse
import os
import sys

# Allow running as python3 evaluator/generate.py from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generators import GENERATORS
from generators.base import DEFAULT_OUTDIR


def main():
    parser = argparse.ArgumentParser(
        description="Generate binary key traces for the benchmark harness.",
    )
    sub = parser.add_subparsers(dest="generator")
    sub.required = True

    # Register each generator as a subcommand
    for name, gen in GENERATORS.items():
        sp = sub.add_parser(
            name, help=gen.description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        gen.add_args(sp)

    args = parser.parse_args()
    gen = GENERATORS[args.generator]
    outdir = getattr(args, "outdir", DEFAULT_OUTDIR)

    print(f"Generator: {gen.name} ({gen.description})")
    print(f"Output:    {outdir}")
    meta = gen.generate(outdir, args)
    print(f"\nDone. Metadata: {meta.name}.meta.json")
    print(f"  load: {meta.load_file} ({meta.load_count:,} keys)")
    print(f"  run:  {meta.run_file} ({meta.run_count:,} keys)")


if __name__ == "__main__":
    main()
