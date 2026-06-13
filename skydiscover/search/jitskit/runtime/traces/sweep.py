#!/usr/bin/env python3
"""Batch trace generation from YAML config.

Reads a YAML listing traces to generate, creates any missing ones.

Usage:
  python3 traces/sweep.py traces/workloads/zipf_theta_sweep.yaml
  python3 traces/sweep.py traces/workloads/adversarial.yaml --dry-run
  python3 traces/sweep.py traces/workloads/synthetic_knobs.yaml --outdir /mnt/ssd/ycsb_data

Config format (YAML):
  name: my_sweep
  traces:
    - generator: zipf
      theta: 0.5
    - generator: scan
    - generator: one_hit
      one_hit_ratio: 0.3
      theta: 0.99
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import yaml
except ImportError:
    yaml = None

from generators import GENERATORS


def _load_yaml(path: str) -> dict:
    text = open(path).read()
    if yaml:
        return yaml.safe_load(text)
    if path.endswith(".json"):
        return json.loads(text)
    print("ERROR: PyYAML not installed. Install: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


def _describe(trace: dict) -> str:
    gen = trace.get("generator", "?")
    parts = [gen]
    for k, v in sorted(trace.items()):
        if k == "generator":
            continue
        if isinstance(v, float):
            parts.append(f"{k}={v:.2f}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def _generate(trace_config: dict, outdir: str, dry_run: bool) -> dict | None:
    gen_name = trace_config.get("generator")
    if not gen_name:
        print("  ERROR: missing 'generator' field", file=sys.stderr)
        return None

    gen = GENERATORS.get(gen_name)
    if not gen:
        print(f"  ERROR: unknown generator '{gen_name}'. "
              f"Available: {', '.join(GENERATORS)}", file=sys.stderr)
        return None

    # Route YAML entries through the generator's own argparse so required
    # args and choices are validated the same way as the CLI.
    argv: list[str] = []
    for k, v in trace_config.items():
        if k == "generator":
            continue
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        else:
            argv.extend([flag, str(v)])
    argv.extend(["--outdir", outdir])

    parser = argparse.ArgumentParser(prog=f"sweep:{gen_name}")
    gen.add_args(parser)
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        print(f"  ERROR: invalid config for '{gen_name}' (exit {e.code})",
              file=sys.stderr)
        return None

    if dry_run:
        print(f"  [dry-run] would generate: {gen_name}")
        return {"load_file": "(dry-run)", "run_file": "(dry-run)",
                "load_count": getattr(args, "load_count", None),
                "run_count": getattr(args, "run_count", None)}

    meta = gen.generate(outdir, args)
    return {"load_file": meta.load_file, "run_file": meta.run_file,
            "load_count": meta.load_count, "run_count": meta.run_count}


def main():
    parser = argparse.ArgumentParser(
        description="Batch-generate traces from a YAML config.",
    )
    parser.add_argument("config", help="Path to YAML config")
    parser.add_argument("--outdir", default="/mnt/ssd/ycsb_data",
                        help="Trace output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be generated")
    args = parser.parse_args()

    config = _load_yaml(args.config)
    traces = config.get("traces", [])

    print(f"Sweep: {config.get('name', '(unnamed)')}")
    print(f"Traces: {len(traces)}")
    print(f"Output: {args.outdir}")
    print()

    generated = []
    for i, t in enumerate(traces, 1):
        print(f"  [{i}/{len(traces)}] {_describe(t)}")
        info = _generate(t, args.outdir, args.dry_run)
        if info:
            generated.append(info)
            if not args.dry_run:
                print(f"    load: {info['load_file']}")
                print(f"    run:  {info['run_file']}")
        print()

    print(f"Done. {len(generated)} traces generated in {args.outdir}")
    if not args.dry_run:
        print(f"\nList traces: python3 traces/list.py {args.outdir}")


if __name__ == "__main__":
    main()
