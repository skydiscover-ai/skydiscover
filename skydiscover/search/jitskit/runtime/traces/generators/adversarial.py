"""Adversarial trace generators that exploit FASTER's FIFO eviction.

Patterns
--------
scan    Sequential cyclic scan over all load keys.  Every key is evicted
        before its next access.  Near-100% miss rate with writes.

belady  Cycle through (in-memory capacity + 1) keys.  Theoretically optimal
        FIFO adversary: provably 0% hit rate.

stride  Cycle through 1.1x the mutable region.  Forces 100% copy-on-write
        (RCU) for every RMW, doubling effective record size.
"""

from __future__ import annotations

import os

import numpy as np

from .base import (
    DEFAULT_LOAD_COUNT,
    DEFAULT_OUTDIR,
    DEFAULT_RUN_COUNT,
    TraceGenerator,
    TraceMetadata,
    check_existing,
    human_count,
    write_keys,
)

# FASTER record-layout constants (must match benchmark_harness.cc / FASTER src)
_RECORD_HEADER = 8   # RecordInfo
_KEY_BYTES     = 8   # uint64_t
_PAGE_SIZE     = 32 << 20   # 32 MiB
_HEAD_PAGES    = 4


def _record_size(value_size: int) -> int:
    raw = _RECORD_HEADER + _KEY_BYTES + value_size
    return (raw + 7) & ~7


def _inmem_capacity(mem_budget_bytes: int, value_size: int) -> int:
    """Records that fit in FASTER's in-memory log (excluding head pages)."""
    rec = _record_size(value_size)
    usable_pages = mem_budget_bytes // _PAGE_SIZE - _HEAD_PAGES
    if usable_pages <= 0:
        raise ValueError(f"Memory budget too small: need >{(_HEAD_PAGES + 1) * _PAGE_SIZE} bytes")
    return (usable_pages * _PAGE_SIZE) // rec


def _mutable_capacity(mem_budget_bytes: int, value_size: int,
                      mutable_fraction: float = 0.9) -> int:
    rec = _record_size(value_size)
    usable_pages = mem_budget_bytes // _PAGE_SIZE - _HEAD_PAGES
    mutable_pages = max(2, int(usable_pages * mutable_fraction))
    return (mutable_pages * _PAGE_SIZE) // rec


def _write_load_sequential(path: str, count: int) -> None:
    """Write sequential keys 0..count-1."""
    if check_existing(path, count):
        return
    print(f"  Generating {count:,} sequential load keys ...")
    np.arange(count, dtype=np.uint64).tofile(path)
    print(f"  -> {path}  ({os.path.getsize(path) / 1e9:.2f} GB)")


def _write_cyclic_run(path: str, working_set: int, total: int) -> None:
    """Write keys cycling through 0..working_set-1, total keys."""
    if check_existing(path, total):
        return
    one_cycle = np.arange(working_set, dtype=np.uint64)
    full_cycles = total // working_set
    remainder = total % working_set

    print(f"  Generating {total:,} cyclic run keys "
          f"({full_cycles} cycles of {working_set:,} + {remainder:,} remainder) ...")
    with open(path, "wb") as f:
        for _ in range(full_cycles):
            one_cycle.tofile(f)
        if remainder:
            one_cycle[:remainder].tofile(f)
    print(f"  -> {path}  ({os.path.getsize(path) / 1e9:.2f} GB)")


# ── Generators ────────────────────────────────────────────────────────────────

class ScanGenerator(TraceGenerator):
    name = "scan"
    category = "adversarial"
    description = "Sequential cyclic scan -- worst-case for FIFO eviction"

    def add_args(self, parser) -> None:
        parser.add_argument("--load-count", type=int, default=DEFAULT_LOAD_COUNT)
        parser.add_argument("--run-count", type=int, default=DEFAULT_RUN_COUNT)
        parser.add_argument("--outdir", default=DEFAULT_OUTDIR)

    def generate(self, outdir: str, args) -> TraceMetadata:
        lc, rc = args.load_count, args.run_count
        lcs, rcs = human_count(lc), human_count(rc)
        load_path = os.path.join(outdir, f"load_scan_{lcs}_raw.dat")
        run_path = os.path.join(outdir, f"run_scan_{lcs}_{rcs}_raw.dat")
        os.makedirs(outdir, exist_ok=True)

        print(f"\n=== Scan: cyclic scan over {lc:,} keys ===")
        _write_load_sequential(load_path, lc)
        _write_cyclic_run(run_path, lc, rc)

        meta = TraceMetadata(
            name=f"scan_{lcs}", category="adversarial",
            generator="ScanGenerator",
            description=f"Sequential cyclic scan, {lcs} keys",
            load_file=os.path.abspath(load_path),
            run_file=os.path.abspath(run_path),
            load_count=lc, run_count=rc,
            params={"load_count": lc, "run_count": rc},
        )
        meta.save(outdir)
        return meta


class BeladyGenerator(TraceGenerator):
    name = "belady"
    category = "adversarial"
    description = "Capacity+1 cycle -- optimal FIFO adversary (0%% hit rate)"

    def add_args(self, parser) -> None:
        parser.add_argument("--load-count", type=int, default=DEFAULT_LOAD_COUNT)
        parser.add_argument("--run-count", type=int, default=DEFAULT_RUN_COUNT)
        g = parser.add_mutually_exclusive_group(required=True)
        g.add_argument("--working-set", type=int, default=None,
                       help="Exact number of cycling keys (capacity+1)")
        g.add_argument("--mem-budget-gb", type=float, default=None,
                       help="Auto-compute working set from FASTER memory model")
        parser.add_argument("--value-size", type=int, default=100,
                            help="Value size in bytes (only used with --mem-budget-gb)")
        parser.add_argument("--outdir", default=DEFAULT_OUTDIR)

    def generate(self, outdir: str, args) -> TraceMetadata:
        lc, rc = args.load_count, args.run_count

        if args.working_set is not None:
            ws = min(args.working_set, lc)
        else:
            mem_bytes = int(args.mem_budget_gb * (1 << 30))
            capacity = _inmem_capacity(mem_bytes, args.value_size)
            ws = min(capacity + 1, lc)

        print(f"\n=== Belady: capacity+1 cycle ===")
        print(f"  working set      : {ws:,} keys")
        if ws >= lc:
            print("  WARNING: dataset fits in memory -- belady has no effect")

        lcs, rcs = human_count(lc), human_count(rc)
        load_path = os.path.join(outdir, f"load_belady_{lcs}_raw.dat")
        run_path = os.path.join(outdir, f"run_belady_{lcs}_{rcs}_raw.dat")
        os.makedirs(outdir, exist_ok=True)

        _write_load_sequential(load_path, lc)
        _write_cyclic_run(run_path, ws, rc)

        meta = TraceMetadata(
            name=f"belady_{lcs}", category="adversarial",
            generator="BeladyGenerator",
            description=f"Capacity+1 adversary (working_set={ws:,})",
            load_file=os.path.abspath(load_path),
            run_file=os.path.abspath(run_path),
            load_count=lc, run_count=rc,
            params={"working_set": ws,
                    "mem_budget_gb": getattr(args, "mem_budget_gb", None),
                    "value_size": getattr(args, "value_size", None)},
        )
        meta.save(outdir)
        return meta


class StrideGenerator(TraceGenerator):
    name = "stride"
    category = "adversarial"
    description = "Mutable-overflow stride -- forces 100%% RCU copies"

    def add_args(self, parser) -> None:
        parser.add_argument("--load-count", type=int, default=DEFAULT_LOAD_COUNT)
        parser.add_argument("--run-count", type=int, default=DEFAULT_RUN_COUNT)
        g = parser.add_mutually_exclusive_group(required=True)
        g.add_argument("--working-set", type=int, default=None,
                       help="Exact number of cycling keys (1.1x mutable region)")
        g.add_argument("--mem-budget-gb", type=float, default=None,
                       help="Auto-compute working set from FASTER memory model")
        parser.add_argument("--value-size", type=int, default=100,
                            help="Value size in bytes (only used with --mem-budget-gb)")
        parser.add_argument("--mutable-fraction", type=float, default=0.9,
                            help="FASTER mutable region fraction (only with --mem-budget-gb)")
        parser.add_argument("--outdir", default=DEFAULT_OUTDIR)

    def generate(self, outdir: str, args) -> TraceMetadata:
        lc, rc = args.load_count, args.run_count

        if args.working_set is not None:
            ws = min(args.working_set, lc)
        else:
            mem_bytes = int(args.mem_budget_gb * (1 << 30))
            mf = args.mutable_fraction
            mut_cap = _mutable_capacity(mem_bytes, args.value_size, mf)
            full_cap = _inmem_capacity(mem_bytes, args.value_size)
            ws = min(int(mut_cap * 1.1), full_cap, lc)

        print(f"\n=== Stride: 1.1x mutable overflow ===")
        print(f"  working set      : {ws:,}")

        lcs, rcs = human_count(lc), human_count(rc)
        load_path = os.path.join(outdir, f"load_stride_{lcs}_raw.dat")
        run_path = os.path.join(outdir, f"run_stride_{lcs}_{rcs}_raw.dat")
        os.makedirs(outdir, exist_ok=True)

        _write_load_sequential(load_path, lc)
        _write_cyclic_run(run_path, ws, rc)

        meta = TraceMetadata(
            name=f"stride_{lcs}", category="adversarial",
            generator="StrideGenerator",
            description=f"Mutable-overflow stride (working_set={ws:,})",
            load_file=os.path.abspath(load_path),
            run_file=os.path.abspath(run_path),
            load_count=lc, run_count=rc,
            params={"working_set": ws,
                    "mem_budget_gb": getattr(args, "mem_budget_gb", None),
                    "value_size": getattr(args, "value_size", None)},
        )
        meta.save(outdir)
        return meta
