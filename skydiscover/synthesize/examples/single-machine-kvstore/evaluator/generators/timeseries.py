"""Time-series workload: keys are timestamps, inserts append at the tail, deletes trim the head.

Procedural rather than trace-driven: the load file holds the initial window [0, W) and the run file
is a placeholder; the harness drives inserts and deletes itself (delete rate via KVSTORE_DELETE_RATE).
"""

from __future__ import annotations

import os

import numpy as np

from .base import (
    DEFAULT_LOAD_COUNT,
    DEFAULT_OUTDIR,
    DEFAULT_SEED,
    TraceGenerator,
    TraceMetadata,
    check_existing,
    human_count,
    write_keys,
)

# Must exceed kChunkSize in benchmark_harness.cc (currently 3200) to pass
# the kTxnCount > kChunkSize guard; 10K gives ~3x headroom against any
# future chunk-size bump. The TS worker path never reads from this file.
_PLACEHOLDER_RUN_COUNT = 10_000


class TimeseriesHDGenerator(TraceGenerator):
    """Time-series head-delete: inserts at tail, deletes from head."""

    name = "timeseries"
    category = "synthetic"
    description = (
        "Time-series head-delete: inserts append at tail, deletes "
        "only at head (TTL / retention-window)"
    )

    def add_args(self, parser) -> None:
        parser.add_argument(
            "--load-count",
            type=int,
            default=DEFAULT_LOAD_COUNT,
            help="Retention window W (default: 250M)",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=DEFAULT_SEED,
            help="Reserved; load file is deterministic today",
        )
        parser.add_argument("--outdir", default=DEFAULT_OUTDIR)

    def generate(self, outdir: str, args) -> TraceMetadata:
        lc = args.load_count
        rc = _PLACEHOLDER_RUN_COUNT
        lcs = human_count(lc)
        rcs = human_count(rc)

        load_path = os.path.join(outdir, f"load_timeseries_{lcs}_raw.dat")
        run_path = os.path.join(outdir, f"run_timeseries_{lcs}_{rcs}_raw.dat")
        os.makedirs(outdir, exist_ok=True)

        print(f"\n=== Time-series head-delete (retention_window={lc:,}) ===")

        # Load = [0, W) sequentially. Order doesn't matter for Upsert idempotence;
        # keeping it monotone matches the "keys are timestamps" mental model.
        if not check_existing(load_path, lc):
            print(f"  Generating {lc:,} load keys [0, {lc:,}) ...")
            np.arange(lc, dtype=np.uint64).tofile(load_path)
            print(f"  -> {load_path}  ({os.path.getsize(load_path) / 1e9:.2f} GB)")

        # Run file is a small placeholder the harness allocates but never reads
        # on the TS path. Its only role: pass the kTxnCount > kChunkSize check
        # (kChunkSize=3200) in benchmark_harness.cc main(). Size is
        # _PLACEHOLDER_RUN_COUNT, chosen just above that threshold with slack.
        # Contents are irrelevant -- any non-empty u64 sequence works.
        if not check_existing(run_path, rc):
            arr = np.zeros(rc, dtype=np.uint64)
            write_keys(run_path, arr)
            del arr

        meta = TraceMetadata(
            name=f"timeseries_{lcs}",
            category="synthetic",
            generator="TimeseriesHDGenerator",
            description=(
                f"Time-series head-delete, W={lcs} "
                "(delete_rate controlled at runtime via KVSTORE_DELETE_RATE)"
            ),
            load_file=os.path.abspath(load_path),
            run_file=os.path.abspath(run_path),
            load_count=lc,
            run_count=rc,
            params={
                "retention_window": lc,
                "seed": args.seed,
                "note": "delete rate is a harness env (KVSTORE_DELETE_RATE), not trace-encoded",
            },
        )
        meta.save(outdir)
        return meta
