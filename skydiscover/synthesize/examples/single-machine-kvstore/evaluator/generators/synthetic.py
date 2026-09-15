"""Transforms of a Zipfian base stream that each exercise one workload knob: one_hit (single-access
keys), bursty (repeated hits in a short window), hotspot (writes concentrated on hot keys), bimodal
(key parity picks a small or large value; see KVSTORE_BIMODAL_VALUES in the harness).
"""

from __future__ import annotations

import os

import numpy as np

from .base import (
    DEFAULT_LOAD_COUNT,
    DEFAULT_OUTDIR,
    DEFAULT_RUN_COUNT,
    DEFAULT_SEED,
    TraceGenerator,
    TraceMetadata,
    check_existing,
    human_count,
    write_keys,
    write_keys_chunked,
)
from .ycsb import ZipfSampler


class OneHitWonderGenerator(TraceGenerator):
    """Inject single-access keys: a fraction r of run-phase ops hit a key never seen again."""

    name = "one_hit"
    category = "synthetic"
    description = "Zipfian + one-hit wonders (single-access pollution)"

    def add_args(self, parser) -> None:
        parser.add_argument(
            "--one-hit-ratio",
            type=float,
            required=True,
            help="Fraction of run ops that are one-hit wonders (0.0-1.0)",
        )
        parser.add_argument(
            "--theta", type=float, default=0.99, help="Zipf skew for the non-one-hit portion"
        )
        parser.add_argument("--load-count", type=int, default=DEFAULT_LOAD_COUNT)
        parser.add_argument("--run-count", type=int, default=DEFAULT_RUN_COUNT)
        parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
        parser.add_argument("--outdir", default=DEFAULT_OUTDIR)

    def generate(self, outdir: str, args) -> TraceMetadata:
        r = args.one_hit_ratio
        theta = args.theta
        lc, rc = args.load_count, args.run_count
        seed = args.seed

        # One-hit keys come from a separate key space above the load keys.
        # They are loaded (so they exist in the store) but accessed exactly once.
        n_one_hit_ops = int(rc * r)

        # Reserve extra keys for one-hit wonders (each unique, used once)
        total_load = lc + n_one_hit_ops

        lcs = human_count(total_load)
        rcs = human_count(rc)
        tag = f"_oh{int(r*100):02d}_t{theta:.2f}".replace(".", "")
        load_path = os.path.join(outdir, f"load_onehit{tag}_{lcs}_raw.dat")
        run_path = os.path.join(outdir, f"run_onehit{tag}_{lcs}_{rcs}_raw.dat")
        os.makedirs(outdir, exist_ok=True)

        print(f"\n=== One-hit wonders: ratio={r}, theta={theta} ===")
        print(f"  base keys: {lc:,}, one-hit keys: {n_one_hit_ops:,}")
        print(f"  total load: {total_load:,}")

        # Load: all keys (base + one-hit), shuffled
        if not check_existing(load_path, total_load):
            rng = np.random.default_rng(seed)
            keys = np.arange(total_load, dtype=np.uint64)
            rng.shuffle(keys)
            write_keys(load_path, keys)
            del keys
            print(f"  -> {load_path}")

        # Run: interleave Zipfian base ops with one-hit accesses
        if not check_existing(run_path, rc):
            zipf = ZipfSampler(lc, theta, seed + 1)
            rng = np.random.default_rng(seed + 2)
            chunk_size = 10_000_000
            one_hit_cursor = lc  # one-hit keys start at index lc

            def chunks():
                nonlocal one_hit_cursor
                remaining = rc
                while remaining > 0:
                    n = min(chunk_size, remaining)
                    # For each op, decide: one-hit or zipf
                    is_one_hit = rng.random(n) < r
                    n_oh = int(is_one_hit.sum())
                    keys = zipf.sample(n)
                    # Clamp to reserved pool: per-chunk sum has binomial
                    # variance; cumulative sum can exceed n_one_hit_ops.
                    # Demote excess positions to zipf so every run key stays
                    # in [0, total_load).
                    budget = total_load - one_hit_cursor
                    if n_oh > budget:
                        true_idx = np.flatnonzero(is_one_hit)
                        is_one_hit[true_idx[budget:]] = False
                        n_oh = budget
                    if n_oh > 0:
                        oh_keys = np.arange(one_hit_cursor, one_hit_cursor + n_oh, dtype=np.uint64)
                        keys[is_one_hit] = oh_keys
                        one_hit_cursor += n_oh
                    yield keys
                    remaining -= n

            print(f"  Generating {rc:,} run keys ({n_one_hit_ops:,} one-hit) ...")
            write_keys_chunked(run_path, chunks(), rc)

        meta = TraceMetadata(
            name=f"onehit{tag}_{lcs}",
            category="synthetic",
            generator="OneHitWonderGenerator",
            description=f"Zipf(theta={theta}) + {r*100:.0f}% one-hit wonders",
            load_file=os.path.abspath(load_path),
            run_file=os.path.abspath(run_path),
            load_count=total_load,
            run_count=rc,
            params={
                "one_hit_ratio": r,
                "theta": theta,
                "seed": seed,
                "base_keys": lc,
                "one_hit_keys": n_one_hit_ops,
            },
        )
        meta.save(outdir)
        return meta


class BurstyGenerator(TraceGenerator):
    """With probability burst_ratio, repeat the same key burst_size times in a row."""

    name = "bursty"
    category = "synthetic"
    description = "Zipfian + bursty accesses (k consecutive hits per key)"

    def add_args(self, parser) -> None:
        parser.add_argument(
            "--burst-size",
            type=int,
            required=True,
            help="Number of consecutive accesses per burst (k)",
        )
        parser.add_argument(
            "--burst-ratio", type=float, default=0.1, help="Fraction of ops that trigger a burst"
        )
        parser.add_argument("--theta", type=float, default=0.99)
        parser.add_argument("--load-count", type=int, default=DEFAULT_LOAD_COUNT)
        parser.add_argument("--run-count", type=int, default=DEFAULT_RUN_COUNT)
        parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
        parser.add_argument("--outdir", default=DEFAULT_OUTDIR)

    def generate(self, outdir: str, args) -> TraceMetadata:
        k = args.burst_size
        br = args.burst_ratio
        theta = args.theta
        lc, rc = args.load_count, args.run_count
        seed = args.seed

        lcs, rcs = human_count(lc), human_count(rc)
        tag = f"_k{k}_br{int(br*100):02d}_t{theta:.2f}".replace(".", "")
        load_path = os.path.join(outdir, f"load_bursty{tag}_{lcs}_raw.dat")
        run_path = os.path.join(outdir, f"run_bursty{tag}_{lcs}_{rcs}_raw.dat")
        os.makedirs(outdir, exist_ok=True)

        print(f"\n=== Bursty: k={k}, burst_ratio={br}, theta={theta} ===")

        # Load: standard shuffled keys
        if not check_existing(load_path, lc):
            rng = np.random.default_rng(seed)
            keys = np.arange(lc, dtype=np.uint64)
            rng.shuffle(keys)
            write_keys(load_path, keys)
            del keys
            print(f"  -> {load_path}")

        # Run: Zipfian with bursts injected
        if not check_existing(run_path, rc):
            zipf = ZipfSampler(lc, theta, seed + 1)
            rng = np.random.default_rng(seed + 2)

            # Generate more base keys than needed (bursts expand the stream)
            # Then truncate to rc
            print(f"  Generating {rc:,} run keys with burst(k={k}, ratio={br}) ...")
            chunk_size = 5_000_000
            written = 0

            with open(run_path, "wb") as f:
                while written < rc:
                    n = min(chunk_size, rc - written)
                    base_keys = zipf.sample(n)
                    is_burst = rng.random(n) < br

                    # Expand bursts: replace single access with k repeats
                    parts: list = []
                    for i in range(n):
                        if written + len(parts) >= rc:
                            break
                        if is_burst[i]:
                            reps = min(k, rc - written - len(parts))
                            parts.extend([base_keys[i]] * reps)
                        else:
                            parts.append(base_keys[i])

                    chunk = np.array(parts[: rc - written], dtype=np.uint64)
                    chunk.tofile(f)
                    written += len(chunk)

            # Verify exact size
            actual = os.path.getsize(run_path) // 8
            print(f"  -> {run_path} ({actual:,} keys)")

        actual_rc = os.path.getsize(run_path) // 8
        meta = TraceMetadata(
            name=f"bursty{tag}_{lcs}",
            category="synthetic",
            generator="BurstyGenerator",
            description=f"Zipf(theta={theta}) + burst(k={k}, ratio={br})",
            load_file=os.path.abspath(load_path),
            run_file=os.path.abspath(run_path),
            load_count=lc,
            run_count=actual_rc,
            params={"burst_size": k, "burst_ratio": br, "theta": theta, "seed": seed},
        )
        meta.save(outdir)
        return meta


class HotspotGenerator(TraceGenerator):
    """Concentrate a fraction of writes on the top-X% keys; reads stay Zipfian."""

    name = "hotspot"
    category = "synthetic"
    description = "Zipfian reads + concentrated writes on hot keys"

    def add_args(self, parser) -> None:
        parser.add_argument(
            "--hot-fraction",
            type=float,
            default=0.01,
            help="Fraction of keys that are 'hot' (default: 0.01 = top 1%%)",
        )
        parser.add_argument(
            "--hot-write-ratio",
            type=float,
            default=0.9,
            help="Fraction of writes targeting hot keys (default: 0.9)",
        )
        parser.add_argument(
            "--write-ratio",
            type=float,
            default=0.5,
            help="Fraction of ops treated as 'write-like' by the "
            "pattern -- should match the harness --setup "
            "(rmw=1.0, 50:50=0.5, 0:100=1.0, 100:0=0.0; "
            "default 0.5)",
        )
        parser.add_argument("--theta", type=float, default=0.99)
        parser.add_argument("--load-count", type=int, default=DEFAULT_LOAD_COUNT)
        parser.add_argument("--run-count", type=int, default=DEFAULT_RUN_COUNT)
        parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
        parser.add_argument("--outdir", default=DEFAULT_OUTDIR)

    def generate(self, outdir: str, args) -> TraceMetadata:
        hf = args.hot_fraction
        hwr = args.hot_write_ratio
        wr = args.write_ratio
        theta = args.theta
        lc, rc = args.load_count, args.run_count
        seed = args.seed

        n_hot = max(1, int(lc * hf))
        lcs, rcs = human_count(lc), human_count(rc)
        tag = f"_hf{int(hf*100):02d}_hwr{int(hwr*100):02d}_t{theta:.2f}".replace(".", "")
        load_path = os.path.join(outdir, f"load_hotspot{tag}_{lcs}_raw.dat")
        run_path = os.path.join(outdir, f"run_hotspot{tag}_{lcs}_{rcs}_raw.dat")
        os.makedirs(outdir, exist_ok=True)

        print(
            f"\n=== Hotspot: hot_keys={n_hot:,} ({hf*100:.1f}%), "
            f"hot_write_ratio={hwr}, theta={theta} ==="
        )

        if not check_existing(load_path, lc):
            rng = np.random.default_rng(seed)
            keys = np.arange(lc, dtype=np.uint64)
            rng.shuffle(keys)
            write_keys(load_path, keys)
            del keys
            print(f"  -> {load_path}")

        # Run keys: reads from Zipfian, writes concentrated on hot set.
        # The harness decides which ops are reads vs writes based on --setup.
        # We encode the ACCESS PATTERN here: keys are drawn Zipfian, then a random
        # fraction (wr) of ops are marked write-like and, of those, a fraction (hwr)
        # are redirected to the hot set. The harness assigns read/write based on
        # workload_id, so this approximates hot-write behavior with 50:50 or RMW workloads.
        if not check_existing(run_path, rc):
            zipf = ZipfSampler(lc, theta, seed + 1)
            rng = np.random.default_rng(seed + 2)
            chunk_size = 10_000_000

            def chunks():
                remaining = rc
                while remaining > 0:
                    n = min(chunk_size, remaining)
                    # Base: Zipfian
                    keys = zipf.sample(n)
                    # For wr of ops (the write-like ones), bias toward hot set
                    is_write_like = rng.random(n) < wr
                    use_hot = is_write_like & (rng.random(n) < hwr)
                    n_hot_ops = int(use_hot.sum())
                    if n_hot_ops > 0:
                        keys[use_hot] = rng.integers(0, n_hot, size=n_hot_ops, dtype=np.uint64)
                    yield keys
                    remaining -= n

            print(f"  Generating {rc:,} hotspot run keys ...")
            write_keys_chunked(run_path, chunks(), rc)

        meta = TraceMetadata(
            name=f"hotspot{tag}_{lcs}",
            category="synthetic",
            generator="HotspotGenerator",
            description=f"Zipf(theta={theta}) + {hwr*100:.0f}% writes on top {hf*100:.1f}% keys",
            load_file=os.path.abspath(load_path),
            run_file=os.path.abspath(run_path),
            load_count=lc,
            run_count=rc,
            params={
                "hot_fraction": hf,
                "hot_write_ratio": hwr,
                "write_ratio": wr,
                "theta": theta,
                "seed": seed,
                "n_hot_keys": n_hot,
            },
        )
        meta.save(outdir)
        return meta


class BimodalGenerator(TraceGenerator):
    """Zipf stream where key parity selects a small or large value size. The trace is key-only; the
    harness applies sizes when KVSTORE_BIMODAL_VALUES=1.
    """

    name = "bimodal"
    category = "synthetic"
    description = "Zipfian with bimodal per-key value size (needs harness support)"

    def add_args(self, parser) -> None:
        parser.add_argument("--theta", type=float, default=0.99, help="Zipf skew (default: 0.99)")
        parser.add_argument("--load-count", type=int, default=DEFAULT_LOAD_COUNT)
        parser.add_argument("--run-count", type=int, default=DEFAULT_RUN_COUNT)
        parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
        parser.add_argument("--outdir", default=DEFAULT_OUTDIR)

    def generate(self, outdir: str, args) -> TraceMetadata:
        theta = args.theta
        lc, rc = args.load_count, args.run_count
        seed = args.seed

        lcs, rcs = human_count(lc), human_count(rc)
        tag = f"_t{theta:.2f}".replace(".", "")
        load_path = os.path.join(outdir, f"load_bimodal{tag}_{lcs}_raw.dat")
        run_path = os.path.join(outdir, f"run_bimodal{tag}_{lcs}_{rcs}_raw.dat")
        os.makedirs(outdir, exist_ok=True)

        print(f"\n=== Bimodal: theta={theta} (even keys=small, odd keys=large) ===")

        if not check_existing(load_path, lc):
            rng = np.random.default_rng(seed)
            keys = np.arange(lc, dtype=np.uint64)
            rng.shuffle(keys)
            write_keys(load_path, keys)
            del keys
            print(f"  -> {load_path}")

        if not check_existing(run_path, rc):
            zipf = ZipfSampler(lc, theta, seed + 1)
            chunk_size = 10_000_000

            def chunks():
                remaining = rc
                while remaining > 0:
                    n = min(chunk_size, remaining)
                    yield zipf.sample(n)
                    remaining -= n

            print(f"  Generating {rc:,} Zipfian(theta={theta}) run keys ...")
            write_keys_chunked(run_path, chunks(), rc)

        meta = TraceMetadata(
            name=f"bimodal{tag}_{lcs}",
            category="synthetic",
            generator="BimodalGenerator",
            description=(
                f"Zipf(theta={theta}) -- harness picks value size "
                f"from key parity when KVSTORE_BIMODAL_VALUES=1"
            ),
            load_file=os.path.abspath(load_path),
            run_file=os.path.abspath(run_path),
            load_count=lc,
            run_count=rc,
            params={
                "theta": theta,
                "seed": seed,
                "note": "set KVSTORE_BIMODAL_VALUES=1 at bench time",
            },
        )
        meta.save(outdir)
        return meta
