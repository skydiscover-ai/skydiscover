"""YCSB trace generators: Zipfian and Uniform.

Pure Python/numpy implementation -- no dependency on FASTER's gen_keys binary.
Produces identical traces to gen_keys.cc at default parameters (seed=211,
theta=0.99, 250M load, 1B run).

The Zipfian generator faithfully ports FASTER's ZipfGenerator algorithm
(inverse-CDF method with exact zeta computation).
"""

from __future__ import annotations

import os
import time

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


# ── Zipfian inverse-CDF (matches FASTER's ZipfGenerator.cs) ──────────────────

def _compute_zeta(n: int, theta: float) -> float:
    """Compute zeta(n, theta) = sum_{i=1}^{n} 1/i^theta.

    Vectorized over 1M-element chunks for speed.  ~2s for n=250M, theta=0.99.
    """
    total = 0.0
    chunk = 1 << 20  # 1M
    for start in range(1, n + 1, chunk):
        end = min(start + chunk, n + 1)
        i = np.arange(start, end, dtype=np.float64)
        total += np.sum(1.0 / np.power(i, theta))
    return total


class ZipfSampler:
    """YCSB-style bounded Zipfian sampler over [0, n).

    By default, the sampler scrambles its output through a deterministic
    random permutation of [0, n). This preserves the Zipf distribution
    exactly — same hotset size, same probabilities, same coefficient —
    but destroys the rank-to-identity correlation YCSB-style samplers
    produce by default (where key=0 is hottest, key=1 next, etc.).
    Without scrambling, implementations that hardcode "low-indexed keys
    are hot" exploit this generator artifact rather than specializing to
    the Zipf distribution itself.

    Pass scramble=False to opt out (e.g., to reproduce pre-scramble
    baselines). `perm_seed` is saved in the trace metadata so the same
    permutation can be recreated for audit/debug.
    """

    def __init__(self, n: int, theta: float, seed: int,
                 scramble: bool = True, perm_seed: int | None = None):
        self.n = n
        self.theta = theta
        self.rng = np.random.default_rng(seed)

        print(f"  Computing zeta(n={n:,}, theta={theta}) ...")
        t0 = time.monotonic()
        self.zetan = _compute_zeta(n, theta)
        self.zeta2 = _compute_zeta(2, theta)
        self.alpha = 1.0 / (1.0 - theta)
        self.eta = (1.0 - (2.0 / n) ** (1.0 - theta)) / (1.0 - self.zeta2 / self.zetan)
        dt = time.monotonic() - t0
        print(f"  zeta={self.zetan:.6f}  ({dt:.1f}s)")

        if scramble:
            self.perm_seed = perm_seed if perm_seed is not None else (seed ^ 0xDEADBEEF)
            print(f"  Materializing scramble permutation (perm_seed={self.perm_seed}) ...")
            t0 = time.monotonic()
            self.perm = np.random.default_rng(self.perm_seed).permutation(n).astype(np.uint64)
            dt = time.monotonic() - t0
            print(f"  perm ready ({dt:.1f}s, {self.perm.nbytes / 1e9:.2f} GB)")
        else:
            self.perm = None
            self.perm_seed = None

    def sample(self, size: int) -> np.ndarray:
        """Generate `size` Zipfian keys in [0, n)."""
        u = self.rng.uniform(0.0, 1.0, size)
        uz = u * self.zetan

        result = np.floor(
            self.n * np.power(self.eta * u - self.eta + 1.0, self.alpha)
        ).astype(np.uint64)

        # Special cases: uz < 1 → 0, uz < 1+0.5^theta → 1
        mask1 = uz < (1.0 + 0.5 ** self.theta)
        result[mask1] = 1
        mask0 = uz < 1.0
        result[mask0] = 0

        # Clamp to valid range
        np.clip(result, 0, self.n - 1, out=result)

        # Scramble: map rank-space key -> uniformly-random key identity.
        # Distribution is preserved exactly; only numeric identity changes.
        if self.perm is not None:
            result = self.perm[result]

        return result


# ── Generators ────────────────────────────────────────────────────────────────

class ZipfGenerator(TraceGenerator):
    name = "zipf"
    category = "ycsb"
    description = "Zipfian distribution (configurable theta, default 0.99)"

    def add_args(self, parser) -> None:
        parser.add_argument("--theta", type=float, default=0.99,
                            help="Zipf skew parameter")
        parser.add_argument("--load-count", type=int, default=DEFAULT_LOAD_COUNT,
                            help="Number of load keys")
        parser.add_argument("--run-count", type=int, default=DEFAULT_RUN_COUNT,
                            help="Number of run keys")
        parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                            help="PRNG seed")
        parser.add_argument("--no-scramble", action="store_true",
                            help="Disable the rank->identity permutation. "
                                 "Produces raw YCSB-style Zipf where key=0 is "
                                 "hottest; intended only for reproducing pre-scramble baselines.")
        parser.add_argument("--perm-seed", type=int, default=None,
                            help="Seed for scramble permutation (default: seed ^ 0xDEADBEEF)")
        parser.add_argument("--outdir", default=DEFAULT_OUTDIR,
                            help="Output directory")

    def generate(self, outdir: str, args) -> TraceMetadata:
        theta = args.theta
        load_count = args.load_count
        run_count = args.run_count
        seed = args.seed
        scramble = not getattr(args, "no_scramble", False)
        perm_seed = getattr(args, "perm_seed", None)
        # Resolve effective perm_seed up-front so it's correct in metadata
        # even when the run file already exists on disk (skipped regeneration).
        effective_perm_seed = (perm_seed if perm_seed is not None
                               else ((seed + 1) ^ 0xDEADBEEF)) if scramble else None

        # Build filenames -- always include theta for clarity
        theta_tag = f"_t{theta:.2f}".replace(".", "")
        lc = human_count(load_count)
        rc = human_count(run_count)
        load_path = os.path.join(outdir, f"load_zipf{theta_tag}_{lc}_raw.dat")
        run_path = os.path.join(outdir, f"run_zipf{theta_tag}_{lc}_{rc}_raw.dat")

        os.makedirs(outdir, exist_ok=True)

        # Load keys: 0..load_count-1, Fisher-Yates shuffled
        if not check_existing(load_path, load_count):
            print(f"  Generating {load_count:,} load keys (shuffled 0..{load_count-1}) ...")
            rng = np.random.default_rng(seed)
            keys = np.arange(load_count, dtype=np.uint64)
            rng.shuffle(keys)
            write_keys(load_path, keys)
            del keys
            size_gb = os.path.getsize(load_path) / 1e9
            print(f"  -> {load_path}  ({size_gb:.2f} GB)")

        # Run keys: Zipfian(theta) over [0, load_count)
        if not check_existing(run_path, run_count):
            # Advance the seed past the shuffle state. Use a separate seed
            # derived from the main seed for the run keys, matching gen_keys.cc
            # behavior where the same mt19937_64 is shared.
            zipf = ZipfSampler(load_count, theta, seed + 1,
                               scramble=scramble, perm_seed=perm_seed)

            chunk_size = 10_000_000  # 10M keys per chunk
            def chunks():
                remaining = run_count
                while remaining > 0:
                    n = min(chunk_size, remaining)
                    yield zipf.sample(n)
                    remaining -= n

            scramble_tag = "scrambled" if scramble else "ordered (rank=identity)"
            print(f"  Generating {run_count:,} Zipfian(theta={theta}, {scramble_tag}) run keys ...")
            write_keys_chunked(run_path, chunks(), run_count)

        trace_name = f"zipf{theta_tag}_{lc}"
        meta = TraceMetadata(
            name=trace_name,
            category="ycsb",
            generator="ZipfGenerator",
            description=f"YCSB Zipfian theta={theta} ({'scrambled' if scramble else 'ordered'}), {lc} load, {rc} run",
            load_file=os.path.abspath(load_path),
            run_file=os.path.abspath(run_path),
            load_count=load_count,
            run_count=run_count,
            params={"theta": theta, "seed": seed,
                    "scramble": scramble,
                    "perm_seed": effective_perm_seed if scramble else None,
                    "load_count": load_count, "run_count": run_count},
        )
        meta.save(outdir)
        return meta


class UniformGenerator(TraceGenerator):
    name = "uniform"
    category = "ycsb"
    description = "Uniform random distribution"

    def add_args(self, parser) -> None:
        parser.add_argument("--load-count", type=int, default=DEFAULT_LOAD_COUNT,
                            help="Number of load keys")
        parser.add_argument("--run-count", type=int, default=DEFAULT_RUN_COUNT,
                            help="Number of run keys")
        parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                            help="PRNG seed")
        parser.add_argument("--outdir", default=DEFAULT_OUTDIR,
                            help="Output directory")

    def generate(self, outdir: str, args) -> TraceMetadata:
        load_count = args.load_count
        run_count = args.run_count
        seed = args.seed

        lc = human_count(load_count)
        rc = human_count(run_count)
        load_path = os.path.join(outdir, f"load_uniform_{lc}_raw.dat")
        run_path = os.path.join(outdir, f"run_uniform_{lc}_{rc}_raw.dat")

        os.makedirs(outdir, exist_ok=True)

        if not check_existing(load_path, load_count):
            print(f"  Generating {load_count:,} load keys (shuffled) ...")
            rng = np.random.default_rng(seed)
            keys = np.arange(load_count, dtype=np.uint64)
            rng.shuffle(keys)
            write_keys(load_path, keys)
            del keys
            size_gb = os.path.getsize(load_path) / 1e9
            print(f"  -> {load_path}  ({size_gb:.2f} GB)")

        if not check_existing(run_path, run_count):
            rng = np.random.default_rng(seed + 1)
            chunk_size = 10_000_000

            def chunks():
                remaining = run_count
                nonlocal rng
                while remaining > 0:
                    n = min(chunk_size, remaining)
                    yield rng.integers(0, load_count, size=n, dtype=np.uint64)
                    remaining -= n

            print(f"  Generating {run_count:,} uniform run keys ...")
            write_keys_chunked(run_path, chunks(), run_count)

        meta = TraceMetadata(
            name=f"uniform_{lc}",
            category="ycsb",
            generator="UniformGenerator",
            description=f"YCSB Uniform, {lc} load, {rc} run",
            load_file=os.path.abspath(load_path),
            run_file=os.path.abspath(run_path),
            load_count=load_count,
            run_count=run_count,
            params={"seed": seed, "load_count": load_count, "run_count": run_count},
        )
        meta.save(outdir)
        return meta
