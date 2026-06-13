"""Base classes and utilities for trace generators.

All generators produce the same binary format: flat arrays of little-endian
uint64_t keys.  The benchmark harness auto-detects key counts from file size.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

# ── Constants (shared across generators) ─────────────────────────────────────
DEFAULT_LOAD_COUNT = 250_000_000   # 250M (matches FASTER SIGMOD'18)
DEFAULT_RUN_COUNT  = 1_000_000_000 # 1B
DEFAULT_SEED       = 211           # matches FASTER benchmark
DEFAULT_OUTDIR     = "/mnt/ssd/ycsb_data"


@dataclass
class TraceMetadata:
    """Metadata for a generated trace pair, written as a sidecar .meta.json."""

    name: str                   # e.g. "zipf_t099_250M"
    category: str               # "ycsb", "adversarial", "real"
    generator: str              # class name, e.g. "ZipfGenerator"
    description: str            # human-readable one-liner
    load_file: str              # absolute path to load_*.dat
    run_file: str               # absolute path to run_*.dat
    load_count: int             # keys in load file
    run_count: int              # keys in run file
    params: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    file_format: str = "uint64_le"

    def save(self, outdir: str) -> str:
        """Write metadata to <outdir>/<name>.meta.json.  Returns path."""
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        path = os.path.join(outdir, f"{self.name}.meta.json")
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "TraceMetadata":
        with open(path) as f:
            return cls(**json.load(f))


class TraceGenerator(ABC):
    """Abstract base for all trace generators."""

    name: str          # CLI subcommand name
    category: str      # "ycsb", "adversarial", "real"
    description: str   # one-line help text

    @abstractmethod
    def add_args(self, parser) -> None:
        """Add generator-specific CLI arguments to an argparse subparser."""

    @abstractmethod
    def generate(self, outdir: str, args) -> TraceMetadata:
        """Generate trace files in outdir, return metadata."""


# ── Shared helpers ────────────────────────────────────────────────────────────

def write_keys(path: str, keys: np.ndarray) -> None:
    """Write a uint64 numpy array to a binary file."""
    keys.astype(np.uint64).tofile(path)


def write_keys_chunked(path: str, chunk_iter, total: int) -> None:
    """Stream chunks of uint64 keys to a binary file with progress."""
    written = 0
    t0 = time.monotonic()
    with open(path, "wb") as f:
        for chunk in chunk_iter:
            chunk.astype(np.uint64).tofile(f)
            written += len(chunk)
            elapsed = time.monotonic() - t0
            if elapsed > 5.0 and written < total:
                rate = written / elapsed / 1e6
                print(f"    {written/1e6:.0f}M / {total/1e6:.0f}M keys "
                      f"({rate:.1f}M keys/s)", flush=True)
    dt = time.monotonic() - t0
    size_gb = os.path.getsize(path) / 1e9
    print(f"  -> {path}  ({size_gb:.2f} GB, {dt:.1f}s)")


def human_count(n: int) -> str:
    """Format a key count for filenames: 250000000 -> '250M'."""
    if n >= 1_000_000_000 and n % 1_000_000_000 == 0:
        return f"{n // 1_000_000_000}G"
    if n >= 1_000_000 and n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    if n >= 1_000 and n % 1_000 == 0:
        return f"{n // 1_000}K"
    return str(n)


def check_existing(path: str, expected_count: int) -> bool:
    """Return True if file exists with the expected key count."""
    if not os.path.exists(path):
        return False
    actual = os.path.getsize(path) // 8
    if actual == expected_count:
        print(f"  already exists: {path} ({actual:,} keys)")
        return True
    return False
