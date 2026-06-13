"""Trace generator registry.

To add a new generator:
  1. Create a module in this package (e.g., burstiness.py)
  2. Define a class inheriting TraceGenerator
  3. Import and add it to _ALL_GENERATORS below
"""

from .adversarial import BeladyGenerator, ScanGenerator, StrideGenerator
from .base import TraceGenerator, TraceMetadata
from .real import (
    MetaKVGenerator,
    TencentPhotoGenerator,
    TwitterGenerator,
    WikimediaGenerator,
)
from .synthetic import (
    BimodalGenerator,
    BurstyGenerator,
    HotspotGenerator,
    OneHitWonderGenerator,
)
from .timeseries import TimeseriesHDGenerator
from .ycsb import UniformGenerator, ZipfGenerator

_ALL_GENERATORS: list[TraceGenerator] = [
    # YCSB (standard benchmarks)
    ZipfGenerator(),
    UniformGenerator(),
    # Adversarial (exploit FIFO weaknesses)
    ScanGenerator(),
    BeladyGenerator(),
    StrideGenerator(),
    # Synthetic knobs (Zipfian + one dimension varied)
    OneHitWonderGenerator(),
    BurstyGenerator(),
    HotspotGenerator(),
    BimodalGenerator(),
    # Time-series head-delete (procedural; delete rate is a harness env)
    TimeseriesHDGenerator(),
    # Real-world
    MetaKVGenerator(),
    TwitterGenerator(),
    WikimediaGenerator(),
    TencentPhotoGenerator(),
]

GENERATORS: dict[str, TraceGenerator] = {g.name: g for g in _ALL_GENERATORS}

__all__ = ["GENERATORS", "TraceGenerator", "TraceMetadata"]
