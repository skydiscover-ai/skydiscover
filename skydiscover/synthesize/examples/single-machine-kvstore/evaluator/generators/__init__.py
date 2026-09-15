"""Trace generator registry. To add one: subclass TraceGenerator in a new module and list it in
_ALL_GENERATORS.
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
