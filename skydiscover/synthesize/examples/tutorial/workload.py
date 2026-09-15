"""The tutorial workload and its scorer.

An application hammers ~100 hot keys, but every 120 operations a batch job scans 140 one-time keys
through a 128-entry cache. replay(create_cache) scores the hit rate on the application's own
traffic; the cache must expose get, put, and size and never exceed capacity. The trace is generated
by seed here; replay(create_cache, trace="my_trace.csv") takes your own key,kind export
(python workload.py --export trace.csv shows the shape).
"""

import csv
import random
from collections import OrderedDict

CAPACITY = 128
HOT_KEYS = 100
TAIL_KEYS = 500
BLOCK_OPS = 120
SCAN_OPS = 140
BLOCKS = 55
SEED = 7


def make_trace(seed=SEED):
    """Fixed, seeded list of (key, kind) pairs; kind is 'app' or 'scan'."""
    rng = random.Random(seed)
    hot = [f"user:{i}" for i in range(HOT_KEYS)]
    hot_w = [1.0 / (i + 1) ** 0.1 for i in range(HOT_KEYS)]
    tail = [f"item:{i}" for i in range(TAIL_KEYS)]
    tail_w = [1.0 / (i + 1) ** 0.8 for i in range(TAIL_KEYS)]
    trace, scan_id = [], 0
    for _ in range(BLOCKS):
        for _ in range(BLOCK_OPS):
            if rng.random() < 0.95:
                trace.append((rng.choices(hot, hot_w)[0], "app"))
            else:
                trace.append((rng.choices(tail, tail_w)[0], "app"))
        for i in range(SCAN_OPS):
            trace.append((f"scan:{scan_id}:{i}", "scan"))
        scan_id += 1
    return trace


def load_trace(path):
    """Load a trace exported from your own system: one key,kind row per access."""
    with open(path, newline="") as fh:
        return [(row[0], row[1]) for row in csv.reader(fh) if len(row) >= 2]


def replay(create_cache, capacity=CAPACITY, seed=SEED, trace=None):
    """Replay a trace against a cache and return application-traffic hit rate.

    trace may be a path to a key,kind CSV (your own export) or a list of
    (key, kind) pairs; by default the generated trace for seed is used.
    """
    if isinstance(trace, str):
        trace = load_trace(trace)
    cache = create_cache(capacity)
    app_hits = app_ops = 0
    for step, (key, kind) in enumerate(trace if trace is not None else make_trace(seed)):
        hit = cache.get(key) == ("v", key)
        if kind == "app":
            app_ops += 1
            app_hits += hit
        if not hit:
            cache.put(key, ("v", key))
        if cache.size() > capacity:
            raise AssertionError(f"cache exceeded capacity at op {step}")
    return {"app_hit_rate": round(app_hits / app_ops, 4), "app_ops": app_ops}


class FIFO:
    """The tutorial baseline, also importable by generated benchmark scripts."""

    def __init__(self, capacity):
        self.capacity, self.data = capacity, OrderedDict()

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        if key not in self.data and len(self.data) >= self.capacity:
            self.data.popitem(last=False)
        self.data[key] = value

    def size(self):
        return len(self.data)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "--export":
        with open(sys.argv[2], "w", newline="") as fh:
            csv.writer(fh).writerows(make_trace())
        print(f"wrote {sys.argv[2]} ({len(make_trace())} accesses, one key,kind row each)")
        sys.exit(0)

    print("FIFO baseline:", replay(FIFO))
