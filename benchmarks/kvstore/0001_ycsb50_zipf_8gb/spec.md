# Task 0001 — `ycsb50_zipf_8gb`

**Goal:** synthesize a correct, high-throughput single-node key-value store specialized to the spec
below, and beat the tuned baseline.

## API contract

Implement `IKVStore` (see `_harness/kvstore_interface.h`) and `create_kvstore()`:

| op | semantic |
|----|----------|
| `Read(key, out)` | point read; returns the bytes of the last `Upsert`/`RMW` for `key` |
| `Upsert(key, value)` | blind insert-or-overwrite |
| `RMW(key, mod, n)` | atomic read-modify-write (integer-add counter semantic) |
| `Delete(key)` | remove `key` |

All ops except `Init`/`StartSession`/`StopSession` must be thread-safe. C++17 + POSIX threads only;
no external dependencies; written from first principles.

## Correctness property (HARD GATE)

`get(k) == last_put(k)` — a read returns exactly the bytes (and size) most recently written, with no
torn reads and no lost updates under concurrency (paper §2, §3.1), plus crash-recovery/durability
matched to the requirement card. The evaluator runs **6 consistency tests**; **any** failure scores
`0` regardless of throughput.

## Workload signature

| field | value |
|-------|-------|
| mix | **50:50** read / upsert |
| distribution | **Zipf(0.99)** |
| value size | 100 bytes |
| key count | 250,000,000 |
| memory budget | **8 GB** |
| threads | 16 (pinned to one socket) |
| baseline to beat | **FASTER** (most-favorable published config) |

## Scoring

`combined_score` = **peak throughput (Mops/s)** across all iterations, **after** the correctness gate
passes (else `0`, `validity: 0`). Leading indicators (cache-hit ratio, eviction rate, disk IOPS,
p50/p99/p999 latency, memory utilization) flow into `metrics` for the dashboard.

## Environment tiers

- **hardware** (`search.database.mode: ltm`) — bare-metal NVMe at the scratch dir, NUMA, cgroup memory
  cap, `perf`. The faithful path for this 8 GB budget.
- **portable** (`search.database.mode: inmem`) — in-memory budget, runs anywhere; for quickstart/CI
  with a smaller representative spec.

Baselines for this hardware tier are in `../_baselines/results.json` (hardware-tagged).
