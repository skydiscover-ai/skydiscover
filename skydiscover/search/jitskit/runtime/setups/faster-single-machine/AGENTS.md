# SkyKV — Agent Instructions

You are implementing a high-performance concurrent key-value store from scratch.
Your goal is to **beat** the native FASTER system (SIGMOD'18) on throughput while
satisfying FASTER's crash recovery guarantee (the monotonicity property).

## Before You Start

Read and deeply understand these files:

- `setups/faster-single-machine/cards/system_card.json` — API, consistency guarantees, optimization objective
- `setups/faster-single-machine/cards/trace_card.json` — dataset and workload parameters
- `interface/kvstore_interface.h` — The `IKVStore` abstract class you must implement

Think carefully about what data structures, concurrency strategies, and memory layouts
you need. Consider the tradeoffs before writing code. If you need more context, look up
relevant research — papers, blog posts, or reference implementations.

## Rules

**Do NOT modify any file outside `interface/generated/`.** The following are read-only:

- `interface/kvstore_interface.h` — the interface contract
- `interface/benchmark_harness.cc` — the performance benchmark
- `interface/consistency_harness.cc` — the correctness tests
- `interface/CMakeLists.txt` — the build configuration
- `agent-pipeline/analyze.py` — the evaluation script
- `cards/` — all spec cards

## What to Implement

Write `interface/generated/kvstore_impl.cc`:

1. `#include "../kvstore_interface.h"` at the top
2. Class inheriting `IKVStore`, implementing all pure virtual methods
3. Factory: `IKVStore* create_kvstore() { return new YourClass(); }` (no `extern "C"`)
4. C++17 + `<pthread.h>` only — no external libraries
5. `Read`, `Upsert`, `RMW` must be thread-safe
6. Concurrent `RMW` on the same key must be atomic (no lost updates)
7. Must retain all 250M loaded keys
8. Cache decisions must adapt to runtime access patterns. Do not hardcode which keys are hot, and do not bias the layout toward the benchmark's specific key distribution.

## Build

```bash
cd interface && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) kvstore_bench consistency_test
```

## Evaluation Pipeline

**Correctness is a gate.** Do not run performance benchmarks until correctness passes.

```bash
LOAD=/mnt/ssd/ycsb_data/load_zipf_250M_raw.dat
RUN=/mnt/ssd/ycsb_data/run_zipf_250M_1000M_raw.dat
STORAGE=/mnt/ssd/kvstore_data
MEM_BUDGET=$((4 * 1024 * 1024 * 1024))  # 4 GB in-memory budget
```

### Stage 1: Correctness (MUST PASS before Stage 2)

These tests verify FASTER's core crash recovery guarantee (SIGMOD'18, Section 6.5):
for any two updates r1, r2 issued in order by a thread, the recovered state after
a crash includes {none, r1 only, both} — never r2 without r1. The guarantee applies
to ALL update types (RMW and Upsert). Reads must never return torn/corrupted values.

```bash
# Test 1: Crash Recovery (Fuzzy Checkpoint) — RMW writers active during Checkpoint()
numactl --cpunodebind=0 --membind=0 ./build/consistency_test 1 8 $LOAD $MEM_BUDGET $STORAGE

# Test 2: Crash Recovery (No Checkpoint) — RMW, crash without explicit checkpoint
numactl --cpunodebind=0 --membind=0 ./build/consistency_test 2 8 $LOAD $MEM_BUDGET $STORAGE

# Test 3: Upsert Crash Recovery (Fuzzy Checkpoint) — Upsert writers during Checkpoint()
numactl --cpunodebind=0 --membind=0 ./build/consistency_test 3 8 $LOAD $MEM_BUDGET $STORAGE

# Test 4: Upsert Crash Recovery (No Checkpoint) — Upsert, crash without checkpoint
numactl --cpunodebind=0 --membind=0 ./build/consistency_test 4 8 $LOAD $MEM_BUDGET $STORAGE

# Test 5: Read Correctness — concurrent reads must not see torn values during RMW
numactl --cpunodebind=0 --membind=0 ./build/consistency_test 5 8 $LOAD $MEM_BUDGET $STORAGE
```

If any test prints `FAILED`, fix the implementation and re-test.

### Stage 2: Performance

Benchmark the target workload at the specified thread counts.
Workload IDs: 0=YCSB-A, 1=RMW_100, 2=YCSB-B, 3=YCSB-C, 4=W_0_100,
              5=TIMESERIES_HD (time-series head-delete).

**Workload 5 (time-series head-delete)** has a stricter Delete contract:
- Keys are monotone timestamps. Inserts append at tail, deletes only
  remove from head.
- After the run, the harness samples ~16K keys from `[0, head)` and
  ~16K from `[head, tail)` and calls `Read()` on each. Deleted keys
  MUST NOT be readable; live keys MUST be readable. Any mismatch is an
  integrity failure and throughput is rejected.
- Delete rate is controlled by env `KVSTORE_DELETE_RATE` (default 1.0 =
  stationary window). The run-key file is procedural (placeholder); the
  harness owns atomic head/tail counters and the key stream. Do not
  bypass `Delete()` to chase Mops/s.

```bash
# Usage: ./build/kvstore_bench <workload_id> <threads> $LOAD $RUN
KVSTORE_VALIDATE_LOAD_KEYS=0 numactl --cpunodebind=0 --membind=0 \
  ./build/kvstore_bench <WL_ID> <THREADS> $LOAD $RUN
```

## Iteration Strategy

1. Build -> correctness tests -> fix until all pass
2. Benchmark at low thread count first (fast feedback)
3. If performance is poor, rethink the design — don't just tweak parameters
4. Keep iterating until you exceed FASTER on the target workload

## C++ Reminders

- `std::mutex` is NOT copyable/movable — use `new[]` or `unique_ptr<T[]>`
- `std::atomic<T>` requires trivially copyable T
