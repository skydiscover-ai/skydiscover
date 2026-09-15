# Task: single-node key-value store

Build a single-node, embedded key-value store for a dataset larger than memory. It runs under a
YCSB-style read/write workload and must be fast on the benchmark and correct as a production store:
every acknowledged write is durable and reads back its exact bytes, memory stays within the given
budget, keys are arbitrary (not just this trace's dense integers), and under pressure it evicts or
rejects, never silently drops, corrupts, or misfiles acknowledged data.

Tune it for the workload below, but it must stay correct on any other value size, key distribution,
or read/write mix, and past the end of the benchmark window; being slower there is fine, being
wrong is not.

Paths are relative to this directory.

## Score
- Workload: 4 KB values, 50:50 read/write, 8 GB memory budget, zipf θ=0.99, 16 threads.
- Metric: throughput (Mops/s) over a 30-second timed run, after loading the dataset.
- Only implementations that pass every test are benchmarked. Correctness is pass/fail, not part of
  the score.
- The dataset is larger than memory, so the benchmark runs on a Linux machine with a local SSD
  (see the README).

## Interface
- Header: `evaluator/kvstore_interface.h`. Implement the full `IKVStore` interface.
- Entry point: `IKVStore* create_kvstore()`.

## Build and benchmark
- `evaluator/benchmark_harness.cc` measures throughput; `evaluator/consistency_harness.cc` checks
  crash recovery and concurrent reads at full dataset size. `evaluator/CMakeLists.txt` builds a
  candidate with either; `evaluator/generate.py` generates the workload.
- `IMPL=<file> bash evaluator/score.sh` builds one implementation and runs both harnesses. Without
  `IMPL` it scores the reference.

## Testing
- Start from the four correctness tests in `evaluator/tests/`: exact bytes, bounded memory, crash
  durability, delete durability. Copy them and their `test.sh` into the run's suite and admit
  each with `validate_test.py --seed`; they need no mutant. Write further tests in the same shape.
- A test you write must pass `evaluator/reference_kvstore.cc` and fail a broken copy of it that you
  write with exactly the defect the test is for (`validate_test.py --mutant`).
- There is no `Recover()`, and `Checkpoint()` returns void. To test recovery, destroy the store and
  create a fresh one with `create_kvstore()` + `InitExtended()` on the same storage path. To test a
  hard crash, fork a child that writes and checkpoints, kill it, and recover in the parent.
- Values go up to the maximum value size, and a read must return the exact bytes written: put a
  random nonce in the value and check it round-trips.

## What you are given
- `spec/requirements.json`, `spec/workload.json`: hardware and compiler facts, the operating
  point, and the workload the run's own specification builds on.
- `evaluator/kvstore_interface.h`: the interface.
- `evaluator/reference_kvstore.cc`: a correct implementation.
- `evaluator/tests/`: four correctness tests and the `test.sh` that builds and runs them.
- `evaluator/benchmark_harness.cc`, `consistency_harness.cc`, `generate.py`, `CMakeLists.txt`,
  `score.sh`: the benchmark, the crash-recovery check, the workload, the build, and the one script
  that runs them.
