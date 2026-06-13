# Setup: FASTER Single-Machine

Evaluation setup based on [FASTER: A Concurrent Key-Value Store with In-Place Updates](https://www.microsoft.com/en-us/research/publication/faster-a-concurrent-key-value-store-with-in-place-updates/) (Chandramouli et al., SIGMOD 2018).

## Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EVOLUTION LOOP (per iteration)                      │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ 1. PROMPT CONSTRUCTION                                               │   │
│  │                                                                      │   │
│  │  Inputs:                                                             │   │
│  │    - cards/system_card.json     (API, monotonicity property,         │   │
│  │                                  read correctness, memory constraint)│   │
│  │    - cards/trace_card.json      (250M keys, value size, workload,    │   │
│  │                                  distribution -- patched at runtime) │   │
│  │    - interface/kvstore_interface.h  (IKVStore abstract class)        │   │
│  │    - Previous benchmark feedback (Mops/s, per-op breakdown,          │   │
│  │                                   % of baseline, memory usage)       │   │
│  │                                                                      │   │
│  │  Output: text prompt for the agent                                   │   │
│  └──────────────────────────┬───────────────────────────────────────────┘   │
│                             ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ 2. AGENT (Claude Code or Codex)                                      │   │
│  │                                                                      │   │
│  │  Input:  prompt + workspace (cards, interface, previous impl)        │   │
│  │  Action: reads specs, researches, writes/improves kvstore_impl.cc    │   │
│  │  Output: interface/generated/kvstore_impl.cc  (C++17, single file)   │   │
│  └──────────────────────────┬───────────────────────────────────────────┘   │
│                             ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ 3. BUILD                                                             │   │
│  │                                                                      │   │
│  │  Command: cmake + make -j                                            │   │
│  │  Produces: kvstore_bench, consistency_test binaries                  │   │
│  │  On failure: BUILD_FAILED → feedback to agent, skip to next iter     │   │
│  └──────────────────────────┬───────────────────────────────────────────┘   │
│                             ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ 4. CORRECTNESS (evaluators/consistency_harness.cc)                   │   │
│  │                                                                      │   │
│  │  Test 1: Crash recovery monotonicity (fuzzy checkpoint)              │   │
│  │    - Writers do paired RMWs DURING Checkpoint()                      │   │
│  │    - Crash + recover → verify: for each pair (r1,r2) issued in      │   │
│  │      order, recovered state is {none, r1 only, both} -- never       │   │
│  │      r2 without r1                                                   │   │
│  │                                                                      │   │
│  │  Test 2: Crash recovery monotonicity (no checkpoint)                 │   │
│  │    - Paired writes, then crash WITHOUT calling Checkpoint()          │   │
│  │    - Same monotonicity verification                                  │   │
│  │                                                                      │   │
│  │  On failure: CORRECTNESS_FAILED → feedback to agent, skip benchmark  │   │
│  └──────────────────────────┬───────────────────────────────────────────┘   │
│                             ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ 5. BENCHMARK (interface/benchmark_harness.cc)                        │   │
│  │                                                                      │   │
│  │  For each (memory_budget, thread_count):                             │   │
│  │    - Set cgroup v2 memory limit (if budget > 0)                      │   │
│  │    - Load 250M keys with nonce-based validation                      │   │
│  │    - Run 30-second timed workload (50:50, RMW, 100:0, or 0:100)     │   │
│  │    - Measure: total Mops/s, per-op breakdown (reads/upserts/rmws),   │   │
│  │              load time, validation time, store memory usage           │   │
│  │    - Compare against baseline (FASTER) at same budget                │   │
│  │                                                                      │   │
│  │  Output per run: wlN_tM_memXg.txt with all metrics                  │   │
│  └──────────────────────────┬───────────────────────────────────────────┘   │
│                             ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ 6. FEEDBACK + CHECKPOINT                                             │   │
│  │                                                                      │   │
│  │  - agent-pipeline/format_feedback.py parses raw output → structured feedback  │   │
│  │  - Track best Mops/s, save best_impl.cc to runs/              │   │
│  │  - Both backends retain full session context across iterations        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Repeat for N iterations                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## What the agent sees (example feedback for iteration 5)

```
Read AGENTS.md for the task specification.

You have 50 tool calls for this iteration. Use them wisely.

## Iteration 5 -- Improve Your Implementation

Your previous implementation (642 lines) is at interface/generated/kvstore_impl.cc.

## Benchmark Results (50:50)
  FASTER baseline: mem=8GB → 0.86 Mops/s
  FASTER baseline: mem=32GB → 54.48 Mops/s
  [mem=8GB]
  - t=16: 0.57 Mops/s (35625 ops/s/thread, load=28.1s @ 8.9M keys/s, validate=160.3s, store_mem=6.12/8.00 GB, 66.3% of FASTER)
    ops: reads=0.29 upserts=0.28 rmws=0.00 Mops/s
  [mem=32GB]
  - t=16: 3.98 Mops/s (248750 ops/s/thread, load=19.5s @ 12.8M keys/s, validate=5.1s, store_mem=27.4/32.0 GB, 7.3% of FASTER)
    ops: reads=1.99 upserts=1.99 rmws=0.00 Mops/s

## Build & Correctness
cd interface/build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc) kvstore_bench consistency_test
numactl --cpunodebind=0 --membind=0 ./build/consistency_test 1 8 /mnt/ssd/ycsb_data/load_zipf_250M_raw.dat 4294967296 /mnt/ssd/kvstore_consistency_...
numactl --cpunodebind=0 --membind=0 ./build/consistency_test 2 8 /mnt/ssd/ycsb_data/load_zipf_250M_raw.dat 4294967296 /mnt/ssd/kvstore_consistency_...

Fix any failures. Do NOT run benchmarks.

IMPORTANT: Correctness is a GATE. Both crash recovery tests MUST pass.
```

## Contents

| Directory | What |
|-----------|------|
| `cards/` | System card (API, guarantees), trace card (workload), design hints |
| `evaluators/` | Correctness tests: monotonicity property (crash recovery), read correctness |
| `baselines/` | FASTER reference numbers per memory budget |
| `scripts/` | Convenience wrappers: `run_ltm_8gb_only.sh`, `run_all.sh`, etc. |
| `logs/` | Experiment results organized by experimenter |
| `CLAUDE.md` | Agent instruction template (Claude backend) |
| `AGENTS.md` | Agent instruction template (Codex backend) |

## System properties tested

Each evaluator test verifies a specific property from the paper.

| Property | Evaluator test | Paper reference | Paper wording |
|----------|---------------|-----------------|---------------|
| Monotonicity property | Test 1: fuzzy checkpoint (RMW) | Section 6.5, "Recovery and Consistency in Faster" | "For any two update requests r1 and r2 issued (in order) by a thread, the state after recovery includes the effects of (1) none; (2) only r1; or (3) both r1 and r2. In other words, the state after recovery cannot include the effects of r2 without also including r1." |
| Monotonicity property | Test 2: no checkpoint (RMW) | Section 6.5 | Same property, but crash without explicit Checkpoint() |
| Monotonicity property | Test 3: fuzzy checkpoint (Upsert) | Section 6.5 + Table 2 | Same property for blind updates, which follow a different update path (Table 2) |
| Monotonicity property | Test 4: no checkpoint (Upsert) | Section 6.5 + Table 2 | Same, no checkpoint |
| Read correctness | Test 5: concurrent reads during writes | Section 4 | "user threads read and modify record values in the safety of epoch protection" -- reads must never observe torn/corrupted values |

## Agent session model

| Backend | Session | How context is maintained |
|---------|---------|--------------------------|
| Claude | Persistent (`--continue`) | Agent retains full conversation context across iterations. |
| Codex | Persistent (`codex exec resume <thread_id>`) | First iteration captures `thread_id` from `--json` output. Subsequent iterations resume with full context. Stress tested to 1.13M tokens. |

**Codex compaction requirement**: Do NOT set `model_context_window` or `model_auto_compact_token_limit` in `~/.codex/config.toml`. With explicit custom values, auto-compaction breaks -- it only checks token count between turns, not between tool calls within a turn ([openai/codex#16033](https://github.com/openai/codex/issues/16033)). With defaults (256K), compaction works correctly and sessions can run for 100+ iterations.

## How to run

```bash
# Quick: 8GB budget, 100 iterations
bash setups/faster-single-machine/scripts/run_ltm_8gb_only.sh

# Full control:
bash agent-pipeline/run.sh \
  --backend codex \
  --mode ltm \
  --distribution zipf \
  --setup 50:50 \
  --iterations 100 \
  --model gpt-5.4
```
