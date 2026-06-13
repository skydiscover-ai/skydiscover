# SkyKV: LLM-Generated KV Store Benchmark

An evaluation framework that tasks LLM coding agents (Claude, Codex) with
building a high-performance concurrent key-value store **from scratch** and
benchmarks it against [FASTER](https://www.microsoft.com/en-us/research/publication/faster-a-concurrent-key-value-store-with-in-place-updates/)
(SIGMOD'18) under identical conditions.

## Quick Start

```bash
# 1. Generate traces
python3 traces/generate.py zipf --outdir /mnt/ssd/ycsb_data

# 2. Run the evolution loop
bash agent-pipeline/run.sh --backend claude --mode ltm --distribution zipf \
    --setup 50:50 --iterations 50
```

## Project Structure

```
skykv-claude/
├── agent-pipeline/              # Core agent evolution loop
│   ├── run.sh                   #   CLI entry point
│   ├── orchestrator.py          #   Build → test → bench → feedback loop
│   ├── critique.py              #   Review critique agent
│   ├── critique_audit.py        #   Adversarial audit (test gate)
│   ├── format_feedback.py       #   Benchmark → agent feedback
│   ├── memory.py                #   Codex session rotation
│   ├── checkpoint.py            #   Leaderboard tracking
│   └── prompts/                 #   Prompt templates
│
├── traces/                      # Trace generation & analysis
│   ├── generate.py              #   Unified CLI for all trace types
│   ├── sweep.py                 #   Experiment sweep from YAML
│   ├── list.py                  #   List available traces
│   ├── generators/              #   ycsb, adversarial, synthetic, real
│   ├── belady/                  #   Belady optimal cache simulator
│   └── workloads/               #   Sweep configs (YAML)
│
├── interface/                   # Shared C++ harness (read-only for agent)
│   ├── kvstore_interface.h      #   IKVStore + CacheStats
│   ├── benchmark_harness.cc     #   Benchmark (auto-detects trace sizes)
│   └── generated/               #   Agent writes kvstore_impl.cc here
│
├── setups/faster-single-machine/
│   ├── cards/                   #   system_card.json, trace_card.json
│   ├── baselines/               #   FASTER reference numbers
│   ├── seeds_faster/            #   FASTER adapter (vendored source)
│   └── scripts/                 #   Experiment launcher scripts
│
├── runs/                        # Experiment results and logs
└── unit-tests/                  # Pipeline unit tests
```

## Pipeline

```
agent-pipeline/run.sh --backend claude --mode ltm --setup 50:50 --distribution zipf
  │
  ▼
orchestrator.py: for each iteration:
  1. PROMPT       (cards + feedback + critique)
  2. GENERATOR    (Claude/Codex → kvstore_impl.cc)
  3. BUILD        (cmake + make)
  4. CORRECTNESS  (5 crash-recovery tests)
  4b. AUDIT GATE  (adversarial tests)
  5. BENCHMARK    (Mops/s, cache hit ratio)
  6. TRACK BEST   (best_impl.cc, leaderboard)
  7. REVIEWER     (feedback → next prompt)
  8. AUDITOR      (write new tests → gate)
```

## Async Read

**The evaluation default is the async path** — any `kvstore_bench` invocation pipelines Reads through `IKVStore::ReadAsync` unless you explicitly opt out. Two paths exist:

> For the design rationale and how this maps to FASTER's paper-level async contract (Context*, CompletionCallback, CompletePending, pending queue), see **[docs/ASYNC_READ_DESIGN.md](docs/ASYNC_READ_DESIGN.md)**.


- `KVSTORE_ASYNC_EVAL=1` **(default, no env needed)** — Reads flow through `IKVStore::ReadAsync` + a per-worker ring (size `KVSTORE_PIPELINE_DEPTH`, default 256). Stores that override `ReadAsync` pipeline real I/O across up to 256 outstanding Reads per thread; stores that don't get a transparent sync-wrap.
- `KVSTORE_ASYNC_EVAL=0` — Reads go through `bool Read()` directly. No ring, no atomics, no pipeline. Matches pre-pipeline behavior for apples-to-apples comparison with older numbers.

The startup banner prints which path is active. `Upsert`/`RMW`/`Delete` stay sync — the paper shows only Read consistently goes Pending on LTM.

**What agents need to know.** `bool Read(...)` is still pure-virtual and required. `ReadAsync` / `CompletePending` are **optional overrides, and they're a pair — override both or neither:**

- **Pure in-memory store** (hash table, trie, etc.): do nothing. The defaults — sync-wrap `ReadAsync` + no-op `CompletePending` — work correctly. Your `Read` is called inline, nothing queues, nothing needs draining. Zero extra code.
- **Disk-tier store** (hybrid RAM + SSD with overlapping I/O): override both. `ReadAsync` submits I/O without blocking and can return `OpStatus::Pending`; `CompletePending(false)` walks your ready completions and fires each slot's `done=1`; `CompletePending(true)` blocks until the session has no outstanding I/O. If you override only `ReadAsync` and actually return `Pending`, the default no-op `CompletePending` won't drive your I/O — the harness harvest loop will spin on `slot->done` forever. That's the one failure mode to avoid.

**What `CompletePending(bool wait)` does.** It drains the thread's pending-ops queue. When your `ReadAsync` returns `OpStatus::Pending`, you've stashed the caller's `ReadSlot*` somewhere and kicked off I/O — but the completion (writing `slot->out` / `slot->status` / `slot->done=1`) has NOT happened yet. When the I/O eventually lands, your store should enqueue the completion; `CompletePending(false)` is the signal from the harness/caller to "now walk your ready-completions and fire them" (non-blocking — return fast even if some are still in-flight). `CompletePending(true)` blocks until every op submitted by this session has completed. Someone has to call this periodically: if nothing drains the queue, it grows without bound and internal backpressure stalls the store. The harness calls `CompletePending(false)` every 64 ops on the async path and every 1600 ops on the sync path, then `CompletePending(true)` at thread exit — mirroring native FASTER's benchmark cadence.

### Interface surface

```cpp
enum class OpStatus : uint8_t { Ok, NotFound, Pending };
struct ReadSlot { uint64_t key; GenValue out; std::atomic<uint8_t> done{0}; OpStatus status; void* user; };

class IKVStore {
    virtual bool Read(uint64_t, GenValue&) = 0;                            // unchanged
    virtual OpStatus ReadAsync(ReadSlot* slot);                            // default: wraps Read() sync
    virtual void CompletePending(bool wait);                               // default: no-op
    // Upsert / RMW / Delete / Init / Checkpoint / ... unchanged
};
```

Slot lifetime contract: caller keeps the slot alive until `ReadAsync` returns `Ok`/`NotFound`, or `slot->done.load(acquire) == 1`, or `CompletePending(true)` returns. See `interface/kvstore_interface.h` for the full contract.

### Validation — does our async match FASTER's async?

Representative fair test: 16 threads / YCSB-C (100% reads) / single NVMe / 208 B records (matched: our seed's Value vs native's `FASTER_VALUE_PADDING=200`) / **13 GB FASTER hlog memory matched across all three** (our seed deducts 3 GB from the 16 GB total budget; native is passed 13 GB directly) / **scrambled Zipf** trace (see Traces section) / OS page cache dropped between runs / fresh store dir per run:

| Config | Mops/s |
|---|---|
| A) Our harness + sync path (FASTER seed) | **1.27** |
| B) Our harness + async path (FASTER seed) | **1.35** |
| C) Native FASTER C++ (`benchmark_ltm200`, its own async design) | **1.35** |

**B ≈ C, to three significant figures.** At matched conditions the async eval path matches native FASTER — the framework adds no measurable tax. Sync path (A) is 6% slower than async, the expected benefit of overlapping disk I/O when storage saturates. At low thread counts where storage has slack the two paths cross and sync is faster by a few % (pipeline overhead dominates); toggle with `KVSTORE_ASYNC_EVAL=0` if your workload is in that regime.

The absolute gap to FASTER's paper-default 100 B records (~2.1 Mops/s at 16T) is pure compile-time record-size specialization by native, not framework overhead — confirmed by rebuilding native at 200 B records (shown above) and watching its number drop into line with ours.

#### Under memory pressure (cgroup-enforced, YCSB-A, 8 GB budget)

Same harness config, but stress-tested with cgroup v2 at 20 GB (= 8 GB FASTER budget + 12 GB harness/trace overhead, matching `eval_sweep.sh` / `fig10_runner.sh` convention). YCSB-A (50:50 read/upsert) triggers write-side Pending frequently because the hlog buffer is tight:

| Config | Mops/s |
|---|---|
| Our harness + async path (FASTER seed) | **0.66** |
| Native FASTER C++ | **0.70** |
| Gap | **6%** |

Findings from getting to this match:
- The seed's write paths (`Upsert`/`RMW`/`Delete`) used to call `store_->CompletePending(true)` inline on `Status::Pending`, which drained every in-flight op on the thread — flushing the pipelined Reads. Fixed: match native's lazy-drain pattern.
- `load_worker` never called `CompletePending` periodically during the load phase. Added it every 1600 ops, matching native's `thread_setup_store`.
- The checksum array (1 GB at 250 M keys × 4 B) was unconditionally allocated even with `KVSTORE_VALIDATE_LOAD_KEYS=0`. Fixed: conditional on the env flag.
- **Cross-thread auditor** (enabled for A/B/W_0_100 workloads) spawns a separate session that spins in our seed's sync `Read` wrapper (`CompletePending(false)` without `Refresh()`). Under cgroup-induced disk pressure, its session epoch freezes, and FASTER cannot safely evict hlog pages while any session is behind. The whole store stalls on safe-epoch waits — 14× throughput collapse. Toggle with `KVSTORE_AUDIT=0` for perf A/B; proper fix (inject `Refresh()` into the sync spin loop) is a follow-up.

### Running

```bash
# Async path (default)
interface/build/kvstore_bench 3 16 <load> <run> <budget> <storage>

# Sync path (rollback / matches pre-pipeline results)
KVSTORE_ASYNC_EVAL=0 interface/build/kvstore_bench ...

# Deeper/shallower pipeline, drain less often
KVSTORE_PIPELINE_DEPTH=1600 KVSTORE_COMPLETE_PENDING_INTERVAL=1600 \
    interface/build/kvstore_bench ...
```

## Traces

See **[traces/README.md](traces/README.md)** for full documentation.

**Zipf traces are scrambled by default.** The `ZipfSampler` applies a deterministic random permutation `perm[]` to its output, so hot keys are distributed uniformly across the key space rather than clustered at low integer values (which is a YCSB-sampler artifact, not a property of the Zipf distribution). This prevents implementations from exploiting key-identity assumptions like "keys < 64M are hot → flat array." Distribution shape, hotset size, and all spec-card parameters are preserved exactly. Opt out with `--no-scramble` on the generator to reproduce pre-scramble baselines. `perm_seed` is recorded in each trace's `meta.json`. Applies to all Zipf-backed generators (`zipf`, `onehit`, `bursty`, `hotspot`, `bimodal`); real traces (`metakv`, `twitter`) and adversarial traces are left untouched.

```bash
python3 traces/generate.py zipf --theta 0.5 --outdir /mnt/ssd/ycsb_data
python3 traces/list.py /mnt/ssd/ycsb_data
python3 traces/sweep.py traces/workloads/zipf_theta_sweep.yaml
python3 traces/belady/run.py /mnt/ssd/ycsb_data --mem-budget-gb 8 --value-size 100
```

## `agent-pipeline/run.sh` Flags

| Flag | Values | Description |
|------|--------|-------------|
| `--backend` | `claude` / `codex` | Agent to drive |
| `--mode` | `inmem` / `ltm` | In-memory or larger-than-memory |
| `--distribution` | `zipf`, `uniform`, `scan`, `belady`, `stride`, `metakv`, `twitter<N>` | Named trace |
| `--trace-load/run` | paths | Custom trace files (alt to `--distribution`) |
| `--setup` | `rmw`, `50:50`, `100:0`, `0:100`, `all` | Read/write mix |
| `--iterations` | int (50) | Evolution iterations |
| `--value-size` | bytes (8/100) | Value size |
| `--mem-budget` | GB (0/"8 32") | Memory budget |
| `--critique` | `off`/`review`/`audit`/`full` | Critique pipeline |
| `--seed` | name | Start from seed impl (e.g., `faster`) |
