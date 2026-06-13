# Feedback Mechanism

## Thesis

System profilers are the new "gradients." In gradient-free LLM-driven optimization, the KPI (Mops/s) is the loss function, but leading indicators (cache hit ratios, memory utilization, I/O bandwidth, latency percentiles, OS-level counters) provide the directional signal — the "gradient" — that guides the LLM to structurally rewrite code.

## Feedback Levels

`--feedback-level <minimal|rich>` (env: `SKYKV_FEEDBACK_LEVEL`)

### `minimal`

```
## Benchmark Results (RMW)
  [mem=8GB]
  - t=16: 11.05 Mops/s
```

### `rich` (default)

```
## Benchmark Results (RMW)
  [mem=8GB]
  - t=16: 45.20 Mops/s (2825000.00 ops/s/thread, load=32.5s @ 7.7M keys/s,
    store_mem=7.50 / 8.00 GB (94% of budget))
    ops: reads=0.00 upserts=0.00 rmws=45.20 Mops/s
    read_cache_hit: 0.00% (0 / 250000000)
    rmw_cache_hit: 95.20% (1285440000 / 1350720000)
    cache_hit_total: 80.34% (1285440000 / 1600720000)
    cache_size_ratio: 7.50 / 23.28 GB (32.2%)
    cache_budget_util: 7.50 / 8.00 GB (93.8%)
    disk_io: read_iops=12345 write_iops=6789 read_mb=482.3 write_mb=265.1 read_mb_s=16.1 write_mb_s=8.8
    op_latency: p50=0.31 p99=823.45 p999=4521.20 max=12345.67 us (n=29340)
    eviction_rate: 45000 evictions (1500.0 evictions/s)
    memory_util: 7.5 / 8 GB (94%), 0.5 GB unused
    perf: context_switches=88165, page_faults=2462610, major_faults=0, cpu_migrations=3
```

## Metrics Reference

| Category | Metric | Source | Level |
|----------|--------|--------|-------|
| **KPI** | `Mops/s`, `ops/s/thread` | benchmark_harness.cc | both |
| **Op mix** | `op_breakdown` (reads/upserts/rmws Mops/s) | benchmark_harness.cc | rich |
| **Cache** | `read_cache_hit`, `rmw_cache_hit`, `cache_hit_total` | `IKVStore::GetCacheStats()` | rich |
| **Cache budget** | `cache_size_ratio`, `cache_budget_util` | `IKVStore::GetCacheStats()` | rich |
| **Memory** | `store_mem`, `memory_util` | `/proc/self/status` | rich |
| **I/O** | `disk_io` (read/write IOPS, MB/s) | `/proc/diskstats` delta | rich |
| **Latency** | `op_latency` (p50/p99/p999/max us) | Harness sampling (1-in-1024) | rich |
| **Eviction** | `eviction_rate` (evictions/s) | `CacheStats::evictions` | rich |
| **OS** | `context_switches`, `page_faults`, `major_faults`, `cpu_migrations` | `perf stat` | rich |
| **OS (bare metal)** | `IPC`, `cache_miss_rate`, `LLC_load_misses`, `branch_miss_rate` | `perf stat` HW PMU | rich |
| **Load** | `load_time`, `load_rate` | benchmark_harness.cc | rich |
| **Errors** | OOM, timeout, validation, build, correctness, audit | orchestrator | both |

## Error Diagnosis

Both levels report errors. `rich` adds phase detection for ambiguous kills:

| Error | `minimal` | `rich` adds |
|-------|-----------|-------------|
| `TIMED OUT` | status only | load progress, stall location |
| `OOM KILLED` | status only | store_mem, budget, cgroup_limit |
| `KILLED DURING LOAD PHASE` | status only | last progress line |
| `KILLED DURING RUN PHASE` | status only | (none) |
| `VALIDATION FAILED` | status only | per-field counts, integrity detail |

## Architecture

```
benchmark_harness.cc ──> bench_output/wl1_t16_mem8g.txt
       |
/proc/diskstats ──> (delta computed in harness)
       |
perf stat ──> bench_output/perf_wl1_t16_mem8g.txt
       |
       v
format_feedback.py (parse + filter by level)
       |
       v
orchestrator.py (PromptBuilder._benchmark_feedback)
       |
       v
Agent prompt (next iteration)
```

## Controls

| Flag | What | Default |
|------|------|---------|
| `--feedback-level minimal\|rich` | Diagnostic detail in feedback | `rich` |
| `--show-baseline` | "X% of FASTER" in feedback | `off` |
| `--critique off\|review\|audit\|full` | Reviewer / auditor agents | `off` |
| `--audit-every N` | Auditor batch frequency | `1` |

These are orthogonal — each can be toggled independently.

## Ablation

```bash
# Convenience script: runs matched minimal + rich pairs
bash agent-pipeline/scripts/run_feedback_ablation.sh \
  --backend codex --mode ltm --distribution zipf --setup rmw --iterations 20

# Manual
bash agent-pipeline/run.sh --backend codex --mode ltm --distribution zipf \
  --setup rmw --iterations 20 --feedback-level minimal

bash agent-pipeline/run.sh --backend codex --mode ltm --distribution zipf \
  --setup rmw --iterations 20 --feedback-level rich
```

## Files

| File | Role |
|------|------|
| `format_feedback.py` | Parse bench output + perf stat, format by level |
| `orchestrator.py` | Wire feedback level, wrap bench with perf stat, error diagnosis |
| `run.sh` | CLI flag, exports `SKYKV_FEEDBACK_LEVEL` |
| `benchmark_harness.cc` | Emits all metrics (throughput, cache, memory, I/O, latency, evictions) |
| `kvstore_interface.h` | `CacheStats` struct (cache hits + evictions) |
| `scripts/run_feedback_ablation.sh` | Ablation convenience script |
