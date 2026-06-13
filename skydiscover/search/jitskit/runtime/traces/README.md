# traces/ -- Trace Generation & Analysis

## End-to-End: New Workload → Agent Experiment

### Step 1: Generate the trace

```bash
# Single trace
python3 traces/generate.py zipf --theta 0.5 --outdir /mnt/ssd/ycsb_data

# Batch from YAML
python3 traces/sweep.py traces/workloads/zipf_theta_sweep.yaml
```

### Step 2: Run the agent

```bash
bash agent-pipeline/run.sh --backend claude --mode ltm \
    --trace-load /mnt/ssd/ycsb_data/load_zipf_t050_250M_raw.dat \
    --trace-run  /mnt/ssd/ycsb_data/run_zipf_t050_250M_1000M_raw.dat \
    --setup 50:50 --mem-budget 8 --value-size 100
```

That's it. There are no manual card-writing steps.

### How cards work (auto-generated)

The orchestrator auto-generates both cards from `run.sh` flags -- you never
write them by hand.

**trace_card.json** -- auto-filled per experiment from CLI flags:

| Field | Filled from |
|-------|-------------|
| `load_keys`, `run_keys` | Auto-detected from trace file size |
| `value_size` | `--value-size` |
| `distribution` | `--distribution` name or "Custom trace" for `--trace-load` |
| `workload` | `--setup` |
| `load_file`, `run_file` | `--trace-load`/`--trace-run` or `--distribution` paths |

**system_card.json** -- auto-filled per experiment:

| Field | Filled from |
|-------|-------------|
| `memory_constraint.budget_gb` | `--mem-budget` |
| `memory_constraint.budget_bytes` | `--mem-budget` (converted) |
| Hardware, API, correctness | Static (same for all experiments) |

Templates live at `setups/faster-single-machine/cards/`. The orchestrator
copies them into the agent workspace and patches the `_PATCHED_AT_RUNTIME_`
placeholders before the agent sees them.

**Example: what the agent sees for `--distribution zipf --setup 50:50 --mem-budget 8`:**

trace_card.json:
```json
{
  "dataset": {
    "load_keys": 250000000,
    "run_keys": 1000000000,
    "value_size": 100,
    "load_file": "/mnt/ssd/ycsb_data/load_zipf_t099_250M_raw.dat",
    "run_file": "/mnt/ssd/ycsb_data/run_zipf_t099_250M_1000M_raw.dat"
  },
  "distribution": "Zipfian θ=0.99",
  "workload": "50:50"
}
```

system_card.json:
```json
{
  "memory_constraint": { "budget_gb": [8], "budget_bytes": 8589934592 },
  "hardware": { "cpu": "64 vCPU", "memory": "256 GB DDR4" }
}
```

### Varying dimensions

**Different trace** (key access pattern) → generate a new trace, pass via
`--trace-load`/`--trace-run` or `--distribution`:
```bash
python3 traces/generate.py zipf --theta 0.5 --outdir /mnt/ssd/ycsb_data
bash agent-pipeline/run.sh ... --trace-load /mnt/ssd/ycsb_data/load_zipf_t050_250M_raw.dat \
                                --trace-run  /mnt/ssd/ycsb_data/run_zipf_t050_250M_1000M_raw.dat
```

**Different memory budget** → just change `--mem-budget` (system_card auto-updates):
```bash
bash agent-pipeline/run.sh ... --distribution zipf --mem-budget 16
```

**Different workload mix** → just change `--setup` (trace_card auto-updates):
```bash
bash agent-pipeline/run.sh ... --distribution zipf --setup rmw
```

**Different value size** → just change `--value-size` (trace_card auto-updates):
```bash
bash agent-pipeline/run.sh ... --distribution zipf --value-size 1000
```

---

## Available Generators

### YCSB (Standard)

| Generator | Knobs | Example |
|-----------|-------|---------|
| `zipf` | `--theta` (0.5-0.99) | `python3 traces/generate.py zipf --theta 0.5` |
| `uniform` | -- | `python3 traces/generate.py uniform` |

### Adversarial (FIFO-hostile)

| Generator | Knobs | Example |
|-----------|-------|---------|
| `scan` | -- | `python3 traces/generate.py scan` |
| `belady` | `--working-set` or `--mem-budget-gb` | `python3 traces/generate.py belady --working-set 50000001` |
| `stride` | `--working-set` or `--mem-budget-gb` | `python3 traces/generate.py stride --working-set 45000000` |

### Synthetic (Zipfian + one knob)

| Generator | Knobs | Example |
|-----------|-------|---------|
| `one_hit` | `--one-hit-ratio` | `python3 traces/generate.py one_hit --one-hit-ratio 0.3` |
| `bursty` | `--burst-size`, `--burst-ratio` | `python3 traces/generate.py bursty --burst-size 10` |
| `hotspot` | `--hot-fraction`, `--hot-write-ratio` | `python3 traces/generate.py hotspot --hot-fraction 0.01` |

### Real-World

Production cache traces from published research. Downloaded from libCacheSim's
S3 and converted to our binary format.

| Generator | Source | Description |
|-----------|--------|-------------|
| `metakv` | Berg OSDI'20 | Meta Cachelib KV: 82M unique keys, 1.64B requests. Highly skewed. |
| `twitter` | Yang OSDI'20 | Twitter Twemcache: 54 clusters. Default cluster 18. 10% sampled. |

```bash
# Meta KV (downloads 1.68 GB, produces ~82M-key load + ~1.64B-key run)
python3 traces/generate.py metakv --outdir /mnt/ssd/ycsb_data

# Twitter cluster 18
python3 traces/generate.py twitter --cluster 18 --outdir /mnt/ssd/ycsb_data

# Twitter cluster 52 (different access pattern)
python3 traces/generate.py twitter --cluster 52 --outdir /mnt/ssd/ycsb_data
```

Real traces have native sizes (not 250M/1B). The harness auto-detects
key counts from file size, so they work without any special configuration.

**Note**: real traces carry per-access op types (GET/SET), but the harness
applies a fixed workload mix (`--setup`) to the key stream. So these traces
give real **access distributions**, not real **op mixes**.

---

## Belady Optimal Bound

Theoretical best cache hit rate for any trace + memory budget:

```bash
python3 traces/belady/run.py /mnt/ssd/ycsb_data --mem-budget-gb 8 --value-size 100
python3 traces/belady/run.py /mnt/ssd/ycsb_data --mem-budget-gb 4,8,16,24 --value-size 100
```

`--value-size` is needed because the trace only contains keys. Belady needs
it to compute how many records fit in the budget (record = 8B header + 8B
key + value_size, 8B-aligned, minus ~3GB hash table overhead).

---

## Architecture

```
traces/
├── generate.py              # Single-trace CLI
├── sweep.py                 # Batch generate from YAML
├── list.py                  # List available traces
├── generators/
│   ├── base.py              # TraceGenerator ABC + helpers
│   ├── ycsb.py              # Zipfian, Uniform
│   ├── adversarial.py       # Scan, Belady, Stride
│   ├── synthetic.py         # One-hit, Bursty, Hotspot
│   └── real.py              # Meta KV, Twitter
├── belady/                  # Belady optimal simulator
└── workloads/               # Batch configs (YAML, trace params only)
```

### Trace format

Flat arrays of little-endian `uint64_t` (8 bytes each). Two files per trace:
- `load_<name>_<N>_raw.dat` -- N unique keys to bulk-insert
- `run_<name>_<N>_<M>_raw.dat` -- M keys for the timed benchmark

File names encode parameters: `load_zipf_t050_250M_raw.dat` = Zipf theta=0.50,
250M load keys.

---

## Adding a New Generator

1. Create `traces/generators/mydim.py` with a class inheriting `TraceGenerator`
2. Register in `traces/generators/__init__.py`
3. Run: `python3 traces/generate.py mydim --my-knob 0.5 --outdir /mnt/ssd/ycsb_data`
4. Use: `bash agent-pipeline/run.sh ... --trace-load <load.dat> --trace-run <run.dat>`
