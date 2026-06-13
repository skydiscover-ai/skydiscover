# setups/ -- Evaluation Setups

Each subfolder is a self-contained evaluation setup with everything specific to one reference system.

## Structure

```
setups/<setup-name>/
├── cards/              # What to build: API, guarantees, workload
│   ├── system_card.json
│   ├── trace_card.json
│   └── design_hints.json
├── evaluators/         # How to test: correctness harnesses for this setup's guarantees
├── baselines/          # What to compare against: reference system numbers
├── scripts/            # Convenience wrappers for common experiment configs
└── logs/               # Experiment results organized by experimenter
```

## Current setups

| Setup | Reference system | Optimization | Guarantees tested |
|-------|-----------------|-------------|-------------------|
| `faster-single-machine` | FASTER (SIGMOD 2018) | Throughput (ops/sec) | Monotonicity property, read correctness |

## How to add a new setup

1. `mkdir -p setups/<name>/{cards,evaluators,baselines,scripts,logs}`
2. **cards/system_card.json** -- API and guarantees (use the reference paper's exact words)
3. **cards/trace_card.json** -- dataset and workload (runtime-patched fields use `_PATCHED_AT_RUNTIME_`)
4. **evaluators/consistency_harness.cc** -- correctness tests for this setup's guarantees
5. **baselines/** -- reference system numbers (CSV or markdown)
6. **scripts/** -- convenience wrappers that call `bash agent-pipeline/run.sh`
7. Set `SKYKV_CARD_SET="<name>"` in `agent-pipeline/run.sh`

## What's shared vs what's per-setup

| Shared (reused across setups) | Per-setup |
|------------------------------|-----------|
| `agent-pipeline/` -- orchestrator, pipeline scripts | `cards/` -- system and trace specs |
| `interface/kvstore_interface.h` -- abstract KV store API | `evaluators/` -- correctness tests |
| `interface/benchmark_harness.cc` -- performance harness | `baselines/` -- reference numbers |
| `scripts/` -- trace generation, plotting | `scripts/` -- experiment wrappers |

## Design principles

- **system_card** states requirements, not implementations. It says *what* the store must do, not *how*.
- **trace_card** has runtime-variable fields marked `_PATCHED_AT_RUNTIME_`. The orchestrator fills these in.
- **design_hints** is opt-in. It describes a reference system's architecture as suggestions, not requirements.
- **evaluators** test the guarantees stated in the system card. Each setup defines its own correctness tests.
- All terminology should match the reference paper's exact wording.
