# `jitskit` — agentic KV-store synthesis as a search strategy

Wraps the Jitskit multi-agent runtime (planner → coder → critic → auditor,
whiteboard memory, instrumented bare-metal evaluator) **as-is** and exposes it
to SkyDiscover as a search strategy, the multi-agent sibling of `claude_code`.

Unlike `claude_code` (which always runs the agent in Docker), `jitskit` runs the
runtime's `run.sh` **in a host subprocess** so the agent keeps full host access
(numactl / perf / cgroup / `/mnt/ssd`) — it must measure the hardware it is
optimizing for. The runtime is **never edited**.

## Layout

```
search/jitskit/
├── controller.py   # config -> run.sh flags/env, launch on host, read leaderboard.json
├── database.py     # minimal store (mirrors claude_code/database.py)
└── runtime/        # the jitskit runtime, VENDORED in-tree (committed, not a submodule) = PROJECT_DIR
```

## Setup

The runtime ships **in this repo** — there is nothing to fetch or init. Just:

```bash
export ANTHROPIC_API_KEY=...
```

To run against a *different* local checkout of the runtime (e.g. the private
`skykv-claude` dev tree), point the strategy at it:

```yaml
search:
  type: jitskit
  database:
    runtime_dir: /path/to/skykv-claude   # the PROJECT_DIR run.sh cd's into
```

## Configuration

Knobs ride on `search.database` (a `JitsKitConfig`) — they are parsed, unlike
bespoke top-level blocks which `config.from_dict` silently drops:

```yaml
language: cpp
search:
  type: jitskit
  database:
    backend: claude          # claude | codex
    mode: ltm                # inmem | ltm
    model: claude-sonnet-4-6
    workload: "50:50"        # -> run.sh --setup (workload mix)
    distribution: "zipf(0.99)"
    value_size: 100
    mem_budget_gb: [8, 32]   # LIST allowed (invariant I4)
    threads: [16]
    critique_mode: full      # off | review | audit | full
    feedback_level: rich     # minimal | rich
    audit_every: 15
```

Run it:

```bash
skydiscover-run benchmarks/kvstore/0001_ycsb50_zipf_8gb/initial_program.cc \
                benchmarks/kvstore/0001_ycsb50_zipf_8gb/evaluator \
                -s jitskit -i 50
```

The reported score is the runtime's own peak Mops/s from `leaderboard.json`.
SkyDiscover does **not** re-measure: the controller sets `skip_test_rescore =
True`, so the Runner's `mode="test"` re-eval is bypassed (invariant I1).

## Acceptance test (hardware-gated, not in CI)

The unit tests in `tests/search/test_jitskit.py` lock the wrapper contract. The
full **bit-identical** check must run on the bare-metal tier:

1. `bash runtime/agent-pipeline/run.sh --backend claude --mode ltm --setup 50:50 \
   --distribution zipf --value-size 100 --mem-budget 8 --threads 16 --iterations 2`
2. `skydiscover-run ... -s jitskit -i 2` with the equivalent `JitsKitConfig`.
3. Assert the produced `best_impl.cc` is byte-identical and the reported peak
   Mops/s equals that run's own `leaderboard.json` value (no re-measurement).

Any divergence is a wrapper-wiring bug by definition.

## Known constraint — concurrent same-spec runs

`run.sh:327` derives `RUN_KEY` from the spec and cannot be overridden via env, so
the runtime keys its workspace by `{backend}_{mode}_{run_key}` **without** a
timestamp. Two concurrent runs of the *same spec* share a workspace and would
collide; the controller logs a warning when it detects this. Isolate by pointing
`runtime_dir` at distinct checkouts.
