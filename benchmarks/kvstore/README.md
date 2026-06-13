# `kvstore` — single-node KV-store synthesis tasks

Each task asks a solver to **synthesize a correct, high-throughput key-value store specialized to a
spec** and beat a tuned baseline. Any search strategy can attempt the *same* task — that
comparability is the point:

```bash
# the agentic multi-agent synthesizer
skydiscover-run -c benchmarks/kvstore/0001_ycsb50_zipf_8gb/config.yaml -s jitskit
# the single-agent baseline, SAME task (the paper's headline comparison)
skydiscover-run -c benchmarks/kvstore/0001_ycsb50_zipf_8gb/config.yaml -s claude_code
```

## Layout

```
benchmarks/kvstore/
├── README.md
├── resolver.py              # maps params.task -> a checked-in task dir (path-free runs)
├── _harness/                # ONE canonical C++ harness (see "Harness" below) — populated from runtime
├── _baselines/results.json  # FASTER/F2/RocksDB/Redis numbers, hardware-tagged
└── 0001_ycsb50_zipf_8gb/     # a task instance
    ├── spec.md              # human-facing statement (API, property, workload, budget, baseline)
    ├── config.yaml          # machine-facing spec (parses against the real SkyDiscover schema)
    ├── initial_program.cc   # trivial in-memory seed implementing IKVStore, with EVOLVE-BLOCK markers
    └── evaluator/           # build -> 6 consistency tests (HARD GATE) -> benchmark -> combined_score
```

`_`-prefixed dirs (`_harness`, `_baselines`) are **shared**, not tasks — the resolver and any
task-discovery skip them.

## Conventions

- **Task id:** `NNNN_<slug>` (stable, sortable). The slug encodes the workload signature.
- **Score:** `combined_score` = peak throughput (Mops/s) across iterations, **after** the 6-test
  correctness gate passes; a failing candidate scores `0` with `validity: 0`.
- **Knobs** live under `search.database` (a `JitsKitConfig`); the human spec lives under
  `benchmark.params`. Tier = `search.database.mode` (`ltm` hardware / `inmem` portable).

## Harness (population — vendoring decision, plan §8.1)

`_harness/` holds the ONE canonical copy of `kvstore_interface.h`, `benchmark_harness.cc`,
`consistency_harness.cc`, `CMakeLists.txt` — deduping the runtime's forked copies (the evaluator copy
is canonical). It is **not vendored into this repo unilaterally** (it is private C++, and how to
vendor — submodule vs synced copy — is an open team decision). Populate it one of two ways:

```bash
# A) from the runtime submodule (recommended):
ln -s ../../skydiscover/search/jitskit/runtime/_harness benchmarks/kvstore/_harness
# B) a synced copy with a CI drift-check (if the team prefers a vendored copy).
```

Until `_harness/` is populated, the evaluator returns `validity: 0` with a clear error (it never
pretends to pass), and the seed won't compile (its `#include "../_harness/kvstore_interface.h"` is
unresolved). `-s jitskit` does not need `_harness/` — it reads the runtime's own `leaderboard.json`.

## Adding a task

1. `cp -r 0001_ycsb50_zipf_8gb NNNN_<slug>` and edit `spec.md` + `config.yaml`
   (`benchmark.params.task` and the `search.database` workload/distribution/value_size/mem_budget).
2. Add the baselines for the new cell to `_baselines/results.json` (hardware-tagged).
3. The evaluator is shared — no code change unless the API/property changes.
