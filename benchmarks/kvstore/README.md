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
├── _harness -> ../../skydiscover/search/jitskit/runtime/interface   # symlink: the ONE harness (no copy)
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

## Harness (single source — no copy)

`_harness` is a **symlink** to the runtime's `interface/` (`skydiscover/search/jitskit/runtime/
interface`) — the exact `kvstore_interface.h`, `benchmark_harness.cc`, `consistency_harness.cc`,
`CMakeLists.txt` that jitskit's own loop compiles. **There is no second copy**; it can never drift.

The non-jitskit evaluator stages the candidate **flat** alongside these files and builds on the host
(hence the seed's flat `#include "kvstore_interface.h"`). This path is **build-tier — verify on a box
with the C++ toolchain**; it is not exercised in CI. `-s jitskit` does **not** use this evaluator at
all — it reads the runtime's own `leaderboard.json`.

## Adding a task

1. `cp -r 0001_ycsb50_zipf_8gb NNNN_<slug>` and edit `spec.md` + `config.yaml`
   (`benchmark.params.task` and the `search.database` workload/distribution/value_size/mem_budget).
2. Add the baselines for the new cell to `_baselines/results.json` (hardware-tagged).
3. The evaluator is shared — no code change unless the API/property changes.
