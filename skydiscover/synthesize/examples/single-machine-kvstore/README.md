# Single-Machine Key-Value Store

Build a key-value store for a dataset larger than memory, scored on throughput, and required to
survive crashes without losing acknowledged writes. Needs a Linux machine with a local SSD, `cmake`,
and a C++17 compiler; a run takes several hours.

## Run it

Wire the skill once with `uv run skydiscover init` and restart your coding agent (see the
[quick start](../../README.md#-quick-start)). Then, from the repo root:

```
/skysynth build the key-value store specified in skydiscover/synthesize/examples/single-machine-kvstore/task.md
```

The result lands in `outputs/synthesize/<slug>_<timestamp>/best/`. To score one implementation
yourself:

```
IMPL=<file> bash skydiscover/synthesize/examples/single-machine-kvstore/evaluator/score.sh
```

Without `IMPL` it scores the reference implementation.

## What's here

| Path | What it is |
|---|---|
| `task.md` | what to build, how it is scored, the interface; the input to `/skysynth` |
| `spec/` | hardware and workload facts the run's specification builds on |
| `evaluator/kvstore_interface.h` | the interface a candidate implements |
| `evaluator/reference_kvstore.cc` | a correct implementation |
| `evaluator/benchmark_harness.cc` | throughput, the score |
| `evaluator/consistency_harness.cc` | crash recovery and concurrent reads under load |
| `evaluator/generate.py`, `evaluator/generators/` | the workload |
| `evaluator/CMakeLists.txt` | builds a candidate with either harness |
| `evaluator/score.sh` | builds one implementation, runs the consistency check, then the benchmark |
| `evaluator/tests/` | four correctness tests (exact bytes, bounded memory, crash durability, delete durability) and the `test.sh` that builds and runs them against `$SKYDISCOVER_IMPL` |

A candidate is checked twice and measured once. The workflow runs `evaluator/tests/` on it during the run.
`score.sh` then builds it with the harness, runs the consistency check (1M keys), and measures
throughput; that number is the score.
