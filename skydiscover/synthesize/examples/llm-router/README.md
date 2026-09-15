# Specialized LLM Routers

A model router decides, for each request, which LLM answers it. Two tenants share one rate-limited
fleet of 12 models from 3 providers and want different things: tenant A is interactive (answers must
start within 2.5 s), tenant B is batch (10-minute deadline, wants a low bill). Build one router per
tenant that beats a single router trained on both, on quality, cost, and on-time answers.

Quality and price are real per-request measurements from
[LLMRouterBench](https://github.com/ynulihao/LLMRouterBench); the benchmark uses simulated time, so
a full trace replays in seconds with no live API calls. Runs on any machine.

## Run it

Wire the skill once with `uv run skydiscover init` and restart your coding agent (see the
[quick start](../../README.md#-quick-start)). Build the traces once, from this directory:

```bash
hf download NPULH/LLMRouterBench --repo-type dataset --local-dir evaluator/benchmark/.data/lrb
tar xzf evaluator/benchmark/.data/lrb/bench-release.tar.gz -C evaluator/benchmark/.data/lrb   # 1.3 GB
python evaluator/benchmark/data/make_tenant_traces.py
```

Then, from the repo root:

```
/skysynth build the specialist routers specified in skydiscover/synthesize/examples/llm-router/task.md
```

The result lands in `outputs/synthesize/<slug>_<timestamp>/best/`. `task.md` fixes the protocol:
train on the train split, tune on val with five evaluations, and score the held-out test split once.
To replay one router yourself, from this directory:

```
python evaluator/benchmark/replay_tenants.py evaluator/benchmark/.data/traces/trace_tenantA_val.jsonl evaluator.reference_router.ReferenceRouter
```

## What's here

| Path | What it is |
|---|---|
| `task.md` | what to build, how it is scored, the interface, the rules, and the data protocol; the input to `/skysynth` |
| `evaluator/` | the interface, a correct reference router, and the baseline router to beat |
| `evaluator/benchmark/` | the replay benchmark (`replay.py`, `replay_tenants.py`) and the fleet description (`env_card.yaml`) |
| `evaluator/benchmark/data/` | the trace builders, the frozen prompt ids, and the workload cards |
| `evaluator/benchmark/baseline/`, `evaluator/benchmark/artifacts/` | how the baseline router was trained, and its saved predictions so it replays on any machine |

Downloaded data and generated traces live under `evaluator/benchmark/.data/` (gitignored).
