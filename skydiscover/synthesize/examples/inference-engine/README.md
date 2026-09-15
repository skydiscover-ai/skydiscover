# Specialized Inference Engine

Build an inference engine for `Qwen/Qwen3-4B` on one NVIDIA L4, for a workload where many requests
share a long prompt prefix. Scored on generated tokens per second; outputs must match the reference
model. Needs an L4 the coding agent can reach.

## Run it

Wire the skill once with `uv run skydiscover init` and restart your coding agent (see the
[quick start](../../README.md#-quick-start)). Then, from the repo root:

```
/skysynth build the llm-inference engine specified in skydiscover/synthesize/examples/inference-engine/task.md
```

The result lands in `outputs/synthesize/<slug>_<timestamp>/best/`. To measure one engine
yourself, start its server and run `python evaluator/benchmark.py --url http://localhost:8000`.

## What's here

| Path | What it is |
|---|---|
| `task.md` | what to build, the workload, the hardware, and the accuracy bar; the input to `/skysynth` |
| `evaluator/benchmark.py` | the throughput benchmark |
| `evaluator/accuracy_prompts.json` | the accuracy test data (below) |

An inference engine has no small reference implementation, so the agents write the tests during
the run against the reference model's outputs.

### Accuracy test

For every prompt in `accuracy_prompts.json`, the engine's greedy output must match `Qwen/Qwen3-4B`
under HuggingFace transformers token for token. The reference runs live, so no expected outputs are
stored. The prompts are short code completions whose continuations are near-deterministic, so an
exact match is well defined.

- `single_prompts`: five prompts, one at a time. Checks the model itself.
- `shared_prefix_sets`: four sets of one prefix plus four continuations, run as one batch sharing
  the prefix, each compared to the reference on prefix + continuation alone. Checks that sharing a
  prefix never changes a request's output.
