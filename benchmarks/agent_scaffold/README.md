# Agent scaffold discovery

Evolve the **agent harness** (tool policy, memory, retries, stopping rules)
rather than a single pure function. Addresses [issue #1](https://github.com/skydiscover-ai/skydiscover/issues/1)
with a DGM/ShinkaEvolve-style discovery object on a closed synthetic suite.

## What is evolved

`run_agent(question, tools) -> str` inside `initial_program.py`.

## Tools (fixed by the evaluator)

| Tool | Role |
|------|------|
| `search(query)` | Resolve a keyword to an entity id |
| `lookup(entity, field)` | Read KB fields (`capital`, `country`, `currency`, `river`, `population_m`, `neighbor`) |
| `calculate(expression)` | Tiny arithmetic helper |
| `budget_remaining()` | Remaining tool-call budget (max 12) |

## Metrics

- `accuracy` — held-out task exact match
- `train_accuracy` — in-suite accuracy (secondary)
- `efficiency` — fewer tool calls is better
- `combined_score` — weighted mix favoring held-out accuracy

## Run

```bash
uv run skydiscover-run benchmarks/agent_scaffold/initial_program.py \
  benchmarks/agent_scaffold/evaluator.py \
  -c benchmarks/agent_scaffold/config.yaml \
  -s best_of_n -i 50
```

Offline smoke (no LLM):

```bash
python benchmarks/agent_scaffold/evaluator.py benchmarks/agent_scaffold/initial_program.py
```
