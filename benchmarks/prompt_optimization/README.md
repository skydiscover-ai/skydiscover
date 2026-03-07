# Prompt Optimization Benchmark

Evolves plain-text LLM instruction prompts using SkyDiscover's evolutionary search.

## How It Works

1. Start with a seed prompt (e.g., a one-line instruction)
2. Evaluate the prompt by running it on a QA task and measuring exact-match accuracy
3. Use an LLM to rewrite the prompt guided by performance feedback and context from other candidates
4. Repeat — the population of prompts improves over generations

Key config: `language: text` and `diff_based_generation: false` — prompts are fully rewritten each iteration (not diffed like code).

## Benchmarks

| Task | Dataset | Metric | Dir |
|------|---------|--------|-----|
| Multi-hop QA | [HotPotQA](https://hotpotqa.github.io/) | Exact match accuracy (0–1) | `hotpot_qa/` |

## Quick Start

```bash
cd benchmarks/prompt_optimization/hotpot_qa

# Install deps
pip install dspy litellm bm25s pystemmer datasets diskcache ujson

# Set API key
export OPENAI_API_KEY=...

# Run (first run downloads ~1.3GB of data)
uv run skydiscover-run initial_prompt.txt evaluator.py -c config_adaevolve.yaml -i 100
```

See `hotpot_qa/README.md` for full details.
