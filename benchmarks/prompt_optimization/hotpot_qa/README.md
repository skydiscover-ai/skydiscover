# Prompt Optimization: HotPotQA

Evolves natural-language instruction prompts for multi-hop question answering on [HotPotQA](https://hotpotqa.github.io/) using SkyDiscover.

Unlike code benchmarks, this evolves **plain text prompts** (not Python code). The evaluator uses [DSPy](https://dspy.ai/) with BM25 retrieval to score prompts on exact match accuracy.

## Setup

### 1. Install dependencies

```bash
pip install dspy litellm bm25s pystemmer datasets diskcache ujson
```

### 2. Set your API key

```bash
export OPENAI_API_KEY=...
```

### 3. First run downloads data automatically

The evaluator downloads on first use:
- **HotPotQA dataset** from HuggingFace (~300MB)
- **Wikipedia abstracts corpus** (~1GB, for BM25 retrieval)
- **BM25 index** (built and cached locally)

These are cached in the `hotpot_qa/` directory and reused on subsequent runs.

## Run

```bash
# From the repo root:

# AdaEvolve (multi-island adaptive search)
uv run skydiscover-run \
  benchmarks/prompt_optimization/hotpot_qa/initial_prompt.txt \
  benchmarks/prompt_optimization/hotpot_qa/evaluator.py \
  -c benchmarks/prompt_optimization/hotpot_qa/config_adaevolve.yaml -i 100

# EvoX (co-evolutionary search)
uv run skydiscover-run \
  benchmarks/prompt_optimization/hotpot_qa/initial_prompt.txt \
  benchmarks/prompt_optimization/hotpot_qa/evaluator.py \
  -c benchmarks/prompt_optimization/hotpot_qa/config_evox.yaml -i 100

# Or use the reproduce script (runs AdaEvolve + EvoX in parallel):
bash scripts/reproduce/prompt_opt.sh
```

## Scoring

- **combined_score**: Exact match accuracy on 300 training examples (0.0–1.0). Validation and test sets are held out.
- **prompt_length**: Length of the evolved prompt
- Each evaluation runs ~300 LLM calls with BM25 retrieval

## Files

| File | Description |
|------|-------------|
| `initial_prompt.txt` | Seed prompt: "Given the fields `question`, `passages`, produce the fields `answer`." |
| `evaluator.py` | DSPy-based evaluator with BM25 retrieval and exact match scoring |
| `config_adaevolve.yaml` | AdaEvolve config |
| `config_evox.yaml` | EvoX config |
| `requirements.txt` | Python dependencies |

## Notes

- Uses `language: text` and `diff_based_generation: false` — prompts are evolved as full rewrites, not diffs
- Evaluator model is `gpt-5-mini` (configured in evaluator, separate from evolution model)
- BM25 retriever uses disk caching for fast repeated queries
