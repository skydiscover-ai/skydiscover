# LLM-SQL — Column Reordering for Prefix Caching

When rows of a table are serialized into LLM prompts sequentially, consecutive rows that share leading column values can reuse cached prefixes. This task evolves a column-reordering strategy that maximizes prefix-cache hit rates across multiple real-world datasets without altering the underlying data.

## Setup

1. **Download the datasets** (~69 MB total):

   ```bash
   cd benchmarks/ADRS/llm_sql
   bash download_dataset.sh
   ```

   This downloads 5 CSV datasets into `datasets/`:
   - `movies.csv` — Rotten Tomatoes movie reviews (~9 MB)
   - `beer.csv` — Beer review dataset (~2.5 MB)
   - `BIRD.csv` — BIRD text-to-SQL dataset (~34 MB)
   - `PDMX.csv` — PDMX metadata dataset (~7.4 MB)
   - `products.csv` — Amazon product catalog (~16 MB)

2. **Set your API key:**

   ```bash
   export OPENAI_API_KEY=...
   ```

## Run

From the repo root:

```bash
uv run skydiscover-run \
  benchmarks/ADRS/llm_sql/initial_program.py \
  benchmarks/ADRS/llm_sql/eval \
  -c benchmarks/ADRS/llm_sql/config.yaml \
  -s [your_algorithm] \
  -i 100
```

## Scoring

Combined score: `0.95 * average_hit_rate + 0.05 * (12 - min(12, avg_runtime)) / 12`

- **Hit rate** (95% weight): prefix-cache hit count normalized across 5 datasets
- **Runtime** (5% weight): wall-clock seconds for the reordering algorithm

## Files

| File | Description |
|------|-------------|
| `initial_program.py` | Baseline `Evolved` class with `reorder()` method to evolve |
| `eval/` | Containerized evaluator — scores programs on prefix hit rate and runtime across 5 datasets |
| `config.yaml` | Task-specific config (LLM, evaluator timeout, system prompt) |
| `solver.py` | Base `Algorithm` class and greedy baseline |
| `utils.py` | Prefix hit count evaluation utilities |
| `download_dataset.sh` | Script to download required CSV datasets |
