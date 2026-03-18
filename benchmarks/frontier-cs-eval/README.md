# Frontier-CS Benchmark

Evolves C++ solutions for [Frontier-CS](https://github.com/facebookresearch/Frontier-CS) algorithmic optimization problems using SkyDiscover.

## Setup

```bash
# 1. Clone Frontier-CS
cd benchmarks/frontier-cs-eval
git clone https://github.com/FrontierCS/Frontier-CS.git

# 2. Start the judge server (requires Docker)
cd Frontier-CS/algorithmic
docker compose up -d

# 3. Install dependencies (from project root)
cd ../../..
uv sync --extra frontier-cs

# 4. Set your API key
export OPENAI_API_KEY=...
```

## Run

Supported algorithms: `adaevolve`, `evox`, `openevolve`, `gepa`, `shinkaevolve`


Single problem:
```bash
cd benchmarks/frontier-cs-eval
FRONTIER_CS_PROBLEM=0 uv run skydiscover-run initial_program.cpp eval \
  -c config.yaml -s [search_algorithm] -i 50
```

All problems in parallel:
```bash
uv run python run_all_frontiercs.py --search [search_algorithm] --iterations 50 --workers 6
```

## Evaluate best programs (post-discovery)

```bash
uv run python run_best_programs_frontiercs.py
```

## Analyze results

```bash
uv run python combine_results.py   # merge training/testing scores into CSV
uv run python analyze_results.py   # generate plots and statistics
```

## Files

| File | Description |
|------|-------------|
| `initial_program.cpp` | Seed C++ program |
| `eval/` | Containerized evaluator — evaluates C++ solutions via Frontier-CS docker judge |
| `config.yaml` | Config with system prompt template |
| `run_all_frontiercs.py` | Parallelizes evolution across all problems |
| `run_best_programs_frontiercs.py` | Re-evaluates best programs after evolution |
| `combine_results.py` | Combines training/testing scores into CSV |
| `analyze_results.py` | Generates score analysis plots and statistics |

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | API key |
| `FRONTIER_CS_PROBLEM` | `0` | Problem ID to evolve |
| `JUDGE_URLS` | `http://localhost:8081` | Comma-separated judge server URLs |
