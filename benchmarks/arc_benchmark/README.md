# ARC Benchmark

Evolves ARC-AGI visual reasoning task solutions using SkyDiscover.

## Setup

### 1. Download ARC data

Clone the ARC-AGI-2 repo and convert the data:

```bash
cd benchmarks/arc_benchmark
git clone https://github.com/arcprize/ARC-AGI-2.git /tmp/ARC-AGI-2
OUT_DIR=./data uv run python convert_arc_agi2_data.py /tmp/ARC-AGI-2
rm -rf /tmp/ARC-AGI-2
```

This creates 4 files in `data/`:
- `arc-agi_training_challenges.json` (1000 tasks)
- `arc-agi_training_solutions.json`
- `arc-agi_evaluation_challenges.json` (120 tasks)
- `arc-agi_evaluation_solutions.json`

### 2. Set your API key

```bash
export OPENAI_API_KEY=...
```

## Run a single task

ARC requires a per-task config (each task has unique training examples as the prompt). Use `generate_config.py` to create one, then run with any search backend:

```bash
cd benchmarks/arc_benchmark

# Generate task-specific config
TASK_NUM=0 ARC_TASK_FILE=training CONFIG_OUT=./config_task_0.yaml \
  uv run python generate_config.py

# Run with any backend
uv run skydiscover-run initial_program.py eval \
  -c config_task_0.yaml -s [your_algorithm] -i 30

# Or with evox, openevolve, gepa:
uv run skydiscover-run initial_program.py eval \
  -c config_task_0.yaml -s [your_algorithm] -i 30
```

## Run all evaluation tasks

```bash
cd benchmarks/arc_benchmark
export ARC_TASK_FILE=evaluation

NUM_TASKS=$(uv run python -c "import json; print(len(json.load(open('data/arc-agi_evaluation_challenges.json'))))")

for i in $(seq 0 $((NUM_TASKS - 1))); do
  TASK_NUM=$i CONFIG_OUT=./config_task_${i}.yaml uv run python generate_config.py
  TASK_NUM=$i uv run skydiscover-run initial_program.py eval \
    -c config_task_${i}.yaml -s [your_algorithm] -i 30 \
    -o outputs/eval_task_${i}
done
```

## Post-discovery test evaluation

After the discovery process, evaluate the best program on held-out test inputs:

```bash
TASK_NUM=0 ARC_TASK_FILE=evaluation \
  OUTS_DIR=./outputs/eval_task_0/adaevolve \
  uv run python post_discovery_eval.py
```

## Config: GPT vs Gemini

Edit `config.yaml` — comment the GPT block and uncomment the Gemini block, or override with `--model`:

```bash
uv run skydiscover-run ... -m gemini/gemini-3-pro-preview
```

## Files

| File | Description |
|------|-------------|
| `initial_program.py` | Seed program with two transform functions to evolve |
| `eval/` | Containerized evaluator — scores programs on pass@2 + cell accuracy |
| `config.yaml` | Base config template (prompt injected by generate_config.py) |
| `generate_config.py` | Injects task-specific training examples into config as system prompt |
| `post_discovery_eval.py` | Evaluates best program on held-out test inputs |
| `convert_arc_agi2_data.py` | Converts raw ARC-AGI-2 data to benchmark format |
| `requirements.txt` | Dependencies (numpy) |

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | API key |
| `ARC_TASK_FILE` | `training` | `training` or `evaluation` |
| `TASK_NUM` | `0` | Task index within the dataset |
| `BASE_CONFIG` | `./config.yaml` | Base config template path |
| `CONFIG_OUT` | `./config_task_{N}.yaml` | Output path for generated config |
| `DATA_ROOT` | `./data` | Path to ARC data directory |
| `MAX_ITERATIONS` | (from config) | Override `max_iterations` at runtime |
| `ARC_EVAL_INCLUDE_TEST` | `0` | Set to `1` to also run the held-out test inputs during evolution |
| `ARC_EVAL_USE_TEST_FOR_SCORE` | `0` | Set to `1` to average train and test scores into `combined_score` (only used when `ARC_EVAL_INCLUDE_TEST=1`) |
