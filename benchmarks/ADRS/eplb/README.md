# Expert Parallelism Load Balancer (EPLB)

This benchmark uses SkyDiscover to optimize the Expert Parallelism Load Balancer (EPLB) algorithm for Mixture-of-Expert (MoE) models. The goal is to rearrange and replicate experts across GPUs to balance load, while keeping the rearrangement algorithm itself fast.

## Setup

1. **Install PyTorch** (required by the evaluator):

   ```bash
   uv pip install torch
   ```

2. **Download the workload file** from [Hugging Face](https://huggingface.co/datasets/abmfy/eplb-openevolve) into this directory:

   ```bash
   cd benchmarks/ADRS/eplb
   wget https://huggingface.co/datasets/abmfy/eplb-openevolve/resolve/main/expert-load.json
   ```

3. **Set your API key:**

   ```bash
   export OPENAI_API_KEY=...
   ```

## Run

From the repo root:

```bash
uv run skydiscover-run \
  benchmarks/ADRS/eplb/initial_program.py \
  benchmarks/ADRS/eplb/eval \
  -c benchmarks/ADRS/eplb/config.yaml \
  -s [your_algorithm] \
  -i 100 \
  -o eplb_output
```

Or from this directory:

```bash
uv run skydiscover-run initial_program.py eval \
  -c config.yaml \
  -s [your_algorithm] \
  -i 100
```

## Evaluate a saved program

```bash
python evaluate_best_program.py
```

## Files

| File | Description |
|------|-------------|
| `initial_program.py` | Baseline `rebalance_experts` function to evolve |
| `eval/` | Containerized evaluator — scores programs on load-balance quality and execution speed |
| `config.yaml` | Task-specific config (LLM, evaluator timeout, system prompt) |
| `evaluate_best_program.py` | Standalone script to evaluate a saved best program |
| `expert-load.json` | Workload data (must be downloaded — see Setup) |
