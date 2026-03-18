# Cloudcast — Multi-Cloud Data Transfer Optimization

Broadcast a dataset from a source cloud region to multiple destinations at minimum total cost. The evolved `search_algorithm` constructs routing topologies (relay trees, Steiner-like structures) that exploit shared intermediate hops across cloud providers.

Based on the Skyplane/Cloudcast system (NSDI'24).

## Setup

1. **Download the dataset** (network profiles and evaluation configs):

   ```bash
   cd benchmarks/ADRS/cloudcast
   bash download_dataset.sh
   ```

   This downloads:
   - `profiles/cost.csv` — egress cost ($/GB) per region pair
   - `profiles/throughput.csv` — measured throughput (bps) per region pair
   - `examples/config/*.json` — 5 network configurations used for evaluation (intra-AWS, intra-Azure, intra-GCP, inter-cloud)

2. **Set your API key:**

   ```bash
   export OPENAI_API_KEY=...
   ```

## Run

From the repo root:

```bash
uv run skydiscover-run \
  benchmarks/ADRS/cloudcast/initial_program.py \
  benchmarks/ADRS/cloudcast/eval \
  -c benchmarks/ADRS/cloudcast/config.yaml \
  -s [your_algorithm] \
  -i 100
```

## Files

| File | Description |
|------|-------------|
| `initial_program.py` | Baseline `search_algorithm` function to evolve |
| `eval/` | Containerized evaluator — scores programs on total transfer cost across 5 network configs |
| `config.yaml` | Task-specific config (LLM, evaluator timeout, system prompt) |
| `simulator.py` | Broadcast cost simulator |
| `broadcast.py` | `BroadCastTopology` data structure |
| `utils.py` | Graph construction from profile CSVs |
| `download_dataset.sh` | Script to download required data files |
