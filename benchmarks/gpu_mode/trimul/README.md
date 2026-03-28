# GPU Mode: Triangle Multiplicative Update (TriMul)

Evolve a Triton kernel for the TriMul operator using SkyDiscover.

Core operation for AlphaFold3, Chai, Protenix protein structure models.

## Quick Start

From the repo root:

```bash
uv run skydiscover-run \
  benchmarks/gpu_mode/trimul/initial_program.py \
  benchmarks/gpu_mode/trimul/eval \
  -c benchmarks/gpu_mode/trimul/config.yaml \
  -s [your_algorithm] -i 50
```

## Scoring

- **Correctness:** Must match reference output (rtol=0.02, atol=0.02 vs PyTorch reference)
- **Score:** `SCORE_SCALE / geom_mean_us` where `SCORE_SCALE = 3000.0`
- Higher is better (faster runtime = higher score)

## Modal Cloud GPU Support

```bash
GPUMODE_USE_MODAL=true GPUMODE_MODAL_GPU=H100 \
  uv run skydiscover-run \
  benchmarks/gpu_mode/trimul/initial_program.py \
  benchmarks/gpu_mode/trimul/eval \
  -c benchmarks/gpu_mode/trimul/config.yaml \
  -s [your_algorithm] -i 50
```
