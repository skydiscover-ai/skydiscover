# GPU Mode: RGB to Grayscale

Evolve a Triton kernel for RGB to Grayscale conversion using SkyDiscover.

**Formula:** `Y = 0.2989 * R + 0.5870 * G + 0.1140 * B`

## Quick Start

From the repo root:

```bash
uv run skydiscover-run \
  benchmarks/gpu_mode/grayscale/initial_program.py \
  benchmarks/gpu_mode/grayscale/eval \
  -c benchmarks/gpu_mode/grayscale/config.yaml \
  -s [your_algorithm] -i 50
```

## Scoring

- **Correctness:** Must pass all test cases (rtol/atol=1e-4 vs PyTorch reference)
- **Score:** `SCORE_SCALE / geom_mean_us` where `SCORE_SCALE = 3000.0`
- Higher is better (faster runtime = higher score)

## Modal Cloud GPU Support

```bash
GPUMODE_USE_MODAL=true GPUMODE_MODAL_GPU=H100 \
  uv run skydiscover-run \
  benchmarks/gpu_mode/grayscale/initial_program.py \
  benchmarks/gpu_mode/grayscale/eval \
  -c benchmarks/gpu_mode/grayscale/config.yaml \
  -s [your_algorithm] -i 50
```
