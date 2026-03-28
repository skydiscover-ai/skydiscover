# GPU Mode: Float16 Vector Addition

Evolve a Triton kernel for float16 vector addition using SkyDiscover.

**Operation:** `C = A + B` (element-wise, float16)

## Quick Start

From the repo root:

```bash
uv run skydiscover-run \
  benchmarks/gpu_mode/vecadd/initial_program.py \
  benchmarks/gpu_mode/vecadd/eval \
  -c benchmarks/gpu_mode/vecadd/config.yaml \
  -s [your_algorithm] -i 50
```

## Scoring

- **Correctness weight:** 0.3 (must return float16, rtol/atol=1e-3)
- **Speedup weight:** 1.0 (geometric mean vs PyTorch reference, capped at 10x)
- **Combined:** `0.3 * correctness + speedup`

## Modal Cloud GPU Support

```bash
GPUMODE_USE_MODAL=true GPUMODE_MODAL_GPU=H100 \
  uv run skydiscover-run \
  benchmarks/gpu_mode/vecadd/initial_program.py \
  benchmarks/gpu_mode/vecadd/eval \
  -c benchmarks/gpu_mode/vecadd/config.yaml \
  -s [your_algorithm] -i 50
```
