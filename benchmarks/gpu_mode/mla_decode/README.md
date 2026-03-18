# GPU Mode: Multi-Head Latent Attention (MLA) Decode

Evolve a Triton kernel for the MLA decode operator using SkyDiscover.

Core attention mechanism from DeepSeek-V2/V3, used for efficient inference with compressed KV cache via LoRA projections and RoPE.

## Quick Start

From the repo root:

```bash
uv run skydiscover-run \
  benchmarks/gpu_mode/mla_decode/initial_program.py \
  benchmarks/gpu_mode/mla_decode/eval \
  -c benchmarks/gpu_mode/mla_decode/config.yaml \
  -s [your_algorithm] -i 50
```

## Scoring

- **Correctness:** Must match reference MLA output (rtol=0.06, atol=0.06 in bfloat16)
- **Score:** `SCORE_SCALE / geom_mean_us` where `SCORE_SCALE = 3000.0`
- Higher is better (faster runtime = higher score)

## Modal Cloud GPU Support

**Note:** This benchmark requires an H200 GPU (141GB VRAM). The H100 (80GB) does not have enough memory.

```bash
GPUMODE_USE_MODAL=true GPUMODE_MODAL_GPU=H200 \
  uv run skydiscover-run \
  benchmarks/gpu_mode/mla_decode/initial_program.py \
  benchmarks/gpu_mode/mla_decode/eval \
  -c benchmarks/gpu_mode/mla_decode/config.yaml \
  -s [your_algorithm] -i 50
```
