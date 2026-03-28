# GPU Mode: Triton Kernel Optimization

Evolve high-performance GPU kernels using SkyDiscover. Each benchmark provides a reference PyTorch implementation and scores submissions by runtime — faster is better. Pure PyTorch submissions are accepted; Triton is not required.

## Benchmarks

| Benchmark | Operation | Tolerance | GPU |
|-----------|-----------|-----------|-----|
| [`vecadd`](vecadd/) | Float16 element-wise `C = A + B` | rtol/atol=1e-3 | H100 |
| [`grayscale`](grayscale/) | RGB → Grayscale (`0.2989R + 0.5870G + 0.1140B`) | rtol/atol=1e-4 | H100 |
| [`trimul`](trimul/) | Triangle multiplicative update (AlphaFold3/Chai/Protenix) | rtol/atol=0.02 | H100 |
| [`mla_decode`](mla_decode/) | Multi-head latent attention decode (DeepSeek-V2/V3) | rtol/atol=0.06 (bfloat16) | **H200** |

## Quick Start

```bash
# Run on local GPU
uv run skydiscover-run \
  benchmarks/gpu_mode/trimul/initial_program.py \
  benchmarks/gpu_mode/trimul/eval \
  -c benchmarks/gpu_mode/trimul/config.yaml \
  -s [your_algorithm] \
  -i 50

# Run on Modal cloud GPU (set GPU type per benchmark)
GPUMODE_USE_MODAL=true GPUMODE_MODAL_GPU=H100 \
  uv run skydiscover-run \
  benchmarks/gpu_mode/trimul/initial_program.py \
  benchmarks/gpu_mode/trimul/eval \
  -c benchmarks/gpu_mode/trimul/config.yaml \
  -s [your_algorithm] \
  -i 50
```

> **Note:** `mla_decode` requires `GPUMODE_MODAL_GPU=H200` — H100 (80GB) does not have enough VRAM.

## Writing a Submission

Your program must define a `custom_kernel(data)` function. The `data` argument is problem-specific (see each benchmark's `reference.py` for the exact type). Return the computed result.

```python
# EVOLVE-BLOCK-START
import torch
import triton
import triton.language as tl

def custom_kernel(data):
    # data is a problem-specific input (tensor, dataclass, etc.)
    # return the computed result
    ...
# EVOLVE-BLOCK-END
```

## Scoring

All benchmarks use the same formula:

```
combined_score = SCORE_SCALE / geom_mean_us
```

`geom_mean_us` is the geometric mean of kernel runtimes in microseconds across all benchmark cases. Higher score = faster kernel. `SCORE_SCALE` is `3000.0` for all current benchmarks.

`vecadd` uses a different combined formula (`0.3 * correctness + speedup`) — see its README for details.

## Evaluation Pipeline

The shared evaluator (`shared_eval.py`) handles both local and Modal paths:

1. **Correctness** — runs all `TEST_CASES` from `reference.py`, checks output against reference within tolerance
2. **Warmup** — runs one benchmark case briefly to trigger Triton JIT compilation
3. **Benchmark** — times `BENCHMARK_CASES` using CUDA events, repeats until error < 0.1% or time budget is exhausted
4. **Score** — geometric mean of benchmark runtimes → `SCORE_SCALE / geom_mean_us`

## Directory Structure

```
gpu_mode/
├── shared_eval.py       # Shared evaluator (correctness + benchmarking logic)
├── modal_eval.py        # Modal cloud GPU runners (H100, A100, L40S, T4, H200)
├── vecadd/              # Float16 vector addition
├── grayscale/           # RGB → grayscale conversion
├── trimul/              # Triangle multiplicative update
└── mla_decode/          # MLA decode (DeepSeek attention)

# Each benchmark contains:
#   initial_program.py   — starting kernel
#   eval/            — containerized evaluator directory
#   reference.py         — reference kernel, test/benchmark cases, SCORE_SCALE
#   config.yaml          — search config
#   requirements.txt     — dependencies
```
