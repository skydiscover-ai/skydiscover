"""
Reference implementation for Grayscale Triton kernel.
Y = 0.2989 R + 0.5870 G + 0.1140 B

Input:  (H, W, 3) float32 RGB tensor
Output: (H, W) float32 grayscale tensor
"""

import torch

# ---------------------------------------------------------------------------
# Scoring and benchmark configuration (read by shared_eval.py)
# ---------------------------------------------------------------------------

SCORE_SCALE = 3000.0

# grayscale uses CUDA events timing, 0.1% rel error, 120s wall clock timeout
BENCH_USE_CUDA_EVENTS = True
BENCH_REL_ERROR = 0.001
BENCH_WALL_TIMEOUT_NS = 120e9
BENCH_NO_GRAD = False
BENCH_MAX_REPEATS = 100
BENCH_MAX_TIME_NS = 10e9
BENCH_WARMUP_STYLE = 'tiny_benchmark'

# ---------------------------------------------------------------------------
# Test / benchmark cases
# ---------------------------------------------------------------------------

TEST_CASES = [
    {"size": 256, "seed": 42},
    {"size": 512, "seed": 123},
    {"size": 1024, "seed": 456},
    {"size": 2048, "seed": 789},
]

BENCHMARK_CASES = [
    {"size": 1024, "seed": 1001},
    {"size": 2048, "seed": 1002},
    {"size": 4096, "seed": 1003},
    {"size": 8192, "seed": 1004},
]

# ---------------------------------------------------------------------------
# Reference kernel
# ---------------------------------------------------------------------------


def ref_kernel(data):
    """Reference: Y = 0.2989 R + 0.5870 G + 0.1140 B"""
    rgb, output = data
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=rgb.device, dtype=rgb.dtype)
    output[...] = torch.sum(rgb * weights, dim=-1)
    return output


def generate_input(size, seed):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    x = torch.rand(size, size, 3, device="cuda", dtype=torch.float32, generator=gen).contiguous()
    y = torch.empty(size, size, device="cuda", dtype=torch.float32).contiguous()
    return x, y


def check_implementation(data, submission_output, rtol=1e-4, atol=1e-4):
    ref_output = ref_kernel(data)
    if submission_output.shape != ref_output.shape:
        return False, f"Shape mismatch: expected {ref_output.shape}, got {submission_output.shape}"
    if torch.allclose(submission_output, ref_output, rtol=rtol, atol=atol):
        return True, "Match"
    diff = torch.abs(submission_output.float() - ref_output.float())
    return False, f"Output mismatch: max_diff={diff.max().item():.6f}"


# ---------------------------------------------------------------------------
# Self-contained reference code for Modal remote execution
# ---------------------------------------------------------------------------

MODAL_REFERENCE_CODE = r'''
import torch

def ref_kernel(data):
    rgb, output = data
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=rgb.device, dtype=rgb.dtype)
    output[...] = torch.sum(rgb * weights, dim=-1)
    return output

def generate_input(size, seed):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    x = torch.rand(size, size, 3, device="cuda", dtype=torch.float32, generator=gen).contiguous()
    y = torch.empty(size, size, device="cuda", dtype=torch.float32).contiguous()
    return x, y

def check_implementation(data, submission_output, rtol=1e-4, atol=1e-4):
    ref_output = ref_kernel(data)
    if submission_output.shape != ref_output.shape:
        return False, f"Shape mismatch: expected {ref_output.shape}, got {submission_output.shape}"
    if torch.allclose(submission_output, ref_output, rtol=rtol, atol=atol):
        return True, "Match"
    diff = torch.abs(submission_output.float() - ref_output.float())
    return False, f"Output mismatch: max_diff={diff.max().item():.6f}"
'''
