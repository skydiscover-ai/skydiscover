"""
Reference implementation for float16 vector addition Triton kernel.
C = A + B
"""

import math
try:
    import torch
except ImportError:
    torch = None  # Modal-only mode — functions below won't be called locally

# ---------------------------------------------------------------------------
# Reward parameters
# ---------------------------------------------------------------------------

CORRECTNESS_WEIGHT = 0.3
SPEED_WEIGHT = 1.0
SPEED_MAX_REWARD = 10.0

# ---------------------------------------------------------------------------
# Test / benchmark cases
# ---------------------------------------------------------------------------

TEST_CASES = [
    {"N": 256, "seed": 42},
    {"N": 512, "seed": 123},
    {"N": 1024, "seed": 456},
    {"N": 2048, "seed": 789},
]

BENCHMARK_CASES = [
    {"N": 1024, "seed": 1001},
    {"N": 2048, "seed": 1002},
    {"N": 4096, "seed": 1003},
    {"N": 8192, "seed": 1004},
]

# ---------------------------------------------------------------------------
# Reference kernel
# ---------------------------------------------------------------------------


def ref_kernel(data):
    a, b = data
    return a + b


def generate_input(N, seed):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    a = torch.randn(N, N, device="cuda", dtype=torch.float16, generator=gen)
    b = torch.randn(N, N, device="cuda", dtype=torch.float16, generator=gen)
    return (a, b)


def check_implementation(data, output, rtol=1e-3, atol=1e-3):
    ref_out = ref_kernel(data)
    if output.shape != ref_out.shape:
        return False, f"Shape mismatch: expected {ref_out.shape}, got {output.shape}"
    if output.dtype != torch.float16:
        return False, f"Dtype mismatch: expected float16, got {output.dtype}"
    if torch.allclose(output, ref_out, rtol=rtol, atol=atol):
        return True, "Match"
    diff = torch.abs(output.float() - ref_out.float())
    return False, f"Output mismatch: max_diff={diff.max().item():.6f}"


# ---------------------------------------------------------------------------
# Self-contained reference code for Modal execution
# ---------------------------------------------------------------------------

MODAL_REFERENCE_CODE = '''
import torch

def ref_kernel(data):
    a, b = data
    return a + b

def generate_input(N, seed):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    a = torch.randn(N, N, device="cuda", dtype=torch.float16, generator=gen)
    b = torch.randn(N, N, device="cuda", dtype=torch.float16, generator=gen)
    return (a, b)

def check_implementation(data, output, rtol=1e-3, atol=1e-3):
    ref_out = ref_kernel(data)
    if output.shape != ref_out.shape:
        return False, f"Shape mismatch: expected {ref_out.shape}, got {output.shape}"
    if output.dtype != torch.float16:
        return False, f"Dtype mismatch: expected float16, got {output.dtype}"
    if torch.allclose(output, ref_out, rtol=rtol, atol=atol):
        return True, "Match"
    diff = torch.abs(output.float() - ref_out.float())
    return False, f"Output mismatch: max_diff={diff.max().item():.6f}"
'''
