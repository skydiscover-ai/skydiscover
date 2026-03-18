"""
Reference implementation for Triangle Multiplicative Update (TriMul) Triton kernel.
Core operation for AlphaFold3, Chai, Protenix protein structure models.
Same test cases, benchmarks, generate_input, ref_kernel, and check_implementation.
"""

import math
import torch
from torch import nn, einsum

# ---------------------------------------------------------------------------
# Scoring and benchmark configuration (read by shared_eval.py)
# ---------------------------------------------------------------------------

SCORE_SCALE = 3000.0

# trimul uses CUDA events timing, 0.1% rel error, 120s wall clock timeout
BENCH_USE_CUDA_EVENTS = True
BENCH_REL_ERROR = 0.001
BENCH_WALL_TIMEOUT_NS = 120e9
BENCH_NO_GRAD = False
BENCH_MAX_REPEATS = 100
BENCH_MAX_TIME_NS = 10e9
BENCH_WARMUP_STYLE = 'tiny_benchmark'

# ---------------------------------------------------------------------------
# Test / benchmark cases — full set from discover task.yml
# ---------------------------------------------------------------------------

TEST_CASES = [
    {"seqlen": 32, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 9371, "nomask": True, "distribution": "normal"},
    {"seqlen": 32, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 1092, "nomask": False, "distribution": "normal"},
    {"seqlen": 64, "bs": 2, "dim": 256, "hiddendim": 128, "seed": 2291, "nomask": True, "distribution": "normal"},
    {"seqlen": 64, "bs": 2, "dim": 256, "hiddendim": 128, "seed": 210284, "nomask": False, "distribution": "normal"},
    {"seqlen": 128, "bs": 1, "dim": 768, "hiddendim": 128, "seed": 81934, "nomask": True, "distribution": "normal"},
    {"seqlen": 256, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 1932, "nomask": True, "distribution": "normal"},
    {"seqlen": 256, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 10432, "nomask": False, "distribution": "normal"},
    {"seqlen": 768, "bs": 2, "dim": 128, "hiddendim": 128, "seed": 731, "nomask": True, "distribution": "normal"},
    {"seqlen": 1024, "bs": 1, "dim": 384, "hiddendim": 128, "seed": 53121, "nomask": False, "distribution": "normal"},
    {"seqlen": 1024, "bs": 1, "dim": 768, "hiddendim": 128, "seed": 31, "nomask": True, "distribution": "normal"},
    {"seqlen": 1024, "bs": 1, "dim": 768, "hiddendim": 128, "seed": 4921, "nomask": False, "distribution": "normal"},
    {"seqlen": 32, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 937321, "nomask": True, "distribution": "cauchy"},
    {"seqlen": 64, "bs": 2, "dim": 256, "hiddendim": 128, "seed": 2291, "nomask": True, "distribution": "cauchy"},
    {"seqlen": 128, "bs": 1, "dim": 768, "hiddendim": 128, "seed": 8134, "nomask": True, "distribution": "cauchy"},
    {"seqlen": 256, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 932, "nomask": True, "distribution": "cauchy"},
    {"seqlen": 768, "bs": 2, "dim": 128, "hiddendim": 128, "seed": 31, "nomask": True, "distribution": "cauchy"},
    {"seqlen": 1024, "bs": 1, "dim": 384, "hiddendim": 128, "seed": 5321, "nomask": False, "distribution": "cauchy"},
    {"seqlen": 1024, "bs": 1, "dim": 768, "hiddendim": 128, "seed": 491, "nomask": False, "distribution": "cauchy"},
]

BENCHMARK_CASES = [
    {"seqlen": 256, "bs": 2, "dim": 128, "hiddendim": 128, "seed": 9371, "nomask": True, "distribution": "normal"},
    {"seqlen": 768, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 381, "nomask": True, "distribution": "cauchy"},
    {"seqlen": 256, "bs": 2, "dim": 384, "hiddendim": 128, "seed": 2301, "nomask": False, "distribution": "normal"},
    {"seqlen": 512, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 12819, "nomask": True, "distribution": "normal"},
    {"seqlen": 1024, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 381, "nomask": True, "distribution": "cauchy"},
    {"seqlen": 768, "bs": 1, "dim": 384, "hiddendim": 128, "seed": 481, "nomask": False, "distribution": "normal"},
    {"seqlen": 1024, "bs": 1, "dim": 384, "hiddendim": 128, "seed": 23291, "nomask": True, "distribution": "normal"},
]

# ---------------------------------------------------------------------------
# Reference kernel
# ---------------------------------------------------------------------------


class _TriMul(nn.Module):
    def __init__(self, dim, hidden_dim, device="cuda"):
        super().__init__()
        self.norm = nn.LayerNorm(dim, device=device)
        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, device=device)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, device=device)
        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, device=device)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, device=device)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, device=device)
        self.to_out_norm = nn.LayerNorm(hidden_dim, device=device)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, device=device)

    def forward(self, x, mask):
        x = self.norm(x)
        left = self.left_proj(x)
        right = self.right_proj(x)
        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask
        left = left * self.left_gate(x).sigmoid()
        right = right * self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()
        out = einsum('... i k d, ... j k d -> ... i j d', left, right)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def ref_kernel(data):
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        input_tensor, mask, weights, config = data
        trimul = _TriMul(dim=config["dim"], hidden_dim=config["hidden_dim"],
                         device=input_tensor.device)
        trimul.norm.weight = nn.Parameter(weights['norm.weight'])
        trimul.norm.bias = nn.Parameter(weights['norm.bias'])
        trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'])
        trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'])
        trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'])
        trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'])
        trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'])
        trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'])
        trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'])
        trimul.to_out.weight = nn.Parameter(weights['to_out.weight'])
        return trimul(input_tensor, mask)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


def generate_input(seqlen, bs, dim, hiddendim, seed, nomask, distribution="normal"):
    hidden_dim = hiddendim
    config = {"hidden_dim": hidden_dim, "dim": dim}
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)

    if distribution == "cauchy":
        u = torch.empty((bs, seqlen, seqlen, dim), device="cuda", dtype=torch.float32)
        u.uniform_(0.0, 1.0, generator=gen)
        input_tensor = 2.0 * torch.tan(math.pi * (u - 0.5))
    else:
        input_tensor = torch.randn(
            (bs, seqlen, seqlen, dim), device='cuda', dtype=torch.float32, generator=gen
        ).contiguous()

    if nomask:
        mask = torch.ones(bs, seqlen, seqlen, device="cuda")
    else:
        mask = torch.randint(0, 2, (bs, seqlen, seqlen), device="cuda", generator=gen).float()

    weights = {
        "norm.weight": torch.randn(dim, device="cuda"),
        "norm.bias": torch.randn(dim, device="cuda"),
        "left_proj.weight": torch.randn(hidden_dim, dim, device="cuda") / math.sqrt(hidden_dim),
        "right_proj.weight": torch.randn(hidden_dim, dim, device="cuda") / math.sqrt(hidden_dim),
        "left_gate.weight": torch.randn(hidden_dim, dim, device="cuda") / math.sqrt(hidden_dim),
        "right_gate.weight": torch.randn(hidden_dim, dim, device="cuda") / math.sqrt(hidden_dim),
        "out_gate.weight": torch.randn(hidden_dim, dim, device="cuda") / math.sqrt(hidden_dim),
        "to_out_norm.weight": torch.randn(hidden_dim, device="cuda"),
        "to_out_norm.bias": torch.randn(hidden_dim, device="cuda"),
        "to_out.weight": torch.randn(dim, hidden_dim, device="cuda") / math.sqrt(dim),
    }
    return (input_tensor, mask, weights, config)


def check_implementation(data, submission_output, rtol=2e-2, atol=2e-2):
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        ref_output = ref_kernel(data)
        if ref_output.shape != submission_output.shape:
            return False, f"Shape mismatch: {ref_output.shape} vs {submission_output.shape}"
        if torch.allclose(ref_output.float(), submission_output.float(), rtol=rtol, atol=atol):
            return True, "Match"
        diff = torch.abs(ref_output.float() - submission_output.float())
        return False, f"max_diff={diff.max().item():.6f}, avg_diff={diff.mean().item():.6f}"
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


# ---------------------------------------------------------------------------
# Self-contained reference code for Modal remote execution
# ---------------------------------------------------------------------------

MODAL_REFERENCE_CODE = r'''
import math
import torch
from torch import nn, einsum


class _TriMul(nn.Module):
    def __init__(self, dim, hidden_dim, device="cuda"):
        super().__init__()
        self.norm = nn.LayerNorm(dim, device=device)
        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, device=device)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, device=device)
        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, device=device)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, device=device)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, device=device)
        self.to_out_norm = nn.LayerNorm(hidden_dim, device=device)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, device=device)

    def forward(self, x, mask):
        x = self.norm(x)
        left = self.left_proj(x)
        right = self.right_proj(x)
        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask
        left = left * self.left_gate(x).sigmoid()
        right = right * self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()
        out = einsum('... i k d, ... j k d -> ... i j d', left, right)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def ref_kernel(data):
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        input_tensor, mask, weights, config = data
        trimul = _TriMul(dim=config["dim"], hidden_dim=config["hidden_dim"],
                         device=input_tensor.device)
        trimul.norm.weight = nn.Parameter(weights['norm.weight'])
        trimul.norm.bias = nn.Parameter(weights['norm.bias'])
        trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'])
        trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'])
        trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'])
        trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'])
        trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'])
        trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'])
        trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'])
        trimul.to_out.weight = nn.Parameter(weights['to_out.weight'])
        return trimul(input_tensor, mask)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


def generate_input(seqlen, bs, dim, hiddendim, seed, nomask, distribution="normal"):
    hidden_dim = hiddendim
    config = {"hidden_dim": hidden_dim, "dim": dim}
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)

    if distribution == "cauchy":
        u = torch.empty((bs, seqlen, seqlen, dim), device="cuda", dtype=torch.float32)
        u.uniform_(0.0, 1.0, generator=gen)
        input_tensor = 2.0 * torch.tan(math.pi * (u - 0.5))
    else:
        input_tensor = torch.randn(
            (bs, seqlen, seqlen, dim), device='cuda', dtype=torch.float32, generator=gen
        ).contiguous()

    if nomask:
        mask = torch.ones(bs, seqlen, seqlen, device="cuda")
    else:
        mask = torch.randint(0, 2, (bs, seqlen, seqlen), device="cuda", generator=gen).float()

    weights = {
        "norm.weight": torch.randn(dim, device="cuda"),
        "norm.bias": torch.randn(dim, device="cuda"),
        "left_proj.weight": torch.randn(hidden_dim, dim, device="cuda") / math.sqrt(hidden_dim),
        "right_proj.weight": torch.randn(hidden_dim, dim, device="cuda") / math.sqrt(hidden_dim),
        "left_gate.weight": torch.randn(hidden_dim, dim, device="cuda") / math.sqrt(hidden_dim),
        "right_gate.weight": torch.randn(hidden_dim, dim, device="cuda") / math.sqrt(hidden_dim),
        "out_gate.weight": torch.randn(hidden_dim, dim, device="cuda") / math.sqrt(hidden_dim),
        "to_out_norm.weight": torch.randn(hidden_dim, device="cuda"),
        "to_out_norm.bias": torch.randn(hidden_dim, device="cuda"),
        "to_out.weight": torch.randn(dim, hidden_dim, device="cuda") / math.sqrt(dim),
    }
    return (input_tensor, mask, weights, config)


def check_implementation(data, submission_output, rtol=2e-2, atol=2e-2):
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        ref_output = ref_kernel(data)
        if ref_output.shape != submission_output.shape:
            return False, f"Shape mismatch: {ref_output.shape} vs {submission_output.shape}"
        if torch.allclose(ref_output.float(), submission_output.float(), rtol=rtol, atol=atol):
            return True, "Match"
        diff = torch.abs(ref_output.float() - submission_output.float())
        return False, f"max_diff={diff.max().item():.6f}, avg_diff={diff.mean().item():.6f}"
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn
'''
