"""
Reference implementation for MLA Decode (Multi-Head Latent Attention) Triton kernel.
Same test cases, benchmarks, generate_input, ref_kernel, and check_implementation.
"""

import math
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Scoring and benchmark configuration (read by shared_eval.py)
# ---------------------------------------------------------------------------

SCORE_SCALE = 3000.0

# MLA uses wall-clock timing, 1% rel error, no wall clock timeout, torch.no_grad()
BENCH_USE_CUDA_EVENTS = False
BENCH_REL_ERROR = 0.01
BENCH_WALL_TIMEOUT_NS = None
BENCH_NO_GRAD = True
BENCH_MAX_REPEATS = 100
BENCH_MAX_TIME_NS = 10e9
BENCH_WARMUP_STYLE = 'timed_calls'

# ---------------------------------------------------------------------------
# Model classes (needed by both reference and submissions)
# ---------------------------------------------------------------------------


class RoPE(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        theta = 10000 ** (-torch.arange(0, d_model // 2, dtype=torch.bfloat16) / (d_model // 2))
        self.register_buffer("theta", theta)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        seq_len = x.size(-2)
        d_model = x.size(-1)
        assert d_model == self.d_model
        seq_idx = torch.arange(start_pos, start_pos + seq_len, device=x.device)
        idx_theta = torch.einsum('s,d->sd', seq_idx, self.theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=-1)
        cos = idx_theta2.cos().to(torch.bfloat16)
        sin = idx_theta2.sin().to(torch.bfloat16)
        return x * cos + self.rotate_half(x) * sin


class KVCache(nn.Module):
    def __init__(self, kv_cache_shape: tuple, **kwargs) -> None:
        super().__init__(**kwargs)
        self.register_buffer('data', torch.zeros(kv_cache_shape, dtype=torch.bfloat16))
        self.seq_len = 0
        self.zero()

    def zero(self) -> None:
        self.data.zero_()

    def get_data(self) -> torch.Tensor:
        return self.data

    def forward(self, c_kv: torch.Tensor) -> torch.Tensor:
        assert self.seq_len + c_kv.size(1) <= self.data.size(1), "KV Cache Exceeded"

        self.data = self.data.to(c_kv.dtype)
        self.data[
            :, self.seq_len: self.seq_len + c_kv.size(1), :
        ] = c_kv
        self.seq_len += c_kv.size(1)

        return self.data[:, :self.seq_len], self.seq_len


@dataclass
class Config:
    batch_size: int
    dim: int
    n_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    seq_len: int
    max_seq_len: int
    kv_cache_shape: tuple
    Q_proj_down_weight: torch.Tensor
    Q_proj_up_weight: torch.Tensor
    KV_proj_down_weight: torch.Tensor
    KV_proj_up_weight: torch.Tensor
    wo_weight: torch.Tensor


class MLA(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.nope_head_dim = config.qk_nope_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.Q_proj_down = nn.Linear(self.dim, self.q_lora_rank, dtype=torch.bfloat16, bias=False)
        self.KV_proj_down = nn.Linear(self.dim, self.kv_lora_rank + self.rope_head_dim, dtype=torch.bfloat16, bias=False)
        self.Q_proj_up = nn.Linear(self.q_lora_rank, (self.nope_head_dim + self.rope_head_dim) * self.n_heads, dtype=torch.bfloat16, bias=False)
        self.KV_proj_up = nn.Linear(self.kv_lora_rank, (self.nope_head_dim + self.v_head_dim) * self.n_heads, dtype=torch.bfloat16, bias=False)
        self.q_rope = RoPE(self.rope_head_dim)
        self.k_rope = RoPE(self.rope_head_dim)
        self.wo = nn.Linear(self.v_head_dim * self.n_heads, self.dim, dtype=torch.bfloat16, bias=False)
        self.eps = 1e-6

    def forward(self, x: torch.Tensor, kv_cache: KVCache) -> torch.Tensor:
        batch_size, seq_len, model_dim = x.size()

        q_lora = self.Q_proj_down(x)
        kv_lora = self.KV_proj_down(x)
        kv_lora, kv_len = kv_cache(kv_lora)
        query_pos = kv_len - 1

        q_nope_and_rope = self.Q_proj_up(q_lora).view(
            batch_size, seq_len, self.n_heads, self.nope_head_dim + self.rope_head_dim)
        q_nope, q_rope = torch.split(q_nope_and_rope, [self.nope_head_dim, self.rope_head_dim], dim=-1)

        kv_nope, k_rope = torch.split(kv_lora, [self.kv_lora_rank, self.rope_head_dim], dim=-1)
        kv_nope = self.KV_proj_up(kv_nope).view(
            batch_size, kv_len, self.n_heads, self.nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv_nope, [self.nope_head_dim, self.v_head_dim], dim=-1)

        q_rope = q_rope.permute(0, 2, 1, 3)
        q_rope = self.q_rope(q_rope, start_pos=query_pos)

        q_nope = q_nope.permute(0, 2, 1, 3)
        q = torch.concat([q_nope, q_rope], dim=-1)

        k_rope = k_rope[:, None, :, :]
        k_rope = self.k_rope(k_rope).expand(-1, self.n_heads, -1, -1)
        k_nope = k_nope.permute(0, 2, 1, 3)
        k = torch.concat([k_nope, k_rope], dim=-1)

        v = v.permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.rope_head_dim + self.nope_head_dim)
        attn = F.softmax(scores, dim=-1).to(torch.bfloat16)
        y = torch.matmul(attn, v).view(batch_size, 1, -1)
        y = self.wo(y)

        return y, kv_cache.get_data()


# ---------------------------------------------------------------------------
# Test / benchmark cases — from discover task.yml
# ---------------------------------------------------------------------------

TEST_CASES = [
    {"batchsize": 128, "dim": 7168, "dq": 1536, "prefill": 128, "seed": 9247},
    {"batchsize": 128, "dim": 7168, "dq": 1536, "prefill": 512, "seed": 2197},
    {"batchsize": 128, "dim": 7168, "dq": 1536, "prefill": 1024, "seed": 9107},
    {"batchsize": 128, "dim": 7168, "dq": 1536, "prefill": 2048, "seed": 5291},
]

BENCHMARK_CASES = [
    {"batchsize": 128, "dim": 7168, "dq": 1536, "prefill": 4096, "seed": 9817},
    {"batchsize": 128, "dim": 7168, "dq": 1536, "prefill": 6144, "seed": 5291},
]


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------


def generate_input(batchsize, dim, dq, prefill, seed):
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)

    Q_proj_down_weight = torch.randn((dq, dim), dtype=torch.bfloat16, generator=gen, device='cuda') / math.sqrt(dim)
    KV_proj_down_weight = torch.randn((512 + 64, dim), dtype=torch.bfloat16, generator=gen, device='cuda') / math.sqrt(dim)
    Q_proj_up_weight = torch.randn(((128 + 64) * 128, dq), dtype=torch.bfloat16, generator=gen, device='cuda') / math.sqrt(dq)
    KV_proj_up_weight = torch.randn(((128 + 128) * 128, 512), dtype=torch.bfloat16, generator=gen, device='cuda') / math.sqrt(512)
    wo_weight = torch.randn((dim, 128 * 128), dtype=torch.bfloat16, generator=gen, device='cuda') / math.sqrt(128 * 128)

    config = Config(
        batch_size=batchsize,
        dim=dim,
        q_lora_rank=dq,
        n_heads=128,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        seq_len=1,
        max_seq_len=8192,
        kv_cache_shape=(batchsize, 8192, 512 + 64),
        Q_proj_down_weight=Q_proj_down_weight,
        Q_proj_up_weight=Q_proj_up_weight,
        KV_proj_down_weight=KV_proj_down_weight,
        KV_proj_up_weight=KV_proj_up_weight,
        wo_weight=wo_weight,
    )
    x = torch.randn((config.batch_size, 1, config.dim), dtype=torch.bfloat16, generator=gen, device='cuda')

    kv_cache = KVCache((config.batch_size, config.max_seq_len, config.kv_lora_rank + config.qk_rope_head_dim)).to('cuda')
    pre_filled_cache = torch.randn(
        (config.batch_size, prefill, config.kv_lora_rank + config.qk_rope_head_dim),
        dtype=torch.bfloat16, generator=gen, device='cuda')
    kv_cache(pre_filled_cache)

    return config, x, kv_cache


# ---------------------------------------------------------------------------
# Reference kernel
# ---------------------------------------------------------------------------


def ref_kernel(data):
    config, x, kv_cache = data

    model = MLA(config).to('cuda')
    model.Q_proj_down.weight = nn.Parameter(config.Q_proj_down_weight)
    model.Q_proj_up.weight = nn.Parameter(config.Q_proj_up_weight)
    model.KV_proj_down.weight = nn.Parameter(config.KV_proj_down_weight)
    model.KV_proj_up.weight = nn.Parameter(config.KV_proj_up_weight)
    model.wo.weight = nn.Parameter(config.wo_weight)

    output, kv_data = model(x, kv_cache)
    return output, kv_data


# ---------------------------------------------------------------------------
# Correctness checking
# ---------------------------------------------------------------------------


@torch.no_grad()
def _verbose_allclose(received, expected, rtol=1e-05, atol=1e-08, max_print=5):
    if received.shape != expected.shape:
        return False, [f"SIZE MISMATCH. received shape: {received.shape}, expected shape: {expected.shape}"]

    diff = torch.abs(received.to(torch.float32) - expected.to(torch.float32))
    tolerance = atol + rtol * torch.abs(expected.to(torch.float32))
    tol_mismatched = diff > tolerance
    nan_mismatched = torch.logical_xor(torch.isnan(received), torch.isnan(expected))
    posinf_mismatched = torch.logical_xor(torch.isposinf(received), torch.isposinf(expected))
    neginf_mismatched = torch.logical_xor(torch.isneginf(received), torch.isneginf(expected))
    mismatched = torch.logical_or(
        torch.logical_or(tol_mismatched, nan_mismatched),
        torch.logical_or(posinf_mismatched, neginf_mismatched),
    )

    mismatched_indices = torch.nonzero(mismatched)
    num_mismatched = mismatched.count_nonzero().item()

    if num_mismatched >= 1:
        mismatch_details = [f"Number of mismatched elements: {num_mismatched}"]
        for index in mismatched_indices[:max_print]:
            i = tuple(index.tolist())
            mismatch_details.append(f"ERROR at {i}: {received[i]} {expected[i]}")
        if num_mismatched > max_print:
            mismatch_details.append(f"... and {num_mismatched - max_print} more mismatched elements.")
        return False, mismatch_details

    return True, [f"Maximum error: {torch.max(diff)}"]


def check_implementation(data, submission_output, rtol=2e-2, atol=8e-3):
    """Check submission output against reference. Returns (passed: bool, msg: str)."""
    import gc
    output_mla, output_kv = submission_output

    # Move submission output to CPU and free GPU memory before running ref kernel
    output_mla_cpu = output_mla.cpu()
    output_kv_cpu = output_kv.cpu()
    del output_mla, output_kv
    gc.collect()
    torch.cuda.empty_cache()

    config, x, kv_cache = data
    with torch.no_grad():
        expected_mla, expected_kv = ref_kernel((config, x, kv_cache))

    # Move ref output to CPU and free GPU memory before comparison
    expected_mla_cpu = expected_mla.cpu()
    expected_kv_cpu = expected_kv.cpu()
    del expected_mla, expected_kv
    gc.collect()
    torch.cuda.empty_cache()

    good_mla, reasons_mla = _verbose_allclose(output_mla_cpu, expected_mla_cpu, rtol=rtol, atol=atol)
    good_kv, reasons_kv = _verbose_allclose(output_kv_cpu, expected_kv_cpu, rtol=rtol, atol=atol)

    if not good_mla:
        return False, "MLA output mismatch: " + " ".join(reasons_mla)
    if not good_kv:
        return False, "KV cache mismatch: " + " ".join(reasons_kv)

    return True, "Match"


# ---------------------------------------------------------------------------
# Self-contained reference code for Modal remote execution
# ---------------------------------------------------------------------------

MODAL_REFERENCE_CODE = r'''
import math
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


class RoPE(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        theta = 10000 ** (-torch.arange(0, d_model // 2, dtype=torch.bfloat16) / (d_model // 2))
        self.register_buffer("theta", theta)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        seq_len = x.size(-2)
        d_model = x.size(-1)
        assert d_model == self.d_model
        seq_idx = torch.arange(start_pos, start_pos + seq_len, device=x.device)
        idx_theta = torch.einsum('s,d->sd', seq_idx, self.theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=-1)
        cos = idx_theta2.cos().to(torch.bfloat16)
        sin = idx_theta2.sin().to(torch.bfloat16)
        return x * cos + self.rotate_half(x) * sin


class KVCache(nn.Module):
    def __init__(self, kv_cache_shape: tuple, **kwargs) -> None:
        super().__init__(**kwargs)
        self.register_buffer('data', torch.zeros(kv_cache_shape, dtype=torch.bfloat16))
        self.seq_len = 0
        self.zero()

    def zero(self) -> None:
        self.data.zero_()

    def get_data(self) -> torch.Tensor:
        return self.data

    def forward(self, c_kv: torch.Tensor) -> torch.Tensor:
        assert self.seq_len + c_kv.size(1) <= self.data.size(1), "KV Cache Exceeded"
        self.data = self.data.to(c_kv.dtype)
        self.data[:, self.seq_len: self.seq_len + c_kv.size(1), :] = c_kv
        self.seq_len += c_kv.size(1)
        return self.data[:, :self.seq_len], self.seq_len


@dataclass
class Config:
    batch_size: int
    dim: int
    n_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    seq_len: int
    max_seq_len: int
    kv_cache_shape: tuple
    Q_proj_down_weight: torch.Tensor
    Q_proj_up_weight: torch.Tensor
    KV_proj_down_weight: torch.Tensor
    KV_proj_up_weight: torch.Tensor
    wo_weight: torch.Tensor


class MLA(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.nope_head_dim = config.qk_nope_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.Q_proj_down = nn.Linear(self.dim, self.q_lora_rank, dtype=torch.bfloat16, bias=False)
        self.KV_proj_down = nn.Linear(self.dim, self.kv_lora_rank + self.rope_head_dim, dtype=torch.bfloat16, bias=False)
        self.Q_proj_up = nn.Linear(self.q_lora_rank, (self.nope_head_dim + self.rope_head_dim) * self.n_heads, dtype=torch.bfloat16, bias=False)
        self.KV_proj_up = nn.Linear(self.kv_lora_rank, (self.nope_head_dim + self.v_head_dim) * self.n_heads, dtype=torch.bfloat16, bias=False)
        self.q_rope = RoPE(self.rope_head_dim)
        self.k_rope = RoPE(self.rope_head_dim)
        self.wo = nn.Linear(self.v_head_dim * self.n_heads, self.dim, dtype=torch.bfloat16, bias=False)
        self.eps = 1e-6

    def forward(self, x: torch.Tensor, kv_cache: KVCache) -> torch.Tensor:
        batch_size, seq_len, model_dim = x.size()
        q_lora = self.Q_proj_down(x)
        kv_lora = self.KV_proj_down(x)
        kv_lora, kv_len = kv_cache(kv_lora)
        query_pos = kv_len - 1
        q_nope_and_rope = self.Q_proj_up(q_lora).view(
            batch_size, seq_len, self.n_heads, self.nope_head_dim + self.rope_head_dim)
        q_nope, q_rope = torch.split(q_nope_and_rope, [self.nope_head_dim, self.rope_head_dim], dim=-1)
        kv_nope, k_rope = torch.split(kv_lora, [self.kv_lora_rank, self.rope_head_dim], dim=-1)
        kv_nope = self.KV_proj_up(kv_nope).view(
            batch_size, kv_len, self.n_heads, self.nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv_nope, [self.nope_head_dim, self.v_head_dim], dim=-1)
        q_rope = q_rope.permute(0, 2, 1, 3)
        q_rope = self.q_rope(q_rope, start_pos=query_pos)
        q_nope = q_nope.permute(0, 2, 1, 3)
        q = torch.concat([q_nope, q_rope], dim=-1)
        k_rope = k_rope[:, None, :, :]
        k_rope = self.k_rope(k_rope).expand(-1, self.n_heads, -1, -1)
        k_nope = k_nope.permute(0, 2, 1, 3)
        k = torch.concat([k_nope, k_rope], dim=-1)
        v = v.permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.rope_head_dim + self.nope_head_dim)
        attn = F.softmax(scores, dim=-1).to(torch.bfloat16)
        y = torch.matmul(attn, v).view(batch_size, 1, -1)
        y = self.wo(y)
        return y, kv_cache.get_data()


def ref_kernel(data):
    config, x, kv_cache = data
    model = MLA(config).to('cuda')
    model.Q_proj_down.weight = nn.Parameter(config.Q_proj_down_weight)
    model.Q_proj_up.weight = nn.Parameter(config.Q_proj_up_weight)
    model.KV_proj_down.weight = nn.Parameter(config.KV_proj_down_weight)
    model.KV_proj_up.weight = nn.Parameter(config.KV_proj_up_weight)
    model.wo.weight = nn.Parameter(config.wo_weight)
    output, kv_data = model(x, kv_cache)
    return output, kv_data


def generate_input(batchsize, dim, dq, prefill, seed):
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    Q_proj_down_weight = torch.randn((dq, dim), dtype=torch.bfloat16, generator=gen, device='cuda') / math.sqrt(dim)
    KV_proj_down_weight = torch.randn((512 + 64, dim), dtype=torch.bfloat16, generator=gen, device='cuda') / math.sqrt(dim)
    Q_proj_up_weight = torch.randn(((128 + 64) * 128, dq), dtype=torch.bfloat16, generator=gen, device='cuda') / math.sqrt(dq)
    KV_proj_up_weight = torch.randn(((128 + 128) * 128, 512), dtype=torch.bfloat16, generator=gen, device='cuda') / math.sqrt(512)
    wo_weight = torch.randn((dim, 128 * 128), dtype=torch.bfloat16, generator=gen, device='cuda') / math.sqrt(128 * 128)
    config = Config(
        batch_size=batchsize, dim=dim, q_lora_rank=dq, n_heads=128,
        kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64,
        v_head_dim=128, seq_len=1, max_seq_len=8192,
        kv_cache_shape=(batchsize, 8192, 512 + 64),
        Q_proj_down_weight=Q_proj_down_weight, Q_proj_up_weight=Q_proj_up_weight,
        KV_proj_down_weight=KV_proj_down_weight, KV_proj_up_weight=KV_proj_up_weight,
        wo_weight=wo_weight,
    )
    x = torch.randn((config.batch_size, 1, config.dim), dtype=torch.bfloat16, generator=gen, device='cuda')
    kv_cache = KVCache((config.batch_size, config.max_seq_len, config.kv_lora_rank + config.qk_rope_head_dim)).to('cuda')
    pre_filled_cache = torch.randn(
        (config.batch_size, prefill, config.kv_lora_rank + config.qk_rope_head_dim),
        dtype=torch.bfloat16, generator=gen, device='cuda')
    kv_cache(pre_filled_cache)
    return config, x, kv_cache


@torch.no_grad()
def _verbose_allclose(received, expected, rtol=1e-05, atol=1e-08, max_print=5):
    if received.shape != expected.shape:
        return False, [f"SIZE MISMATCH. received shape: {received.shape}, expected shape: {expected.shape}"]
    diff = torch.abs(received.to(torch.float32) - expected.to(torch.float32))
    tolerance = atol + rtol * torch.abs(expected.to(torch.float32))
    tol_mismatched = diff > tolerance
    nan_mismatched = torch.logical_xor(torch.isnan(received), torch.isnan(expected))
    posinf_mismatched = torch.logical_xor(torch.isposinf(received), torch.isposinf(expected))
    neginf_mismatched = torch.logical_xor(torch.isneginf(received), torch.isneginf(expected))
    mismatched = torch.logical_or(
        torch.logical_or(tol_mismatched, nan_mismatched),
        torch.logical_or(posinf_mismatched, neginf_mismatched),
    )
    mismatched_indices = torch.nonzero(mismatched)
    num_mismatched = mismatched.count_nonzero().item()
    if num_mismatched >= 1:
        mismatch_details = [f"Number of mismatched elements: {num_mismatched}"]
        for index in mismatched_indices[:max_print]:
            i = tuple(index.tolist())
            mismatch_details.append(f"ERROR at {i}: {received[i]} {expected[i]}")
        if num_mismatched > max_print:
            mismatch_details.append(f"... and {num_mismatched - max_print} more mismatched elements.")
        return False, mismatch_details
    return True, [f"Maximum error: {torch.max(diff)}"]


def check_implementation(data, submission_output, rtol=2e-2, atol=8e-3):
    import gc
    output_mla, output_kv = submission_output
    # Move submission output to CPU and free GPU memory before running ref kernel
    output_mla_cpu = output_mla.cpu()
    output_kv_cpu = output_kv.cpu()
    del output_mla, output_kv
    gc.collect()
    torch.cuda.empty_cache()
    config, x, kv_cache = data
    with torch.no_grad():
        expected_mla, expected_kv = ref_kernel((config, x, kv_cache))
    # Move ref output to CPU and free GPU memory before comparison
    expected_mla_cpu = expected_mla.cpu()
    expected_kv_cpu = expected_kv.cpu()
    del expected_mla, expected_kv
    gc.collect()
    torch.cuda.empty_cache()
    good_mla, reasons_mla = _verbose_allclose(output_mla_cpu, expected_mla_cpu, rtol=rtol, atol=atol)
    good_kv, reasons_kv = _verbose_allclose(output_kv_cpu, expected_kv_cpu, rtol=rtol, atol=atol)
    if not good_mla:
        return False, "MLA output mismatch: " + " ".join(reasons_mla)
    if not good_kv:
        return False, "KV cache mismatch: " + " ".join(reasons_kv)
    return True, "Match"
'''
