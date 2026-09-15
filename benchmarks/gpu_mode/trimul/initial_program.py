# EVOLVE-BLOCK-START
"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
import triton
import triton.language as tl
from torch import einsum, nn


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum(
            "... i k d, ... j k d -> ... i j d", left.to(torch.bfloat16), right.to(torch.bfloat16)
        )

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights["norm.weight"].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights["left_proj.weight"].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights["right_proj.weight"].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights["left_gate.weight"].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights["right_gate.weight"].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights["out_gate.weight"].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights["to_out_norm.weight"].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights["to_out.weight"].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights["norm.bias"].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights["to_out_norm.bias"].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output


# EVOLVE-BLOCK-END
