# EVOLVE-BLOCK-START
"""
Initial Grayscale submission with Triton kernel.
Y = 0.2989 R + 0.5870 G + 0.1140 B
"""

import torch
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel(
    rgb_ptr,
    out_ptr,
    H,
    W,
    stride_h,
    stride_w,
    stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pixels = H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels

    h_idx = offsets // W
    w_idx = offsets % W

    r_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 0 * stride_c
    g_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 1 * stride_c
    b_ptr = rgb_ptr + h_idx * stride_h + w_idx * stride_w + 2 * stride_c

    r = tl.load(r_ptr, mask=mask)
    g = tl.load(g_ptr, mask=mask)
    b = tl.load(b_ptr, mask=mask)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    out_offsets = h_idx * W + w_idx
    tl.store(out_ptr + out_offsets, gray, mask=mask)


def custom_kernel(data):
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    rgb = rgb.contiguous()
    stride_h, stride_w, stride_c = rgb.stride()
    n_pixels = H * W
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel[grid](
        rgb,
        output,
        H,
        W,
        stride_h,
        stride_w,
        stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


# EVOLVE-BLOCK-END
