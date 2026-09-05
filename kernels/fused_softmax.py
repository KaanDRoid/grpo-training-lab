"""Triton tutorial 02: one-program-per-row numerically stable softmax."""

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(output_ptr, input_ptr, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    values = tl.load(input_ptr + row * n_cols + offsets, mask=mask, other=-float("inf"))
    values = values.to(tl.float32)
    shifted = values - tl.max(values, axis=0)
    numerator = tl.exp(shifted)
    denominator = tl.sum(numerator, axis=0)
    probabilities = numerator / denominator
    tl.store(output_ptr + row * n_cols + offsets, probabilities, mask=mask)


def fused_softmax(x: torch.Tensor) -> torch.Tensor:
    """Compute softmax over the last dimension of a contiguous CUDA matrix."""
    if not x.is_cuda:
        raise ValueError("fused_softmax requires a CUDA tensor")
    if x.ndim != 2:
        raise ValueError(f"fused_softmax expects [rows, cols], got {tuple(x.shape)}")
    if not x.is_contiguous():
        raise ValueError("fused_softmax requires a contiguous tensor")
    rows, cols = x.shape
    if rows == 0 or cols == 0:
        return torch.empty_like(x)

    block_size = triton.next_power_of_2(cols)
    if block_size > 65_536:
        raise ValueError(f"row width {cols} exceeds the tutorial kernel limit")
    num_warps = 8 if block_size >= 2_048 else 4
    output = torch.empty_like(x)
    _softmax_kernel[(rows,)](
        output, x, n_cols=cols, BLOCK_SIZE=block_size, num_warps=num_warps
    )
    return output
