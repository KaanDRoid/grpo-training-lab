"""Triton tutorial 01: vector addition, rewritten and validated locally."""

import torch
import triton
import triton.language as tl


@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(axis=0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return ``x + y`` using a one-dimensional Triton launch."""
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("vector_add requires CUDA tensors")
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {tuple(x.shape)} != {tuple(y.shape)}")
    if x.dtype != y.dtype:
        raise ValueError(f"dtype mismatch: {x.dtype} != {y.dtype}")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("vector_add requires contiguous tensors")

    output = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return output
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
