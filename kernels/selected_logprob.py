"""Forward-only selected-token logprob kernels.

A memory-efficient selected-token logprob operator.  It uses the
Liger-style decomposition: PyTorch performs reliable tensor-core GEMMs in small
row chunks, while Triton reduces each chunk's vocabulary dimension without materializing a full
``log_softmax`` tensor.

The vocabulary reduction is two-stage so real model widths such as Qwen2.5's 151,936 entries do
not require one enormous power-of-two Triton program:

1. each program reduces one 1,024-column tile to ``(max, sum(exp(x-max)))`` in fp32;
2. one program per row combines those partials and subtracts the final log-normalizer from the
   selected target logit.

The original public wrappers remain explicitly forward-only. A separate autograd wrapper adds a
recompute-in-backward path: one Triton kernel forms ``upstream * (one_hot - softmax)`` for a row
chunk, then PyTorch/cuBLAS applies the linear-layer Jacobians. This separation prevents accidental
use of an unvalidated backward while keeping the forward baseline directly benchmarkable.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


_VOCAB_TILE = 1024
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _partial_lse_kernel(
    logits_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    n_cols: tl.constexpr,
    n_tiles: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    tile = tl.program_id(axis=1)
    offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    values = tl.load(
        logits_ptr + row * n_cols + offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)

    tile_max = tl.max(values, axis=0)
    tile_sum = tl.sum(tl.exp(values - tile_max), axis=0)
    partial_offset = row * n_tiles + tile
    tl.store(partial_max_ptr + partial_offset, tile_max)
    tl.store(partial_sum_ptr + partial_offset, tile_sum)


@triton.jit
def _finalize_logprob_kernel(
    logits_ptr,
    target_ids_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    output_ptr,
    log_normalizer_ptr,
    n_cols: tl.constexpr,
    n_tiles: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    tile_offsets = tl.arange(0, BLOCK_SIZE)
    tile_mask = tile_offsets < n_tiles
    partial_offsets = row * n_tiles + tile_offsets
    partial_max = tl.load(
        partial_max_ptr + partial_offsets,
        mask=tile_mask,
        other=-float("inf"),
    )
    partial_sum = tl.load(
        partial_sum_ptr + partial_offsets,
        mask=tile_mask,
        other=0.0,
    )

    row_max = tl.max(partial_max, axis=0)
    row_sum = tl.sum(partial_sum * tl.exp(partial_max - row_max), axis=0)
    log_normalizer = row_max + tl.log(row_sum)

    target_id = tl.load(target_ids_ptr + row)
    target_logit = tl.load(logits_ptr + row * n_cols + target_id).to(tl.float32)
    tl.store(output_ptr + row, target_logit - log_normalizer)
    tl.store(log_normalizer_ptr + row, log_normalizer)


@triton.jit
def _logprob_backward_kernel(
    logits_ptr,
    target_ids_ptr,
    log_normalizer_ptr,
    grad_output_ptr,
    grad_logits_ptr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    tile = tl.program_id(axis=1)
    offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    logits = tl.load(
        logits_ptr + row * n_cols + offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    target_id = tl.load(target_ids_ptr + row)
    log_normalizer = tl.load(log_normalizer_ptr + row)
    upstream = tl.load(grad_output_ptr + row).to(tl.float32)
    probability = tl.exp(logits - log_normalizer)
    one_hot = offsets == target_id
    grad_logits = upstream * (one_hot.to(tl.float32) - probability)
    tl.store(grad_logits_ptr + row * n_cols + offsets, grad_logits, mask=mask)


def _validate_targets(target_ids: torch.Tensor, rows: int, vocab_size: int, device) -> None:
    if not target_ids.is_cuda or target_ids.device != device:
        raise ValueError("target_ids must be on the same CUDA device as the inputs")
    if target_ids.ndim != 1 or target_ids.shape[0] != rows:
        raise ValueError(f"target_ids must have shape ({rows},), got {tuple(target_ids.shape)}")
    if target_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("target_ids must use torch.int32 or torch.int64")
    if rows and bool(torch.any((target_ids < 0) | (target_ids >= vocab_size)).item()):
        raise ValueError(f"target_ids must lie in [0, {vocab_size})")


def _selected_logprob_unchecked(
    logits: torch.Tensor, target_ids: torch.Tensor, *, return_log_normalizer: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    rows, vocab_size = logits.shape
    output = torch.empty(rows, device=logits.device, dtype=torch.float32)
    log_normalizer = torch.empty(rows, device=logits.device, dtype=torch.float32)
    if rows == 0:
        return (output, log_normalizer) if return_log_normalizer else output

    n_tiles = triton.cdiv(vocab_size, _VOCAB_TILE)
    partial_max = torch.empty((rows, n_tiles), device=logits.device, dtype=torch.float32)
    partial_sum = torch.empty_like(partial_max)
    _partial_lse_kernel[(rows, n_tiles)](
        logits,
        partial_max,
        partial_sum,
        n_cols=vocab_size,
        n_tiles=n_tiles,
        BLOCK_SIZE=_VOCAB_TILE,
        num_warps=8,
    )

    final_block = triton.next_power_of_2(n_tiles)
    if final_block > 1024:
        raise ValueError(f"vocabulary size {vocab_size} exceeds the current 1,048,576 limit")
    _finalize_logprob_kernel[(rows,)](
        logits,
        target_ids,
        partial_max,
        partial_sum,
        output,
        log_normalizer,
        n_cols=vocab_size,
        n_tiles=n_tiles,
        BLOCK_SIZE=final_block,
        num_warps=4,
    )
    return (output, log_normalizer) if return_log_normalizer else output


def _logprob_backward_unchecked(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    log_normalizer: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    rows, vocab_size = logits.shape
    grad_logits = torch.empty_like(logits)
    if rows == 0:
        return grad_logits
    n_tiles = triton.cdiv(vocab_size, _VOCAB_TILE)
    _logprob_backward_kernel[(rows, n_tiles)](
        logits,
        target_ids,
        log_normalizer,
        grad_output.contiguous(),
        grad_logits,
        n_cols=vocab_size,
        BLOCK_SIZE=_VOCAB_TILE,
        num_warps=8,
    )
    return grad_logits


def selected_logprob_from_logits(
    logits: torch.Tensor, target_ids: torch.Tensor
) -> torch.Tensor:
    """Return selected logprobs ``[N]`` from contiguous logits ``[N,V]`` in fp32."""
    if not logits.is_cuda:
        raise ValueError("selected_logprob_from_logits requires CUDA logits")
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [N,V], got {tuple(logits.shape)}")
    if not logits.is_contiguous():
        raise ValueError("logits must be contiguous")
    if logits.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported logits dtype: {logits.dtype}")
    if logits.shape[1] == 0:
        raise ValueError("vocabulary dimension must be non-empty")
    if torch.is_grad_enabled() and logits.requires_grad:
        raise RuntimeError("selected_logprob_from_logits is forward-only; backward is not implemented")

    target_ids = target_ids.contiguous()
    _validate_targets(target_ids, logits.shape[0], logits.shape[1], logits.device)
    return _selected_logprob_unchecked(logits, target_ids)


def chunked_linear_selected_logprob(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target_ids: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    chunk_rows: int = 16,
) -> torch.Tensor:
    """Compute selected logprobs for ``hidden @ weight.T + bias`` without full ``[N,V]`` storage.

    ``hidden`` is ``[N,H]``, ``weight`` is ``[V,H]``, and ``target_ids`` is ``[N]``.  Only a
    ``[min(chunk_rows,N),V]`` logits chunk is live at once; no vocabulary-sized softmax output is
    created.  The returned values are fp32 regardless of input precision.
    """
    if not hidden.is_cuda or not weight.is_cuda or hidden.device != weight.device:
        raise ValueError("hidden and weight must be on the same CUDA device")
    if hidden.ndim != 2 or weight.ndim != 2:
        raise ValueError("hidden and weight must both be rank-2 tensors")
    if hidden.shape[1] != weight.shape[1]:
        raise ValueError(
            f"hidden width {hidden.shape[1]} does not match weight width {weight.shape[1]}"
        )
    if hidden.dtype != weight.dtype or hidden.dtype not in _SUPPORTED_DTYPES:
        raise ValueError("hidden and weight must have the same supported floating dtype")
    if not hidden.is_contiguous() or not weight.is_contiguous():
        raise ValueError("hidden and weight must be contiguous")
    if chunk_rows < 1:
        raise ValueError(f"chunk_rows must be positive, got {chunk_rows}")
    if weight.shape[0] == 0:
        raise ValueError("vocabulary dimension must be non-empty")
    if bias is not None:
        if (
            not bias.is_cuda
            or bias.device != hidden.device
            or bias.dtype != hidden.dtype
            or bias.ndim != 1
            or bias.shape[0] != weight.shape[0]
            or not bias.is_contiguous()
        ):
            raise ValueError("bias must be contiguous [V] with the same device and dtype")
    if torch.is_grad_enabled() and (
        hidden.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)
    ):
        raise RuntimeError("chunked_linear_selected_logprob is forward-only; backward is not implemented")

    rows, vocab_size = hidden.shape[0], weight.shape[0]
    target_ids = target_ids.contiguous()
    _validate_targets(target_ids, rows, vocab_size, hidden.device)
    output = torch.empty(rows, device=hidden.device, dtype=torch.float32)

    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        logits = F.linear(hidden[start:end], weight, bias)
        output[start:end] = _selected_logprob_unchecked(logits, target_ids[start:end])
    return output


class _ChunkedLinearSelectedLogprob(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, target_ids, bias, chunk_rows):
        ctx.chunk_rows = int(chunk_rows)
        ctx.has_bias = bias is not None
        bias_to_save = bias if bias is not None else hidden.new_empty(0)
        ctx.save_for_backward(hidden, weight, target_ids, bias_to_save)

        if hidden.dtype == torch.float64:
            logits = F.linear(hidden, weight, bias)
            return torch.log_softmax(logits, dim=-1).gather(
                1, target_ids[:, None]
            ).squeeze(1)
        # Function.forward runs with grad mode disabled, so the explicit forward-only wrapper is
        # safe to reuse here without silently dropping an outer autograd graph.
        return chunked_linear_selected_logprob(
            hidden, weight, target_ids, bias, chunk_rows=ctx.chunk_rows
        )

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight, target_ids, saved_bias = ctx.saved_tensors
        bias = saved_bias if ctx.has_bias else None
        need_hidden, need_weight, _, need_bias, _ = ctx.needs_input_grad
        grad_hidden = torch.empty_like(hidden) if need_hidden else None
        grad_weight = torch.zeros_like(weight) if need_weight else None
        grad_bias = torch.zeros_like(bias) if need_bias else None

        for start in range(0, hidden.shape[0], ctx.chunk_rows):
            end = min(start + ctx.chunk_rows, hidden.shape[0])
            hidden_chunk = hidden[start:end]
            target_chunk = target_ids[start:end]
            logits = F.linear(hidden_chunk, weight, bias)

            if hidden.dtype == torch.float64:
                probabilities = torch.softmax(logits, dim=-1)
                grad_logits = -probabilities
                grad_logits.scatter_add_(
                    1,
                    target_chunk[:, None],
                    torch.ones_like(target_chunk[:, None], dtype=grad_logits.dtype),
                )
                grad_logits.mul_(grad_output[start:end, None])
            else:
                _, log_normalizer = _selected_logprob_unchecked(
                    logits, target_chunk, return_log_normalizer=True
                )
                grad_logits = _logprob_backward_unchecked(
                    logits, target_chunk, log_normalizer, grad_output[start:end]
                )

            if need_hidden:
                grad_hidden[start:end] = grad_logits @ weight
            if need_weight:
                grad_weight.add_(grad_logits.transpose(0, 1) @ hidden_chunk)
            if need_bias:
                grad_bias.add_(grad_logits.sum(dim=0))

        return grad_hidden, grad_weight, None, grad_bias, None


def chunked_linear_selected_logprob_autograd(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target_ids: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    chunk_rows: int = 16,
) -> torch.Tensor:
    """Autograd-enabled selected logprob with chunked recomputation in backward.

    CUDA fp16/bf16/fp32 uses the Triton forward and logit-gradient kernels. Float64 uses an
    intentionally small PyTorch path so ``torch.autograd.gradcheck`` can inspect the exact custom
    backward formula at double precision. CPU is supported only for this float64 gradcheck path.
    """
    if hidden.ndim != 2 or weight.ndim != 2:
        raise ValueError("hidden and weight must both be rank-2 tensors")
    if hidden.shape[1] != weight.shape[1]:
        raise ValueError(
            f"hidden width {hidden.shape[1]} does not match weight width {weight.shape[1]}"
        )
    if hidden.device != weight.device or hidden.dtype != weight.dtype:
        raise ValueError("hidden and weight must share device and dtype")
    if hidden.dtype != torch.float64 and (
        not hidden.is_cuda or hidden.dtype not in _SUPPORTED_DTYPES
    ):
        raise ValueError("runtime backward requires CUDA fp16, bf16, or fp32")
    if not hidden.is_contiguous() or not weight.is_contiguous():
        raise ValueError("hidden and weight must be contiguous")
    if chunk_rows < 1:
        raise ValueError(f"chunk_rows must be positive, got {chunk_rows}")
    if weight.shape[0] == 0:
        raise ValueError("vocabulary dimension must be non-empty")
    if bias is not None and (
        bias.device != hidden.device
        or bias.dtype != hidden.dtype
        or bias.ndim != 1
        or bias.shape[0] != weight.shape[0]
        or not bias.is_contiguous()
    ):
        raise ValueError("bias must be contiguous [V] with the same device and dtype")

    target_ids = target_ids.contiguous()
    if target_ids.device != hidden.device:
        raise ValueError("target_ids must be on the same device as hidden")
    if target_ids.ndim != 1 or target_ids.shape[0] != hidden.shape[0]:
        raise ValueError(
            f"target_ids must have shape ({hidden.shape[0]},), got {tuple(target_ids.shape)}"
        )
    if target_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("target_ids must use torch.int32 or torch.int64")
    if target_ids.numel() and bool(
        torch.any((target_ids < 0) | (target_ids >= weight.shape[0])).item()
    ):
        raise ValueError(f"target_ids must lie in [0, {weight.shape[0]})")
    return _ChunkedLinearSelectedLogprob.apply(
        hidden, weight, target_ids, bias, chunk_rows
    )
