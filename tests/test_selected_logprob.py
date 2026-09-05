"""Correctness and contract tests for the forward-only selected-logprob reproduction."""

import pytest
import torch
import torch.nn.functional as F

from kernels.selected_logprob import (
    chunked_linear_selected_logprob,
    selected_logprob_from_logits,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("shape", [(1, 3), (7, 257), (5, 4097), (2, 65_537)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_selected_logprob_matches_fp32_reference(shape, dtype):
    rows, vocab = shape
    torch.manual_seed(rows * vocab)
    logits = torch.randn(rows, vocab, device="cuda", dtype=dtype) * 3
    target_ids = torch.randint(0, vocab, (rows,), device="cuda")

    expected = torch.log_softmax(logits.float(), dim=-1).gather(
        1, target_ids[:, None]
    ).squeeze(1)
    actual = selected_logprob_from_logits(logits, target_ids)
    tolerance = 2e-4 if dtype == torch.float16 else 3e-5
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def test_selected_logprob_handles_qwen_vocab_and_large_logits():
    vocab = 151_936
    logits = torch.full((2, vocab), -10_000.0, device="cuda", dtype=torch.float32)
    target_ids = torch.tensor([vocab - 1, 17], device="cuda")
    logits[0, vocab - 1] = 10_000.0
    logits[1, 17] = 9_999.0

    actual = selected_logprob_from_logits(logits, target_ids)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("chunk_rows", [1, 3, 8])
def test_chunked_linear_matches_fp32_reference(use_bias, chunk_rows):
    rows, hidden_size, vocab = 7, 32, 151_936
    torch.manual_seed(1234 + chunk_rows)
    hidden = torch.randn(rows, hidden_size, device="cuda", dtype=torch.float16)
    weight = torch.randn(vocab, hidden_size, device="cuda", dtype=torch.float16) / hidden_size**0.5
    bias = torch.randn(vocab, device="cuda", dtype=torch.float16) if use_bias else None
    target_ids = torch.randint(0, vocab, (rows,), device="cuda")

    logits = F.linear(hidden, weight, bias)
    expected = torch.log_softmax(logits.float(), dim=-1).gather(
        1, target_ids[:, None]
    ).squeeze(1)
    actual = chunked_linear_selected_logprob(
        hidden, weight, target_ids, bias, chunk_rows=chunk_rows
    )
    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)


def test_forward_only_contract_is_explicit():
    logits = torch.randn(2, 17, device="cuda", requires_grad=True)
    target_ids = torch.tensor([0, 16], device="cuda")
    with pytest.raises(RuntimeError, match="forward-only"):
        selected_logprob_from_logits(logits, target_ids)


def test_target_validation_rejects_out_of_range_ids():
    logits = torch.randn(2, 17, device="cuda")
    with pytest.raises(ValueError, match="must lie"):
        selected_logprob_from_logits(logits, torch.tensor([0, 17], device="cuda"))
