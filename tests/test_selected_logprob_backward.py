"""Gradient checks for the custom selected-logprob autograd path."""

import pytest
import torch
import torch.nn.functional as F

from kernels.selected_logprob import chunked_linear_selected_logprob_autograd


@pytest.mark.parametrize("use_bias", [False, True])
def test_custom_backward_passes_fp64_gradcheck(use_bias):
    torch.manual_seed(71)
    hidden = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(5, dtype=torch.float64, requires_grad=True) if use_bias else None
    targets = torch.tensor([0, 4], dtype=torch.int64)

    def function(x, w, b):
        return chunked_linear_selected_logprob_autograd(
            x, w, targets, b, chunk_rows=1
        )

    assert torch.autograd.gradcheck(
        function,
        (hidden, weight, bias),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("use_bias", [False, True])
def test_triton_backward_matches_torch_autograd(dtype, use_bias):
    torch.manual_seed(99)
    rows, hidden_size, vocab = 5, 11, 257
    hidden_ref = torch.randn(
        rows, hidden_size, device="cuda", dtype=dtype, requires_grad=True
    )
    weight_ref = torch.randn(
        vocab, hidden_size, device="cuda", dtype=dtype, requires_grad=True
    ) / hidden_size**0.5
    weight_ref = weight_ref.detach().requires_grad_(True)
    bias_ref = (
        torch.randn(vocab, device="cuda", dtype=dtype, requires_grad=True)
        if use_bias else None
    )
    targets = torch.randint(0, vocab, (rows,), device="cuda")
    upstream = torch.randn(rows, device="cuda", dtype=torch.float32)

    expected = torch.log_softmax(
        F.linear(hidden_ref, weight_ref, bias_ref).float(), dim=-1
    ).gather(1, targets[:, None]).squeeze(1)
    expected_inputs = (hidden_ref, weight_ref) + ((bias_ref,) if use_bias else ())
    expected_grads = torch.autograd.grad(expected, expected_inputs, upstream)

    hidden = hidden_ref.detach().clone().requires_grad_(True)
    weight = weight_ref.detach().clone().requires_grad_(True)
    bias = bias_ref.detach().clone().requires_grad_(True) if use_bias else None
    actual = chunked_linear_selected_logprob_autograd(
        hidden, weight, targets, bias, chunk_rows=3
    )
    actual_inputs = (hidden, weight) + ((bias,) if use_bias else ())
    actual_grads = torch.autograd.grad(actual, actual_inputs, upstream)

    forward_tol = 4e-4 if dtype == torch.float16 else 4e-5
    grad_tol = 2e-2 if dtype == torch.float16 else 2e-4
    torch.testing.assert_close(actual, expected, atol=forward_tol, rtol=forward_tol)
    for got, want in zip(actual_grads, expected_grads):
        torch.testing.assert_close(got, want, atol=grad_tol, rtol=grad_tol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_qwen_vocab_hidden_gradient_with_frozen_weight():
    torch.manual_seed(151_936)
    rows, hidden_size, vocab = 3, 16, 151_936
    hidden_ref = torch.randn(
        rows, hidden_size, device="cuda", dtype=torch.float32, requires_grad=True
    )
    weight = torch.randn(vocab, hidden_size, device="cuda", dtype=torch.float32) / hidden_size**0.5
    targets = torch.tensor([0, 65_536, vocab - 1], device="cuda")
    upstream = torch.tensor([0.5, -1.25, 2.0], device="cuda")

    expected = torch.log_softmax(F.linear(hidden_ref, weight), dim=-1).gather(
        1, targets[:, None]
    ).squeeze(1)
    expected_grad, = torch.autograd.grad(expected, hidden_ref, upstream)

    hidden = hidden_ref.detach().clone().requires_grad_(True)
    actual = chunked_linear_selected_logprob_autograd(
        hidden, weight, targets, chunk_rows=2
    )
    actual_grad, = torch.autograd.grad(actual, hidden, upstream)
    torch.testing.assert_close(actual, expected, atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(actual_grad, expected_grad, atol=4e-4, rtol=4e-4)
