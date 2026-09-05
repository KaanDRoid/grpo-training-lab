"""Numerical checks for the first two official Triton tutorial patterns."""

import pytest
import torch

from kernels.fused_softmax import fused_softmax
from kernels.vector_add import vector_add


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("size", [1, 127, 1024, 100_003])
def test_vector_add_matches_torch(size):
    torch.manual_seed(size)
    x = torch.randn(size, device="cuda", dtype=torch.float32)
    y = torch.randn(size, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(vector_add(x, y), x + y, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("shape", [(1, 3), (7, 257), (32, 1000), (4, 4096)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fused_softmax_matches_fp32_reference(shape, dtype):
    torch.manual_seed(shape[0] * shape[1])
    x = torch.randn(*shape, device="cuda", dtype=dtype) * 3
    expected = torch.softmax(x.float(), dim=-1).to(dtype)
    actual = fused_softmax(x)
    tolerance = 2e-3 if dtype == torch.float16 else 2e-6
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(
        actual.float().sum(dim=-1), torch.ones(shape[0], device="cuda"),
        atol=tolerance, rtol=tolerance,
    )


def test_fused_softmax_stays_finite_on_large_logits():
    x = torch.tensor([[10_000.0, 9_999.0, -10_000.0]], device="cuda")
    actual = fused_softmax(x)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, torch.softmax(x, dim=-1), atol=2e-6, rtol=2e-6)
