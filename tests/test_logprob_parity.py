import torch

from grpo_trainer import behavior_logprobs_for_loss, logprob_parity_stats


def _stats(deltas):
    old = torch.zeros(1, len(deltas))
    current = torch.tensor(deltas).view(1, -1)
    mask = torch.ones_like(current, dtype=torch.long)
    return logprob_parity_stats(current, old, mask)


def test_small_distributed_drift_passes():
    stats = _stats([0.01] * 100)
    assert stats["acceptable"]
    assert stats["max"] == stats["p99"] == stats["mean"]


def test_single_fp16_tail_outlier_uses_hard_ceiling_not_old_brittle_max():
    stats = _stats([0.0] * 999 + [0.25])
    assert stats["acceptable"]
    assert stats["max"] == 0.25
    assert stats["p99"] < 0.20


def test_systematic_mismatch_fails_mean_guard():
    stats = _stats([0.0] * 50 + [0.21] * 50)
    assert not stats["acceptable"]
    assert stats["p99"] > 0.20
    assert stats["tail_tokens"] == 50


def test_bounded_short_batch_tail_is_diagnostic_not_a_sample_size_dependent_gate():
    assert _stats([0.0] * 19 + [0.25])["acceptable"]
    stats = _stats([0.0] * 18 + [0.25, 0.21])
    assert stats["acceptable"]
    assert stats["tail_tokens"] == 2


def test_extreme_outlier_and_nonfinite_values_fail():
    assert not _stats([0.36] + [0.0] * 99)["acceptable"]
    assert not _stats([float("nan")])["acceptable"]


def test_hf_recompute_can_use_a_looser_diagnostic_tail_ceiling():
    old = torch.zeros(1, 100)
    current = torch.tensor([0.75] + [0.0] * 99).view(1, -1)
    mask = torch.ones_like(current, dtype=torch.long)
    assert logprob_parity_stats(current, old, mask, hard_max=1.0)["acceptable"]
    assert not logprob_parity_stats(current, old, mask, hard_max=0.35)["acceptable"]


def test_synchronous_hf_uses_detached_current_but_vllm_keeps_rollout_scores():
    rollout = torch.tensor([[-2.0]])
    current = torch.tensor([[-1.5]], requires_grad=True)
    hf_old = behavior_logprobs_for_loss(rollout, current, "hf", True)
    assert hf_old.item() == -1.5
    assert not hf_old.requires_grad
    assert behavior_logprobs_for_loss(rollout, current, "vllm", True) is rollout
    assert behavior_logprobs_for_loss(rollout, current, "hf", False) is rollout
