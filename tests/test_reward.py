import pytest
import torch

from reward import arithmetic_reward


def test_exact_reward_uses_last_integer_and_rejects_missing_answer():
    rewards = arithmetic_reward(
        ["first 10, final 12", "12 then 13", "no numeric answer"],
        [12, 12, 12],
    )
    torch.testing.assert_close(rewards, torch.tensor([1.0, 0.0, 0.0]))


def test_relative_distance_reward_is_bounded_and_exact_at_one():
    rewards = arithmetic_reward(
        ["100", "75", "-100", "nothing"],
        [100, 100, 100, 100],
        mode="relative_distance",
    )
    torch.testing.assert_close(rewards, torch.tensor([1.0, 0.75, 0.0, 0.0]))


def test_relative_distance_handles_zero_answer():
    rewards = arithmetic_reward(["0", "1", "-1"], [0, 0, 0], mode="relative_distance")
    torch.testing.assert_close(rewards, torch.tensor([1.0, 0.0, 0.0]))


def test_unknown_reward_mode_fails_loudly():
    with pytest.raises(ValueError, match="unknown arithmetic reward mode"):
        arithmetic_reward(["1"], [1], mode="learned_vibes")
