"""
Numerical-correctness tests -- verifying the GRPO loss/advantage/logprob math.
Pure PyTorch, CPU only, no GPU / no model. Run:  python test_grpo_loss.py   (or: pytest -q)

Every expected value below is hand-derived (see comments) so these tests actually pin behavior,
not just "it ran". The masking test is the regression guard for the real bug that was fixed:
grpo_loss must ignore padded positions no matter what junk they contain.
"""

import math

import torch

from grpo_loss import grpo_loss, group_relative_advantage, entropy_bonus_grad
from compute_logprobs import selective_log_softmax, completion_gather_positions


def test_group_relative_advantage_no_std():
    # groups of 2: [1,0] -> mean 0.5 -> centered [0.5, -0.5]
    rewards = torch.tensor([1., 0., 1., 0.])
    adv = group_relative_advantage(rewards, group_size=2, std_normalize=False)
    torch.testing.assert_close(adv, torch.tensor([0.5, -0.5, 0.5, -0.5]))


def test_group_relative_advantage_with_std():
    # torch.std is UNBIASED (n-1): std([1,0]) = sqrt(0.5) = 0.70710678
    # adv = (r-mean)/(std+1e-6) ~= [0.7071, -0.7071]
    rewards = torch.tensor([1., 0., 1., 0.])
    adv = group_relative_advantage(rewards, group_size=2, std_normalize=True)
    s = math.sqrt(0.5)
    torch.testing.assert_close(
        adv, torch.tensor([0.5 / s, -0.5 / s, 0.5 / s, -0.5 / s]), atol=1e-4, rtol=1e-4)


def test_group_relative_advantage_rejects_invalid_groups():
    for rewards, group_size in ((torch.tensor([1.]), 1), (torch.tensor([1., 2., 3.]), 2)):
        try:
            group_relative_advantage(rewards, group_size=group_size)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid GRPO grouping should fail before silently mixing prompts")


def test_grpo_loss_masking_ignores_padding():
    # B=2, T=3. old=0 everywhere. logprobs carry JUNK in masked positions (big values ->
    # huge ratio) that MUST NOT affect the loss.
    logprobs = torch.tensor([[0., 0., 5.],
                             [0., 5., 5.]])
    old = torch.zeros(2, 3)
    advantages = torch.tensor([2., -4.])
    mask = torch.tensor([[1, 1, 0],
                         [1, 0, 0]])
    # Hand derivation (clip_eps=0.2, ratio=1 on unmasked positions):
    #   row0 (adv=2): per_token = -min(1*2, clamp(1)*2) = -2 at j0,j1 -> seq mean = -2
    #   row1 (adv=-4): per_token = -min(1*-4, 1*-4) = +4 at j0        -> seq mean = +4
    #   seq_mean loss = mean(-2, 4) = 1.0   ; token_mean loss = (-2-2+4)/3 = 0.0
    seq = grpo_loss(logprobs, old, advantages, mask, aggregation="seq_mean")
    tok = grpo_loss(logprobs, old, advantages, mask, aggregation="token_mean")
    torch.testing.assert_close(seq, torch.tensor(1.0))
    torch.testing.assert_close(tok, torch.tensor(0.0))

    # regression guard: changing ONLY the masked-out entries must not move the loss.
    logprobs2 = logprobs.clone()
    logprobs2[0, 2] = -999.
    logprobs2[1, 1] = 123.
    seq2 = grpo_loss(logprobs2, old, advantages, mask, aggregation="seq_mean")
    torch.testing.assert_close(seq, seq2)


def test_grpo_loss_clipping():
    # single unmasked token, positive advantage, ratio > 1+eps -> clipped branch caps the gain.
    # logprob-old = ln(2) -> ratio=2. adv=1, eps=0.2 -> unclipped=2, clipped=1.2, min=1.2 -> loss=-1.2
    logprobs = torch.tensor([[math.log(2.0)]])
    old = torch.tensor([[0.0]])
    adv = torch.tensor([1.0])
    mask = torch.tensor([[1]])
    loss = grpo_loss(logprobs, old, adv, mask, clip_eps=0.2, aggregation="seq_mean")
    torch.testing.assert_close(loss, torch.tensor(-1.2), atol=1e-6, rtol=0)


def test_grpo_loss_requires_reference_for_kl():
    values = torch.zeros(1, 1)
    try:
        grpo_loss(values, values, torch.ones(1), torch.ones(1, 1), kl_coef=0.1)
    except ValueError as exc:
        assert "ref_logprobs" in str(exc)
    else:
        raise AssertionError("positive KL coefficient without a reference must fail loudly")


def test_entropy_bonus_grad_matches_autograd():
    logits = torch.tensor([[1.0, 2.0, 0.5], [0.0, 0.0, 3.0]], requires_grad=True)
    p = torch.softmax(logits, dim=-1)
    H = -(p * torch.log(p)).sum(dim=-1)          # [B]
    H.sum().backward()                            # rows independent -> per-row dH/dz
    analytic = entropy_bonus_grad(p.detach(), H.detach())
    torch.testing.assert_close(analytic, logits.grad, atol=1e-5, rtol=1e-4)


def test_selective_log_softmax():
    # uniform logits -> log(1/3); and a known non-uniform row
    logits = torch.tensor([[0., 0., 0.],
                           [1., 0., 0.]])
    idx = torch.tensor([0, 0])
    got = selective_log_softmax(logits, idx)
    lse_row1 = math.log(math.e + 1 + 1)          # logsumexp([1,0,0])
    torch.testing.assert_close(
        got, torch.tensor([math.log(1 / 3), 1.0 - lse_row1]), atol=1e-5, rtol=1e-5)


def test_completion_gather_positions_shift():
    # the alignment arithmetic: pos[b,j] = (P_b - 1) + j, clamped to [0, S-1]
    prompt_lens = torch.tensor([2, 3])
    pos = completion_gather_positions(prompt_lens, T=3, S=8)
    torch.testing.assert_close(pos, torch.tensor([[1, 2, 3], [2, 3, 4]]))
    # clamp check: tiny S must not index out of range
    pos_clamped = completion_gather_positions(torch.tensor([6]), T=4, S=7)
    assert int(pos_clamped.max()) <= 6

    try:
        completion_gather_positions(torch.tensor([0]), T=1, S=1)
    except ValueError:
        pass
    else:
        raise AssertionError("an empty prompt has no valid next-token prediction position")


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    _run_all()
