"""Masked GRPO objective and group-relative advantages.

Cross-entropy has logit gradient softmax - one_hot; sampled-token log probability
has gradient one_hot - softmax.

Current and behavior logprobs and the completion mask have shape [B,T], aligned
with completion_ids. Causal recomputation uses logits at prompt_len - 1 + j to
score completion token j.
"""

import torch


def grpo_loss(logprobs, old_logprobs, advantages, completion_mask,
              clip_eps=0.2, kl_coef=0.0, ref_logprobs=None,
              aggregation="seq_mean"):
    """
    logprobs, old_logprobs : [B, T]  current-policy / rollout-time-policy per-token logprobs
                                      (completion tokens only -- see ALIGNMENT CONTRACT above)
    advantages             : [B]     group-relative advantage, broadcast over T
    completion_mask        : [B, T]  1 on real completion tokens, 0 on padding. REQUIRED --
                                      without it, padding tokens silently corrupt the loss and
                                      its scale drifts with how much padding is in the batch.
    kl_coef=0.0             -> Dr.GRPO-style: drop the reference-model term entirely (saves ~3GB
                                on a T4 since you never need a second model copy resident)
    aggregation            : "seq_mean"   -> original GRPO: per-sequence masked mean, then batch
                                             mean (each sequence weighted equally = length-normalized)
                             "token_mean" -> DAPO/Dr.GRPO-style: global masked token mean; removes
                                             per-sequence length weighting. (Dr.GRPO's exact form
                                             divides by a *constant* max length -- swap the
                                             denominator to (B * T) if you want that precise variant.)
    """
    assert logprobs.shape == old_logprobs.shape == completion_mask.shape, (
        f"[B,T] mismatch: logprobs {logprobs.shape}, old {old_logprobs.shape}, "
        f"mask {completion_mask.shape}"
    )
    assert advantages.shape[0] == logprobs.shape[0], (
        f"advantages batch {advantages.shape[0]} != logprobs batch {logprobs.shape[0]}"
    )
    if kl_coef < 0:
        raise ValueError(f"kl_coef must be non-negative, got {kl_coef}")
    if kl_coef > 0 and ref_logprobs is None:
        raise ValueError("ref_logprobs is required when kl_coef > 0")
    if ref_logprobs is not None and ref_logprobs.shape != logprobs.shape:
        raise ValueError(
            f"ref_logprobs must match logprobs: {ref_logprobs.shape} != {logprobs.shape}"
        )

    mask = completion_mask.to(dtype=torch.bool)
    # Mask BEFORE exponentiation. Masking only the final loss is insufficient because an extreme
    # padding value can overflow exp() and produce inf * 0 == NaN during the reduction.
    log_ratio = torch.where(mask, logprobs - old_logprobs, torch.zeros_like(logprobs))
    ratio = torch.exp(log_ratio)
    adv = advantages.unsqueeze(-1)  # [B,1] -> broadcast over T
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    per_token_loss = -torch.min(unclipped, clipped)  # [B, T]

    if kl_coef > 0 and ref_logprobs is not None:
        # k3 estimator (Schulman) -- low-variance, non-negative KL estimate
        ref_delta = torch.where(mask, ref_logprobs - logprobs, torch.zeros_like(logprobs))
        kl = torch.exp(ref_delta) - ref_delta - 1
        per_token_loss = per_token_loss + kl_coef * kl

    mask = mask.to(per_token_loss.dtype)
    if aggregation == "seq_mean":
        seq_tokens = mask.sum(-1).clamp_min(1.0)
        per_seq = (per_token_loss * mask).sum(-1) / seq_tokens
        return per_seq.mean()
    elif aggregation == "token_mean":
        return (per_token_loss * mask).sum() / mask.sum().clamp_min(1.0)
    else:
        raise ValueError(f"unknown aggregation {aggregation!r}")


def group_relative_advantage(rewards, group_size, std_normalize=True):
    """
    rewards : [B] flat, grouped in chunks of `group_size` (GRPO's "group" = same prompt,
              multiple sampled completions). Rows MUST be prompt-major so each group is one
              prompt's completions (rollout_client emits them this way).
    std_normalize=True  : original GRPO -- (r - group_mean) / group_std  (scales by within-group
                          reward std). NOTE: this is the *difficulty/std* normalization.
    std_normalize=False : Dr.GRPO ablation -- drop the std term, keep mean-centering: (r - group_mean).
                          Dr.GRPO argues the std term introduces a question-difficulty bias.
                          (This toggle is NOT the length-bias fix -- that lives in grpo_loss's
                          `aggregation` denominator. Don't conflate the two.)
    Run BOTH on the same batch and diff the advantages -- that diff IS the numerical-correctness /
    paper-validation check.
    """
    if group_size < 2:
        raise ValueError("group_size must be at least 2 for group-relative advantages")
    if rewards.ndim != 1 or rewards.numel() % group_size:
        raise ValueError(
            f"rewards must be flat and divisible by group_size={group_size}; "
            f"got shape {tuple(rewards.shape)}"
        )
    r = rewards.view(-1, group_size)
    mean = r.mean(dim=1, keepdim=True)
    if std_normalize:
        std = r.std(dim=1, keepdim=True) + 1e-6
        advantages = (r - mean) / std
    else:
        advantages = r - mean
    return advantages.view(-1)


def entropy_bonus_grad(softmax_probs, entropy):
    """
    dH/dz_j = -softmax_j * (log softmax_j + H)   -- the numerically fragile term.
    Derivation (verified): H = -Σ_k p_k log p_k, dp_k/dz_j = p_j(δ_kj - p_k)
      dH/dz_j = -Σ_k p_j(δ_kj - p_k)(log p_k + 1) = -p_j(log p_j + 1) + p_j Σ_k p_k(log p_k + 1)
              = -p_j(log p_j + 1) + p_j(1 - H) = -p_j(log p_j + H).
    Only needed if you implement the kernel's entropy backward (not currently wired into
    grpo_loss). Kept here as a reference
    formula, not wired into grpo_loss above (entropy bonus coefficient defaults to unused).
    """
    log_p = torch.log(softmax_probs.clamp_min(1e-12))
    return -softmax_probs * (log_p + entropy.unsqueeze(-1))
