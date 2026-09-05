"""Current-policy completion log probabilities.

For a prompt of length P, completion token j is predicted by logits[P + j - 1].
Both backends return aligned completion-only scores.

The naive backend materializes full [B,S,V] logits. The selected backend gathers
prediction-position hidden states from the Qwen backbone and applies the chunked
selected-logprob autograd operator.
"""

import torch


def selective_log_softmax(logits, index):
    """logits [..., V], index [...] (integer) -> logprob of index token, shape [...].
    Clarity version (full log_softmax); the fused kernel replaces this."""
    logp = torch.log_softmax(logits, dim=-1)
    return logp.gather(-1, index.unsqueeze(-1)).squeeze(-1)


def completion_gather_positions(prompt_lens, T, S):
    """prompt_lens [B] (int) -> gather positions [B, T] = (P-1)+j, clamped to [0, S-1].
    Pulled out as its own function precisely so the shift arithmetic can be unit-tested
    without a model (see test_grpo_loss.py)."""
    if prompt_lens.ndim != 1:
        raise ValueError(f"prompt_lens must be rank 1, got shape {tuple(prompt_lens.shape)}")
    if T < 1 or S < 1:
        raise ValueError(f"T and S must be positive, got T={T}, S={S}")
    if torch.any(prompt_lens < 1):
        raise ValueError("every prompt must contain at least one token")
    B = prompt_lens.shape[0]
    j = torch.arange(T, device=prompt_lens.device).view(1, T)
    pos = (prompt_lens.view(B, 1) - 1) + j
    return pos.clamp_(0, S - 1)


def build_full_sequences(prompt_ids_list, completion_ids, completion_mask, pad_id):
    """Concatenate prompt + real completion tokens per row, right-pad to a common length.
    Returns input_ids [B,S], attention_mask [B,S], prompt_lens [B]."""
    B, T = completion_ids.shape
    fulls, prompt_lens = [], []
    for b in range(B):
        p = list(prompt_ids_list[b])
        L = int(completion_mask[b].sum().item())
        c = completion_ids[b, :L].tolist()
        fulls.append(p + c)
        prompt_lens.append(len(p))
    S = max((len(f) for f in fulls), default=1) or 1
    input_ids = torch.full((B, S), pad_id, dtype=torch.long)
    attn = torch.zeros((B, S), dtype=torch.long)
    for b, f in enumerate(fulls):
        input_ids[b, :len(f)] = torch.tensor(f, dtype=torch.long)
        attn[b, :len(f)] = 1
    return input_ids, attn, torch.tensor(prompt_lens, dtype=torch.long)


def _causal_lm_backbone_and_head(model):
    """Return the adapter-instrumented backbone and output head for the validated Qwen path."""
    causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    config = getattr(causal_lm, "config", None)
    if getattr(config, "model_type", None) != "qwen2":
        raise ValueError(
            "selected logprob backend is validated only for Qwen2/Qwen2.5 causal LMs"
        )
    prefix = getattr(causal_lm, "base_model_prefix", "model")
    backbone = getattr(causal_lm, prefix, None)
    get_output_embeddings = getattr(causal_lm, "get_output_embeddings", None)
    head = get_output_embeddings() if get_output_embeddings is not None else None
    if backbone is None or head is None or not hasattr(head, "weight"):
        raise ValueError("causal LM does not expose a compatible backbone and output head")
    return backbone, head


def _compute_selected_completion_logprobs(
    model, input_ids, attn, prompt_lens, completion_ids, completion_mask, chunk_rows
):
    from kernels.selected_logprob import chunked_linear_selected_logprob_autograd

    backbone, head = _causal_lm_backbone_and_head(model)
    output = backbone(
        input_ids=input_ids,
        attention_mask=attn,
        use_cache=False,
        return_dict=True,
    )
    hidden = output.last_hidden_state
    B, S, _ = hidden.shape
    T = completion_ids.shape[1]
    positions = completion_gather_positions(prompt_lens.to(hidden.device), T, S)
    batch_idx = torch.arange(B, device=hidden.device).view(B, 1).expand(B, T)
    active = completion_mask.to(hidden.device).bool()
    prediction_hidden = hidden[batch_idx, positions][active].contiguous()
    targets = completion_ids.to(hidden.device)[active].contiguous()
    if prediction_hidden.shape[0] == 0:
        return torch.zeros((B, T), device=hidden.device, dtype=torch.float32) + hidden.sum() * 0.0

    weight = head.weight
    bias = getattr(head, "bias", None)
    if not weight.is_contiguous() or (bias is not None and not bias.is_contiguous()):
        raise ValueError("selected logprob backend requires a contiguous output head")
    selected = chunked_linear_selected_logprob_autograd(
        prediction_hidden,
        weight,
        targets,
        bias,
        chunk_rows=chunk_rows,
    )
    return torch.zeros((B, T), device=hidden.device, dtype=selected.dtype).masked_scatter(
        active, selected
    )


def compute_completion_logprobs(
    model,
    prompt_ids_list,
    completion_ids,
    completion_mask,
    pad_id,
    *,
    backend="naive",
    chunk_rows=8,
):
    """
    model              : HF/PEFT CausalLM (grad enabled -- this is the CURRENT policy)
    prompt_ids_list    : list[list[int]] length B, prompt-major (row b's prompt token ids)
    completion_ids     : [B, T] right-padded (from rollout_client.RolloutBatch)
    completion_mask    : [B, T] 1 on real completion tokens
    returns            : logprobs [B, T] aligned 1:1 with completion_ids, grad-enabled,
                         zeroed on padded positions.
    """
    input_ids, attn, prompt_lens = build_full_sequences(
        prompt_ids_list, completion_ids, completion_mask, pad_id)
    device = next(model.parameters()).device
    input_ids, attn = input_ids.to(device), attn.to(device)

    if backend == "selected":
        if not input_ids.is_cuda:
            raise ValueError("selected logprob backend requires CUDA")
        if chunk_rows < 1:
            raise ValueError("chunk_rows must be positive")
        return _compute_selected_completion_logprobs(
            model,
            input_ids,
            attn,
            prompt_lens,
            completion_ids,
            completion_mask,
            chunk_rows,
        )
    if backend != "naive":
        raise ValueError(f"unknown logprob backend {backend!r}")

    out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits                       # [B, S, V]  (the memory bottleneck; see note)
    B, S, V = logits.shape
    T = completion_ids.shape[1]

    pos = completion_gather_positions(prompt_lens.to(device), T, S)      # [B, T]
    token_ids = completion_ids.to(device)
    batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, T)

    # Accumulate the reduction in fp32, but gather only [B,T] target values. Compared with
    # gathering [B,T,V] first, this avoids another vocab-sized activation and is the difference
    # between a usable and an immediate-OOM reference path on an 8 GB consumer GPU.
    log_normalizer = torch.logsumexp(logits.float(), dim=-1)             # [B, S]
    selected_lse = log_normalizer.gather(1, pos)                          # [B, T]
    selected_logits = logits[batch_idx, pos, token_ids].float()           # [B, T]
    logp = selected_logits - selected_lse
    return logp * completion_mask.to(device).to(logp.dtype)
