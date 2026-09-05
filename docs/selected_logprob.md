# Selected-logprob forward, backward, and trainer integration

The operator combines PyTorch/cuBLAS GEMM with a Triton reduction and bounds the number of live
logits rows. It reproduces the decomposition used in existing fused linear-loss implementations,
including Liger; no new kernel method is claimed.

## Numerical contract

For one row of logits `z` and sampled token `t`, only this scalar is required:

```text
log p(t) = z[t] - logsumexp(z)
```

The vocabulary is divided into 1,024-element tiles. Each tile produces two fp32 values:

```text
m_q = max(z_q)
s_q = sum(exp(z_q - m_q))
```

A second Triton kernel combines the tiles without losing the max-subtraction stability:

```text
m = max_q(m_q)
logsumexp(z) = m + log(sum_q(s_q * exp(m_q - m)))
```

Masked lanes load negative infinity for the maximum. This is what permits non-power-of-two
vocabularies such as Qwen2.5's 151,936 entries without allowing padding to affect the result.

`chunked_linear_selected_logprob()` evaluates `hidden @ weight.T + bias` a small number of rows at
a time. At most `[chunk_rows,V]` logits are live, no `[N,V]` log-softmax output is created, and the
final output is `[N]` fp32. This original benchmark wrapper remains explicitly forward-only and
raises if a caller attempts to attach it to an autograd graph.

`chunked_linear_selected_logprob_autograd()` is the separately validated training path. For
`log p(t) = z[t] - logsumexp(z)`, its logit derivative is:

```text
G_j = upstream * (1[j=t] - softmax(z)_j)
grad_hidden = G @ weight
grad_weight = G.T @ hidden
grad_bias = sum_rows(G)
```

Backward recomputes one logits row chunk, uses Triton to form `G`, applies the three linear
Jacobians with PyTorch/cuBLAS, and releases that chunk before continuing. The sign is deliberately
the reverse of cross-entropy's `softmax - one_hot` because this function returns positive logprob,
not negative log-likelihood.

## RTX 5060 result

Command:

```powershell
.\.venv\Scripts\python.exe benchmarks\benchmark_selected_logprob.py --csv benchmarks\selected_logprob_rtx5060.csv
```

Configuration: `N=128`, `H=256`, fp16 GEMM, `chunk_rows=8`, five timed repetitions. Peak numbers
are transient PyTorch CUDA allocations above the already-resident inputs and weights.

| Vocab | Naive ms | Chunked ms | Naive peak MiB | Chunked peak MiB | Max abs error |
|---:|---:|---:|---:|---:|---:|
| 8,191 | 0.52 | 2.23 | 10.00 | 0.25 | 3.815e-6 |
| 32,768 | 1.13 | 2.13 | 40.00 | 1.00 | 9.537e-7 |
| 65,537 | 1.06 | 4.94 | 80.00 | 3.00 | 1.907e-6 |
| 151,936 | 2.31 | 5.37 | 186.38 | 4.64 | 9.537e-7 |

At the Qwen vocabulary width, the measured transient allocation falls by about **40x (97.5%)**.
The Python chunk loop makes this baseline about **2.3x slower** than the naive operation. That is
a memory/latency tradeoff from processing logits in small chunks. Backward measurements and
correctness checks are reported below.

## Backward result

The fp64 path passes `torch.autograd.gradcheck` with and without bias. Independent CUDA tests
compare fp16/fp32 `grad_hidden`, `grad_weight`, and `grad_bias` against ordinary PyTorch autograd,
including a 151,936-wide frozen-weight case.

RTX 5060 benchmark configuration: `N=128`, `H=256`, fp16, frozen output weight, random upstream
gradient, `chunk_rows=8`, three repetitions.

| Vocab | Naive backward ms | Chunked backward ms | Naive peak MiB | Chunked peak MiB | Hidden-grad max error |
|---:|---:|---:|---:|---:|---:|
| 8,191 | 0.51 | 6.04 | 14.12 | 0.56 | 4.883e-4 |
| 32,768 | 1.21 | 11.74 | 56.13 | 1.69 | 2.441e-4 |
| 65,537 | 2.32 | 11.55 | 112.13 | 4.19 | 4.883e-4 |
| 151,936 | 5.27 | 15.88 | 260.69 | 7.14 | 1.953e-3 |

At Qwen width, transient backward allocation falls by about **36.5x (97.3%)** while this baseline
is about **3x slower**. The measured error is the expected fp16 storage/accumulation difference;
fp32 gradient tests use a much tighter `2e-4` tolerance.

## Opt-in trainer integration

`compute_completion_logprobs(..., backend="selected")` bypasses `Qwen2ForCausalLM.forward`, runs
the PEFT-instrumented Qwen backbone directly, and gathers only the hidden rows that predict active
completion tokens. Those rows and the frozen `lm_head` weight are passed to the custom autograd
operator. The default remains `backend="naive"`, and non-Qwen architectures are rejected rather
than silently assuming their output-head semantics.

Use it from the trainer with:

```powershell
.\.venv\Scripts\python.exe grpo_trainer.py --backend hf --steps 1 --smoke --seed 17 --logprob-backend selected --logprob-chunk-rows 8
```

On the real Qwen2.5-0.5B model, a two-prompt/four-generation benchmark measured the complete
current-policy recompute plus backward path:

| Backend | Time | Incremental peak allocation | Active logprob mean |
|---|---:|---:|---:|
| Naive `[B,S,V]` | 462.94 ms | 1381.86 MiB | -0.0739093 |
| Selected, 8-row chunks | 589.71 ms | 88.71 MiB | -0.0739093 |

The active logprobs matched exactly. Incremental peak allocation fell **15.58x (93.6%)** for a
**1.27x** latency cost. Across all 336 trainable LoRA gradient tensors, cosine similarity was
`0.9999956` and relative L2 error was `0.00299` (fp16). The reproducible report is
`benchmarks/trainer_logprob_backend_rtx5060.json`.

End-to-end single-prompt HF smoke peak allocation fell from 1.64 GiB to 1.11 GiB with identical
reward/loss and gradient norms 37.9523 versus 37.9345. The WSL/vLLM path also completed rollout,
selected backward, optimizer update, adapter hot-reload, and prefix-cache reset: reward 0.75,
gradient norm 39.52, parity max/mean 0.0159/0.0022, trainer peak 1.48 GiB.

## Verification coverage

- fp16 and fp32 logits;
- vocab widths `3`, `257`, `4,097`, `65,537`, and `151,936`;
- large-logit stability (`+10,000/-10,000`);
- target bounds and forward-only contract;
- chunk sizes `1`, `3`, and `8`, with and without bias;
- comparison to an fp32 PyTorch `log_softmax` reference.
- fp64 custom-backward gradcheck, with and without bias;
- CUDA fp16/fp32 comparisons for all three linear gradients;
- Qwen-vocabulary hidden gradient with a frozen output weight.
- masked/padded trainer-backend output and full parameter-gradient comparison;
- real Qwen LoRA forward/backward memory and gradient-quality benchmark;
- end-to-end HF and WSL/vLLM optimizer-step smoke tests.
