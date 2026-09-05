# Qwen2.5-1.5B three-seed curriculum pilot

This experiment tests whether the 1.5B-Instruct capacity result also produces a repeatable
held-out learning gain. It is deliberately a small three-seed pilot; the seed, not each correlated
generation, is the statistical unit.

## Protocol

- model: local Qwen2.5-1.5B-Instruct, fp16 rank-16 LoRA;
- shared-model Hugging Face rollout and selected-logprob backward;
- seeds 41, 43, and 47;
- 24 steps per seed, two unique prompts per step, four generations per prompt;
- 12 steps of 7-digit operands followed by 12 steps of 8-digit operands;
- 48 unique training prompts and 24 disjoint held-out prompts per seed;
- 96 fixed-seed held-out rollouts before and after each run;
- learning rate `1e-5`;
- exact on-policy HF behavior logprobs from a detached current recomputation.

The final point matters numerically. Cached generation and full-sequence recomputation are the same
theoretical policy but can differ for a few fp16 tokens. In a synchronous shared-model backend no
optimizer update occurs between them, so the loss uses `current_logprobs.detach()` as the behavior
policy. Raw generation-score drift is still recorded and guarded by hard max `<=0.35` and mean
`<=0.03`; it cannot perturb the importance ratio.

Example command:

```powershell
.\.venv\Scripts\python.exe curriculum_experiment.py --model .\Qwen2.5-1.5B-Instruct --run-label qwen1p5b --steps 24 --prompts-per-step 2 --stage-steps 12 --operand-digits 7,8 --seed 41 --heldout-per-stage 12 --hf-recompute-behavior-logprobs --output-dir results
```

## Per-seed result

| Seed | Baseline | Trained | Change | Non-zero-gradient steps | Peak VRAM |
|---:|---:|---:|---:|---:|---:|
| 41 | 57/96 | 57/96 | 0.00 pp | 9/24 | 4.87 GiB |
| 43 | 68/96 | 64/96 | -4.17 pp | 5/24 | 4.60 GiB |
| 47 | 57/96 | 56/96 | -1.04 pp | 9/24 | 4.60 GiB |

Pooled counts are 182/288 (63.19%) before and 177/288 (61.46%) after. These counts are descriptive,
not 288 independent experimental replications.

## Run-level confidence interval

The aggregate uses a two-sided 95% Student-t interval over the three paired run deltas (df=2):

| Slice | Mean change | 95% CI |
|---|---:|---:|
| 7-digit | +0.69 pp | [-2.29, +3.68] pp |
| 8-digit | -4.17 pp | [-13.13, +4.80] pp |
| Overall | -1.74 pp | **[-7.12, +3.65] pp** |

The interval contains zero and the point estimate is negative. This 24-step, four-generation
exact-reward protocol does not demonstrate a 1.5B held-out gain.

Only 23 of 72 optimizer steps received non-zero group-relative signal. The 1.5B model often emits
four identical-reward samples for a prompt, so binary exact match still causes gradient starvation.
The subsequent [reward ablation](reward_signal_ablation.md) tested larger groups and shaped
rewards using a separate training-only screen. Held-out scores were excluded from candidate selection.

## Artifacts and verification

- `results/qwen1p5b_curriculum_7d_8d_seed{41,43,47}.json/.csv`;
- `results/qwen1p5b_curriculum_multiseed_summary.json/.csv`;
- `aggregate_curriculum_runs.py` for protocol validation and Student-t aggregation.

The seed-43 adapter was loaded onto a fresh base model and reproduced its recorded evaluation
exactly: 64/96 with rollout SHA-256
`f20c20f4ab62c3c690162e2f844a575a7d0b85adaa34647ed4df4d9f238f30c8`.
