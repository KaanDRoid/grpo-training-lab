# 1.5B reward-signal ablation

## Question and predeclared boundary

The three-seed 1.5B experiment found exact-reward variance on only 23/72 optimizer steps. This
ablation asks whether either (a) eight generations or (b) a deterministic shaped arithmetic
reward creates more within-prompt GRPO signal than four-generation exact match. Candidate
selection uses training prompts only. Held-out prompts and exact scores are not used for screening,
and final evaluation remains binary exact match for every condition.

The shaped training reward parses the last integer and computes
`max(0, 1 - abs(prediction-answer) / max(abs(answer), 1))`. A missing integer receives zero and an
exact answer receives one. It is a programmatic verifier, not a learned reward model.

## Training-only signal screen

One frozen Qwen2.5-1.5B-Instruct policy generated eight completions for each of 144 prompts from
new curriculum seeds 53, 59, and 61. The four-generation measurements use the first four members
of those same groups, controlling the sampled completion pool.

| Condition | Signal groups | Fraction | Mean within-group std |
|---|---:|---:|---:|
| Exact, 4 generations | 40/144 | 27.78% | 0.1469 |
| Exact, 8 generations | 50/144 | 34.72% | 0.1597 |
| Relative-distance, 4 generations | 53/144 | 36.81% | 0.0185 |
| Relative-distance, 8 generations | 61/144 | 42.36% | 0.0209 |

Relative-distance with four generations produced slightly more signal-bearing groups than exact
match with eight, without doubling training completions. It was therefore selected for the paired
training check. Exact×8 was retained as a third arm to test whether its higher-quality binary
signal justified the compute cost.

## Paired seed-67 training result

All three runs share the same base model, 48 training prompts, 24 disjoint held-out prompts,
rollout/evaluation seeds, 24 optimizer steps, learning rate, naive logprob backend, and exact
four-sample held-out evaluation. The comparison tool rejects differing schedules, held-out pairs,
or baseline rollout hashes. Every baseline was exactly 56/96.

| Training condition | Signal groups | Non-zero steps | Training completions | Peak VRAM | Held-out exact | Change |
|---|---:|---:|---:|---:|---:|---:|
| Exact×4 | 12/48 | 11/24 | 192 | 4.96 GiB | **64/96** | +8.33 pp |
| Relative-distance×4 | 15/48 | 13/24 | 192 | 4.60 GiB | 60/96 | +4.17 pp |
| Exact×8 | 15/48 | 12/24 | 384 | 6.52 GiB | 63/96 | +7.29 pp |

By stage, exact×4 changed 7-digit/8-digit correct counts from 34/22 to 35/29;
relative-distance×4 reached 34/26; exact×8 reached 34/29.

Both interventions reduced gradient starvation, but neither beat exact×4 on the held-out exact
metric. Dense shaping added three signal groups and two non-zero optimizer steps yet finished four
correct samples below exact×4. Eight generations added three signal groups at twice the completion
budget and 1.56 GiB more peak allocation, then finished one correct sample below exact×4. This is a
single paired mechanism pilot, not an uncertainty estimate. In particular, seed 67's positive
exact×4 result does not override the earlier three-seed mean of -1.74 pp.

Exact×4 remains the reference protocol. Both alternatives are retained as negative results;
additional seeds were not run for this ablation.
The retained adapter was loaded into a fresh base-model process and reproduced 64/96 with rollout
SHA-256 `fc03b9d548e1ffb7760c754fd03fd16d2eca4458ce68bca8ceb59f7756905996` exactly.

## Numerical note

Seed 67 exposed one cached-generation/full-recompute fp16 token with logprob drift 0.4125 while the
68-token mean was 0.0085. A token-level reproducer showed this was a real batch-path numerical tail,
not a causal-shift or selected-backend error. In synchronous HF recompute mode the PPO denominator
is `current_logprobs.detach()`, so raw generation scores are diagnostic only. That mode now permits
a 1.0 single-token diagnostic ceiling while retaining the 0.03 mean guard. vLLM and HF runs that
actually use rollout scores retain the stricter 0.35 ceiling.

## Artifacts

- `results/reward_signal_probe_1p5b.json`
- `results/reward_ablation_seed67_summary.json` and `.csv`
- `results/ablation_exact_g4_curriculum_7d_8d_seed67.json` and adapter
- `results/ablation_relative_g4_curriculum_7d_8d_seed67.json` and adapter
- `results/ablation_exact_g8_curriculum_7d_8d_seed67.json` and adapter
- `benchmarks/inspect_hf_parity_outlier.py`
