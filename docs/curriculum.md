# Deterministic multi-prompt curriculum

The single-prompt reward-curve run saturated after step 8. This experiment replaces that repeated
prompt with a restart-stable stream of distinct arithmetic problems and makes the train/evaluation
split explicit.

## Split and schedule contract

`ArithmeticCurriculum` derives each stage from the absolute optimizer step, not the requested final
run length. Extending a 12-step checkpoint to 24 steps therefore cannot rewrite its earlier data.
Domain-separated SHA-256 seeds make prompt generation independent of global Python RNG state and
identical across process restarts. Duplicate pairs are rejected, and held-out generation receives
the complete training pair set as an exclusion list.

The first pilot used one prompt per step and 4/5/6-digit operands. It was deliberately retained as
a negative result: all 4-digit groups were correct and most harder groups were either all correct
or all wrong. Only one of 24 steps had reward variance, so exact-match GRPO produced only one
non-zero gradient and held-out accuracy stayed at 41/96. A curriculum must target the model's
learning frontier; merely changing prompts is insufficient.

The calibrated protocol is:

- Qwen2.5-0.5B-Instruct with rank-16 fp16 LoRA;
- seed 29, 24 steps, learning rate `1e-5`;
- two prompts per step and four generations per prompt;
- 12 steps of 5-digit operands followed by 12 steps of 6-digit operands;
- 48 unique training pairs;
- 12 held-out pairs per difficulty, four samples each (96 rollouts total);
- zero train/held-out pair overlap;
- identical held-out prompts and sampling seeds before and after training.

Command:

```powershell
.\.venv\Scripts\python.exe curriculum_experiment.py --steps 24 --prompts-per-step 2 --stage-steps 12 --operand-digits 5,6 --seed 29 --heldout-per-stage 12 --output-dir results
```

## Result

| Held-out slice | Baseline | Trained | Absolute change |
|---|---:|---:|---:|
| 5-digit operands | 23/48 (47.92%) | 32/48 (66.67%) | +18.75 pp |
| 6-digit operands | 8/48 (16.67%) | 27/48 (56.25%) | +39.58 pp |
| Overall | 31/96 (32.29%) | 59/96 (61.46%) | +29.17 pp |

Six of the 12 five-digit steps and four of the 12 six-digit steps produced non-zero gradients. The
other groups still had identical exact-match rewards, but the calibrated batch supplied ten useful
optimizer steps instead of the pilot's one. Mean training rewards were 0.59375 and 0.53125 for the
two stages. Peak allocated VRAM was 2.39 GiB.

This is a strong within-protocol result, not a broad arithmetic claim: it is one seed and 24
held-out problems from the same generated distributions. The report retains per-prompt counts so a
larger multi-seed experiment can be added without overstating this run.

## Parity, reproducibility, and export

The larger prompt batch exposed one fp16 tail token with an absolute rollout/recompute difference
of 0.2288 while the step mean was 0.0081. The original max-only 0.20 guard was therefore replaced
with a hard max `<= 0.35` and mean `<= 0.03`; p99 and tail counts remain diagnostics because their
meaning is unstable for very short completions. This still rejects broad shift/tokenization errors
while tolerating bounded batch-shape-sensitive tail tokens.

Two complete executions produced byte-identical training CSV and adapter tensors:

- training schedule SHA-256: `05246ce176f28b6b9a29068aae4ddfefbc576c79aaf76e845fc2dedbb1cef36d`;
- CSV SHA-256: `51DF608849801EB679E5014148F2639C60FD102850A2250117DDE3DB0540F047`;
- adapter safetensors SHA-256: `12DC3DC8C173EAD0FCF2EFC1EBB00D022281D95375FCD057678ABC48F3E73F56`.

A fresh base model plus the saved adapter reproduced the recorded 59/96 result and rollout hash
exactly:

```powershell
.\.venv\Scripts\python.exe verify_curriculum_adapter.py results\curriculum_5d_6d_seed29.json
```

Machine-readable artifacts are `results/curriculum_5d_6d_seed29.json` and `.csv`; the local adapter
directory is ignored by Git.
