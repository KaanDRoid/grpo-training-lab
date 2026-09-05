# Deterministic GRPO reward-curve experiment

This experiment asks a narrow question: can the validated single-GPU GRPO loop move a real
Qwen2.5-0.5B LoRA policy toward exact integer-addition rewards without breaking rollout/current
logprob parity?

It is not presented as a general arithmetic benchmark. The policy trains for 24 steps on one fixed
six-digit addition prompt. Evaluation therefore reports that prompt separately from 16 held-out
prompts sampled from the same six-digit distribution.

## Fixed protocol

- model: local `Qwen2.5-0.5B-Instruct`;
- LoRA rank 16, fp16, gradient checkpointing;
- 24 steps, one prompt per step, four 32-token generations;
- learning rate `1e-5`, experiment seed `17`;
- evaluation seed `20260904`;
- training-prompt evaluation: 8 repeated entries × 4 samples = 32 rollouts;
- held-out evaluation: 16 distinct prompts × 4 samples = 64 rollouts;
- identical sampling seeds before and after training.

Command:

```powershell
.\.venv\Scripts\python.exe reward_curve_experiment.py --steps 24 --seed 17 --heldout-prompts 16 --output-dir results
```

## Result

| Evaluation slice | Baseline | Trained | Absolute change |
|---|---:|---:|---:|
| Repeated training prompt | 20/32 (62.5%) | 32/32 (100%) | +37.5 pp |
| Held-out arithmetic | 14/64 (21.875%) | 22/64 (34.375%) | +12.5 pp |

For the held-out prompts, six improved, one worsened, and nine were unchanged when comparing the
number of correct samples out of four. The run improved the training-prompt result and produced a
descriptive held-out gain. Sixteen prompts do not establish broad generalization.

Training reward by four-step block:

| Steps | Mean rollout reward |
|---:|---:|
| 0–3 | 0.8125 |
| 4–7 | 0.8750 |
| 8–11 | 1.0000 |
| 12–15 | 1.0000 |
| 16–19 | 1.0000 |
| 20–23 | 1.0000 |

Only steps 0, 2, 3, 6, and 7 produced non-zero gradients. From step 8 onward every member of each
four-sample GRPO group received reward 1, so mean-centered group-relative advantage correctly
became zero. The [multi-prompt curriculum](curriculum.md) tests training beyond this saturated task.

Peak trainer allocation was 1.88 GiB. The rollout/current-policy maximum logprob difference stayed
below the 0.20 guard throughout and fell near zero after the output distribution concentrated.

## Reproducibility and export checks

Two full executions with the same seeds produced byte-identical training CSV and LoRA tensor files:

- CSV SHA-256: `AAFD4D1F07F9D20747622D2B435AA7735A4D54E90E7931C7CC30ED4475A47A71`;
- adapter safetensors SHA-256: `F38E08BFF02E4FE12B477E8FEBBBF090ECE4CCEBAA5EE60E5C898CE2350DB021`.

PEFT serializes `target_modules` from a set, which initially made only the JSON list order vary
between processes. The exporter now sorts that field and atomically writes canonical metadata.

Finally, `verify_reward_curve_adapter.py` loaded the exported adapter onto a fresh base model and
exactly reproduced both recorded evaluation objects, including completion hashes:

```text
saved adapter reload: EXACT EVALUATION MATCH (training=32/32, heldout=22/64)
```

Machine-readable artifacts are in `results/reward_curve_seed17.json` and
`results/reward_curve_seed17.csv`. The 35.2 MB adapter is retained locally but ignored by Git.
