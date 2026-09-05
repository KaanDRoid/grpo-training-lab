"""Measure GRPO reward variance without updating the policy or touching held-out prompts."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import torch

from arithmetic_curriculum import ArithmeticCurriculum, unpack_examples
from curriculum_experiment import atomic_json
from grpo_trainer import build_trainer_model, format_instruct_prompt
from hf_rollout_client import HFRolloutClient
from lora_fp16_config import GEN, MODEL
from reward import arithmetic_reward, expand_answers
from reward_curve_experiment import seed_everything


def _condition_stats(groups: list[torch.Tensor]) -> dict[str, float | int]:
    stacked = torch.stack(groups)
    widths = stacked.max(dim=1).values - stacked.min(dim=1).values
    return {
        "groups": stacked.shape[0],
        "signal_groups": int((widths > 1e-8).sum().item()),
        "signal_fraction": float((widths > 1e-8).float().mean().item()),
        "mean_reward": float(stacked.mean().item()),
        "mean_within_group_std": float(stacked.std(dim=1, unbiased=True).mean().item()),
    }


@torch.inference_mode()
def probe(
    model,
    tokenizer,
    *,
    seeds: list[int],
    operand_digits: tuple[int, ...],
    steps_per_stage: int,
    prompts_per_step: int,
    group_sizes: tuple[int, ...],
) -> dict:
    maximum_group = max(group_sizes)
    generation = replace(GEN, num_generations=maximum_group)
    client = HFRolloutClient(model, tokenizer, generation, device="cuda")
    grouped: dict[str, list[torch.Tensor]] = {
        f"{mode}_g{size}": []
        for mode in ("exact", "relative_distance")
        for size in group_sizes
    }
    per_seed = []
    digest = hashlib.sha256()

    torch.cuda.reset_peak_memory_stats()
    for seed in seeds:
        curriculum = ArithmeticCurriculum(
            seed=seed,
            operand_digits=operand_digits,
            steps_per_stage=steps_per_stage,
        )
        seed_groups = {key: [] for key in grouped}
        total_steps = steps_per_stage * len(operand_digits)
        for step in range(total_steps):
            examples = curriculum.batch_for_step(step, prompts_per_step)
            prompts, answers = unpack_examples(examples)
            formatted = [format_instruct_prompt(tokenizer, prompt) for prompt in prompts]
            rollout = client.generate(formatted, seed=seed + step)
            digest.update(rollout.completion_ids.numpy().tobytes())
            digest.update(rollout.attention_mask.numpy().tobytes())
            expanded = expand_answers(answers, maximum_group)
            rewards = {
                mode: arithmetic_reward(rollout.completions, expanded, mode=mode)
                for mode in ("exact", "relative_distance")
            }
            for prompt_index in range(prompts_per_step):
                start = prompt_index * maximum_group
                for mode, values in rewards.items():
                    for size in group_sizes:
                        key = f"{mode}_g{size}"
                        group = values[start:start + size]
                        grouped[key].append(group)
                        seed_groups[key].append(group)
        per_seed.append(
            {
                "seed": seed,
                "conditions": {
                    key: _condition_stats(groups) for key, groups in seed_groups.items()
                },
            }
        )

    return {
        "protocol": {
            "model": str(MODEL.model),
            "seeds": seeds,
            "operand_digits": list(operand_digits),
            "steps_per_stage": steps_per_stage,
            "prompts_per_step": prompts_per_step,
            "group_sizes": list(group_sizes),
            "selection_data": "training prompts only; no held-out prompts or scores",
            "generation": asdict(generation),
        },
        "overall": {key: _condition_stats(groups) for key, groups in grouped.items()},
        "per_seed": per_seed,
        "rollout_sha256": digest.hexdigest(),
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
    }


def _csv_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(","))
    if not parsed or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("values must be comma-separated positive integers")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=_csv_ints, default=(53, 59, 61))
    parser.add_argument("--operand-digits", type=_csv_ints, default=(7, 8))
    parser.add_argument("--stage-steps", type=int, default=12)
    parser.add_argument("--prompts-per-step", type=int, default=2)
    parser.add_argument("--group-sizes", type=_csv_ints, default=(4, 8))
    parser.add_argument("--model", default=MODEL.model)
    parser.add_argument(
        "--output", type=Path, default=Path("results/reward_signal_probe_1p5b.json")
    )
    args = parser.parse_args()
    if max(args.group_sizes) > 16:
        raise ValueError("group-size probe is capped at 16 on the local 8 GB GPU")
    if min(args.stage_steps, args.prompts_per_step) < 1:
        raise ValueError("stage-steps and prompts-per-step must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this probe")

    seed_everything(args.seeds[0])
    model, tokenizer = build_trainer_model(args.model, "cuda")
    result = probe(
        model,
        tokenizer,
        seeds=list(args.seeds),
        operand_digits=args.operand_digits,
        steps_per_stage=args.stage_steps,
        prompts_per_step=args.prompts_per_step,
        group_sizes=args.group_sizes,
    )
    result["protocol"]["model"] = args.model
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
