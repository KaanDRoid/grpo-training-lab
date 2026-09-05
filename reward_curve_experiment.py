"""Deterministic baseline-vs-trained GRPO experiment for the local RTX 5060.

The experiment deliberately reports both the repeated training prompt and held-out arithmetic.
Improvement on the first is memorization/task optimization; only the second is evidence of
within-distribution generalization.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import torch

from grpo_trainer import build_trainer_model, format_instruct_prompt, train
from hf_rollout_client import HFRolloutClient
from lora_fp16_config import GEN, MODEL
from reward import arithmetic_reward, expand_answers, make_arithmetic_batch


TRAIN_PROMPT = "Compute the sum. Reply with ONLY the integer, nothing else.\n847293 + 581947 ="
TRAIN_ANSWER = 1_429_240


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@torch.inference_mode()
def evaluate_dataset(
    model,
    tokenizer,
    prompts: list[str],
    answers: list[int],
    *,
    seed: int,
    batch_size: int,
) -> dict[str, Any]:
    generation = replace(GEN, num_generations=4, temperature=1.0, top_p=1.0)
    client = HFRolloutClient(model, tokenizer, generation, device=str(next(model.parameters()).device))
    all_rewards: list[float] = []
    all_completions: list[str] = []
    digest = hashlib.sha256()

    for batch_index, start in enumerate(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[start:start + batch_size]
        batch_answers = answers[start:start + batch_size]
        formatted = [format_instruct_prompt(tokenizer, prompt) for prompt in batch_prompts]
        rollout = client.generate(formatted, seed=seed + batch_index)
        rewards = arithmetic_reward(
            rollout.completions,
            expand_answers(batch_answers, generation.num_generations),
        )
        all_rewards.extend(rewards.tolist())
        all_completions.extend(rollout.completions)
        digest.update(rollout.completion_ids.numpy().tobytes())
        digest.update(rollout.attention_mask.numpy().tobytes())

    correct = int(sum(all_rewards))
    total = len(all_rewards)
    samples_per_prompt = generation.num_generations
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "samples": total,
        "correct_per_prompt": [
            int(sum(all_rewards[start:start + samples_per_prompt]))
            for start in range(0, total, samples_per_prompt)
        ],
        "rollout_sha256": digest.hexdigest(),
        "examples": all_completions[:8],
    }


def evaluate_policy(model, tokenizer, *, eval_seed: int, heldout_prompts: int, batch_size: int):
    repeated_prompts = [TRAIN_PROMPT] * 8
    repeated_answers = [TRAIN_ANSWER] * len(repeated_prompts)
    heldout, heldout_answers = make_arithmetic_batch(
        heldout_prompts, seed=eval_seed + 10_000, max_operand=999_999
    )
    return {
        "training_prompt": evaluate_dataset(
            model,
            tokenizer,
            repeated_prompts,
            repeated_answers,
            seed=eval_seed,
            batch_size=batch_size,
        ),
        "heldout_arithmetic": evaluate_dataset(
            model,
            tokenizer,
            heldout,
            heldout_answers,
            seed=eval_seed + 1_000,
            batch_size=batch_size,
        ),
    }


def write_metrics_csv(path: Path, metrics: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "step", "reward_mean", "loss", "grad_norm", "parity_max", "parity_mean",
        "peak_vram_gib", "rollout_sha256",
    ]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric in metrics:
            writer.writerow({field: metric[field] for field in fields})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--eval-seed", type=int, default=20_260_904)
    parser.add_argument("--heldout-prompts", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.steps < 1 or args.heldout_prompts < 1 or args.eval_batch_size < 1:
        raise ValueError("steps, heldout-prompts, and eval-batch-size must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this experiment")

    seed_everything(args.seed)
    model, tokenizer = build_trainer_model(MODEL.small_model, "cuda")
    baseline = evaluate_policy(
        model,
        tokenizer,
        eval_seed=args.eval_seed,
        heldout_prompts=args.heldout_prompts,
        batch_size=args.eval_batch_size,
    )
    print("baseline:", json.dumps(baseline, sort_keys=True))

    metrics = train(
        steps=args.steps,
        prompts_per_step=1,
        model_name=MODEL.small_model,
        device="cuda",
        lr=args.lr,
        rollout_backend="hf",
        max_operand=999_999,
        smoke=True,
        seed=args.seed,
        model_instance=model,
        tokenizer_instance=tokenizer,
    )

    trained = evaluate_policy(
        model,
        tokenizer,
        eval_seed=args.eval_seed,
        heldout_prompts=args.heldout_prompts,
        batch_size=args.eval_batch_size,
    )
    print("trained:", json.dumps(trained, sort_keys=True))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"reward_curve_seed{args.seed}"
    adapter_dir = args.output_dir / f"{stem}_adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    # PEFT stores target_modules internally as a set, so its JSON list order can vary with
    # Python's per-process hash seed even when every tensor is identical. Canonicalize metadata
    # so the complete exported adapter, not only its safetensors file, is reproducible.
    adapter_config_path = adapter_dir / "adapter_config.json"
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    if isinstance(adapter_config.get("target_modules"), list):
        adapter_config["target_modules"] = sorted(adapter_config["target_modules"])
    _atomic_json(adapter_config_path, adapter_config)
    report = {
        "experiment": {
            "steps": args.steps,
            "seed": args.seed,
            "learning_rate": args.lr,
            "eval_seed": args.eval_seed,
            "heldout_prompts": args.heldout_prompts,
            "samples_per_prompt": 4,
            "training_prompt": TRAIN_PROMPT,
            "training_answer": TRAIN_ANSWER,
            "generation": asdict(GEN),
            "model": MODEL.small_model,
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
        },
        "baseline": baseline,
        "trained": trained,
        "delta": {
            name: trained[name]["accuracy"] - baseline[name]["accuracy"]
            for name in baseline
        },
        "training_metrics": metrics,
        "adapter_path": str(adapter_dir),
    }
    json_path = args.output_dir / f"{stem}.json"
    csv_path = args.output_dir / f"{stem}.csv"
    _atomic_json(json_path, report)
    write_metrics_csv(csv_path, metrics)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"saved adapter {adapter_dir}")


if __name__ == "__main__":
    main()
