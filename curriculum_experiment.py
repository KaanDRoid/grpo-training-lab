"""Deterministic multi-prompt arithmetic curriculum experiment on the local GPU."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import torch

from arithmetic_curriculum import ArithmeticCurriculum, ArithmeticExample, unpack_examples
from grpo_trainer import build_trainer_model, train
from lora_fp16_config import GEN, MODEL
from reward import REWARD_MODES
from reward_curve_experiment import evaluate_dataset, seed_everything


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def canonicalize_adapter_config(adapter_dir: Path) -> None:
    config_path = adapter_dir / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if isinstance(config.get("target_modules"), list):
        config["target_modules"] = sorted(config["target_modules"])
    atomic_json(config_path, config)


@torch.inference_mode()
def evaluate_curriculum(
    model,
    tokenizer,
    heldout: dict[int, list[ArithmeticExample]],
    *,
    eval_seed: int,
    batch_size: int,
) -> dict[str, Any]:
    by_digits: dict[str, Any] = {}
    total_correct = 0
    total_samples = 0
    digest = hashlib.sha256()
    for stage, (digits, examples) in enumerate(heldout.items()):
        prompts, answers = unpack_examples(examples)
        result = evaluate_dataset(
            model,
            tokenizer,
            prompts,
            answers,
            seed=eval_seed + stage * 1_000,
            batch_size=batch_size,
        )
        by_digits[str(digits)] = result
        total_correct += result["correct"]
        total_samples += result["samples"]
        digest.update(bytes.fromhex(result["rollout_sha256"]))
    return {
        "accuracy": total_correct / total_samples,
        "correct": total_correct,
        "samples": total_samples,
        "rollout_sha256": digest.hexdigest(),
        "by_operand_digits": by_digits,
    }


def write_curriculum_csv(path: Path, metrics: list[dict[str, Any]]) -> None:
    fields = [
        "step", "curriculum_stage", "operand_digits", "prompt", "reward_mean", "loss",
        "reward_mode", "exact_reward_mean", "signal_groups", "exact_signal_groups",
        "grad_norm", "parity_max", "parity_p99", "parity_mean", "parity_tail_tokens",
        "active_completion_tokens", "peak_vram_gib", "rollout_sha256",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric in metrics:
            row = {field: metric.get(field) for field in fields}
            row["prompt"] = " || ".join(metric["prompts"])
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def stage_training_summary(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for digits in sorted({metric["operand_digits"] for metric in metrics}):
        selected = [metric for metric in metrics if metric["operand_digits"] == digits]
        result[str(digits)] = {
            "steps": len(selected),
            "mean_reward": sum(metric["reward_mean"] for metric in selected) / len(selected),
            "mean_exact_reward": sum(
                metric["exact_reward_mean"] for metric in selected
            ) / len(selected),
            "signal_groups": sum(metric["signal_groups"] for metric in selected),
            "exact_signal_groups": sum(metric["exact_signal_groups"] for metric in selected),
            "nonzero_gradient_steps": sum(metric["grad_norm"] > 0 for metric in selected),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--prompts-per-step", type=int, default=2)
    parser.add_argument("--stage-steps", type=int, default=12)
    parser.add_argument(
        "--operand-digits", default="5,6",
        help="comma-separated curriculum stages; the calibrated local default skips easy 4-digit sums",
    )
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--eval-seed", type=int, default=20_260_904)
    parser.add_argument("--heldout-per-stage", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--reward-mode", choices=REWARD_MODES, default="exact")
    parser.add_argument("--num-generations", type=int, default=GEN.num_generations)
    parser.add_argument("--logprob-backend", choices=("naive", "selected"), default="naive")
    parser.add_argument("--logprob-chunk-rows", type=int, default=8)
    parser.add_argument("--hf-recompute-behavior-logprobs", action="store_true")
    parser.add_argument("--model", default=MODEL.small_model)
    parser.add_argument(
        "--run-label", default="",
        help="optional filesystem-safe prefix used to distinguish model families",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    if min(
        args.steps,
        args.prompts_per_step,
        args.stage_steps,
        args.heldout_per_stage,
        args.eval_batch_size,
    ) < 1:
        raise ValueError("step, batch, and held-out counts must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_-]*", args.run_label):
        raise ValueError("run-label may contain only letters, digits, underscores, and hyphens")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this experiment")
    if args.num_generations < 2:
        raise ValueError("num-generations must be at least two for GRPO")

    try:
        operand_digits = tuple(int(value.strip()) for value in args.operand_digits.split(","))
    except ValueError as exc:
        raise ValueError("operand-digits must be a comma-separated integer list") from exc
    curriculum = ArithmeticCurriculum(
        seed=args.seed,
        operand_digits=operand_digits,
        steps_per_stage=args.stage_steps,
    )
    training_examples = curriculum.training_examples(args.steps, args.prompts_per_step)
    heldout = curriculum.heldout_examples(args.heldout_per_stage, exclude=training_examples)
    training_keys = {example.key for example in training_examples}
    heldout_keys = {example.key for examples in heldout.values() for example in examples}
    if training_keys & heldout_keys:
        raise RuntimeError("training/evaluation split leakage detected")

    schedule_payload = [asdict(example) for example in training_examples]
    schedule_bytes = json.dumps(schedule_payload, separators=(",", ":")).encode("utf-8")
    schedule_sha256 = hashlib.sha256(schedule_bytes).hexdigest()

    seed_everything(args.seed)
    model, tokenizer = build_trainer_model(args.model, "cuda")
    training_generation = replace(GEN, num_generations=args.num_generations)
    baseline = evaluate_curriculum(
        model,
        tokenizer,
        heldout,
        eval_seed=args.eval_seed,
        batch_size=args.eval_batch_size,
    )
    print("baseline:", json.dumps(baseline, sort_keys=True))

    metrics = train(
        steps=args.steps,
        prompts_per_step=args.prompts_per_step,
        model_name=args.model,
        device="cuda",
        lr=args.lr,
        rollout_backend="hf",
        seed=args.seed,
        model_instance=model,
        tokenizer_instance=tokenizer,
        curriculum=curriculum,
        hf_recompute_behavior_logprobs=args.hf_recompute_behavior_logprobs,
        reward_mode=args.reward_mode,
        generation_config=training_generation,
        logprob_backend=args.logprob_backend,
        logprob_chunk_rows=args.logprob_chunk_rows,
    )
    trained = evaluate_curriculum(
        model,
        tokenizer,
        heldout,
        eval_seed=args.eval_seed,
        batch_size=args.eval_batch_size,
    )
    print("trained:", json.dumps(trained, sort_keys=True))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_label = "_".join(f"{digits}d" for digits in curriculum.operand_digits)
    label_prefix = f"{args.run_label}_" if args.run_label else ""
    stem = f"{label_prefix}curriculum_{stage_label}_seed{args.seed}"
    adapter_dir = args.output_dir / f"{stem}_adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    canonicalize_adapter_config(adapter_dir)

    per_stage_delta = {
        digits: trained["by_operand_digits"][digits]["accuracy"]
        - baseline["by_operand_digits"][digits]["accuracy"]
        for digits in baseline["by_operand_digits"]
    }
    report = {
        "experiment": {
            "steps": args.steps,
            "prompts_per_step": args.prompts_per_step,
            "learning_rate": args.lr,
            "eval_seed": args.eval_seed,
            "eval_batch_size": args.eval_batch_size,
            "heldout_per_stage": args.heldout_per_stage,
            "samples_per_prompt": args.num_generations,
            "curriculum": curriculum.to_config(),
            "generation": asdict(training_generation),
            "evaluation_generation": asdict(replace(GEN, num_generations=4)),
            "model": args.model,
            "hf_recompute_behavior_logprobs": args.hf_recompute_behavior_logprobs,
            "reward_mode": args.reward_mode,
            "logprob_backend": args.logprob_backend,
            "logprob_chunk_rows": args.logprob_chunk_rows,
        },
        "split": {
            "training_examples": len(training_examples),
            "heldout_examples": len(heldout_keys),
            "overlap": 0,
            "training_schedule_sha256": schedule_sha256,
            "training_schedule": schedule_payload,
            "heldout_pairs": {
                str(digits): [asdict(example) for example in examples]
                for digits, examples in heldout.items()
            },
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
        },
        "baseline": baseline,
        "trained": trained,
        "delta": {
            "overall_accuracy": trained["accuracy"] - baseline["accuracy"],
            "by_operand_digits": per_stage_delta,
        },
        "training_by_operand_digits": stage_training_summary(metrics),
        "training_metrics": metrics,
        "adapter_path": str(adapter_dir),
    }
    json_path = args.output_dir / f"{stem}.json"
    csv_path = args.output_dir / f"{stem}.csv"
    atomic_json(json_path, report)
    write_curriculum_csv(csv_path, metrics)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"saved adapter {adapter_dir}")


if __name__ == "__main__":
    main()
