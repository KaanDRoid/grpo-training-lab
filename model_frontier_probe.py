"""Find deterministic arithmetic prompts with mixed group rewards for a model checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from arithmetic_curriculum import ArithmeticCurriculum, unpack_examples
from grpo_trainer import build_trainer_model
from reward_curve_experiment import evaluate_dataset, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--operand-digits", default="3,4,5,6")
    parser.add_argument("--prompts-per-stage", type=int, default=6)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    digits = tuple(int(value.strip()) for value in args.operand_digits.split(","))
    curriculum = ArithmeticCurriculum(seed=args.seed, operand_digits=digits)
    examples_by_digits = curriculum.heldout_examples(args.prompts_per_stage)

    seed_everything(args.seed)
    model, tokenizer = build_trainer_model(str(args.model), "cuda")
    report = {}
    for stage, (operand_digits, examples) in enumerate(examples_by_digits.items()):
        prompts, answers = unpack_examples(examples)
        result = evaluate_dataset(
            model,
            tokenizer,
            prompts,
            answers,
            seed=args.seed + stage * 1_000,
            batch_size=args.batch_size,
        )
        report[str(operand_digits)] = {
            "accuracy": result["accuracy"],
            "correct": result["correct"],
            "samples": result["samples"],
            "prompts": [
                {
                    "a": example.a,
                    "b": example.b,
                    "answer": example.answer,
                    "correct_samples": correct,
                }
                for example, correct in zip(examples, result["correct_per_prompt"])
            ],
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
