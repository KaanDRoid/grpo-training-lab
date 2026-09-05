"""Validate and summarize paired reward-signal curriculum runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from curriculum_experiment import atomic_json


_MATCHED_EXPERIMENT_KEYS = (
    "steps",
    "prompts_per_step",
    "learning_rate",
    "eval_seed",
    "eval_batch_size",
    "heldout_per_stage",
    "curriculum",
    "model",
    "hf_recompute_behavior_logprobs",
    "logprob_backend",
    "logprob_chunk_rows",
)


def summarize(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if len(reports) < 2:
        raise ValueError("at least two ablation reports are required")
    reference = reports[0]
    matched_protocol = {
        key: reference["experiment"][key] for key in _MATCHED_EXPERIMENT_KEYS
    }
    schedule_hash = reference["split"]["training_schedule_sha256"]
    heldout_pairs = reference["split"]["heldout_pairs"]
    baseline_hash = reference["baseline"]["rollout_sha256"]
    for report in reports[1:]:
        candidate = {key: report["experiment"][key] for key in _MATCHED_EXPERIMENT_KEYS}
        if candidate != matched_protocol:
            raise ValueError("ablation reports differ outside reward/group-size conditions")
        if report["split"]["training_schedule_sha256"] != schedule_hash:
            raise ValueError("ablation reports use different training schedules")
        if report["split"]["heldout_pairs"] != heldout_pairs:
            raise ValueError("ablation reports use different held-out prompts")
        if report["baseline"]["rollout_sha256"] != baseline_hash:
            raise ValueError("ablation reports do not share the same sampled baseline")

    rows = []
    for report in reports:
        experiment = report["experiment"]
        metrics = report["training_metrics"]
        group_size = experiment["generation"]["num_generations"]
        row = {
            "condition": f"{experiment['reward_mode']}_g{group_size}",
            "reward_mode": experiment["reward_mode"],
            "group_size": group_size,
            "training_completions": (
                experiment["steps"] * experiment["prompts_per_step"] * group_size
            ),
            "baseline_correct": report["baseline"]["correct"],
            "trained_correct": report["trained"]["correct"],
            "heldout_samples": report["trained"]["samples"],
            "delta_correct": report["trained"]["correct"] - report["baseline"]["correct"],
            "delta_accuracy": report["trained"]["accuracy"] - report["baseline"]["accuracy"],
            "signal_groups": sum(metric["signal_groups"] for metric in metrics),
            "exact_signal_groups": sum(metric["exact_signal_groups"] for metric in metrics),
            "total_prompt_groups": experiment["steps"] * experiment["prompts_per_step"],
            "nonzero_gradient_steps": sum(metric["grad_norm"] > 0 for metric in metrics),
            "peak_vram_gib": max(metric["peak_vram_gib"] for metric in metrics),
            "max_raw_parity": max(metric["parity_max"] for metric in metrics),
            "max_raw_parity_mean": max(metric["parity_mean"] for metric in metrics),
            "by_operand_digits": {
                digit: {
                    "baseline_correct": report["baseline"]["by_operand_digits"][digit]["correct"],
                    "trained_correct": report["trained"]["by_operand_digits"][digit]["correct"],
                    "samples": report["trained"]["by_operand_digits"][digit]["samples"],
                }
                for digit in report["trained"]["by_operand_digits"]
            },
        }
        rows.append(row)

    return {
        "method": {
            "design": "paired single-seed mechanism pilot",
            "selection": "candidate mechanisms selected using training-prompt reward variance only",
            "evaluation": "binary exact match on an identical held-out prompt/sample set",
            "warning": "one paired seed ranks these mechanisms but does not estimate population uncertainty",
        },
        "matched_protocol": matched_protocol,
        "training_schedule_sha256": schedule_hash,
        "baseline_rollout_sha256": baseline_hash,
        "conditions": rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat_rows = [{key: value for key, value in row.items() if key != "by_operand_digits"}
                 for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("results/reward_ablation_seed67_summary.json")
    )
    args = parser.parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    result = summarize(reports)
    atomic_json(args.output, result)
    write_csv(args.output.with_suffix(".csv"), result["conditions"])
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    print(f"wrote {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
