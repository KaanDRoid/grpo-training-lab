"""Aggregate deterministic curriculum reports with a run-level Student-t interval."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any

from curriculum_experiment import atomic_json


_T_975 = {
    1: 12.7062047364,
    2: 4.3026527297,
    3: 3.1824463053,
    4: 2.7764451052,
    5: 2.5705818356,
    6: 2.4469118511,
    7: 2.3646242510,
    8: 2.3060041352,
    9: 2.2621571629,
    10: 2.2281388520,
    11: 2.2009851601,
    12: 2.1788128297,
    13: 2.1603686565,
    14: 2.1447866879,
    15: 2.1314495456,
    16: 2.1199052992,
    17: 2.1098155778,
    18: 2.1009220402,
    19: 2.0930240544,
    20: 2.0859634473,
    21: 2.0796138447,
    22: 2.0738730679,
    23: 2.0686576104,
    24: 2.0638985616,
    25: 2.0595385528,
    26: 2.0555294386,
    27: 2.0518305165,
    28: 2.0484071418,
    29: 2.0452296421,
    30: 2.0422724563,
}


def student_t_95_interval(values: list[float]) -> dict[str, float | int]:
    if len(values) < 2:
        raise ValueError("at least two independent runs are required for a confidence interval")
    degrees_of_freedom = len(values) - 1
    if degrees_of_freedom not in _T_975:
        raise ValueError("exact t critical values are provided for at most 31 runs")
    mean = statistics.fmean(values)
    sample_std = statistics.stdev(values)
    standard_error = sample_std / math.sqrt(len(values))
    margin = _T_975[degrees_of_freedom] * standard_error
    return {
        "runs": len(values),
        "mean": mean,
        "sample_std": sample_std,
        "standard_error": standard_error,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def _protocol(report: dict[str, Any]) -> dict[str, Any]:
    experiment = report["experiment"]
    protocol = {
        key: experiment[key]
        for key in (
            "steps",
            "prompts_per_step",
            "learning_rate",
            "eval_seed",
            "heldout_per_stage",
            "samples_per_prompt",
            "curriculum",
            "generation",
            "model",
            "hf_recompute_behavior_logprobs",
        )
    }
    protocol["curriculum"] = {
        key: value
        for key, value in protocol["curriculum"].items()
        if key != "seed"
    }
    # Reports predating the reward ablation used exact match implicitly.
    protocol["reward_mode"] = experiment.get("reward_mode", "exact")
    return protocol


def aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if len(reports) < 2:
        raise ValueError("at least two reports are required")
    protocol = _protocol(reports[0])
    if any(_protocol(report) != protocol for report in reports[1:]):
        raise ValueError("curriculum reports use different experimental protocols")
    seeds = [report["experiment"]["curriculum"]["seed"] for report in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError("each report must use a distinct curriculum seed")

    digits = list(reports[0]["baseline"]["by_operand_digits"])
    run_rows = []
    for seed, report in zip(seeds, reports):
        run_rows.append(
            {
                "seed": seed,
                "baseline_correct": report["baseline"]["correct"],
                "trained_correct": report["trained"]["correct"],
                "samples": report["baseline"]["samples"],
                "delta": report["trained"]["accuracy"] - report["baseline"]["accuracy"],
                "nonzero_gradient_steps": sum(
                    metric["grad_norm"] > 0 for metric in report["training_metrics"]
                ),
                "peak_vram_gib": max(
                    metric["peak_vram_gib"] for metric in report["training_metrics"]
                ),
                "max_raw_parity": max(
                    metric["parity_max"] for metric in report["training_metrics"]
                ),
                "max_raw_parity_mean": max(
                    metric["parity_mean"] for metric in report["training_metrics"]
                ),
            }
        )

    total_samples = sum(row["samples"] for row in run_rows)
    total_baseline = sum(row["baseline_correct"] for row in run_rows)
    total_trained = sum(row["trained_correct"] for row in run_rows)
    by_digits = {}
    for digit in digits:
        deltas = [
            report["trained"]["by_operand_digits"][digit]["accuracy"]
            - report["baseline"]["by_operand_digits"][digit]["accuracy"]
            for report in reports
        ]
        by_digits[digit] = {
            "delta_accuracy_run_level": student_t_95_interval(deltas),
            "pooled_baseline_correct": sum(
                report["baseline"]["by_operand_digits"][digit]["correct"]
                for report in reports
            ),
            "pooled_trained_correct": sum(
                report["trained"]["by_operand_digits"][digit]["correct"]
                for report in reports
            ),
            "pooled_samples": sum(
                report["baseline"]["by_operand_digits"][digit]["samples"]
                for report in reports
            ),
        }

    return {
        "method": {
            "confidence_interval": "two-sided 95% Student-t over independent run deltas",
            "statistical_unit": "curriculum seed",
            "warning": "three runs give a very wide pilot interval; pooled samples are descriptive",
        },
        "protocol": protocol,
        "seeds": seeds,
        "runs": run_rows,
        "overall": {
            "delta_accuracy_run_level": student_t_95_interval(
                [row["delta"] for row in run_rows]
            ),
            "pooled_baseline_correct": total_baseline,
            "pooled_trained_correct": total_trained,
            "pooled_samples": total_samples,
            "pooled_baseline_accuracy": total_baseline / total_samples,
            "pooled_trained_accuracy": total_trained / total_samples,
            "pooled_delta_accuracy": (total_trained - total_baseline) / total_samples,
        },
        "by_operand_digits": by_digits,
        "training": {
            "nonzero_gradient_steps": sum(
                row["nonzero_gradient_steps"] for row in run_rows
            ),
            "total_steps": sum(len(report["training_metrics"]) for report in reports),
            "max_peak_vram_gib": max(row["peak_vram_gib"] for row in run_rows),
            "max_raw_parity": max(row["max_raw_parity"] for row in run_rows),
            "max_raw_parity_mean": max(row["max_raw_parity_mean"] for row in run_rows),
        },
    }


def write_runs_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/qwen1p5b_curriculum_multiseed_summary.json"),
    )
    args = parser.parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    summary = aggregate(reports)
    atomic_json(args.output, summary)
    csv_path = args.output.with_suffix(".csv")
    write_runs_csv(csv_path, summary["runs"])
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
