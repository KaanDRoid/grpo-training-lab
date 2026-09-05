"""Reload a curriculum adapter and reproduce its recorded held-out evaluation."""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from arithmetic_curriculum import ArithmeticExample
from curriculum_experiment import evaluate_curriculum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    model_name = report["experiment"]["model"]
    adapter_path = Path(report["adapter_path"])
    heldout = {
        int(digits): [ArithmeticExample(**pair) for pair in pairs]
        for digits, pairs in report["split"]["heldout_pairs"].items()
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
    model = PeftModel.from_pretrained(base, adapter_path).to("cuda")
    actual = evaluate_curriculum(
        model,
        tokenizer,
        heldout,
        eval_seed=report["experiment"]["eval_seed"],
        batch_size=report["experiment"].get("eval_batch_size", 4),
    )
    if actual != report["trained"]:
        raise AssertionError(
            "reloaded adapter evaluation differs from the recorded trained result\n"
            f"expected={report['trained']}\nactual={actual}"
        )
    print(
        "saved curriculum adapter reload: EXACT EVALUATION MATCH "
        f"({actual['correct']}/{actual['samples']}, sha256={actual['rollout_sha256']})"
    )


if __name__ == "__main__":
    main()
