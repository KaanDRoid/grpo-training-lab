"""Reload a saved experiment adapter and reproduce its recorded deterministic evaluation."""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from reward_curve_experiment import evaluate_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    model_name = report["experiment"]["model"]
    adapter_path = Path(report["adapter_path"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
    model = PeftModel.from_pretrained(base, adapter_path).to("cuda")
    actual = evaluate_policy(
        model,
        tokenizer,
        eval_seed=report["experiment"]["eval_seed"],
        heldout_prompts=report["experiment"]["heldout_prompts"],
        batch_size=4,
    )
    if actual != report["trained"]:
        raise AssertionError(
            "reloaded adapter evaluation differs from the recorded trained result\n"
            f"expected={report['trained']}\nactual={actual}"
        )
    print(
        "saved adapter reload: EXACT EVALUATION MATCH "
        f"(training={actual['training_prompt']['correct']}/{actual['training_prompt']['samples']}, "
        f"heldout={actual['heldout_arithmetic']['correct']}/{actual['heldout_arithmetic']['samples']})"
    )


if __name__ == "__main__":
    main()
