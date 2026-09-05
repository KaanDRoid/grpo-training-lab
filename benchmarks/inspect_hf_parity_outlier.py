"""Print token-level details for a reproducible cached-generation parity outlier."""

from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arithmetic_curriculum import ArithmeticCurriculum, unpack_examples
from compute_logprobs import compute_completion_logprobs
from grpo_trainer import build_trainer_model, format_instruct_prompt
from hf_rollout_client import HFRolloutClient
from lora_fp16_config import GEN


def main():
    seed = 67
    step = 1
    model, tokenizer = build_trainer_model(".\\Qwen2.5-1.5B-Instruct", "cuda")
    curriculum = ArithmeticCurriculum(seed=seed, operand_digits=(7, 8), steps_per_stage=12)
    prompts, _ = unpack_examples(curriculum.batch_for_step(step, 2))
    formatted = [format_instruct_prompt(tokenizer, prompt) for prompt in prompts]
    batch = HFRolloutClient(model, tokenizer, GEN, device="cuda").generate(
        formatted, seed=seed + step
    )
    prompt_ids = [tokenizer(p, add_special_tokens=True)["input_ids"] for p in batch.prompts]
    current = compute_completion_logprobs(
        model,
        prompt_ids,
        batch.completion_ids,
        batch.attention_mask,
        tokenizer.pad_token_id,
        backend="naive",
    ).detach().cpu()
    old = batch.rollout_logprobs
    delta = (current - old).abs().masked_fill(~batch.attention_mask.bool(), -1)
    flat_order = torch.argsort(delta.flatten(), descending=True)
    for flat_index in flat_order[:8]:
        row = int(flat_index // delta.shape[1])
        column = int(flat_index % delta.shape[1])
        token_id = int(batch.completion_ids[row, column])
        print(
            f"row={row} col={column} token={token_id} text={tokenizer.decode([token_id])!r} "
            f"rollout={float(old[row, column]):+.6f} recompute={float(current[row, column]):+.6f} "
            f"abs_delta={float(delta[row, column]):.6f}"
        )
    print("completions:", [repr(value) for value in batch.completions])


if __name__ == "__main__":
    main()
