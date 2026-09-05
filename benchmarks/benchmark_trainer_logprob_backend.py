"""Compare naive and selected current-policy logprob paths on the real local Qwen model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compute_logprobs import compute_completion_logprobs
from grpo_trainer import build_trainer_model, format_instruct_prompt
from hf_rollout_client import HFRolloutClient
from lora_fp16_config import GEN, MODEL


PROMPTS = [
    "Compute the sum. Reply with ONLY the integer, nothing else.\n847293 + 581947 =",
    "Compute the sum. Reply with ONLY the integer, nothing else.\n73841 + 69427 =",
]


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_backend(model, prompt_ids, batch, pad_id: int, backend: str, chunk_rows: int):
    model.zero_grad(set_to_none=True)
    mask = batch.attention_mask.cuda()
    active = mask.bool()
    upstream = torch.linspace(
        -1.0, 1.0, steps=mask.numel(), device="cuda", dtype=torch.float32
    ).reshape_as(mask) * mask
    torch.cuda.synchronize()
    starting_allocation = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    logprobs = compute_completion_logprobs(
        model,
        prompt_ids,
        batch.completion_ids,
        mask,
        pad_id=pad_id,
        backend=backend,
        chunk_rows=chunk_rows,
    )
    (logprobs * upstream).sum().backward()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1_000
    peak_delta_mib = (torch.cuda.max_memory_allocated() - starting_allocation) / 2**20
    gradients = {
        name: parameter.grad.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    values = logprobs.detach().cpu()
    result = {
        "elapsed_ms": elapsed_ms,
        "peak_incremental_mib": peak_delta_mib,
        "active_logprob_mean": values[active.cpu()].mean().item(),
        "trainable_gradient_tensors": len(gradients),
    }
    del logprobs, upstream
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return result, values, gradients


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/trainer_logprob_backend_rtx5060.json"),
    )
    parser.add_argument("--model", type=Path, default=Path(MODEL.small_model))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--chunk-rows", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model_name = str(args.model)
    model, tokenizer = build_trainer_model(model_name, "cuda")
    client = HFRolloutClient(model, tokenizer, GEN, device="cuda")
    formatted = [format_instruct_prompt(tokenizer, prompt) for prompt in PROMPTS]
    batch = client.generate(formatted, seed=args.seed)
    prompt_ids = [tokenizer(prompt, add_special_tokens=True)["input_ids"] for prompt in batch.prompts]

    naive, naive_values, naive_gradients = run_backend(
        model, prompt_ids, batch, tokenizer.pad_token_id, "naive", args.chunk_rows
    )
    selected, selected_values, selected_gradients = run_backend(
        model, prompt_ids, batch, tokenizer.pad_token_id, "selected", args.chunk_rows
    )
    active = batch.attention_mask.bool()
    value_delta = (selected_values - naive_values).abs()[active]
    if naive_gradients.keys() != selected_gradients.keys():
        raise AssertionError("backends produced different trainable gradient sets")
    gradient_max_abs = max(
        (selected_gradients[name] - naive_gradients[name]).abs().max().item()
        for name in naive_gradients
    )
    naive_squared_norm = sum(
        gradient.double().square().sum().item() for gradient in naive_gradients.values()
    )
    selected_squared_norm = sum(
        gradient.double().square().sum().item() for gradient in selected_gradients.values()
    )
    difference_squared_norm = sum(
        (selected_gradients[name].double() - naive_gradients[name].double()).square().sum().item()
        for name in naive_gradients
    )
    gradient_dot = sum(
        (selected_gradients[name].double() * naive_gradients[name].double()).sum().item()
        for name in naive_gradients
    )
    naive_norm = naive_squared_norm**0.5
    selected_norm = selected_squared_norm**0.5
    report = {
        "model": model_name,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "seed": args.seed,
        "batch": {
            "prompts": len(PROMPTS),
            "generations_per_prompt": GEN.num_generations,
            "active_completion_tokens": int(active.sum()),
        },
        "naive": naive,
        "selected": selected,
        "comparison": {
            "logprob_max_abs": value_delta.max().item(),
            "logprob_mean_abs": value_delta.mean().item(),
            "trainable_gradient_max_abs": gradient_max_abs,
            "naive_gradient_l2": naive_norm,
            "selected_gradient_l2": selected_norm,
            "gradient_relative_l2_error": difference_squared_norm**0.5 / naive_norm,
            "gradient_cosine_similarity": gradient_dot / (naive_norm * selected_norm),
            "memory_reduction_mib": naive["peak_incremental_mib"]
            - selected["peak_incremental_mib"],
            "memory_reduction_ratio": naive["peak_incremental_mib"]
            / selected["peak_incremental_mib"],
            "selected_slowdown": selected["elapsed_ms"] / naive["elapsed_ms"],
        },
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
