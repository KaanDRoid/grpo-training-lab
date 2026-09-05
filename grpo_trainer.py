"""
Single-GPU GRPO trainer -- the RL training backbone.

Loop:  sample prompts -> vLLM rollout (rollout_client) -> reward -> group-relative advantage
       -> recompute CURRENT-policy logprobs (compute_logprobs) -> grpo_loss -> step
       -> periodically push weights back into the colocated vLLM engine (trainer-inference sync).

Both the shared-model HF backend and the WSL/vLLM LoRA hot-reload backend are verified on an 8 GB
RTX 5060 Laptop GPU. The unit-tested pieces (GRPO loss, advantage, logprob alignment, and exact
checkpoint continuation) must remain correct before trusting a reward curve.

NOTE on old_logprobs vs current logprobs: vLLM behavior scores remain distinct from the current
policy recomputation. The optional synchronous-HF mode may use ``current.detach()`` as the exact
on-policy behavior value because no update can occur between rollout and recomputation. Raw
generation-score parity is still monitored in either case.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random

import torch

from arithmetic_curriculum import ArithmeticCurriculum, unpack_examples
from checkpointing import load_checkpoint, save_checkpoint
from lora_fp16_config import MODEL, LORA, VLLM, GEN, GRPO
from rollout_client import VLLMRolloutClient
from grpo_loss import grpo_loss, group_relative_advantage
from compute_logprobs import compute_completion_logprobs
from reward import REWARD_MODES, make_arithmetic_batch, arithmetic_reward, expand_answers


class SimulatedCrash(RuntimeError):
    """Intentional post-checkpoint failure used to validate recovery."""


def rollout_digest(batch):
    """Stable fingerprint of sampled token ids and their real-token mask."""
    encoded = json.dumps(
        {
            "completion_ids": batch.completion_ids.tolist(),
            "attention_mask": batch.attention_mask.tolist(),
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def logprob_parity_stats(logprobs, old_logprobs, completion_mask, *, hard_max=0.35):
    """Summarize same-policy numerical drift without letting one tail token hide broad drift."""
    active_delta = (logprobs - old_logprobs).abs()[completion_mask.bool()].float()
    if not active_delta.numel():
        return {
            "max": 0.0, "p99": 0.0, "mean": 0.0, "active_tokens": 0,
            "tail_tokens": 0, "acceptable": True,
        }
    finite = bool(torch.isfinite(active_delta).all())
    maximum = active_delta.max().item()
    percentile_99 = torch.quantile(active_delta, 0.99).item()
    mean = active_delta.mean().item()
    active_tokens = active_delta.numel()
    tail_tokens = int((active_delta > 0.20).sum().item())
    # A single low-probability sampled token is sensitive to fp16 batch-shape changes. Guard it
    # with a hard ceiling. For short completions p99 and tail fractions become sample-size brittle;
    # the mean catches systematic mismatch while the tail count remains a diagnostic.
    acceptable = finite and maximum <= hard_max and mean <= 0.03
    return {
        "max": maximum,
        "p99": percentile_99,
        "mean": mean,
        "active_tokens": active_tokens,
        "tail_tokens": tail_tokens,
        "acceptable": acceptable,
    }


def behavior_logprobs_for_loss(
    rollout_logprobs, current_logprobs, rollout_backend, hf_recompute
):
    """Choose PPO denominator values without leaking gradients into the behavior policy."""
    if rollout_backend == "hf" and hf_recompute:
        return current_logprobs.detach()
    return rollout_logprobs


def make_run_config(
    *, model_name, rollout_backend, prompts_per_step, lr, sync_every, max_operand, smoke, seed,
    curriculum, logprob_backend, logprob_chunk_rows, smoke_operands, vllm_config,
    hf_recompute_behavior_logprobs, reward_mode, generation_config
):
    """The immutable semantics that must match before a checkpoint can be resumed."""
    return {
        "model_name": str(model_name),
        "rollout_backend": rollout_backend,
        "prompts_per_step": prompts_per_step,
        "lr": lr,
        "sync_every": sync_every,
        "max_operand": max_operand,
        "smoke": smoke,
        "smoke_operands": list(smoke_operands),
        "seed": seed,
        "curriculum": curriculum.to_config() if curriculum is not None else None,
        "logprob_backend": logprob_backend,
        "logprob_chunk_rows": logprob_chunk_rows,
        "hf_recompute_behavior_logprobs": hf_recompute_behavior_logprobs,
        "reward_mode": reward_mode,
        "lora": asdict(LORA),
        "vllm": asdict(vllm_config),
        "generation": asdict(generation_config),
        "grpo": asdict(GRPO),
    }


def format_instruct_prompt(tokenizer, prompt):
    """Use the checkpoint's chat contract when available; fall back to plain text for base LMs."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def build_trainer_model(model_name, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
    model.config.use_cache = False  # incompatible with grad checkpointing
    if MODEL.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # needed for grad-checkpointing + PEFT

    lcfg = LoraConfig(
        r=LORA.r, lora_alpha=LORA.alpha, target_modules=LORA.target_modules,
        lora_dropout=LORA.dropout, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lcfg)
    return model.to(device), tok


def train(steps=5, prompts_per_step=1, model_name=None, device="cuda",
          lr=1e-5, sync_every=1, rollout_backend="hf", max_operand=999_999,
          smoke=False, seed=0, checkpoint_dir=None, resume=False, save_every=1,
          simulate_crash_after=None, model_instance=None, tokenizer_instance=None,
          curriculum=None, logprob_backend="naive", logprob_chunk_rows=8,
          smoke_operands=(847_293, 581_947), vllm_config=None,
          hf_recompute_behavior_logprobs=False, reward_mode="exact", generation_config=None):
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if sync_every < 1 or save_every < 1:
        raise ValueError("sync_every and save_every must be positive")
    if resume and checkpoint_dir is None:
        raise ValueError("--resume requires --checkpoint-dir")
    if simulate_crash_after is not None and checkpoint_dir is None:
        raise ValueError("--simulate-crash-after requires --checkpoint-dir")
    if simulate_crash_after is not None and not 1 <= simulate_crash_after <= steps:
        raise ValueError("--simulate-crash-after must lie in [1, steps]")
    if smoke and curriculum is not None:
        raise ValueError("smoke and curriculum modes are mutually exclusive")
    if logprob_backend not in ("naive", "selected"):
        raise ValueError(f"unknown logprob backend {logprob_backend!r}")
    if logprob_chunk_rows < 1:
        raise ValueError("logprob_chunk_rows must be positive")
    if reward_mode not in REWARD_MODES:
        raise ValueError(f"unknown arithmetic reward mode {reward_mode!r}")
    if logprob_backend == "selected" and not device.startswith("cuda"):
        raise ValueError("selected logprob backend requires a CUDA device")

    model_name = model_name or MODEL.small_model
    vllm_config = vllm_config or VLLM
    generation_config = generation_config or GEN
    if generation_config.num_generations < 2:
        raise ValueError("generation num_generations must be at least two for GRPO")
    if not 0.0 < vllm_config.gpu_memory_utilization < 1.0:
        raise ValueError("vLLM gpu_memory_utilization must lie in (0, 1)")
    if (
        vllm_config.max_num_batched_tokens is not None
        and vllm_config.max_num_batched_tokens < 1
    ):
        raise ValueError("vLLM max_num_batched_tokens must be positive")
    if vllm_config.max_num_seqs is not None and vllm_config.max_num_seqs < 1:
        raise ValueError("vLLM max_num_seqs must be positive")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if (model_instance is None) != (tokenizer_instance is None):
        raise ValueError("model_instance and tokenizer_instance must be supplied together")
    if model_instance is None:
        model, tok = build_trainer_model(model_name, device)
    else:
        model, tok = model_instance, tokenizer_instance
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    run_config = make_run_config(
        model_name=model_name,
        rollout_backend=rollout_backend,
        prompts_per_step=prompts_per_step,
        lr=lr,
        sync_every=sync_every,
        max_operand=max_operand,
        smoke=smoke,
        seed=seed,
        curriculum=curriculum,
        logprob_backend=logprob_backend,
        logprob_chunk_rows=logprob_chunk_rows,
        smoke_operands=smoke_operands,
        vllm_config=vllm_config,
        hf_recompute_behavior_logprobs=hf_recompute_behavior_logprobs,
        reward_mode=reward_mode,
        generation_config=generation_config,
    )
    start_step = 0
    metrics = []
    if resume:
        payload = load_checkpoint(
            checkpoint_dir,
            model=model,
            optimizer=optim,
            expected_run_config=run_config,
        )
        start_step = payload["next_step"]
        metrics = list(payload["metrics"])
        if start_step > steps:
            raise ValueError(
                f"checkpoint is already at step {start_step}, beyond target steps={steps}"
            )
        if simulate_crash_after is not None and simulate_crash_after <= start_step:
            raise ValueError(
                f"simulated crash step {simulate_crash_after} was already completed"
            )
        print(f"resumed checkpoint at step {start_step}; target step {steps}")

    if rollout_backend == "hf":
        from hf_rollout_client import HFRolloutClient
        client = HFRolloutClient(model, tok, generation_config, device=device)
    elif rollout_backend == "vllm":
        client = VLLMRolloutClient(
            model_name, vllm_config, generation_config
        )  # colocated engine, same base
    else:
        raise ValueError(f"unknown rollout backend {rollout_backend!r}")
    if resume and rollout_backend == "vllm":
        # The new worker initially owns base weights; install the restored policy before rollout.
        client.sync_weights(model)
    n = generation_config.num_generations
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    for step in range(start_step, steps):
        if smoke:
            # A fixed, verified hard case makes the one-step hardware check exercise a non-zero
            # policy gradient on this checkpoint instead of depending on a lucky random prompt.
            if prompts_per_step != 1:
                raise ValueError("--smoke requires --prompts-per-step 1")
            prompts = [
                "Compute the sum. Reply with ONLY the integer, nothing else.\n"
                f"{smoke_operands[0]} + {smoke_operands[1]} ="
            ]
            answers = [sum(smoke_operands)]
        elif curriculum is not None:
            examples = curriculum.batch_for_step(step, prompts_per_step)
            prompts, answers = unpack_examples(examples)
        else:
            prompts, answers = make_arithmetic_batch(
                prompts_per_step, seed=seed + step, max_operand=max_operand
            )
        rollout_prompts = [format_instruct_prompt(tok, prompt) for prompt in prompts]
        rollout_seed = seed + step
        torch.manual_seed(rollout_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rollout_seed)

        # --- rollout (behavior policy) ---
        batch = client.generate(
            rollout_prompts, seed=rollout_seed
        )  # B = prompts_per_step * n, prompt-major
        rollout_logprobs = batch.rollout_logprobs.to(device)
        completion_mask = batch.attention_mask.to(device)

        # --- reward + group-relative advantage ---
        answers_expanded = expand_answers(answers, n)
        exact_rewards = arithmetic_reward(batch.completions, answers_expanded).to(device)
        rewards = arithmetic_reward(
            batch.completions, answers_expanded, mode=reward_mode
        ).to(device)
        advantages = group_relative_advantage(rewards, group_size=n,
                                              std_normalize=GRPO.std_normalize)
        reward_groups = rewards.view(-1, n)
        exact_groups = exact_rewards.view(-1, n)
        signal_groups = int(((reward_groups.max(dim=1).values
                              - reward_groups.min(dim=1).values) > 0).sum().item())
        exact_signal_groups = int(((exact_groups.max(dim=1).values
                                    - exact_groups.min(dim=1).values) > 0).sum().item())

        # --- current-policy logprobs (grad on) ---
        prompt_ids_list = [tok(p, add_special_tokens=True)["input_ids"] for p in batch.prompts]
        logprobs = compute_completion_logprobs(
            model,
            prompt_ids_list,
            batch.completion_ids,
            completion_mask,
            tok.pad_token_id,
            backend=logprob_backend,
            chunk_rows=logprob_chunk_rows,
        )

        # Before any update, these are two evaluations of the same policy. Small fp16/cache-path
        # drift is expected; a large delta means tokenization, generation warpers, or the causal
        # shift is wrong and a reward curve would be untrustworthy.
        with torch.no_grad():
            parity_hard_max = (
                1.0 if rollout_backend == "hf" and hf_recompute_behavior_logprobs else 0.35
            )
            parity = logprob_parity_stats(
                logprobs,
                rollout_logprobs,
                completion_mask,
                hard_max=parity_hard_max,
            )
            parity_max = parity["max"]
            parity_p99 = parity["p99"]
            parity_mean = parity["mean"]
        if not parity["acceptable"]:
            raise RuntimeError(
                f"rollout/current logprob parity failed at step {step} before the update: "
                f"max_abs={parity_max:.4f}, p99_abs={parity_p99:.4f}, "
                f"mean_abs={parity_mean:.4f}, tail_tokens={parity['tail_tokens']}/"
                f"{parity['active_tokens']}"
            )
        old_logprobs = behavior_logprobs_for_loss(
            rollout_logprobs,
            logprobs,
            rollout_backend,
            hf_recompute_behavior_logprobs,
        )

        # --- loss + step ---
        loss = grpo_loss(
            logprobs, old_logprobs, advantages, completion_mask,
            clip_eps=GRPO.clip_eps, kl_coef=GRPO.kl_coef, aggregation=GRPO.aggregation)
        optim.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient norm at step {step}: {grad_norm}")
        optim.step()

        # --- trainer -> inference weight sync (trainer-inference communication) ---
        if rollout_backend == "vllm" and (step + 1) % sync_every == 0:
            client.sync_weights(model)

        peak_gb = torch.cuda.max_memory_allocated() / 2**30 if device.startswith("cuda") else 0.0
        metric = {
            "step": step,
            "reward_mean": rewards.mean().item(),
            "rewards": rewards.detach().cpu().tolist(),
            "reward_mode": reward_mode,
            "exact_reward_mean": exact_rewards.mean().item(),
            "exact_rewards": exact_rewards.detach().cpu().tolist(),
            "signal_groups": signal_groups,
            "exact_signal_groups": exact_signal_groups,
            "loss": loss.item(),
            "grad_norm": float(grad_norm),
            "parity_max": parity_max,
            "parity_p99": parity_p99,
            "parity_mean": parity_mean,
            "parity_tail_tokens": parity["tail_tokens"],
            "active_completion_tokens": parity["active_tokens"],
            "peak_vram_gib": peak_gb,
            "rollout_sha256": rollout_digest(batch),
        }
        if curriculum is not None:
            metric["curriculum_stage"] = curriculum.stage_for_step(step)
            metric["operand_digits"] = curriculum.digit_for_step(step)
            metric["prompts"] = prompts
        metrics.append(metric)
        print(
            f"step {step:03d} | reward {metric['reward_mean']:.3f} "
            f"(exact {metric['exact_reward_mean']:.3f}) | "
            f"loss {metric['loss']:+.4f} | grad_norm {metric['grad_norm']:.4f} | "
            f"logprob_delta max/mean "
            f"{parity_max:.4f}/{parity_mean:.4f} | peak_vram {peak_gb:.2f} GiB"
        )

        next_step = step + 1
        should_crash = simulate_crash_after == next_step
        should_save = checkpoint_dir is not None and (
            next_step % save_every == 0 or next_step == steps or should_crash
        )
        if should_save:
            checkpoint_path = save_checkpoint(
                checkpoint_dir,
                next_step=next_step,
                model=model,
                optimizer=optim,
                metrics=metrics,
                run_config=run_config,
            )
            print(f"checkpoint saved: {checkpoint_path}")
        if should_crash:
            raise SimulatedCrash(
                f"intentional crash after completed step {next_step}; checkpoint is durable"
            )

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal single-GPU GRPO smoke trainer")
    parser.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--prompts-per-step", type=int, default=1)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-operand", type=int, default=999_999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--logprob-backend", choices=("naive", "selected"), default="naive",
        help="selected avoids full vocabulary logits with the validated Triton autograd path",
    )
    parser.add_argument("--logprob-chunk-rows", type=int, default=8)
    parser.add_argument("--reward-mode", choices=REWARD_MODES, default="exact")
    parser.add_argument("--num-generations", type=int, default=GEN.num_generations)
    parser.add_argument(
        "--hf-recompute-behavior-logprobs", action="store_true",
        help="for synchronous HF, detach current recomputation as exact on-policy old logprobs",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization", type=float,
        help="override the colocated vLLM memory fraction (the local 1.5B probe uses 0.45)",
    )
    parser.add_argument("--vllm-max-num-batched-tokens", type=int)
    parser.add_argument("--vllm-max-num-seqs", type=int)
    parser.add_argument(
        "--curriculum", action="store_true",
        help="use the deterministic 4/5/6-digit multi-prompt arithmetic curriculum",
    )
    parser.add_argument("--curriculum-stage-steps", type=int, default=8)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument(
        "--simulate-crash-after", type=int,
        help="after this many completed steps, save durably and exit with status 75",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="use a fixed hard prompt verified to produce mixed rewards on the local checkpoint",
    )
    parser.add_argument("--smoke-a", type=int, default=847_293)
    parser.add_argument("--smoke-b", type=int, default=581_947)
    args = parser.parse_args()
    curriculum = (
        ArithmeticCurriculum(seed=args.seed, steps_per_stage=args.curriculum_stage_steps)
        if args.curriculum else None
    )
    vllm_overrides = {}
    if args.vllm_gpu_memory_utilization is not None:
        vllm_overrides["gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
    if args.vllm_max_num_batched_tokens is not None:
        vllm_overrides["max_num_batched_tokens"] = args.vllm_max_num_batched_tokens
    if args.vllm_max_num_seqs is not None:
        vllm_overrides["max_num_seqs"] = args.vllm_max_num_seqs
    vllm_config = replace(VLLM, **vllm_overrides)
    generation_config = replace(GEN, num_generations=args.num_generations)
    try:
        train(
            steps=args.steps,
            prompts_per_step=args.prompts_per_step,
            model_name=args.model,
            device=args.device,
            lr=args.lr,
            rollout_backend=args.backend,
            max_operand=args.max_operand,
            smoke=args.smoke,
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
            resume=args.resume,
            save_every=args.save_every,
            simulate_crash_after=args.simulate_crash_after,
            curriculum=curriculum,
            logprob_backend=args.logprob_backend,
            logprob_chunk_rows=args.logprob_chunk_rows,
            smoke_operands=(args.smoke_a, args.smoke_b),
            vllm_config=vllm_config,
            hf_recompute_behavior_logprobs=args.hf_recompute_behavior_logprobs,
            reward_mode=args.reward_mode,
            generation_config=generation_config,
        )
    except SimulatedCrash as exc:
        print(f"SIMULATED_CRASH: {exc}")
        raise SystemExit(75) from None
