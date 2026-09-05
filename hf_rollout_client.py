"""Single-model Hugging Face rollout backend for local development.

The production-shaped path in :mod:`rollout_client` keeps vLLM and the trainer as separate
components and explicitly synchronizes weights. That is the right integration target, but two
model copies are a poor first smoke test on an 8 GB RTX 5060 Laptop GPU and native Windows is not
a supported vLLM platform.

This backend shares the trainable model with generation. It therefore has no weight-sync step,
but preserves the important RolloutBatch contract: prompt-major samples, generated-token ids,
per-token behavior-policy logprobs, and a right-padded completion mask. It lets us validate the
reward -> advantage -> policy-loss -> optimizer loop locally before moving the integration test
to Linux/WSL or a cloud GPU.
"""

from __future__ import annotations

import torch

from rollout_client import RolloutBatch


class HFRolloutClient:
    """Generate rollouts with the same HF/PEFT model that is being trained."""

    def __init__(self, model, tokenizer, generation_config, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.gen_config = generation_config
        self.device = torch.device(device)
        pad = tokenizer.pad_token_id
        self.pad_token_id = pad if pad is not None else tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("tokenizer must define pad_token_id or eos_token_id")

    @torch.inference_mode()
    def generate(self, prompts: list[str], seed: int | None = None) -> RolloutBatch:
        if not prompts:
            raise ValueError("prompts must not be empty")

        # Keep a restart independent of process-global initialization. The trainer also seeds at
        # the step boundary; doing it here makes the backend contract self-contained.
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        n = self.gen_config.num_generations
        was_training = self.model.training
        previous_padding_side = self.tokenizer.padding_side
        self.model.eval()
        self.tokenizer.padding_side = "left"
        try:
            encoded = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
            )
            encoded = {name: value.to(self.device) for name, value in encoded.items()}
            prompt_width = encoded["input_ids"].shape[1]
            do_sample = self.gen_config.temperature > 0
            output = self.model.generate(
                **encoded,
                do_sample=do_sample,
                temperature=self.gen_config.temperature if do_sample else None,
                top_p=self.gen_config.top_p if do_sample else None,
                top_k=0,
                repetition_penalty=1.0,
                typical_p=1.0,
                epsilon_cutoff=0.0,
                eta_cutoff=0.0,
                max_new_tokens=self.gen_config.max_new_tokens,
                num_return_sequences=n,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
            )
        finally:
            self.tokenizer.padding_side = previous_padding_side
            self.model.train(was_training)

        steps = len(output.scores)
        if steps == 0:
            raise RuntimeError("generation returned no token scores")

        completion_ids = output.sequences[:, prompt_width:prompt_width + steps]
        if completion_ids.shape[1] != steps:
            raise RuntimeError(
                f"generated token/score mismatch: {completion_ids.shape[1]} ids vs {steps} scores"
            )

        # Each score is [B,V]. Process one step at a time so we never stack [B,T,V].
        logprob_columns = []
        for step, scores in enumerate(output.scores):
            token = completion_ids[:, step]
            column = torch.log_softmax(scores.float(), dim=-1).gather(
                1, token.unsqueeze(1)
            ).squeeze(1)
            logprob_columns.append(column)
        rollout_logprobs = torch.stack(logprob_columns, dim=1)
        rollout_logprobs = torch.nan_to_num(rollout_logprobs, nan=0.0, neginf=-1e9)

        attention_mask = torch.ones_like(completion_ids, dtype=torch.long)
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            # EOS is a real sampled token. Only tokens *after* its first occurrence are padding.
            for row in range(completion_ids.shape[0]):
                eos_positions = (completion_ids[row] == eos_id).nonzero(as_tuple=False)
                if eos_positions.numel():
                    first_eos = int(eos_positions[0].item())
                    attention_mask[row, first_eos + 1:] = 0

        ids_cpu = completion_ids.detach().cpu()
        mask_cpu = attention_mask.detach().cpu()
        completions = []
        for ids, mask in zip(ids_cpu, mask_cpu):
            length = int(mask.sum().item())
            completions.append(self.tokenizer.decode(ids[:length], skip_special_tokens=True))

        flat_prompts = [prompt for prompt in prompts for _ in range(n)]
        expected_batch = len(flat_prompts)
        if ids_cpu.shape[0] != expected_batch:
            raise RuntimeError(
                f"expected prompt-major batch {expected_batch}, got {ids_cpu.shape[0]}"
            )

        return RolloutBatch(
            prompts=flat_prompts,
            completions=completions,
            completion_ids=ids_cpu,
            rollout_logprobs=rollout_logprobs.detach().cpu(),
            attention_mask=mask_cpu,
        )

    def sync_weights(self, model, probe_prompt=None):
        """No-op: generation and training intentionally share the exact same model object."""
        if model is not self.model:
            raise ValueError("HFRolloutClient can only sync its shared model instance")
        return None
