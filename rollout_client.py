"""vLLM rollout generation and trainer-to-worker LoRA synchronization.

Validated with vLLM 0.26.0. Generation returns padded completion tokens and each
sampled token's log probability. Weight synchronization exports the PEFT adapter,
reloads it with LoRARequest(load_inplace=True), and resets the prefix cache.

API references: vllm/entrypoints/llm.py, vllm/lora/request.py, vllm/logprobs.py.
"""

import math
import os
import platform
from pathlib import Path
import sys
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class RolloutBatch:
    prompts: list[str]
    completions: list[str]
    completion_ids: torch.Tensor       # [B, T] token ids, right-padded with pad_id
    rollout_logprobs: torch.Tensor     # [B, T] logprob of each sampled token *under the rollout policy*
    attention_mask: torch.Tensor       # [B, T] 1 on real completion tokens, 0 on pad (== completion mask)


class VLLMRolloutClient:
    """
    Colocate-mode client: vLLM engine and trainer share one GPU (memory profile in
    lora_fp16_config.VLLM). Not server-mode -- TRL's server mode wants a separate CUDA
    device, which a single free-tier GPU doesn't have.

    B (batch) is prompt-major: row = prompt_idx * num_generations + sample_idx, so each
    prompt's `num_generations` samples are CONTIGUOUS -- this is exactly what
    grpo_loss.group_relative_advantage(...).view(-1, group_size) expects (group_size ==
    num_generations). Do not reorder.
    """

    def __init__(self, model_name: str, vllm_config, generation_config):
        self.model_name = model_name
        self.vllm_config = vllm_config
        self.gen_config = generation_config

        # tokenizer only needed for the pad id (and for the standalone load-test).
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        pad = self.tokenizer.pad_token_id
        self.pad_token_id = pad if pad is not None else self.tokenizer.eos_token_id

        self.engine = self._init_engine()
        self._adapter_dir = Path(__file__).resolve().parent / ".vllm_lora" / "trainer_policy"
        self._active_lora_request = None
        self._last_probe: Optional[str] = None  # for verify_weight_sync no-op detection

    def _init_engine(self):
        # vLLM/FlashInfer launch build helpers (notably ``ninja``) by executable name.
        # Calling this module through ``/path/to/venv/bin/python`` does not necessarily
        # activate the environment, so make its scripts visible to spawned workers.
        venv_bin = Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin")
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        if venv_bin.is_dir() and str(venv_bin) not in path_entries:
            os.environ["PATH"] = str(venv_bin) + os.pathsep + os.environ.get("PATH", "")

        # vLLM's V2 runner uses Unified Virtual Addressing. WSL2 supports pinned memory/UVA,
        # but vLLM keeps it disabled by default unless this opt-in is present.
        if "microsoft" in platform.release().lower():
            os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
            # The vLLM 0.26 wheel can resolve a CUDA 13.3 compiler alongside 13.0
            # runtime headers. FlashInfer's JIT rejects that mixed toolkit during
            # sampling warm-up. Its sampler is optional; vLLM's native PyTorch path
            # is correct and plenty fast for our four short GRPO generations.
            os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
            # vLLM 0.26's PyTorch/CUDA wheels carry a complete CUDA 13 toolchain inside the
            # environment, but FlashInfer's JIT only probes CUDA_HOME or /usr/local/cuda.
            # Point it at the bundled toolchain instead of requiring a duplicate system install.
            python_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
            bundled_cuda = (
                Path(sys.prefix) / "lib" / python_tag / "site-packages" / "nvidia" / "cu13"
            )
            if "CUDA_HOME" not in os.environ and (bundled_cuda / "bin" / "nvcc").is_file():
                os.environ["CUDA_HOME"] = str(bundled_cuda)
                os.environ["PATH"] = (
                    str(bundled_cuda / "bin") + os.pathsep + os.environ.get("PATH", "")
                )
        from vllm import LLM
        scheduler_limits = {}
        if self.vllm_config.max_num_batched_tokens is not None:
            scheduler_limits["max_num_batched_tokens"] = self.vllm_config.max_num_batched_tokens
        if self.vllm_config.max_num_seqs is not None:
            scheduler_limits["max_num_seqs"] = self.vllm_config.max_num_seqs
        return LLM(
            model=self.model_name,
            dtype=self.vllm_config.dtype,
            gpu_memory_utilization=self.vllm_config.gpu_memory_utilization,
            max_model_len=self.vllm_config.max_model_len,
            enforce_eager=self.vllm_config.enforce_eager,
            enable_lora=self.vllm_config.enable_lora,
            max_loras=self.vllm_config.max_loras,
            max_lora_rank=self.vllm_config.max_lora_rank,
            **scheduler_limits,
        )

    def generate(self, prompts: list[str], seed: int | None = None) -> RolloutBatch:
        """Sample `num_generations` completions per prompt, return with per-token rollout logprobs."""
        from vllm import SamplingParams
        n = self.gen_config.num_generations
        sampling_params = SamplingParams(
            n=n,
            max_tokens=self.gen_config.max_new_tokens,
            temperature=self.gen_config.temperature,
            top_p=getattr(self.gen_config, "top_p", 1.0),
            seed=seed,
            logprobs=1,  # top-1 + the sampled token; we index by the SAMPLED id below.
                         # (logprobs=0 also returns exactly the sampled token; 1 is the
                         # conservative, always-non-empty choice.)
        )
        outputs = self.engine.generate(
            prompts,
            sampling_params,
            lora_request=self._active_lora_request,
        )  # list[RequestOutput], input order

        ids_rows: list[list[int]] = []
        lp_rows: list[list[float]] = []
        flat_prompts: list[str] = []
        completions: list[str] = []

        for req_idx, req in enumerate(outputs):          # one RequestOutput per prompt
            for comp in req.outputs:                     # len == n, in order
                token_ids = list(comp.token_ids)         # generated tokens only
                if comp.logprobs is None:
                    raise ValueError("logprobs is None -> pass SamplingParams(logprobs>=0)")
                # comp.logprobs[i] -> dict[int, Logprob] for generated position i.
                # token_ids[i] is ALWAYS a key (vLLM always includes the sampled token).
                row_lp = []
                for i, t in enumerate(token_ids):
                    v = comp.logprobs[i][t].logprob
                    row_lp.append(0.0 if math.isnan(v) else v)
                ids_rows.append(token_ids)
                lp_rows.append(row_lp)
                # req.prompt can be None if you passed token ids; fall back to input string.
                flat_prompts.append(req.prompt if req.prompt is not None else prompts[req_idx])
                completions.append(comp.text)

        B = len(ids_rows)
        assert B == len(prompts) * n, (
            f"expected prompt-major B={len(prompts)*n}, got {B} -- grouping for GRPO advantage broke"
        )
        T = max((len(r) for r in ids_rows), default=1) or 1

        completion_ids = torch.full((B, T), self.pad_token_id, dtype=torch.long)
        rollout_logprobs = torch.zeros((B, T), dtype=torch.float32)
        attention_mask = torch.zeros((B, T), dtype=torch.long)
        for r, (ids, lps) in enumerate(zip(ids_rows, lp_rows)):
            L = len(ids)
            if L == 0:
                continue
            completion_ids[r, :L] = torch.tensor(ids, dtype=torch.long)
            rollout_logprobs[r, :L] = torch.tensor(lps, dtype=torch.float32)
            attention_mask[r, :L] = 1

        return RolloutBatch(
            prompts=flat_prompts,
            completions=completions,
            completion_ids=completion_ids,
            rollout_logprobs=rollout_logprobs,
            attention_mask=attention_mask,
        )

    @torch.no_grad()
    def sync_weights(self, model, probe_prompt: Optional[str] = None):
        """
        Push the trainer's CURRENT LoRA policy into the colocated vLLM worker.

        WSL forces vLLM's engine into a spawned process, so returning its CUDA model through
        apply_model is invalid. PEFT's adapter checkpoint is the intended small data plane:
        write it, hot-reload the same adapter id in place, and clear cached prefixes.
        """
        is_peft = hasattr(model, "merge_adapter") and hasattr(model, "peft_config")
        if not is_peft:
            raise NotImplementedError(
                "the WSL/vLLM backend currently syncs PEFT LoRA adapters; "
                "full-parameter models require vLLM's weight-transfer data plane"
            )

        self._adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(self._adapter_dir, safe_serialization=True)

        from vllm.lora.request import LoRARequest

        reload_request = LoRARequest(
            lora_name="trainer_policy",
            lora_int_id=1,
            lora_path=str(self._adapter_dir),
            load_inplace=True,
        )
        if not self.engine.llm_engine.add_lora(reload_request):
            raise RuntimeError("vLLM rejected the trainer LoRA adapter update")

        # Future generations use the already-loaded adapter without re-reading disk.
        self._active_lora_request = LoRARequest(
            lora_name="trainer_policy",
            lora_int_id=1,
            lora_path=str(self._adapter_dir),
        )

        # invalidate cached KV/prefix so the next generate() uses the NEW policy
        self.engine.reset_prefix_cache()

        if probe_prompt is not None:
            return self.verify_weight_sync(probe_prompt)

    def verify_weight_sync(self, probe_prompt: str, expected_completion_prefix: Optional[str] = None):
        """
        Post-sync correctness check (the "generations-match correctness check" in the arch
        diagram). Greedy-decode a fixed probe prompt right after sync_weights() and detect a
        SILENT NO-OP sync by warning if the probe output is byte-identical to the previous step's
        (a real, previously-reported colocate failure mode). Optionally also assert a known prefix.
        """
        from vllm import SamplingParams
        greedy = SamplingParams(n=1, max_tokens=32, temperature=0.0)
        out = self.engine.generate(
            [probe_prompt], greedy, lora_request=self._active_lora_request
        )
        completion = out[0].outputs[0].text

        if self._last_probe is not None and completion == self._last_probe:
            # not fatal early on (weights may barely move for one step) but log it loudly:
            # if it NEVER changes across steps, the sync is silently no-op'ing.
            print("[verify_weight_sync] WARNING: probe output unchanged since last sync "
                  "-- if this persists, weight sync is a no-op.")
        self._last_probe = completion

        if expected_completion_prefix is not None:
            assert completion.startswith(expected_completion_prefix), (
                f"weight sync did not take effect: got {completion!r}"
            )
        return completion
