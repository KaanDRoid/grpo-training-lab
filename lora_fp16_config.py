"""FP16 LoRA and rollout defaults for an 8 GB RTX 5060 Laptop GPU.

Full fine-tuning with fp16 weights/gradients, fp32 master weights, and fp32 Adam
moments requires roughly 16 bytes per parameter before activations. At 1.5B
parameters this is approximately 24 GB. LoRA limits optimizer state to adapter
parameters while retaining frozen fp16 base weights.

vLLM memory limits must account for the colocated trainer and KV cache.
See docs/model_1p5b_probe.md for the measured 1.5B configuration.
"""

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_SMALL_MODEL = PROJECT_ROOT / "Qwen2.5-0.5B-Instruct"
LOCAL_TARGET_MODEL = PROJECT_ROOT / "Qwen2.5-1.5B-Instruct"


@dataclass
class ModelConfig:
    # Prefer the checkpoint already present in this repository. This keeps the first smoke run
    # offline and avoids accidentally testing a different (base rather than instruct) checkpoint.
    small_model: str = (
        str(LOCAL_SMALL_MODEL)
        if LOCAL_SMALL_MODEL.is_dir()
        else "Qwen/Qwen2.5-0.5B-Instruct"
    )
    # default model once the loop is validated
    model: str = (
        str(LOCAL_TARGET_MODEL)
        if LOCAL_TARGET_MODEL.is_dir()
        else "Qwen/Qwen2.5-1.5B-Instruct"
    )
    # Keep the first validated path FP16 for parity with the later T4/cloud run. The local RTX
    # 5060 (sm_120) also passed a native BF16 matmul check, so BF16 is a later ablation here.
    dtype: str = "float16"
    gradient_checkpointing: bool = True


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    dropout: float = 0.0


@dataclass
class VLLMConfig:
    dtype: str = "float16"
    # default gpu_memory_utilization=0.9 OOMs instantly when colocated with the trainer on one GPU.
    # RTX 5060 Laptop profile: 8 GB total VRAM. The original T4 profile used 0.30/1024,
    # which is too aggressive when a trainer copy shares this smaller device.
    gpu_memory_utilization: float = 0.24
    max_model_len: int = 256
    # Leave scheduler profiling at vLLM defaults for 0.5B. The 1.5B/8GB probe overrides these to
    # 512/4 so warm-up reflects the real four-generation workload instead of 8,192 prefill tokens.
    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = None
    enforce_eager: bool = True  # skip CUDA-graph capture; saves memory, costs some throughput
    enable_lora: bool = True
    max_loras: int = 1
    max_lora_rank: int = 16


@dataclass
class GenerationConfig:
    # NOTE: prompt_len + max_new_tokens must be <= VLLMConfig.max_model_len (1024). With
    # max_new_tokens=512 that leaves only 512 tokens of prompt headroom -- assert/clamp before
    # generate(), or raise max_model_len / lower max_new_tokens for longer prompts.
    max_new_tokens: int = 32
    num_generations: int = 4  # GRPO group size (== rollout_client B grouping == group_size)
    per_device_prompt_bs: int = 1
    temperature: float = 1.0
    top_p: float = 1.0


@dataclass
class GRPOConfig:
    clip_eps: float = 0.2
    kl_coef: float = 0.0            # 0.0 = Dr.GRPO-style: drop the reference model entirely, saves ~3GB
    std_normalize: bool = True     # advantage std-normalization (was misnamed length_normalize).
                                   # True = original GRPO ((r-mean)/std); False = Dr.GRPO ablation
                                   # (drop std, keep mean-centering). This is the DIFFICULTY/std fix,
                                   # NOT the length fix -- length fix lives in `aggregation` below.
    aggregation: str = "seq_mean"  # grpo_loss reduction: "seq_mean" (original, length-normalized
                                   # per sequence) or "token_mean" (DAPO/Dr.GRPO length fix).


# ---- OOM fallback ladder, in order of first resort ----
# 1. reduce GenerationConfig.num_generations and/or max_new_tokens
# 2. reduce VLLMConfig.max_model_len further (shrinks KV cache pool)
# 3. switch to QLoRA (4-bit base + LoRA adapters)
# 4. drop ModelConfig.model to "Qwen/Qwen2.5-0.5B" permanently

MODEL = ModelConfig()
LORA = LoRAConfig()
VLLM = VLLMConfig()
GEN = GenerationConfig()
GRPO = GRPOConfig()
