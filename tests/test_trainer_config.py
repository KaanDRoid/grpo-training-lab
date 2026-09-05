from dataclasses import replace

import pytest

from grpo_trainer import make_run_config, train
from lora_fp16_config import GEN, VLLM


def _run_config(vllm_config):
    return make_run_config(
        model_name="tiny",
        rollout_backend="vllm",
        prompts_per_step=1,
        lr=1e-5,
        sync_every=1,
        max_operand=99,
        smoke=True,
        seed=7,
        curriculum=None,
        logprob_backend="selected",
        logprob_chunk_rows=8,
        smoke_operands=(12, 34),
        vllm_config=vllm_config,
        hf_recompute_behavior_logprobs=True,
        reward_mode="relative_distance",
        generation_config=replace(GEN, num_generations=8),
    )


def test_vllm_scheduler_limits_are_part_of_immutable_run_config():
    tuned = replace(
        VLLM,
        gpu_memory_utilization=0.80,
        max_num_batched_tokens=512,
        max_num_seqs=4,
    )
    config = _run_config(tuned)
    assert config["vllm"]["gpu_memory_utilization"] == 0.80
    assert config["vllm"]["max_num_batched_tokens"] == 512
    assert config["vllm"]["max_num_seqs"] == 4
    assert config["smoke_operands"] == [12, 34]
    assert config["logprob_backend"] == "selected"
    assert config["hf_recompute_behavior_logprobs"] is True
    assert config["reward_mode"] == "relative_distance"
    assert config["generation"]["num_generations"] == 8


@pytest.mark.parametrize(
    "vllm_config",
    [
        replace(VLLM, gpu_memory_utilization=0.0),
        replace(VLLM, max_num_batched_tokens=0),
        replace(VLLM, max_num_seqs=0),
    ],
)
def test_invalid_vllm_memory_limits_fail_before_model_loading(vllm_config):
    with pytest.raises(ValueError, match="vLLM"):
        train(steps=0, device="cpu", vllm_config=vllm_config)


def test_invalid_reward_and_group_size_fail_before_model_loading():
    with pytest.raises(ValueError, match="reward mode"):
        train(steps=0, device="cpu", reward_mode="mystery")
    with pytest.raises(ValueError, match="at least two"):
        train(steps=0, device="cpu", generation_config=replace(GEN, num_generations=1))
