"""Load the local model in vLLM and verify the RolloutBatch contract."""

from lora_fp16_config import GEN, MODEL, VLLM
from rollout_client import VLLMRolloutClient


def main():
    client = VLLMRolloutClient(MODEL.small_model, VLLM, GEN)
    batch = client.generate(["2 + 2 ="])

    expected_rows = GEN.num_generations
    assert batch.completion_ids.shape[0] == expected_rows
    assert batch.rollout_logprobs.shape == batch.completion_ids.shape
    assert batch.attention_mask.shape == batch.completion_ids.shape
    assert batch.attention_mask.sum().item() > 0
    assert batch.rollout_logprobs[batch.attention_mask.bool()].isfinite().all()

    print("completions:", [repr(text) for text in batch.completions])
    print("completion_ids:", tuple(batch.completion_ids.shape))
    print("rollout_logprobs:", tuple(batch.rollout_logprobs.shape))
    print("completion_tokens:", int(batch.attention_mask.sum().item()))
    print("vLLM rollout smoke test: PASS")


if __name__ == "__main__":
    main()
