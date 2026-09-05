from types import SimpleNamespace

import pytest
import torch

from compute_logprobs import _causal_lm_backbone_and_head, compute_completion_logprobs


class TinyBackbone(torch.nn.Module):
    def __init__(self, vocab=37, hidden=11):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.proj = torch.nn.Linear(hidden, hidden)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden = torch.tanh(self.proj(self.embed(input_ids)))
        return SimpleNamespace(last_hidden_state=hidden)


class TinyQwenLikeLM(torch.nn.Module):
    base_model_prefix = "model"

    def __init__(self, vocab=37, hidden=11, model_type="qwen2"):
        super().__init__()
        self.config = SimpleNamespace(model_type=model_type)
        self.model = TinyBackbone(vocab, hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask=None):
        hidden = self.model(input_ids, attention_mask=attention_mask).last_hidden_state
        return SimpleNamespace(logits=self.lm_head(hidden))


def _batch():
    prompts = [[1, 2, 3], [4, 5], [4, 5]]
    completions = torch.tensor([[6, 7, 0], [8, 9, 10], [11, 0, 0]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 0, 0]])
    return prompts, completions, mask


def test_selected_backend_rejects_unvalidated_architecture():
    model = TinyQwenLikeLM(model_type="not-qwen")
    with pytest.raises(ValueError, match="validated only for Qwen"):
        _causal_lm_backbone_and_head(model)


def test_unknown_backend_is_rejected():
    model = TinyQwenLikeLM()
    prompts, completions, mask = _batch()
    with pytest.raises(ValueError, match="unknown logprob backend"):
        compute_completion_logprobs(
            model, prompts, completions, mask, pad_id=0, backend="mystery"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_selected_backend_matches_naive_outputs_and_all_parameter_gradients():
    torch.manual_seed(2026)
    naive_model = TinyQwenLikeLM().cuda()
    selected_model = TinyQwenLikeLM().cuda()
    selected_model.load_state_dict(naive_model.state_dict())
    prompts, completions, mask = _batch()

    naive = compute_completion_logprobs(
        naive_model, prompts, completions, mask, pad_id=0, backend="naive"
    )
    selected = compute_completion_logprobs(
        selected_model,
        prompts,
        completions,
        mask,
        pad_id=0,
        backend="selected",
        chunk_rows=2,
    )
    torch.testing.assert_close(selected, naive, atol=5e-5, rtol=5e-5)
    assert torch.count_nonzero(selected[~mask.bool()]) == 0

    upstream = torch.randn_like(naive) * mask.cuda()
    (naive * upstream).sum().backward()
    (selected * upstream).sum().backward()
    for (naive_name, naive_parameter), (selected_name, selected_parameter) in zip(
        naive_model.named_parameters(), selected_model.named_parameters()
    ):
        assert naive_name == selected_name
        torch.testing.assert_close(
            selected_parameter.grad,
            naive_parameter.grad,
            atol=3e-4,
            rtol=3e-4,
        )
