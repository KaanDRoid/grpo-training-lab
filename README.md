# GRPO Training Lab

A single-GPU GRPO trainer with LoRA, Hugging Face and vLLM rollouts, resumable checkpoints,
and memory-efficient selected-token log probabilities in Triton. Tested with Qwen2.5-0.5B-Instruct
and Qwen2.5-1.5B-Instruct on an 8 GB RTX 5060 Laptop GPU.

The project focuses on numerical correctness and reproducible experiments: token alignment,
rollout-policy consistency, gradient comparisons, and recovery after interrupted training.

## Results

Measurements below use the RTX 5060. Memory figures are incremental CUDA allocations for the
specified operation, not total GPU memory use.

| Operation | PyTorch reference | Selected-logprob backend | Tradeoff |
|---|---:|---:|---|
| Forward, 151,936-token vocabulary | 186.38 MiB | 4.64 MiB | 2.3× slower; max absolute error 9.54e-7 |
| Backward, frozen output weight | 260.69 MiB | 7.14 MiB | 3× slower; fp16 hidden-gradient max error 0.001953 |
| Qwen2.5-0.5B recompute + backward | 1381.86 MiB | 88.71 MiB | 1.27× slower; gradient cosine 0.9999956 |
| Qwen2.5-1.5B recompute + backward | 1410.81 MiB | 170.00 MiB | 1.22× slower; gradient cosine 0.9997745 |

The [kernel report](docs/selected_logprob.md) covers benchmark shapes, reference comparisons,
and the memory/latency tradeoff. Raw measurements are in [benchmarks/](benchmarks/).

Training results include both improvements and negative findings:

- **0.5B curriculum:** held-out accuracy increased from 31/96 to 59/96 in one 24-step run.
  Repeated execution reproduced the training CSV and adapter hashes. This is a small, single-seed
  result. [Experiment details](docs/curriculum.md)
- **1.5B, three seeds:** mean held-out change was -1.74 percentage points, with a run-level 95%
  interval of [-7.12, +3.65]. This experiment did not establish an improvement.
  [Multi-seed results](docs/model_1p5b_multiseed.md)
- **Reward ablation:** relative-distance rewards and eight-generation groups produced more
  training signal, but neither exceeded four-generation exact-match reward on the paired
  held-out evaluation. [Ablation results](docs/reward_signal_ablation.md)

## Implementation

| Component | Files |
|---|---|
| Masked clipped objective, optional k3 KL, group-relative advantages | [grpo_loss.py](grpo_loss.py) |
| Completion-token alignment and logprob backends | [compute_logprobs.py](compute_logprobs.py) |
| Chunked linear projection, Triton reduction, custom backward | [kernels/selected_logprob.py](kernels/selected_logprob.py) |
| Shared-model Hugging Face rollouts | [hf_rollout_client.py](hf_rollout_client.py) |
| vLLM worker, LoRA reload, prefix-cache invalidation | [rollout_client.py](rollout_client.py) |
| Training loop and configuration | [grpo_trainer.py](grpo_trainer.py), [lora_fp16_config.py](lora_fp16_config.py) |
| Atomic checkpoints for LoRA, optimizer, RNG, and metrics | [checkpointing.py](checkpointing.py) |
| Deterministic arithmetic curriculum and held-out evaluation | [curriculum_experiment.py](curriculum_experiment.py) |

The selected-logprob operator keeps GEMM in PyTorch/cuBLAS and bounds live logits to
`[chunk_rows, vocabulary_size]`. Its backward recomputes these chunks using the derivative
`one_hot - softmax`. The design follows existing fused linear-loss approaches, including
verl, Liger, and Cut Cross-Entropy; it is a reproduction, with no claim of a new kernel method.

## Setup

The tested environments use Python 3.12, PyTorch 2.11.0 with CUDA 12.8 on Windows, and vLLM
0.26.0 with CUDA 13.0 under Ubuntu 24.04 / WSL 2. GPU tests require CUDA and Triton.
Model weights and exported adapters are excluded from the repository. Configuration uses a local
model directory when available and otherwise falls back to the official Hugging Face model ID.

### Windows / Hugging Face

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
.\.venv\Scripts\python.exe -m pytest -q test_grpo_loss.py tests
.\.venv\Scripts\python.exe grpo_trainer.py --backend hf --steps 1 --smoke
```

Use `--logprob-backend selected --logprob-chunk-rows 8` to enable the custom operator. The default
`naive` backend provides the PyTorch reference. The selected backend currently supports Qwen2.

### Linux / WSL 2 / vLLM

Run from the repository directory inside Linux or WSL:

```bash
python3 -m venv "$HOME/.venvs/grpo-lab"
source "$HOME/.venvs/grpo-lab/bin/activate"
python -m pip install -r requirements-wsl.txt
python smoke_vllm.py
python grpo_trainer.py --backend vllm --steps 3 --smoke
```

The vLLM worker loads updated LoRA weights after each optimizer step and clears its prefix cache.
The trainer compares rollout logprobs against recomputed values on every step. The validated 1.5B
configuration needs different memory and scheduler limits; see the [capacity report](docs/model_1p5b_probe.md).

## Experiments

Run the 0.5B curriculum with disjoint training and held-out prompts:

```bash
python curriculum_experiment.py --steps 24 --prompts-per-step 2 --stage-steps 12 --operand-digits 5,6 --seed 29 --heldout-per-stage 12 --output-dir results
```

Exercise checkpoint recovery:

```bash
python grpo_trainer.py --backend hf --steps 2 --smoke --seed 17 --checkpoint-dir .resume_demo/run --simulate-crash-after 1
python grpo_trainer.py --backend hf --steps 2 --smoke --seed 17 --checkpoint-dir .resume_demo/run --resume
```

The first command exits after the simulated interruption. `--steps` is the total target, so the
second command resumes from the saved step. Model, optimizer, RNG state, and numerical metrics
matched an uninterrupted run exactly in the recorded comparison.
[Recovery design and results](docs/checkpoint_resume.md)

## Numerical checks and scope

The local suite passed 80 tests, covering masking, causal token alignment, reward grouping,
checkpoint restore, fp64 gradcheck, and fp16/fp32 comparisons against PyTorch. Padding does not
contribute to the objective. KL regularization requires explicit reference-policy logprobs.

When training consumes rollout scores, the logprob parity guards are 0.35 maximum and 0.03 mean
absolute difference. The optional synchronous HF mode uses detached current-policy recomputation
as behavior logprobs; raw generation scores remain diagnostic, with a 1.0 maximum and 0.03 mean
guard. [Numerical details](docs/reward_signal_ablation.md#numerical-note)

Validation is limited to one GPU, small arithmetic tasks, and the recorded model configurations.
Multi-node training and broad task generalization have not been demonstrated. Benchmark speed
and memory results depend on tensor shapes, dtype, and hardware.
