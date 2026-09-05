# Qwen2.5-1.5B capacity probe on the 8 GB RTX 5060

This probe answers a capacity question, not a learning-curve question: can the same fp16 LoRA
trainer, selected-logprob backward, and colocated vLLM synchronization path run Qwen2.5-1.5B on
the local 8 GB laptop GPU?

## Model selection

The base `Qwen/Qwen2.5-1.5B` model was tested first. Its 2.875 GiB checkpoint loaded and the
HF selected-backend step peaked at 3.24 GiB, proving basic capacity. However, a deterministic
frontier scan returned zero exact rewards across 96 rollouts spanning 3/4/5/6-digit addition. A
binary-reward GRPO run cannot learn when every group reward is zero.

The operational target is therefore `Qwen/Qwen2.5-1.5B-Instruct`, matching the already validated
0.5B-Instruct pipeline. Its safetensors file is 2.875 GiB with SHA-256
`DD924A11B4C220F385B51FFA522DAEA7C9F3D850E31B162BB5661DF483C6D3EE`. Both downloaded model
directories are local/ignored artifacts; the config falls back to the official Hugging Face ID
when the local Instruct directory is absent.

The Instruct frontier scan found mixed exact rewards, including `49762 + 10659` at the chosen seed.
That prompt is used below so the capacity check exercises a real backward and optimizer update.

## Shared-model HF result

Command:

```powershell
.\.venv\Scripts\python.exe grpo_trainer.py --backend hf --steps 1 --smoke --smoke-a 49762 --smoke-b 10659 --seed 41 --model .\Qwen2.5-1.5B-Instruct --logprob-backend selected
```

| Backend | Reward | Loss | Gradient norm | Parity max/mean | Peak allocation |
|---|---:|---:|---:|---:|---:|
| Naive | 0.25 | -0.0026 | 27.5510 | 0.1634 / 0.0283 | 3.60 GiB |
| Selected | 0.25 | -0.0026 | 27.5530 | 0.1634 / 0.0283 | 3.24 GiB |

The two-prompt current-policy benchmark gives a more isolated memory measurement:

| Backend | Recompute + backward | Incremental peak |
|---|---:|---:|
| Naive | 439.77 ms | 1410.81 MiB |
| Selected | 537.41 ms | 170.00 MiB |

Selected reduces the incremental allocation **8.30x (87.9%)** for a **1.22x** latency cost.
Logprob max/mean differences are `1.91e-6 / 1.27e-7`. The sampled benchmark tokens are extremely
high-confidence, so their small gradients amplify relative error: gradient cosine is `0.9997745`,
relative L2 error 2.22%, and maximum absolute error only 0.00286. The actual mixed-reward smoke
gradient norms above differ by less than 0.01%.

The machine-readable benchmark is `benchmarks/trainer_logprob_backend_1p5b_rtx5060.json`.

## Colocated WSL/vLLM result

The 0.5B vLLM defaults are intentionally not reused blindly. At 1.5B, the default 8,192-token
chunked-prefill profile and a 0.45 global memory fraction reported -2.95 GiB available for KV.
Limiting the real workload to 512 batched tokens and four sequences reduced profiling by 0.50 GiB,
but 0.45 still counted the trainer allocation inside the global budget and remained negative.

The validated 8 GB profile is:

- `gpu_memory_utilization=0.80` (a global cap, not 80% exclusively owned by vLLM);
- `max_num_batched_tokens=512`;
- `max_num_seqs=4`;
- `max_model_len=256`, eager mode, fp16;
- selected-logprob trainer chunks of eight rows.

vLLM loaded 3.01 GiB of weights, profiled only 0.04 GiB of peak activation, accounted for the
2.98 GiB trainer process, and created a 0.34 GiB / 12,672-token KV cache. A two-step run then
completed rollout, selected backward, optimizer updates, LoRA hot-reload, and prefix-cache reset:

| Step | Reward | Loss | Gradient norm | Parity max/mean | Trainer peak |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.25 | -0.0055 | 27.8384 | 0.0686 / 0.0080 | 3.95 GiB |
| 1 | 0.75 | +0.0079 | 27.4008 | 0.0883 / 0.0100 | 4.16 GiB |

The second rollout is the important synchronization check: it used the non-zero adapter installed
after step 0 and still matched trainer recomputation inside the parity guard.

```powershell
wsl -d Ubuntu-24.04 -- bash -lc 'cd /mnt/c/path/to/project && "$HOME/.venvs/grpo-lab/bin/python" grpo_trainer.py --backend vllm --steps 2 --smoke --smoke-a 49762 --smoke-b 10659 --seed 41 --model Qwen/Qwen2.5-1.5B-Instruct --logprob-backend selected --vllm-gpu-memory-utilization 0.80 --vllm-max-num-batched-tokens 512 --vllm-max-num-seqs 4'
```

Conclusion: 1.5B fp16 LoRA fits both execution paths on this 8 GB GPU, but colocated vLLM requires
workload-sized scheduler profiling and a global utilization cap that includes the trainer process.
