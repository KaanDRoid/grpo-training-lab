# Fault-tolerant checkpoint and exact resume

The trainer publishes checkpoints in two phases:

1. write and `fsync` an immutable `step_XXXXXXXX.pt` temporary file, then atomically rename it;
2. compute its SHA-256 and atomically replace `latest.json`.

A crash before phase 2 leaves the previous manifest valid. Resume never guesses from directory
ordering: it follows the manifest, verifies the digest, checks the format version and immutable run
configuration, and only then restores state.

Each checkpoint contains:

- the next step number;
- trainable parameters only (the frozen base model reloads from its original checkpoint);
- the complete optimizer state;
- Python, PyTorch CPU, and every CUDA device RNG state;
- per-step metrics and rollout token fingerprints;
- the immutable model, LoRA, generation, GRPO, seed, and backend configuration.

Changing a semantic setting such as the learning rate, group size, model, backend, or seed causes
resume to fail explicitly. Modern PyTorch's `weights_only=False` is used deliberately because this
is a local trusted artifact containing optimizer and Python RNG objects.

## Simulated interruption

```powershell
# Intentionally exits after publishing step 1.
.\.venv\Scripts\python.exe grpo_trainer.py --backend hf --steps 2 --smoke --seed 17 `
  --checkpoint-dir .resume_demo\run --simulate-crash-after 1

# A fresh process resumes at step 1 and runs only the remaining work.
.\.venv\Scripts\python.exe grpo_trainer.py --backend hf --steps 2 --smoke --seed 17 `
  --checkpoint-dir .resume_demo\run --resume
```

`--steps` is the total target, not “additional steps.” `--save-every N` controls regular saves;
the final step and a simulated-crash boundary are always saved.

## Verified results

The real Qwen2.5-0.5B LoRA test compared a two-step uninterrupted run with a one-step crash plus a
fresh-process resume. `compare_checkpoints.py` reported an exact match for:

- every trainable model tensor;
- every AdamW state tensor and scalar;
- Python and PyTorch CPU/CUDA RNG states;
- rollout token SHA-256 values;
- rewards, losses, gradient norms, and logprob-parity metrics.

Allocator peak-memory metrics are intentionally excluded from equality because a fresh process may
have a different allocator high-water mark.

The same crash/resume flow was also run through WSL/vLLM. Step 0 produced a non-zero update
(`grad_norm=39.51`). On restart, the restored adapter was loaded into a newly spawned vLLM worker
before generation; the next rollout/recomputed-logprob maximum difference was `0.0032`.

Three CPU regression tests independently cover exact interrupted continuation, checksum
corruption, and changed-configuration rejection.
