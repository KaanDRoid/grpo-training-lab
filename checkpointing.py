"""Small, explicit, fault-tolerant checkpoints for the single-GPU GRPO trainer."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import torch


CHECKPOINT_VERSION = 1
MANIFEST_NAME = "latest.json"


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is missing, corrupt, or incompatible with the run."""


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise CheckpointError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        saved = state["torch_cuda"]
        if len(saved) != torch.cuda.device_count():
            raise CheckpointError(
                f"CUDA device count changed: checkpoint has {len(saved)}, "
                f"runtime has {torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all(saved)


def trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy only trainable parameters to CPU; the frozen base reloads from ``model_name``."""
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


@torch.no_grad()
def load_trainable_state_dict(
    model: torch.nn.Module, saved: dict[str, torch.Tensor]
) -> None:
    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    missing = sorted(set(current) - set(saved))
    unexpected = sorted(set(saved) - set(current))
    if missing or unexpected:
        raise CheckpointError(
            f"trainable parameter mismatch; missing={missing}, unexpected={unexpected}"
        )
    for name, parameter in current.items():
        value = saved[name]
        if tuple(value.shape) != tuple(parameter.shape):
            raise CheckpointError(
                f"shape mismatch for {name}: checkpoint {tuple(value.shape)} != "
                f"model {tuple(parameter.shape)}"
            )
        parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_checkpoint(
    checkpoint_dir: str | Path,
    *,
    next_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: list[dict[str, Any]],
    run_config: dict[str, Any],
) -> Path:
    """Atomically publish an immutable step checkpoint, then atomically move ``latest``."""
    if next_step < 0:
        raise ValueError("next_step must be non-negative")
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"step_{next_step:08d}.pt"
    final_path = directory / filename
    temporary = directory / f".{filename}.{os.getpid()}.tmp"

    payload = {
        "version": CHECKPOINT_VERSION,
        "next_step": next_step,
        "run_config": run_config,
        "model_trainable": trainable_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "rng": capture_rng_state(),
        "metrics": metrics,
    }
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, final_path)

    manifest = {
        "version": CHECKPOINT_VERSION,
        "checkpoint": filename,
        "next_step": next_step,
        "sha256": _sha256(final_path),
    }
    _atomic_json(directory / MANIFEST_NAME, manifest)
    return final_path


def read_checkpoint(checkpoint_dir: str | Path) -> dict[str, Any]:
    directory = Path(checkpoint_dir).resolve()
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        raise CheckpointError(f"checkpoint manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"invalid checkpoint manifest: {manifest_path}") from exc

    if manifest.get("version") != CHECKPOINT_VERSION:
        raise CheckpointError(f"unsupported checkpoint version: {manifest.get('version')}")
    filename = manifest.get("checkpoint")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise CheckpointError("manifest checkpoint path must be a plain filename")
    checkpoint_path = directory / filename
    if not checkpoint_path.is_file():
        raise CheckpointError(f"checkpoint file not found: {checkpoint_path}")
    actual_digest = _sha256(checkpoint_path)
    if actual_digest != manifest.get("sha256"):
        raise CheckpointError(
            f"checkpoint checksum mismatch: expected {manifest.get('sha256')}, "
            f"got {actual_digest}"
        )

    try:
        # Checkpoints are local, trusted training artifacts. Explicit False is required by
        # modern PyTorch because optimizer and Python RNG states are not weights-only objects.
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CheckpointError(f"could not load checkpoint: {checkpoint_path}") from exc
    if payload.get("version") != CHECKPOINT_VERSION:
        raise CheckpointError(f"payload version mismatch in {checkpoint_path}")
    if payload.get("next_step") != manifest.get("next_step"):
        raise CheckpointError("manifest and payload disagree about next_step")
    return payload


def load_checkpoint(
    checkpoint_dir: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_run_config: dict[str, Any],
) -> dict[str, Any]:
    payload = read_checkpoint(checkpoint_dir)
    if payload.get("run_config") != expected_run_config:
        raise CheckpointError(
            "run configuration differs from the checkpoint; refusing an ambiguous resume\n"
            f"checkpoint={payload.get('run_config')}\ncurrent={expected_run_config}"
        )
    load_trainable_state_dict(model, payload["model_trainable"])
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng_state(payload["rng"])
    return payload
