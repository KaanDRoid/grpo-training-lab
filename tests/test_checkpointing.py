"""Crash/resume correctness tests using a tiny deterministic training loop."""

from pathlib import Path
import random

import pytest
import torch

from checkpointing import CheckpointError, load_checkpoint, read_checkpoint, save_checkpoint


RUN_CONFIG = {"model": "tiny", "lr": 0.03, "seed": 123}


def _new_model_and_optimizer(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(5, 7),
        torch.nn.Tanh(),
        torch.nn.Linear(7, 2),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=RUN_CONFIG["lr"])
    return model, optimizer


def _run_steps(model, optimizer, start, stop, metrics, checkpoint_dir=None):
    for step in range(start, stop):
        # Uses all checkpointed RNG sources. A resume that restores only weights/optimizer will
        # diverge here even though it appears to start successfully.
        scale = 0.5 + random.random()
        inputs = torch.randn(4, 5) * scale
        targets = torch.randn(4, 2)
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        metrics.append({"step": step, "loss": loss.item()})
        if checkpoint_dir is not None:
            save_checkpoint(
                checkpoint_dir,
                next_step=step + 1,
                model=model,
                optimizer=optimizer,
                metrics=metrics,
                run_config=RUN_CONFIG,
            )


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0.0, rtol=0.0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
    else:
        assert left == right


def test_interrupted_run_exactly_matches_uninterrupted_run(tmp_path):
    baseline_model, baseline_optimizer = _new_model_and_optimizer(123)
    baseline_metrics = []
    _run_steps(baseline_model, baseline_optimizer, 0, 5, baseline_metrics)

    interrupted_model, interrupted_optimizer = _new_model_and_optimizer(123)
    interrupted_metrics = []
    checkpoint_dir = tmp_path / "run"
    _run_steps(
        interrupted_model,
        interrupted_optimizer,
        0,
        2,
        interrupted_metrics,
        checkpoint_dir,
    )

    # Simulate a fresh process with unrelated initialization and RNG, then resume.
    resumed_model, resumed_optimizer = _new_model_and_optimizer(999)
    payload = load_checkpoint(
        checkpoint_dir,
        model=resumed_model,
        optimizer=resumed_optimizer,
        expected_run_config=RUN_CONFIG,
    )
    resumed_metrics = payload["metrics"]
    _run_steps(
        resumed_model,
        resumed_optimizer,
        payload["next_step"],
        5,
        resumed_metrics,
    )

    _assert_nested_equal(baseline_model.state_dict(), resumed_model.state_dict())
    _assert_nested_equal(baseline_optimizer.state_dict(), resumed_optimizer.state_dict())
    assert baseline_metrics == resumed_metrics


def test_checksum_detects_corruption(tmp_path):
    model, optimizer = _new_model_and_optimizer(123)
    checkpoint_dir = tmp_path / "corrupt"
    path = save_checkpoint(
        checkpoint_dir,
        next_step=0,
        model=model,
        optimizer=optimizer,
        metrics=[],
        run_config=RUN_CONFIG,
    )
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(CheckpointError, match="checksum mismatch"):
        read_checkpoint(checkpoint_dir)


def test_resume_rejects_changed_run_configuration(tmp_path):
    model, optimizer = _new_model_and_optimizer(123)
    checkpoint_dir = tmp_path / "mismatch"
    save_checkpoint(
        checkpoint_dir,
        next_step=0,
        model=model,
        optimizer=optimizer,
        metrics=[],
        run_config=RUN_CONFIG,
    )
    with pytest.raises(CheckpointError, match="configuration differs"):
        load_checkpoint(
            checkpoint_dir,
            model=model,
            optimizer=optimizer,
            expected_run_config={**RUN_CONFIG, "lr": 1.0},
        )
