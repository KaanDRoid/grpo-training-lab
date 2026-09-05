"""Compare the deterministic training state of two published checkpoint directories."""

import argparse
from typing import Any

import torch

from checkpointing import read_checkpoint


def assert_equal(left: Any, right: Any, path="root") -> None:
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or not torch.equal(left, right):
            raise AssertionError(f"tensor differs at {path}")
        return
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise AssertionError(f"mapping keys differ at {path}")
        for key in left:
            assert_equal(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise AssertionError(f"sequence differs at {path}")
        for index, (a, b) in enumerate(zip(left, right)):
            assert_equal(a, b, f"{path}[{index}]")
        return
    if left != right:
        raise AssertionError(f"value differs at {path}: {left!r} != {right!r}")


def comparable_metrics(metrics):
    # Allocator high-water marks can legitimately differ after process restart. Everything that
    # affects optimization or sampled data remains part of the exact comparison.
    return [{k: v for k, v in metric.items() if k != "peak_vram_gib"} for metric in metrics]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("left")
    parser.add_argument("right")
    args = parser.parse_args()

    left = read_checkpoint(args.left)
    right = read_checkpoint(args.right)
    assert_equal(left["next_step"], right["next_step"], "next_step")
    assert_equal(left["run_config"], right["run_config"], "run_config")
    assert_equal(left["model_trainable"], right["model_trainable"], "model_trainable")
    assert_equal(left["optimizer"], right["optimizer"], "optimizer")
    assert_equal(left["rng"], right["rng"], "rng")
    assert_equal(
        comparable_metrics(left["metrics"]),
        comparable_metrics(right["metrics"]),
        "metrics",
    )
    print(
        f"checkpoint continuation: EXACT MATCH at next_step={left['next_step']} "
        "(model, optimizer, RNG, rollout digests, and numerical metrics)"
    )


if __name__ == "__main__":
    main()
