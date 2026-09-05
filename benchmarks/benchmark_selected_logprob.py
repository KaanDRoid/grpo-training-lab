"""Benchmark correctness, latency, and transient memory across vocabulary widths.

Example:
    python benchmarks/benchmark_selected_logprob.py --csv benchmarks/selected_logprob.csv
"""

import argparse
import csv
import gc
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernels.selected_logprob import chunked_linear_selected_logprob


def naive_linear_selected_logprob(hidden, weight, target_ids):
    logits = F.linear(hidden, weight)
    return torch.log_softmax(logits.float(), dim=-1).gather(
        1, target_ids[:, None]
    ).squeeze(1)


def measure_cuda(fn, repeats):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = None
    for _ in range(repeats):
        result = fn()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / repeats
    peak_bytes = torch.cuda.max_memory_allocated() - baseline
    return result, elapsed_ms, peak_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--chunk-rows", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--vocabs", type=int, nargs="+", default=[8191, 32768, 65537, 151936]
    )
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    records = []
    print("vocab     naive_ms  chunked_ms  naive_peak_MiB  chunked_peak_MiB  max_abs")
    for vocab in args.vocabs:
        torch.manual_seed(vocab)
        hidden = torch.randn(
            args.rows, args.hidden_size, device="cuda", dtype=torch.float16
        )
        weight = torch.randn(
            vocab, args.hidden_size, device="cuda", dtype=torch.float16
        ) / args.hidden_size**0.5
        target_ids = torch.randint(0, vocab, (args.rows,), device="cuda")

        # Compile/warm both paths outside the measured region.
        naive_linear_selected_logprob(hidden, weight, target_ids)
        chunked_linear_selected_logprob(
            hidden, weight, target_ids, chunk_rows=args.chunk_rows
        )

        naive, naive_ms, naive_peak = measure_cuda(
            lambda: naive_linear_selected_logprob(hidden, weight, target_ids), args.repeats
        )
        chunked, chunked_ms, chunked_peak = measure_cuda(
            lambda: chunked_linear_selected_logprob(
                hidden, weight, target_ids, chunk_rows=args.chunk_rows
            ),
            args.repeats,
        )
        max_abs = (naive - chunked).abs().max().item()
        record = {
            "vocab": vocab,
            "rows": args.rows,
            "hidden_size": args.hidden_size,
            "chunk_rows": args.chunk_rows,
            "naive_ms": naive_ms,
            "chunked_ms": chunked_ms,
            "naive_peak_mib": naive_peak / 2**20,
            "chunked_peak_mib": chunked_peak / 2**20,
            "max_abs": max_abs,
        }
        records.append(record)
        print(
            f"{vocab:8d}  {naive_ms:8.2f}  {chunked_ms:10.2f}  "
            f"{record['naive_peak_mib']:14.2f}  {record['chunked_peak_mib']:16.2f}  "
            f"{max_abs:.3e}"
        )

        del hidden, weight, target_ids, naive, chunked

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
