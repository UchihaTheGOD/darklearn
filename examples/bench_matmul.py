"""Benchmark the pure-Python matmul — baseline timing before any optimisation.

Times square matmuls at several sizes and reports seconds per matmul and
approximate FLOPS (2*n^3 per matmul). This is the Month 1 Week 4 baseline;
Month 6 will compare the optional C core against these numbers.

Run:  python3 examples/bench_matmul.py   (from repo root)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darklearn import Tensor


def bench(n: int, reps: int) -> float:
    a = Tensor.randn((n, n), seed=0)
    b = Tensor.randn((n, n), seed=1)
    # warm-up (also checks the op works at this size)
    _ = a @ b
    start = time.perf_counter()
    for _ in range(reps):
        _ = a @ b
    return (time.perf_counter() - start) / reps


def main() -> None:
    print(f"{'n':>5} {'reps':>5} {'sec/matmul':>12} {'MFLOPS':>10}")
    for n, reps in [(16, 20), (32, 10), (64, 5), (128, 2)]:
        secs = bench(n, reps)
        flops = 2.0 * n ** 3 / secs / 1e6
        print(f"{n:>5} {reps:>5} {secs:>12.4f} {flops:>10.2f}")
    print("OK — benchmark complete.")


if __name__ == "__main__":
    main()
