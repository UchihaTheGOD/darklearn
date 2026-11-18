"""Benchmark DarkCore: naive Tensor matmul vs tiled Python vs rows
Python vs C (when built). Prints real measured MFLOPS — nothing here
is estimated or faked.

    python3 examples/bench_darkcore.py   (from repo root)

Correctness is checked first (all kernels must match the naive result
within float32 tolerance); then each kernel is timed. Expectation from
the design notes: tiling alone is only ~1.15x in CPython (bytecode is
the bottleneck, not cache), the rows kernel is ~4x, and C is far faster.
If any kernel is slower than naive, that is reported as-is.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darklearn import Tensor
from darklearn import darkcore as dc


def _bench(fn, n, reps):
    a = Tensor.randn((n, n), seed=0)
    b = Tensor.randn((n, n), seed=1)
    fn(a, b)  # warm-up + correctness path
    start = time.perf_counter()
    for _ in range(reps):
        fn(a, b)
    return (time.perf_counter() - start) / reps


def main() -> None:
    # correctness first: every kernel vs the naive Tensor result
    a = Tensor.randn((17, 13), seed=0)
    b = Tensor.randn((13, 11), seed=1)
    ref = (a @ b)._data
    kernels = {
        "tiled-32": lambda x, y: dc.matmul_tiled(x._data, y._data, 17, 13, 11),
        "rows": lambda x, y: dc.matmul_rows(x._data, y._data, 17, 13, 11),
    }
    if dc.C_AVAILABLE:
        kernels["C"] = lambda x, y: dc.matmul_c(x._data, y._data, 17, 13, 11)
    for name, fn in kernels.items():
        got = fn(a, b)
        err = max(abs(float(p) - float(q)) for p, q in zip(got, ref))
        assert err < 1e-3, f"{name} mismatch: {err}"
    print(f"correctness OK (max err < 1e-3) — C core: "
          f"{'built' if dc.C_AVAILABLE else 'NOT built, Python fallback'}")

    print(f"\n{'n':>5} {'kernel':>10} {'sec/matmul':>12} {'MFLOPS':>10} "
          f"{'vs naive':>9}")
    for n, reps in [(64, 3), (128, 2)]:
        naive_s = _bench(lambda x, y: x @ y, n, reps)
        flops = 2.0 * n ** 3 / naive_s / 1e6
        print(f"{n:>5} {'naive':>10} {naive_s:>12.4f} {flops:>10.2f} "
              f"{'1.00x':>9}")
        rows = [
            ("tiled-16", lambda x, y: dc.matmul_tiled(x._data, y._data, n, n, n, block=16)),
            ("tiled-32", lambda x, y: dc.matmul_tiled(x._data, y._data, n, n, n, block=32)),
            ("rows", lambda x, y: dc.matmul_rows(x._data, y._data, n, n, n)),
        ]
        if dc.C_AVAILABLE:
            rows.append(("C", lambda x, y: dc.matmul_c(x._data, y._data, n, n, n)))
        for name, fn in rows:
            r = 5 if name in ("rows", "C") else reps
            secs = _bench(fn, n, r)
            flops = 2.0 * n ** 3 / secs / 1e6
            print(f"{n:>5} {name:>10} {secs:>12.4f} {flops:>10.2f} "
                  f"{naive_s / secs:>8.2f}x")
    print("OK — benchmark complete.")


if __name__ == "__main__":
    main()
