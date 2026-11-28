"""Benchmark DarkCore backward + fused SGD step (Day 13, Month 6).

Times the two things Day 13 optimises, on fixed graphs, so the
before/after numbers are comparable:

- tape backward on ``((a * b + c).relu() @ W).sum()`` — the graph
  whose small grad temporaries (add/mul/relu/sum) were pooled in
  Day 13. Baseline measured 2026-11-28 *before* the pooling change:
  0.45 ms/step at (8,16)@(16,8), 4.83 ms/step at (32,32)@(32,32).
- ``fused_sgd_step`` over a small model's params (Python loop, and
  the C core once Day 13 wires it in).

    python3 examples/bench_backward.py

Nothing here trains a model — it is a timing harness, and it prints
real measured milliseconds, not estimates.
"""

import sys
import time
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darklearn import Tensor
from darklearn import darkcore as dc
from darklearn.darkcore import Tape

# Baseline (pre-pooling, 2026-11-28, this VM):
BASELINE_BACKWARD = {(8, 16, 8): 0.45, (32, 32, 32): 4.83}


def _time_backward(m, k, n, reps):
    a_data = Tensor.randn((m, k), seed=0).tolist()
    b_data = Tensor.randn((m, k), seed=1).tolist()
    w_data = Tensor.randn((k, n), seed=2).tolist()
    t0 = time.perf_counter()
    for _ in range(reps):
        tape = Tape()
        a = tape.leaf(a_data)
        b = tape.leaf(b_data)
        c = tape.leaf(0.25)
        w = tape.leaf(w_data)
        ((a * b + c).relu() @ w).sum().backward()
    return (time.perf_counter() - t0) / reps * 1000.0


def _time_fused_sgd(sizes, momentum, weight_decay, reps, use_c):
    params = [array("f", [0.5] * n) for n in sizes]
    grads = [array("f", [0.1] * n) for n in sizes]
    state = None
    # Python loop directly, or the dispatcher (C when built, else Python).
    fn = dc.fused_sgd_step if use_c else dc.fused_sgd_step_py
    t0 = time.perf_counter()
    for _ in range(reps):
        state = fn(params, grads, lr=0.05, momentum=momentum,
                   weight_decay=weight_decay, state=state)
    return (time.perf_counter() - t0) / reps * 1000.0


def main():
    print("tape backward ((a*b + c).relu() @ W).sum():")
    print(f"  {'shape':>18} {'ms/step':>9} {'vs baseline':>12}")
    for (m, k, n), reps in [((8, 16, 8), 100), ((32, 32, 32), 50)]:
        ms = _time_backward(m, k, n, reps)
        base = BASELINE_BACKWARD[(m, k, n)]
        print(f"  ({m},{k})@({k},{n}): {ms:>9.2f} "
              f"{base / ms:>11.2f}x")

    # Param sizes of a tiny GPT-ish model: embedding + blocks + head.
    sizes = [384 * 16, 16 * 16, 16 * 64, 64 * 16, 16 * 384, 384]
    total = sum(sizes)
    print(f"\nfused_sgd_step over {total} params "
          f"({len(sizes)} tensors):")
    for momentum, wd in [(0.0, 0.0), (0.9, 0.01)]:
        ms = _time_fused_sgd(sizes, momentum, wd, reps=200, use_c=False)
        print(f"  momentum={momentum} wd={wd}: {ms:.3f} ms/step "
              f"(Python)")
        if hasattr(dc, "fused_sgd_step_c"):
            ms_c = _time_fused_sgd(sizes, momentum, wd, reps=200,
                                   use_c=True)
            print(f"  momentum={momentum} wd={wd}: {ms_c:.3f} ms/step "
                  f"(C)  {ms / ms_c:.1f}x")
    print("OK — backward benchmark complete.")


if __name__ == "__main__":
    main()
