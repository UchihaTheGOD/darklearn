"""Day 13 tests — plain asserts, runnable with pytest or directly:
    python3 tests/test_day13.py

Covers: tape backward pools its small grad temporaries (add/mul/relu/
sum/fused-linear) instead of allocating a fresh array per record, with
grads still matching the Tensor engine; and the C fused-SGD step
matches the pure-Python one (when the C core is built).
"""

import sys
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darklearn import Tensor
from darklearn import darkcore as dc
from darklearn.darkcore import Tape


def _close(a, b, tol=1e-4):
    assert len(a) == len(b), f"length {len(a)} vs {len(b)}"
    err = max(abs(float(x) - float(y)) for x, y in zip(a, b)) if a else 0.0
    assert err < tol, f"max err {err} >= {tol}"
    return err


def _mul_add_graph(tape, a_data, b_data, c_val):
    """(a * b + c).relu().sum() on ``tape``; returns the leaf CoreTensors."""
    a = tape.leaf(a_data)
    b = tape.leaf(b_data)
    c = tape.leaf(c_val)
    ((a * b + c).relu()).sum().backward()
    return a, b, c


def test_backward_pools_small_temporaries():
    # Same-shaped graphs on one tape: the second backward must reuse
    # buffers the first one released (add/mul/relu/sum temporaries are
    # pooled now, not allocated per record).
    a_data = [[0.5, -1.0, 2.0], [1.5, 0.3, -0.7]]
    b_data = [[1.0, 0.5, -2.0], [0.25, 1.5, 0.75]]
    tape = Tape()
    _mul_add_graph(tape, a_data, b_data, 0.25)
    reused_after_first = tape.pool.stats()["reused"]
    _mul_add_graph(tape, a_data, b_data, 0.25)
    reused_after_second = tape.pool.stats()["reused"]
    assert reused_after_second > reused_after_first, (
        f"no buffer reuse across backward passes: "
        f"{reused_after_first} -> {reused_after_second}")
    # ... and the pooled path still gives the same grads as the
    # Tensor engine (broadcast case: (2,3) * (3,) row + scalar).
    a = Tensor(a_data, requires_grad=True)
    b = Tensor([1.0, 0.5, -2.0], requires_grad=True)  # (3,) row
    c = Tensor(0.25, requires_grad=True)
    ((a * b + c).relu()).sum().backward()
    tape2 = Tape()
    ac = tape2.leaf(a_data)
    bc = tape2.leaf([1.0, 0.5, -2.0])
    cc = tape2.leaf(0.25)
    ((ac * bc + cc).relu()).sum().backward()
    _close(ac.grad, a.grad._data, tol=1e-4)
    _close(bc.grad, b.grad._data, tol=1e-4)
    _close(cc.grad, c.grad._data, tol=1e-4)


def test_fused_backward_pools():
    # Fused-linear backward pools the bias grad (and the d / tmp
    # buffers, as before); grads still match the Tensor engine.
    from darklearn import Linear
    x = Tensor.randn((3, 4), seed=11, requires_grad=True)
    layer = Linear(4, 5, seed=12)
    ((x @ layer.weight.T + layer.bias).relu()).sum().backward()
    tape = Tape()
    xc = tape.leaf(x.tolist())
    wc = tape.leaf(layer.weight.tolist())
    bc = tape.leaf(layer.bias.tolist())
    dc.fused_linear_relu(xc, wc, bc).sum().backward()
    # Compare grads now: a Tape replays *all* its records on every
    # backward() call, so a second backward on the same tape would
    # re-accumulate the first graph's grads (stale output grads are
    # still set) — the second graph below is only for the reuse stat.
    _close(xc.grad, x.grad._data, tol=1e-4)
    _close(wc.grad, layer.weight.grad._data, tol=1e-4)
    _close(bc.grad, layer.bias.grad._data, tol=1e-4)
    reused_after_first = tape.pool.stats()["reused"]
    xc2 = tape.leaf(x.tolist())
    wc2 = tape.leaf(layer.weight.tolist())
    bc2 = tape.leaf(layer.bias.tolist())
    dc.fused_linear_relu(xc2, wc2, bc2).sum().backward()
    assert tape.pool.stats()["reused"] > reused_after_first, (
        "fused backward did not reuse pooled buffers")


def test_c_fused_sgd_matches():
    if not dc.C_AVAILABLE:
        print("SKIP test_c_fused_sgd_matches (C core not built)")
        return
    from darklearn import SGD
    # C vs Python fused step, momentum + weight decay, several steps,
    # and against optim.SGD on the same values.
    for momentum, wd in [(0.9, 0.01), (0.0, 0.01), (0.9, 0.0)]:
        p = Tensor([1.0, -2.0, 0.5, 3.0], requires_grad=True)
        p.grad = Tensor([0.3, -0.1, 0.2, -0.4])
        opt = SGD([p], lr=0.1, momentum=momentum, weight_decay=wd)
        p_py = array("f", [1.0, -2.0, 0.5, 3.0])
        g_py = array("f", [0.3, -0.1, 0.2, -0.4])
        p_c = array("f", [1.0, -2.0, 0.5, 3.0])
        g_c = array("f", [0.3, -0.1, 0.2, -0.4])
        state_py = state_c = None
        for _ in range(3):
            opt.step()
            state_py = dc.fused_sgd_step_py(
                [p_py], [g_py], lr=0.1, momentum=momentum,
                weight_decay=wd, state=state_py)
            state_c = dc.fused_sgd_step_c(
                [p_c], [g_c], lr=0.1, momentum=momentum,
                weight_decay=wd, state=state_c)
            _close(p_py, p._data, tol=1e-6)
            _close(p_c, p._data, tol=1e-6)
            _close(p_c, p_py, tol=1e-6)
        if momentum:
            assert state_c is not None and state_c[0] is not None
            assert any(abs(v) > 0 for v in state_c[0])
            # the velocity object is created once and then updated in
            # place across steps — no fresh allocation per step
            before = state_c[0]
            dc.fused_sgd_step_c([p_c], [g_c], lr=0.1, momentum=momentum,
                                weight_decay=wd, state=state_c)
            assert state_c[0] is before
    # a None grad skips that param, in both engines
    q_py = array("f", [1.0, 2.0])
    q_c = array("f", [1.0, 2.0])
    dc.fused_sgd_step_py([q_py, q_py], [None, array("f", [0.5, -0.5])],
                         lr=0.2)
    dc.fused_sgd_step_c([q_c, q_c], [None, array("f", [0.5, -0.5])],
                        lr=0.2)
    _close(q_c, q_py, tol=1e-6)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All Day 13 tests passed.")
