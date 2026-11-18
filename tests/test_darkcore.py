"""DarkCore tests — plain asserts, runnable with pytest or directly:
    python3 tests/test_darkcore.py

Covers: tiled matmul vs naive (exact), C core vs Python (when built),
fused linear+ReLU vs Linear+ReLU, tape backward vs Tensor backward
(same graph, both engines), fused SGD vs SGD, buffer-pool reuse.
"""

import sys
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darklearn import Tensor
from darklearn import darkcore as dc
from darklearn.darkcore import Tape, CoreTensor, BufferPool


def _rand(m, n, seed):
    return Tensor.randn((m, n), seed=seed)


def _close(a, b, tol=1e-4):
    assert len(a) == len(b), f"length {len(a)} vs {len(b)}"
    err = max(abs(float(x) - float(y)) for x, y in zip(a, b)) if a else 0.0
    assert err < tol, f"max err {err} >= {tol}"
    return err


def test_tiled_matches_naive():
    for (m, k, n) in [(5, 7, 3), (33, 17, 41), (64, 64, 64)]:
        a = _rand(m, k, seed=0)
        b = _rand(k, n, seed=1)
        ref = (a @ b)._data
        for block in (16, 32):
            got = dc.matmul_tiled(a._data, b._data, m, k, n, block=block)
            _close(got, ref, tol=1e-3)
        # tiled_into (pooled scratch path) must match too
        out = array("f", bytes(4 * m * n))
        dc.matmul_tiled_into(a._data, b._data, m, k, n, out, block=32)
        _close(out, ref, tol=1e-3)


def test_rows_matches_naive():
    for (m, k, n) in [(5, 7, 3), (33, 17, 41), (64, 64, 64)]:
        a = _rand(m, k, seed=2)
        b = _rand(k, n, seed=3)
        ref = (a @ b)._data
        got = dc.matmul_rows(a._data, b._data, m, k, n)
        # rows dots sum in the same k order, so this is exact
        assert list(got) == list(ref), "rows kernel differs from naive"


def test_c_matches_python():
    if not dc.C_AVAILABLE:
        print("SKIP test_c_matches_python (C core not built)")
        return
    for (m, k, n) in [(17, 13, 11), (64, 64, 64)]:
        a = _rand(m, k, seed=4)
        b = _rand(k, n, seed=5)
        ref = dc.matmul_tiled(a._data, b._data, m, k, n)
        got = dc.matmul_c(a._data, b._data, m, k, n)
        _close(got, ref, tol=1e-3)
        # fused: C vs pure-Python raw kernel
        x = Tensor.randn((3, 4), seed=6)._data
        w = Tensor.randn((5, 4), seed=7)._data
        bb = Tensor.randn((5,), seed=8)._data
        oc, mc = dc.fused_linear_relu_c(x, w, bb, 3, 4, 5)
        op, mp = dc.fused_linear_relu_raw(x, w, bb, 3, 4, 5)
        _close(oc, op, tol=1e-4)
        assert list(mc) == list(mp)


def test_fused_linear_matches():
    from darklearn import Linear, ReLU
    x = Tensor.randn((3, 4), seed=9)
    layer = Linear(4, 5, seed=10)
    ref = ReLU()(layer(x)).tolist()
    got, _ = dc.fused_linear_relu_raw(x._data, layer.weight._data,
                                      layer.bias._data, 3, 4, 5)
    ref_flat = [v for row in ref for v in row]
    _close(got, ref_flat, tol=1e-4)
    # tape-aware fused op, forward part
    tape = Tape()
    xc = tape.leaf(x.tolist())
    wc = tape.leaf(layer.weight.tolist())
    bc = tape.leaf(layer.bias.tolist())
    y = dc.fused_linear_relu(xc, wc, bc)
    _close(y._data, ref_flat, tol=1e-4)


def test_tape_backward_matches_tensor():
    # Same graph in both engines: y = ((a*b + c).relu() @ W).sum()
    a_data = [[0.5, -1.0, 2.0], [1.5, 0.3, -0.7]]
    b_data = [[1.0, 0.5, -2.0], [0.25, 1.5, 0.75]]
    w_data = [[0.4, -1.2], [0.9, 0.3], [-0.6, 1.1]]
    # Tensor engine
    a = Tensor(a_data, requires_grad=True)
    b = Tensor(b_data, requires_grad=True)
    c = Tensor(0.25, requires_grad=True)
    w = Tensor(w_data, requires_grad=True)
    ((a * b + c).relu() @ w).sum().backward()
    # DarkCore tape engine
    tape = Tape()
    ac = tape.leaf(a_data)
    bc = tape.leaf(b_data)
    cc = tape.leaf(0.25)
    wc = tape.leaf(w_data)
    ((ac * bc + cc).relu() @ wc).sum().backward()
    _close(ac.grad, a.grad._data, tol=1e-4)
    _close(bc.grad, b.grad._data, tol=1e-4)
    _close(cc.grad, c.grad._data, tol=1e-4)
    _close(wc.grad, w.grad._data, tol=1e-4)


def test_tape_fused_backward():
    from darklearn import Linear, ReLU
    x = Tensor.randn((3, 4), seed=11, requires_grad=True)
    layer = Linear(4, 5, seed=12)
    ((x @ layer.weight.T + layer.bias).relu()).sum().backward()
    tape = Tape()
    xc = tape.leaf(x.tolist())
    wc = tape.leaf(layer.weight.tolist())
    bc = tape.leaf(layer.bias.tolist())
    dc.fused_linear_relu(xc, wc, bc).sum().backward()
    _close(xc.grad, x.grad._data, tol=1e-4)
    _close(wc.grad, layer.weight.grad._data, tol=1e-4)
    _close(bc.grad, layer.bias.grad._data, tol=1e-4)


def test_fused_sgd_matches():
    from darklearn import SGD
    p = Tensor([1.0, -2.0, 0.5], requires_grad=True)
    p.grad = Tensor([0.3, -0.1, 0.2])
    opt = SGD([p], lr=0.1, momentum=0.9, weight_decay=0.01)
    p2 = array("f", [1.0, -2.0, 0.5])
    g2 = array("f", [0.3, -0.1, 0.2])
    state = None
    for _ in range(3):
        opt.step()
        state = dc.fused_sgd_step([p2], [g2], lr=0.1, momentum=0.9,
                                  weight_decay=0.01, state=state)
        _close(p2, p._data, tol=1e-6)
    # momentum state is real velocity, not zeros
    assert state is not None and any(abs(v) > 0 for v in state[0])
    # momentum 0 + no decay == plain gradient step
    q = array("f", [1.0, 2.0])
    dc.fused_sgd_step([q], [array("f", [0.5, -0.5])], lr=0.2)
    _close(q, [0.9, 2.1], tol=1e-6)  # float32 rounding: never == exactly


def test_buffer_pool_reuse():
    pool = BufferPool()
    b1 = pool.acquire(10)
    b1[0] = 99.0
    pool.release(b1)
    b2 = pool.acquire(10)
    assert b2 is b1, "pool did not hand back the released buffer"
    assert b2[0] == 0.0, "acquired buffer was not zeroed"
    assert pool.stats()["reused"] >= 1
    # tape backward actually reuses scratch (pool stats move)
    tape = Tape()
    a = tape.leaf([[1.0, 2.0], [3.0, 4.0]])
    w = tape.leaf([[0.5, -0.5], [1.0, 0.25]])
    (a @ w).sum().backward()
    assert tape.pool.stats()["reused"] >= 1, "backward did not reuse buffers"
    assert a.grad is not None and w.grad is not None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All DarkCore tests passed.")
