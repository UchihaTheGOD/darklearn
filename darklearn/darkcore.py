"""darklearn.darkcore — DarkCore, darklearn's own engine.

This is the engine the user asked for: our own design, not a copy of
PyTorch / micrograd / tutorial code. The *maths* is universal and cannot
change (matmul is matmul, the chain rule is the chain rule, SGD is SGD) —
what is ours is *how the engine is built*:

1. **Tiled matmul** (``matmul_tiled``): block 32, loop order i-k-j, locals
   cached outside the inner loop, no attribute lookups inside it.
   Honest note (measured, see ``examples/bench_darkcore.py``): in CPython
   tiling alone is only ~1.15x faster than the naive i-j-k loop, because
   the bottleneck is Python bytecode, not cache. So the *fast* Python path
   is ``matmul_rows``: B is transposed once into contiguous rows and each
   dot product runs in C via ``sum(map(mul, ...))`` (~4x). The tiled
   kernel stays as the reference kernel and as the C kernel's layout.
2. **Fused linear**: ``fused_linear_relu`` does matmul + bias + ReLU in
   one pass over x/W — no intermediate (batch, out) tensors at all.
3. **Tape autograd**: no per-op closures, no recursive DFS. A ``Tape``
   holds a flat list of values and a flat list of records
   ``(op_code, out_idx, in_idx_tuple, saved)`` where op codes are small
   ints (1=ADD, 2=MUL, 3=MATMUL, 4=RELU, 5=SUM, 6=FUSED_LINEAR_RELU).
   ``backward()`` is a single iterative loop over the tape in reverse.
4. **Buffer pool**: ``BufferPool`` recycles scratch ``array('f')``
   buffers (transposes and matmul temporaries in backward, plus the
   small grad temporaries of add/mul/relu/sum/fused-linear backward —
   broadcast expansions, elementwise products, reductions — all
   acquired from the pool and released right after the grad is
   accumulated, since accumulation copies). Live outputs are never
   pooled — only scratch that is released after use.
5. **Fused optimizer step**: ``fused_sgd_step`` updates params in place
   in one pass per tensor (grad + weight decay + momentum + step fused,
   no per-param Python objects).

Optional C core: if ``darklearn/_darkcore*.so`` was built (see
``scripts/build_c.py``), ``matmul`` / fused ops dispatch to C; otherwise
they fall back to the pure-Python kernels above. Installing darklearn
never requires C.

``tensor.py`` is untouched and keeps working — DarkCore runs in parallel
to it, and the tests compare the two engines against each other.
"""

from __future__ import annotations

import math
import random
from array import array
from operator import mul

# -- op codes (ints, not strings, not closures) -------------------------------
OP_NONE = 0
OP_ADD = 1
OP_MUL = 2
OP_MATMUL = 3
OP_RELU = 4
OP_SUM = 5
OP_FUSED_LINEAR_RELU = 6

OP_NAMES = {0: "none", 1: "add", 2: "mul", 3: "matmul", 4: "relu",
            5: "sum", 6: "fused_linear_relu"}

DEFAULT_BLOCK = 32

# -- optional C core ----------------------------------------------------------
try:  # never fatal: pure Python is always the fallback
    from . import _darkcore as _C
    C_AVAILABLE = True
except Exception:  # ImportError, missing .so, ABI mismatch, ...
    _C = None
    C_AVAILABLE = False


def c_available() -> bool:
    """True when the compiled C core is importable."""
    return C_AVAILABLE


# -- small helpers ------------------------------------------------------------

def _numel(shape) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def _flatten(data):
    """Flatten nested lists / scalars / array('f') -> (flat_list, shape|None)."""
    if isinstance(data, (int, float)):
        return [float(data)], ()
    if isinstance(data, array):
        return [float(x) for x in data], None
    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            return [], (0,)
        if isinstance(data[0], (list, tuple)):
            inner, _ = _flatten(data[0])
            cols = len(inner)
            flat: list[float] = []
            for row in data:
                f, _ = _flatten(row)
                flat.extend(f)
            return flat, (len(data), cols)
        return [float(x) for x in data], (len(data),)
    raise TypeError(f"cannot build CoreTensor from {type(data)}")


def _transpose_into(data, rows: int, cols: int, out) -> None:
    """out[c*rows + r] = data[r*cols + c] — plain loops, no allocation."""
    for r in range(rows):
        base = r * cols
        for c in range(cols):
            out[c * rows + r] = data[base + c]


# -- buffer pool ---------------------------------------------------------------

class BufferPool:
    """Recycle scratch ``array('f')`` buffers by size.

    Only scratch is pooled: transposes and matmul temporaries created
    inside backward / fused ops, released right after use. Live outputs
    are never handed out from the pool. ``stats()`` reports reuse so the
    benefit is measurable, not assumed.
    """

    def __init__(self, max_per_size: int = 4):
        self._bins: dict[int, list] = {}
        self._max = max_per_size
        self.acquired = 0
        self.reused = 0

    def acquire(self, n: int):
        """Return a zeroed ``array('f')`` of length n (recycled if possible)."""
        b = self._bins.get(n)
        if b:
            self.reused += 1
            buf = b.pop()
            for i in range(n):
                buf[i] = 0.0
            return buf
        self.acquired += 1
        return array("f", bytes(4 * n))

    def acquire_dirty(self, n: int):
        """Return an ``array('f')`` of length n, contents undefined.

        Same recycling as ``acquire`` but without the zeroing pass —
        for callers that overwrite every element before reading it
        (a matmul output, an elementwise product, a masked grad).
        The zeroing loop is pure overhead for those callers: measured
        ~88 us at n=1024 on this VM, nearly as much as the fill itself.
        Never use for a buffer that is accumulated into (+=) or only
        partly written.
        """
        b = self._bins.get(n)
        if b:
            self.reused += 1
            return b.pop()
        self.acquired += 1
        return array("f", bytes(4 * n))

    def release(self, buf) -> None:
        """Return a scratch buffer to the pool (dropped when the bin is full)."""
        b = self._bins.setdefault(len(buf), [])
        if len(b) < self._max:
            b.append(buf)

    def stats(self) -> dict:
        return {"acquired": self.acquired, "reused": self.reused,
                "cached": sum(len(v) for v in self._bins.values())}


_POOL = BufferPool()


# -- raw matmul kernels (operate on flat array('f'), no Tensor objects) -------

def matmul_tiled(a, b, m: int, k: int, n: int, block: int = DEFAULT_BLOCK):
    """Tiled matmul, block size ``block``, loop order i-k-j.

    ``a`` is (m, k), ``b`` is (k, n), both flat row-major ``array('f')``.
    Returns a fresh ``array('f')`` of shape (m, n). All loop bounds are
    locals; the inner loop touches one row of ``a`` (one scalar ``aik``),
    one row of ``b`` and one row of the output — the tiling keeps those
    rows hot. See the module docstring for why this alone is ~1.15x in
    CPython, and why ``matmul_rows`` is the fast Python path.
    """
    out = array("f", bytes(4 * m * n))
    ad, bd, od = a, b, out
    B = block
    for ii in range(0, m, B):
        i_end = ii + B if ii + B < m else m
        for kk in range(0, k, B):
            k_end = kk + B if kk + B < k else k
            for jj in range(0, n, B):
                j_end = jj + B if jj + B < n else n
                for i in range(ii, i_end):
                    a_row = i * k
                    o_row = i * n
                    for k2 in range(kk, k_end):
                        aik = ad[a_row + k2]
                        b_row = k2 * n
                        for j in range(jj, j_end):
                            od[o_row + j] += aik * bd[b_row + j]
    return out


def matmul_tiled_into(a, b, m: int, k: int, n: int, out, block: int = DEFAULT_BLOCK) -> None:
    """Tiled matmul writing into ``out`` (a pooled scratch buffer).

    ``out`` is zeroed first, because pooled buffers may hold stale data.
    """
    for i in range(m * n):
        out[i] = 0.0
    ad, bd, od = a, b, out
    B = block
    for ii in range(0, m, B):
        i_end = ii + B if ii + B < m else m
        for kk in range(0, k, B):
            k_end = kk + B if kk + B < k else k
            for jj in range(0, n, B):
                j_end = jj + B if jj + B < n else n
                for i in range(ii, i_end):
                    a_row = i * k
                    o_row = i * n
                    for k2 in range(kk, k_end):
                        aik = ad[a_row + k2]
                        b_row = k2 * n
                        for j in range(jj, j_end):
                            od[o_row + j] += aik * bd[b_row + j]


def matmul_rows(a, b, m: int, k: int, n: int):
    """Fast pure-Python matmul: transpose B once, dot in C.

    ``B`` is transposed once into a list of contiguous rows (one per
    output column). Each output element is then a dot product of two
    contiguous rows, computed by ``sum(map(mul, ...))`` — the multiply
    and the accumulation both run in C, which is what makes this ~4x
    faster than a Python triple loop. Summation order is k = 0..n-1,
    the same as the naive loop, so results match it bit-for-bit.
    Returns a fresh ``array('f')`` of shape (m, n).
    """
    out = array("f", bytes(4 * m * n))
    matmul_rows_into(a, b, m, k, n, out)
    return out


def matmul_rows_into(a, b, m: int, k: int, n: int, out) -> None:
    """``matmul_rows`` writing into ``out`` (fully overwritten)."""
    a_rows = [list(a[i * k:(i + 1) * k]) for i in range(m)]
    b_cols = [[b[r * n + j] for r in range(k)] for j in range(n)]
    for i in range(m):
        ai = a_rows[i]
        o_row = i * n
        for j in range(n):
            out[o_row + j] = sum(map(mul, ai, b_cols[j]))


def matmul_c(a, b, m: int, k: int, n: int, block: int = DEFAULT_BLOCK):
    """Tiled matmul via the compiled C core. Raises if C is unavailable."""
    if not C_AVAILABLE:
        raise RuntimeError("C core not built — run scripts/build_c.py, "
                           "or use matmul_tiled / matmul_rows instead")
    raw = _C.matmul(a, b, m, k, n, block)
    out = array("f")
    out.frombytes(raw)
    return out


def matmul(a, b, m: int, k: int, n: int, block: int = DEFAULT_BLOCK):
    """DarkCore's default matmul: C tiled kernel when built, else rows.

    ``matmul_tiled`` is the reference tiled kernel (always available);
    this dispatcher picks the fastest available path.
    """
    if C_AVAILABLE:
        return matmul_c(a, b, m, k, n, block)
    return matmul_rows(a, b, m, k, n)


# -- fused linear + relu -------------------------------------------------------

def fused_linear_relu_raw(x, w, b, batch: int, in_f: int, out_f: int):
    """One pass: out = relu(x @ W.T + b). No intermediate tensors.

    ``x`` is (batch, in_f), ``w`` is (out_f, in_f), ``b`` is (out_f,).
    Accumulates i-k-j (one scalar of x fans out over a whole output row),
    then bias + ReLU in a second sweep over the output row only.
    Returns ``(out_array, mask)`` where ``mask`` is a bytearray of 0/1
    (1 where the pre-ReLU value was > 0) — the only thing backward needs.
    """
    out = array("f", bytes(4 * batch * out_f))
    for i in range(batch):
        x_row = i * in_f
        o_row = i * out_f
        for k2 in range(in_f):
            xv = x[x_row + k2]
            for j in range(out_f):
                out[o_row + j] += xv * w[j * in_f + k2]
    mask = bytearray(batch * out_f)
    for i in range(batch * out_f):
        v = out[i] + b[i % out_f]
        if v > 0.0:
            out[i] = v
            mask[i] = 1
        else:
            out[i] = 0.0
    return out, mask


def fused_linear_relu_c(x, w, b, batch: int, in_f: int, out_f: int):
    """Fused linear+ReLU via the C core. Returns (out_array, mask)."""
    if not C_AVAILABLE:
        raise RuntimeError("C core not built — run scripts/build_c.py")
    raw_out, raw_mask = _C.fused_linear_relu(x, w, b, batch, in_f, out_f)
    out = array("f")
    out.frombytes(raw_out)
    return out, bytearray(raw_mask)


# -- tape autograd --------------------------------------------------------------

def _reduce_grad_into(g, grad_shape, target_shape, pool):
    """Reduce a grad (flat array, ``grad_shape``) to ``target_shape``.

    Returns ``(buf, from_pool)``. When the shapes already match,
    ``(g, False)`` — no allocation, nothing to release. Otherwise the
    reduction is computed into a pooled buffer, and the caller must
    ``pool.release(buf)`` after accumulating from it (``_accum``
    copies, so releasing right after is safe). This is what keeps
    tape backward from allocating a fresh array per add/mul record.
    """
    if grad_shape == target_shape:
        return g, False
    if target_shape == ():
        buf = pool.acquire_dirty(1)  # single element, set below
        buf[0] = sum(g)
        return buf, True
    if len(grad_shape) == 2 and len(target_shape) == 1:
        rows, cols = grad_shape
        buf = pool.acquire(cols)  # zeroed: the rows are summed into it
        for r in range(rows):
            base = r * cols
            for c in range(cols):
                buf[c] += g[base + c]
        return buf, True
    raise ValueError(f"cannot reduce grad {grad_shape} to {target_shape}")


def _expand_into(t, shape, pool):
    """Expand ``t``'s data to ``shape`` (the add/mul broadcast rules).

    Returns ``(flat, from_pool)``. When ``t`` already has ``shape``,
    ``(t._data, False)`` — no allocation. Otherwise the expansion is
    written into a pooled buffer, and the caller must release it after
    use. Only the broadcast shapes ``_broadcast_pair`` supports are
    handled here (same shape, scalar, or (m, n) with a (n,) row).
    """
    if t.shape == shape:
        return t._data, False
    if t.shape == ():
        buf = pool.acquire_dirty(_numel(shape))  # fully overwritten below
        v = float(t._data[0])
        for i in range(len(buf)):
            buf[i] = v
        return buf, True
    if len(shape) == 2 and len(t.shape) == 1 and shape[1] == t.shape[0]:
        rows, cols = shape
        buf = pool.acquire_dirty(rows * cols)  # fully overwritten below
        for r in range(rows):
            base = r * cols
            for c in range(cols):
                buf[base + c] = t._data[c]
        return buf, True
    raise ValueError(f"cannot broadcast {t.shape} to {shape}")


def _broadcast_pair(a, b):
    """Return (a_flat, b_flat_expanded, out_shape) for add/mul."""
    if a.shape == b.shape:
        return a._data, b._data, a.shape
    if b.shape == ():
        return a._data, array("f", [float(b._data[0])] * a.numel), a.shape
    if a.shape == ():
        return array("f", [float(a._data[0])] * b.numel), b._data, b.shape
    if len(a.shape) == 2 and len(b.shape) == 1 and a.shape[1] == b.shape[0]:
        rows, cols = a.shape
        return a._data, array("f", [float(b._data[c]) for _ in range(rows)
                                    for c in range(cols)]), a.shape
    if len(b.shape) == 2 and len(a.shape) == 1 and b.shape[1] == a.shape[0]:
        rows, cols = b.shape
        return array("f", [float(a._data[c]) for _ in range(rows)
                           for c in range(cols)]), b._data, b.shape
    raise ValueError(f"cannot broadcast {a.shape} and {b.shape}")


class Tape:
    """The graph: flat lists, no closures, no recursion.

    ``values`` — CoreTensors in creation order. ``records`` — one tuple
    per op: ``(op_code, out_idx, in_idx_tuple, saved)``. ``grads`` —
    one ``array('f')`` (or None) per value, same indexing as ``values``.
    Backward is one iterative loop over ``records`` in reverse.
    """

    def __init__(self):
        self.values: list = []
        self.records: list = []
        self.grads: list = []
        self.pool = BufferPool()

    def leaf(self, data, shape=None):
        """Create a tracked leaf value on this tape."""
        t = CoreTensor(data, shape, requires_grad=True, tape=self,
                       idx=len(self.values))
        self.values.append(t)
        self.grads.append(None)
        return t

    def _register(self, t):
        """Give ``t`` an index on this tape (no-op if already on it)."""
        if t._tape is self:
            return t._idx
        if t._tape is not None:
            raise ValueError("CoreTensor belongs to a different Tape")
        t._tape = self
        t._idx = len(self.values)
        self.values.append(t)
        self.grads.append(None)
        return t._idx

    def _new(self, data, shape, op, ins, saved):
        t = CoreTensor(data, shape, requires_grad=True, tape=self,
                       idx=len(self.values))
        self.values.append(t)
        self.grads.append(None)
        self.records.append((op, t._idx, tuple(i._idx for i in ins), saved))
        return t

    def _accum(self, idx, g) -> None:
        cur = self.grads[idx]
        if cur is None:
            self.grads[idx] = array("f", g)
        else:
            for i in range(len(cur)):
                cur[i] += g[i]

    def backward(self, out) -> None:
        """Reverse-mode autodiff from scalar ``out`` — one reverse loop.

        Replays *every* record on the tape, so call it once per graph:
        a second call re-accumulates the earlier graphs' grads (their
        output grads are still set), it does not start fresh.
        """
        if out._tape is not self:
            raise ValueError("backward() called on a value from another Tape")
        if out.numel != 1:
            raise ValueError("backward() needs a scalar tensor")
        self.grads[out._idx] = array("f", [1.0])
        values, grads, pool = self.values, self.grads, self.pool
        for op, oidx, ins, saved in reversed(self.records):
            g = grads[oidx]
            if g is None:
                continue
            out_t = values[oidx]
            if op == OP_ADD:
                for idx in ins:
                    v = values[idx]
                    if v.requires_grad:
                        red, owned = _reduce_grad_into(g, out_t.shape,
                                                       v.shape, pool)
                        self._accum(idx, red)
                        if owned:
                            pool.release(red)
            elif op == OP_MUL:
                a, b = values[ins[0]], values[ins[1]]
                # grad wrt a = g * b (expanded to out shape), then reduce;
                # grad wrt b = g * a, likewise. The expansions and the
                # products are pooled scratch, released after _accum
                # (which copies, so nothing it keeps points at them).
                if a.requires_grad:
                    b_full, b_owned = _expand_into(b, out_t.shape, pool)
                    ga = pool.acquire_dirty(out_t.numel)
                    for i in range(out_t.numel):
                        ga[i] = g[i] * b_full[i]
                    red, red_owned = _reduce_grad_into(ga, out_t.shape,
                                                       a.shape, pool)
                    self._accum(ins[0], red)
                    if red_owned:
                        pool.release(red)
                    pool.release(ga)
                    if b_owned:
                        pool.release(b_full)
                if b.requires_grad:
                    a_full, a_owned = _expand_into(a, out_t.shape, pool)
                    gb = pool.acquire_dirty(out_t.numel)
                    for i in range(out_t.numel):
                        gb[i] = g[i] * a_full[i]
                    red, red_owned = _reduce_grad_into(gb, out_t.shape,
                                                       b.shape, pool)
                    self._accum(ins[1], red)
                    if red_owned:
                        pool.release(red)
                    pool.release(gb)
                    if a_owned:
                        pool.release(a_full)
            elif op == OP_MATMUL:
                a, b = values[ins[0]], values[ins[1]]
                m, k = a.shape
                _, n = b.shape
                if a.requires_grad:
                    bt = pool.acquire_dirty(n * k)
                    _transpose_into(b._data, k, n, bt)
                    tmp = pool.acquire_dirty(m * k)
                    matmul_rows_into(g, bt, m, n, k, tmp)
                    self._accum(ins[0], tmp)
                    pool.release(tmp)
                    pool.release(bt)
                if b.requires_grad:
                    at = pool.acquire_dirty(k * m)
                    _transpose_into(a._data, m, k, at)
                    tmp = pool.acquire_dirty(k * n)
                    matmul_rows_into(at, g, k, m, n, tmp)
                    self._accum(ins[1], tmp)
                    pool.release(tmp)
                    pool.release(at)
            elif op == OP_RELU:
                v = values[ins[0]]
                if v.requires_grad:
                    mask = saved
                    ga = pool.acquire_dirty(v.numel)
                    for i in range(v.numel):
                        ga[i] = g[i] if mask[i] else 0.0
                    self._accum(ins[0], ga)
                    pool.release(ga)
            elif op == OP_SUM:
                v = values[ins[0]]
                if v.requires_grad:
                    gv = float(g[0])
                    buf = pool.acquire_dirty(v.numel)
                    for i in range(v.numel):
                        buf[i] = gv
                    self._accum(ins[0], buf)
                    pool.release(buf)
            elif op == OP_FUSED_LINEAR_RELU:
                x, w, b = values[ins[0]], values[ins[1]], values[ins[2]]
                batch, in_f = x.shape
                out_f, _ = w.shape
                mask = saved
                d = pool.acquire_dirty(batch * out_f)
                for i in range(batch * out_f):
                    d[i] = g[i] if mask[i] else 0.0
                if b.requires_grad:
                    gb = pool.acquire(out_f)  # zeroed: rows are summed into it
                    for i in range(batch):
                        base = i * out_f
                        for j in range(out_f):
                            gb[j] += d[base + j]
                    self._accum(ins[2], gb)
                    pool.release(gb)
                if x.requires_grad:
                    tmp = pool.acquire_dirty(batch * in_f)
                    matmul_rows_into(d, w._data, batch, out_f, in_f, tmp)
                    self._accum(ins[0], tmp)
                    pool.release(tmp)
                if w.requires_grad:
                    dt = pool.acquire_dirty(out_f * batch)
                    _transpose_into(d, batch, out_f, dt)
                    tmp = pool.acquire_dirty(out_f * in_f)
                    matmul_rows_into(dt, x._data, out_f, batch, in_f, tmp)
                    self._accum(ins[1], tmp)
                    pool.release(tmp)
                    pool.release(dt)
                pool.release(d)
            else:
                raise ValueError(f"unknown op code {op}")


class CoreTensor:
    """A value in DarkCore: flat ``array('f')`` + shape + a Tape slot.

    Values are immutable by convention during a forward/backward pass
    (the optimizer updates leaf data in place only *after* backward).
    ``grad`` is the raw ``array('f')`` accumulated on this value's Tape,
    or None.
    """

    __slots__ = ("_data", "shape", "requires_grad", "_tape", "_idx")

    def __init__(self, data=None, shape=None, requires_grad: bool = False,
                 tape=None, idx: int = -1):
        if data is None:
            data = 0.0
        if isinstance(data, CoreTensor):
            flat = list(data._data)
            inferred = data.shape
        elif isinstance(data, array):
            flat = [float(v) for v in data]
            inferred = shape if shape is not None else (len(flat),)
        else:
            flat, inferred = _flatten(data)
            if inferred is None:
                inferred = shape
        if shape is not None:
            shape_t = tuple(shape)
            if _numel(shape_t) != len(flat):
                raise ValueError(
                    f"shape {shape_t} needs {_numel(shape_t)} elements, "
                    f"got {len(flat)}")
            inferred = shape_t
        self._data: array = array("f", flat)
        self.shape: tuple = tuple(inferred)
        self.requires_grad = bool(requires_grad)
        self._tape = tape
        self._idx = idx

    @classmethod
    def zeros(cls, shape, requires_grad: bool = False):
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        return cls([0.0] * _numel(shape), shape=shape,
                   requires_grad=requires_grad)

    @classmethod
    def randn(cls, shape, seed=None, requires_grad: bool = False):
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        rng = random.Random(seed)
        return cls([rng.gauss(0.0, 1.0) for _ in range(_numel(shape))],
                   shape=shape, requires_grad=requires_grad)

    @property
    def numel(self) -> int:
        return _numel(self.shape)

    @property
    def grad(self):
        if self._tape is None or self._idx < 0:
            return None
        return self._tape.grads[self._idx]

    def tolist(self) -> list:
        if self.shape == ():
            return float(self._data[0])
        if len(self.shape) == 1:
            return [float(x) for x in self._data]
        if len(self.shape) == 2:
            rows, cols = self.shape
            return [[float(self._data[r * cols + c]) for c in range(cols)]
                    for r in range(rows)]
        return [float(x) for x in self._data]

    def item(self) -> float:
        if self.numel != 1:
            raise ValueError(f"item() needs 1 element, got {self.numel}")
        return float(self._data[0])

    def detach(self):
        return CoreTensor(list(self._data), shape=self.shape)

    def zero_grad(self) -> None:
        if self._tape is not None and self._idx >= 0:
            self._tape.grads[self._idx] = None

    def backward(self) -> None:
        if self._tape is None:
            raise ValueError("backward() needs a value on a Tape — "
                             "create leaves with Tape.leaf()")
        self._tape.backward(self)

    # -- tape-aware ops -------------------------------------------------------
    def _tape_for(self, other):
        tape = self._tape or (other._tape if isinstance(other, CoreTensor) else None)
        if tape is None:
            return None, None
        tape._register(self)
        if isinstance(other, CoreTensor):
            tape._register(other)
        return tape, other

    @staticmethod
    def _as_core(x):
        return x if isinstance(x, CoreTensor) else CoreTensor(x)

    def __add__(self, other):
        other = self._as_core(other)
        a_flat, b_flat, shape = _broadcast_pair(self, other)
        out = array("f", [float(a) + float(b) for a, b in zip(a_flat, b_flat)])
        tape, _ = self._tape_for(other)
        if tape is None:
            return CoreTensor(out, shape)
        return tape._new(out, shape, OP_ADD, (self, other), None)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = self._as_core(other)
        a_flat, b_flat, shape = _broadcast_pair(self, other)
        out = array("f", [float(a) * float(b) for a, b in zip(a_flat, b_flat)])
        tape, _ = self._tape_for(other)
        if tape is None:
            return CoreTensor(out, shape)
        return tape._new(out, shape, OP_MUL, (self, other), None)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        other = self._as_core(other)
        if len(self.shape) != 2 or len(other.shape) != 2:
            raise ValueError("matmul only supports 2-d CoreTensors")
        m, k1 = self.shape
        k2, n = other.shape
        if k1 != k2:
            raise ValueError(f"matmul shape mismatch {self.shape} @ {other.shape}")
        out = matmul(self._data, other._data, m, k1, n)
        tape, _ = self._tape_for(other)
        if tape is None:
            return CoreTensor(out, (m, n))
        return tape._new(out, (m, n), OP_MATMUL, (self, other), None)

    def relu(self):
        out = array("f", bytes(4 * self.numel))
        mask = bytearray(self.numel)
        for i, v in enumerate(self._data):
            if v > 0.0:
                out[i] = v
                mask[i] = 1
        if self._tape is None:
            return CoreTensor(out, self.shape)
        self._tape._register(self)
        return self._tape._new(out, self.shape, OP_RELU, (self,), mask)

    def sum(self):
        out = array("f", [sum(self._data)])
        if self._tape is None:
            return CoreTensor(out, ())
        self._tape._register(self)
        return self._tape._new(out, (), OP_SUM, (self,), None)

    def __repr__(self) -> str:
        return (f"CoreTensor(shape={self.shape}, data={self.tolist()}, "
                f"requires_grad={self.requires_grad})")


def fused_linear_relu(x, w, b):
    """Tape-aware fused linear+ReLU: ``relu(x @ W.T + b)`` in one pass.

    ``x`` is (batch, in_f), ``w`` is (out_f, in_f), ``b`` is (out_f,).
    Uses the C kernel when built, else the pure-Python fused kernel.
    """
    batch, in_f = x.shape
    out_f, w_in = w.shape
    if w_in != in_f:
        raise ValueError(f"fused_linear_relu shape mismatch {x.shape} "
                         f"with weight {w.shape}")
    if C_AVAILABLE:
        out, mask = fused_linear_relu_c(x._data, w._data, b._data,
                                        batch, in_f, out_f)
    else:
        out, mask = fused_linear_relu_raw(x._data, w._data, b._data,
                                          batch, in_f, out_f)
    tape = x._tape or w._tape or b._tape
    if tape is None:
        return CoreTensor(out, (batch, out_f))
    for t in (x, w, b):
        tape._register(t)
    return tape._new(out, (batch, out_f), OP_FUSED_LINEAR_RELU,
                     (x, w, b), mask)


# -- fused optimizer step ------------------------------------------------------

def fused_sgd_step_py(params, grads, lr: float, momentum: float = 0.0,
                      weight_decay: float = 0.0, state=None):
    """Pure-Python fused SGD step (the fallback; see ``fused_sgd_step``).

    ``params`` / ``grads`` are parallel lists of ``array('f')`` (a grad
    entry may be None to skip that param). ``state`` is a parallel list
    of velocity ``array('f')`` (or None entries); pass None to create it,
    and keep the returned list across steps. Per element, in one pass:
    ``g += wd * p; v = momentum * v + g; p -= lr * v`` (with momentum 0
    the velocity is skipped: ``p -= lr * g``). Matches ``optim.SGD``.
    Returns the (possibly newly created) momentum state list.
    """
    lr = float(lr)
    mo = float(momentum)
    wd = float(weight_decay)
    if state is None:
        state = [None] * len(params)
    for pi, (p, g) in enumerate(zip(params, grads)):
        if g is None:
            continue
        n = len(p)
        if mo:
            v = state[pi]
            if v is None:
                v = array("f", bytes(4 * n))
                state[pi] = v
            if wd:
                for i in range(n):
                    v[i] = mo * v[i] + g[i] + wd * p[i]
                    p[i] -= lr * v[i]
            else:
                for i in range(n):
                    v[i] = mo * v[i] + g[i]
                    p[i] -= lr * v[i]
        else:
            if wd:
                for i in range(n):
                    p[i] -= lr * (g[i] + wd * p[i])
            else:
                for i in range(n):
                    p[i] -= lr * g[i]
    return state


def fused_sgd_step_c(params, grads, lr: float, momentum: float = 0.0,
                     weight_decay: float = 0.0, state=None):
    """Fused SGD step via the C core (raises if the C core is not built).

    Same arguments, same in-place semantics and same return value as
    ``fused_sgd_step_py``: ``params`` / ``grads`` are parallel lists of
    ``array('f')`` (a grad entry may be None to skip that param), and
    ``state`` is the parallel velocity list, created on first use when
    momentum is on. The C kernel updates each param in place and only
    returns new velocity bytes when it had to create them.
    """
    if not C_AVAILABLE:
        raise RuntimeError("C core not built — run scripts/build_c.py, "
                           "or use fused_sgd_step_py instead")
    if state is None:
        state = [None] * len(params)
    for pi, (p, g) in enumerate(zip(params, grads)):
        if g is None:
            continue
        if len(p) != len(g):
            raise ValueError(f"param {pi}: {len(p)} params vs "
                             f"{len(g)} grads")
        v = state[pi] if momentum else None
        created = _C.fused_sgd(p, g, v, float(lr), float(momentum),
                               float(weight_decay))
        if created is not None:
            vel = array("f")
            vel.frombytes(created)
            state[pi] = vel
    return state


def fused_sgd_step(params, grads, lr: float, momentum: float = 0.0,
                   weight_decay: float = 0.0, state=None):
    """Fused SGD step: the C core when built, else the pure-Python loop.

    Same arguments, same in-place semantics and same return value as
    ``fused_sgd_step_py``; the C kernel (``fused_sgd_step_c``) is used
    per param when the C core is available.
    """
    if C_AVAILABLE:
        return fused_sgd_step_c(params, grads, lr, momentum, weight_decay,
                                state)
    return fused_sgd_step_py(params, grads, lr, momentum, weight_decay,
                             state)


__all__ = [
    "OP_NONE", "OP_ADD", "OP_MUL", "OP_MATMUL", "OP_RELU", "OP_SUM",
    "OP_FUSED_LINEAR_RELU", "OP_NAMES", "DEFAULT_BLOCK",
    "C_AVAILABLE", "c_available",
    "BufferPool", "Tape", "CoreTensor",
    "matmul_tiled", "matmul_tiled_into", "matmul_rows", "matmul_rows_into",
    "matmul_c", "matmul",
    "fused_linear_relu_raw", "fused_linear_relu_c", "fused_linear_relu",
    "fused_sgd_step", "fused_sgd_step_py", "fused_sgd_step_c",
]
