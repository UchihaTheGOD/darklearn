"""Day 8 tests — plain asserts, runnable with pytest or directly:
    python3 tests/test_day8.py
    python3 -m pytest tests/test_day8.py -q

Covers: Tensor.narrow (rows/cols, grads), scaled dot-product
attention (forward values, causal mask, grads), MultiHeadAttention
(head split, causal behaviour, grads), TransformerBlock (shapes,
residual + causal behaviour, grads), and a tiny GPT-style training
smoke test.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darklearn import (
    Adam, Embedding, LayerNorm, Linear, MSELoss,
    MultiHeadAttention, TransformerBlock,
    scaled_dot_product_attention,
    sinusoidal_positional_encoding,
)
from darklearn.autograd import gradcheck
from darklearn.tensor import Tensor


def _close(a, b, tol=1e-4):
    return abs(a - b) < tol


def test_narrow_rows_cols():
    x = Tensor([[1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0]], shape=(3, 4))
    assert x.narrow(0, 1, 2).tolist() == [[5.0, 6.0, 7.0, 8.0],
                                          [9.0, 10.0, 11.0, 12.0]]
    assert x.narrow(1, 1, 2).tolist() == [[2.0, 3.0],
                                          [6.0, 7.0],
                                          [10.0, 11.0]]
    assert x.narrow(1, 0, 4).tolist() == x.tolist()
    assert x.narrow(0, 2, 0).shape == (0, 4)
    # 1-d case.
    v = Tensor([10.0, 20.0, 30.0, 40.0], shape=(4,))
    assert v.narrow(0, 1, 2).tolist() == [20.0, 30.0]
    # Out-of-range slices are rejected.
    for bad in (lambda: x.narrow(2, 0, 1),
                lambda: x.narrow(0, 2, 2),
                lambda: x.narrow(1, 3, 2),
                lambda: x.narrow(0, -1, 1),
                lambda: v.narrow(1, 0, 1)):
        try:
            bad()
        except (ValueError, IndexError):
            pass
        else:
            raise AssertionError("expected ValueError/IndexError")


def test_narrow_grad():
    x = Tensor([[1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0]], shape=(2, 4), requires_grad=True)
    # Column slice: grad scatters back to columns 1..2 only.
    y = x.narrow(1, 1, 2)
    y.sum().backward()
    assert x.grad.tolist() == [[0.0, 1.0, 1.0, 0.0],
                               [0.0, 1.0, 1.0, 0.0]]
    x.zero_grad()
    # Row slice: grad scatters back to row 1 only.
    z = x.narrow(0, 1, 1)
    (z * 2.0).sum().backward()
    assert x.grad.tolist() == [[0.0, 0.0, 0.0, 0.0],
                               [2.0, 2.0, 2.0, 2.0]]
    # 1-d narrow matches numeric grads.
    v = Tensor([1.0, -2.0, 3.0, 0.5], shape=(4,), requires_grad=True)
    assert gradcheck(lambda t: (t.narrow(0, 1, 2) ** 2).sum(), v)


def test_sdpa_forward():
    q = Tensor([[1.0, 0.0], [0.0, 1.0]], shape=(2, 2))
    k = Tensor([[1.0, 0.0], [0.0, 1.0]], shape=(2, 2))
    v = Tensor([[1.0, 2.0], [3.0, 4.0]], shape=(2, 2))
    out = scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 2)
    # scores = [[1/sqrt(2), 0], [0, 1/sqrt(2)]]; softmax by hand.
    s = 1.0 / math.sqrt(2.0)
    e = math.exp(s)
    w = e / (e + 1.0)
    expected = [[w * 1.0 + (1 - w) * 3.0, w * 2.0 + (1 - w) * 4.0],
                [(1 - w) * 1.0 + w * 3.0, (1 - w) * 2.0 + w * 4.0]]
    got = out.tolist()
    for r in range(2):
        for c in range(2):
            assert _close(got[r][c], expected[r][c]), (got, expected)
    # Bad shapes are rejected.
    try:
        scaled_dot_product_attention(q, Tensor([[1.0]], shape=(1, 1)), v)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for mismatched k")


def test_sdpa_causal():
    q = Tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], shape=(3, 2))
    k = Tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], shape=(3, 2))
    v = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], shape=(3, 2))
    out = scaled_dot_product_attention(q, k, v, causal=True)
    # Row 0 can only attend to itself, so it returns v row 0 exactly.
    assert out.tolist()[0] == [1.0, 2.0]
    # Row 1 must differ from the non-causal version (which sees row 2).
    plain = scaled_dot_product_attention(q, k, v, causal=False)
    assert out.tolist()[1] != plain.tolist()[1]
    # Last row sees everything, so it matches the non-causal version.
    assert _close(out.tolist()[2][0], plain.tolist()[2][0])
    assert _close(out.tolist()[2][1], plain.tolist()[2][1])


def test_sdpa_grad():
    q = Tensor([[0.5, -0.5], [1.0, 0.0]], shape=(2, 2), requires_grad=True)
    k = Tensor([[1.0, 0.5], [-1.0, 1.0]], shape=(2, 2), requires_grad=True)
    v = Tensor([[2.0, 1.0], [0.0, -1.0]], shape=(2, 2), requires_grad=True)
    out = scaled_dot_product_attention(q, k, v)
    out.sum().backward()
    for t in (q, k, v):
        assert t.grad is not None
        assert all(math.isfinite(float(x)) for x in t.grad._data)
    # Numeric check on one q element by finite differences.
    analytic = float(q.grad._data[0])
    eps = 1e-3
    orig = float(q._data[0])
    q._data[0] = orig + eps
    plus = scaled_dot_product_attention(q, k, v).sum().item()
    q._data[0] = orig - eps
    minus = scaled_dot_product_attention(q, k, v).sum().item()
    q._data[0] = orig
    assert _close(analytic, (plus - minus) / (2 * eps), tol=1e-2)


def test_mha_forward():
    mha = MultiHeadAttention(8, 2, seed=3)
    x = Tensor.randn((5, 8), seed=11)
    out = mha(x)
    assert out.shape == (5, 8)
    # Same seed -> identical weights -> identical output.
    mha2 = MultiHeadAttention(8, 2, seed=3)
    assert mha2(x).tolist() == out.tolist()
    # One head works too, and 1 head == d_head == d_model.
    assert MultiHeadAttention(4, 1, seed=1)(
        Tensor.randn((3, 4), seed=2)).shape == (3, 4)
    # d_model must be divisible by num_heads.
    try:
        MultiHeadAttention(8, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for 8 % 3")
    try:
        mha(Tensor.randn((5, 7), seed=0))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for wrong features")


def test_mha_causal():
    mha = MultiHeadAttention(8, 2, seed=5, causal=True)
    x1 = Tensor.randn((4, 8), seed=21)
    x2 = Tensor(x1.tolist(), shape=(4, 8))
    # Change only rows 1.. — row 0 of the output must not move.
    for c in range(8):
        x2._data[1 * 8 + c] += 5.0
        x2._data[3 * 8 + c] -= 2.0
    out1 = mha(x1).tolist()
    out2 = mha(x2).tolist()
    assert out1[0] == out2[0]
    assert out1[1] != out2[1]
    # Without the mask the same change DOES move row 0.
    mha_nc = MultiHeadAttention(8, 2, seed=5, causal=False)
    assert mha_nc(x1).tolist()[0] != mha_nc(x2).tolist()[0]


def test_mha_grad():
    mha = MultiHeadAttention(8, 2, seed=9)
    x = Tensor.randn((4, 8), seed=31, requires_grad=True)
    out = mha(x)
    out.sum().backward()
    assert x.grad is not None
    for p in mha.parameters():
        assert p.grad is not None, "every projection must get a grad"
        assert all(math.isfinite(float(v)) for v in p.grad._data)
    # 8 params: q/k/v/out weights + biases.
    assert len(mha.parameters()) == 8


def test_block_forward():
    block = TransformerBlock(8, 2, seed=4)
    x = Tensor.randn((5, 8), seed=41)
    out = block(x)
    assert out.shape == (5, 8)
    # No dropout by default: eval mode changes nothing.
    block.eval()
    assert block(x).tolist() == out.tolist()
    block.train()
    assert block(x).tolist() == out.tolist()
    # Residual: with tiny weights the block output stays near the input.
    assert all(abs(a - b) < 3.0
               for ra, rb in zip(out.tolist(), x.tolist())
               for a, b in zip(ra, rb))


def test_block_causal_and_grad():
    block = TransformerBlock(8, 2, seed=6, causal=True)
    x1 = Tensor.randn((4, 8), seed=51, requires_grad=True)
    x2 = Tensor(x1.tolist(), shape=(4, 8))
    for c in range(8):
        x2._data[2 * 8 + c] += 4.0
    out1 = block(x1)
    out2 = block(x2)
    assert out1.tolist()[0] == out2.tolist()[0]
    assert out1.tolist()[1] == out2.tolist()[1]
    assert out1.tolist()[2] != out2.tolist()[2]
    out1.sum().backward()
    assert x1.grad is not None
    for p in block.parameters():
        assert p.grad is not None
    # Params: attn 8 + ln1 2 + ln2 2 + mlp 4 = 16.
    assert len(block.parameters()) == 16


def test_tiny_gpt_trains():
    from darklearn import CrossEntropyLoss
    ids = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    emb = Embedding(4, 8, seed=2)
    block = TransformerBlock(8, 2, seed=3, causal=True)
    head = Linear(8, 4, seed=4)
    loss_fn = CrossEntropyLoss()
    params = emb.parameters() + block.parameters() + head.parameters()
    opt = Adam(params, lr=0.05)
    first = None
    for step in range(60):
        x = emb(ids[:-1]) + sinusoidal_positional_encoding(len(ids) - 1, 8)
        logits = head(block(x))
        loss = loss_fn(logits, ids[1:])
        if step == 0:
            first = loss.item()
        for p in params:
            p.zero_grad()
        loss.backward()
        opt.step()
    assert first is not None
    assert loss.item() < first * 0.5, (first, loss.item())
    assert loss.item() < 1.0, loss.item()


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All Day 8 tests passed.")
