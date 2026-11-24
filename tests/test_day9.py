"""Day 9 tests — plain asserts, runnable with pytest or directly:
    python3 tests/test_day9.py
    python3 -m pytest tests/test_day9.py -q

Covers: generate() (greedy, sampling, temperature, top-k, eos,
errors), the 1-head MultiHeadAttention gradcheck (numeric vs
analytic), and overfitting a tiny corpus until greedy generation
reproduces it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darklearn import (
    Adam, CrossEntropyLoss, Embedding, Linear, MultiHeadAttention,
    TransformerBlock, sinusoidal_positional_encoding,
)
from darklearn.autograd import gradcheck
from darklearn.generate import generate
from darklearn.tensor import Tensor


class _CycleModel:
    """Fake next-token model: logits peak at (last_id + 1) % vocab."""

    def __init__(self, vocab: int = 4, peak: float = 3.0):
        self.vocab = vocab
        self.peak = peak

    def __call__(self, ids):
        rows = []
        for t in ids:
            row = [0.0] * self.vocab
            row[(int(t) + 1) % self.vocab] = self.peak
            rows.append(row)
        flat = [x for row in rows for x in row]
        return Tensor(flat, shape=(len(ids), self.vocab))


class _FixedModel:
    """Fake next-token model: the same logits row at every position."""

    def __init__(self, logits):
        self.logits = [float(x) for x in logits]

    def __call__(self, ids):
        flat = list(self.logits) * len(ids)
        return Tensor(flat, shape=(len(ids), len(self.logits)))


def test_generate_greedy():
    model = _CycleModel(vocab=4)
    out = generate(model, [0], max_new_tokens=5)
    assert out == [0, 1, 2, 3, 0, 1], out
    # Zero new tokens returns the prompt unchanged (a copy, not an alias).
    prompt = [2, 3]
    out = generate(model, prompt, max_new_tokens=0)
    assert out == [2, 3] and out is not prompt
    assert prompt == [2, 3], "generate must not mutate the prompt"
    # Greedy ignores the seed: same output for any seed.
    assert generate(model, [1], max_new_tokens=4, seed=1) == \
        generate(model, [1], max_new_tokens=4, seed=99)


def test_generate_eos():
    model = _CycleModel(vocab=4)
    out = generate(model, [0], max_new_tokens=10, eos_id=2)
    assert out == [0, 1, 2], out
    # eos never reached: full length is generated.
    out = generate(model, [0], max_new_tokens=3, eos_id=99)
    assert out == [0, 1, 2, 3], out


def test_generate_sample_reproducible():
    model = _FixedModel([2.0, 1.0, 0.0, -1.0])
    a = generate(model, [0], max_new_tokens=8, method="sample", seed=7)
    b = generate(model, [0], max_new_tokens=8, method="sample", seed=7)
    assert a == b, (a, b)
    c = generate(model, [0], max_new_tokens=8, method="sample", seed=8)
    assert c != a, (a, c)
    assert all(0 <= t < 4 for t in a)


def test_generate_sample_low_temperature():
    # Near-zero temperature concentrates all mass on the argmax, so
    # sampling must behave like greedy for any seed.
    model = _FixedModel([3.0, 0.0, 0.0, 0.0])
    for seed in (1, 2, 3):
        out = generate(model, [0], max_new_tokens=5, method="sample",
                       temperature=0.01, seed=seed)
        assert out == [0, 0, 0, 0, 0, 0], out


def test_generate_top_k():
    # top_k=1 forces the argmax for every seed.
    model = _FixedModel([3.0, 2.0, 1.0, 0.0])
    for seed in (1, 2, 3, 4, 5):
        out = generate(model, [0], max_new_tokens=6, method="sample",
                       top_k=1, seed=seed)
        assert out == [0] * 7, out
    # top_k=2 keeps only tokens 0 and 1 (distinct logits, so the
    # threshold is exact); without top_k, tokens 2/3 show up.
    model2 = _FixedModel([0.0, -0.1, -0.2, -0.3])
    seen = set()
    for seed in range(5):
        seen.update(generate(model2, [0], max_new_tokens=10,
                             method="sample", top_k=2, seed=seed)[1:])
    assert seen <= {0, 1} and seen == {0, 1}, seen
    seen_all = set()
    for seed in range(5):
        seen_all.update(generate(model2, [0], max_new_tokens=10,
                                 method="sample", seed=seed)[1:])
    assert 2 in seen_all or 3 in seen_all, seen_all


def test_generate_errors():
    model = _CycleModel()
    for bad in (lambda: generate(model, [0], method="beam"),
                lambda: generate(model, [0], temperature=0.0),
                lambda: generate(model, [0], temperature=-1.0),
                lambda: generate(model, [0], top_k=0),
                lambda: generate(model, []),
                lambda: generate(model, [0], max_new_tokens=-1)):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")

    class _WrongShape:
        def __call__(self, ids):
            return Tensor([1.0, 2.0], shape=(2,))  # 1-d, not (seq, vocab)

    try:
        generate(_WrongShape(), [0], max_new_tokens=1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for wrong logits shape")


def test_mha_one_head_gradcheck():
    # Numeric vs analytic grads through a 1-head causal MHA: the
    # 1-d input is reshaped to (seq, d_model) inside the function,
    # so the existing gradcheck helper covers the whole module
    # (Q/K/V projections, per-head attention, output projection).
    seq, d_model = 2, 4
    mha = MultiHeadAttention(d_model, 1, seed=3, causal=True)
    x0 = Tensor.randn((seq, d_model), seed=5)
    flat = Tensor(list(x0._data), shape=(seq * d_model,),
                  requires_grad=True)
    assert gradcheck(
        lambda t: mha(t.reshape((seq, d_model))).sum(), flat)


def test_overfit_tiny_corpus():
    # Train the tiny GPT on one 5-token sentence until greedy
    # generation from its first two tokens reproduces it exactly.
    from darklearn import BPETokenizer
    TEXT = "the cat sat"
    tok = BPETokenizer()
    tok.train(TEXT * 4, vocab_size=320)
    ids = tok.encode(TEXT)
    assert tok.decode(ids) == TEXT

    d_model = 16
    emb = Embedding(tok.vocab_size, d_model, seed=1)
    blocks = [TransformerBlock(d_model, 2, seed=10 + i, causal=True)
              for i in range(2)]
    head = Linear(d_model, tok.vocab_size, seed=20)
    loss_fn = CrossEntropyLoss()
    params = (emb.parameters() + blocks[0].parameters()
              + blocks[1].parameters() + head.parameters())
    opt = Adam(params, lr=0.05)
    first = None
    for step in range(120):
        x = emb(ids[:-1]) + sinusoidal_positional_encoding(
            len(ids) - 1, d_model)
        for block in blocks:
            x = block(x)
        loss = loss_fn(head(x), ids[1:])
        if step == 0:
            first = loss.item()
        for p in params:
            p.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.01, (first, loss.item())

    def model_fn(seq_ids):
        x = emb(seq_ids) + sinusoidal_positional_encoding(
            len(seq_ids), d_model)
        for block in blocks:
            x = block(x)
        return head(x)

    gen = generate(model_fn, ids[:2], max_new_tokens=len(ids) - 2)
    assert gen == ids, (gen, ids)
    assert tok.decode(gen) == TEXT


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All Day 9 tests passed.")
