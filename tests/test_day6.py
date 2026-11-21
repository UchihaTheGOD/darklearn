"""Day 6 tests — plain asserts, runnable with pytest or directly:
    python3 tests/test_day6.py
    python3 -m pytest tests/test_day6.py -q

Covers: BPETokenizer (learn merges, encode/decode, save/load),
TextDataset.from_token_ids.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darklearn import BPETokenizer, DataLoader, TextDataset

CORPUS = ("the cat sat on the mat the dog ran to the mat "
          "the cat and the dog sat on the mat " * 10)


def test_bpe_learns_merges():
    tok = BPETokenizer().train(CORPUS, vocab_size=300, min_freq=2)
    assert tok.num_merges > 0
    assert len(tok) == 256 + tok.num_merges
    # Frequent words collapse to single tokens.
    the_ids = tok.encode("the")
    assert len(the_ids) == 1
    assert tok.vocab[the_ids[0]] == b"the"
    assert len(tok.encode("the cat sat")) < len("the cat sat")
    # A merge's bytes are the concatenation of its parts.
    for rank, (a, b) in enumerate(tok.merges):
        assert tok.vocab[256 + rank] == tok.vocab[a] + tok.vocab[b]
    # Merges never cross a word boundary: no merged token contains a space.
    for token_id, token_bytes in tok.vocab.items():
        if token_id >= 256:
            assert b" " not in token_bytes


def test_bpe_roundtrip():
    tok = BPETokenizer().train(CORPUS, vocab_size=300, min_freq=2)
    texts = [
        "the cat sat on the mat",
        "hello, world! 123",
        "tabs\tand\nnewlines\n\nand  double  spaces",
        "unicode: café ☕ naïve",
        "",
        "a",
    ]
    for text in texts:
        assert tok.decode(tok.encode(text)) == text
    # Untrained tokenizer (no merges) also round-trips: byte fallback.
    fresh = BPETokenizer()
    assert fresh.decode(fresh.encode("anything at all!?")) == \
        "anything at all!?"
    assert fresh.encode("abc") == [97, 98, 99]


def test_bpe_save_load():
    tok = BPETokenizer().train(CORPUS, vocab_size=300, min_freq=2)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bpe.json"
        tok.save(path)
        loaded = BPETokenizer.load(path)
    assert len(loaded) == len(tok)
    assert loaded.merges == tok.merges
    assert loaded.vocab == tok.vocab
    sample = "the cat sat on the mat"
    assert loaded.encode(sample) == tok.encode(sample)
    assert loaded.decode(loaded.encode(sample)) == sample
    # Bad format file is rejected.
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.json"
        bad.write_text('{"format": "not-bpe"}', encoding="utf-8")
        try:
            BPETokenizer.load(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("loading a non-BPE file should raise")


def test_bpe_vocab_size_limit():
    # Cap respected: vocab never exceeds the requested size.
    tok = BPETokenizer().train(CORPUS, vocab_size=260, min_freq=2)
    assert len(tok) == 260
    assert tok.num_merges == 4
    # min_freq stops learning early: a rare pair never merges.
    rare = BPETokenizer().train("abc xyz", vocab_size=512, min_freq=5)
    assert rare.num_merges == 0
    assert len(rare) == 256


def test_bpe_deterministic():
    a = BPETokenizer().train(CORPUS, vocab_size=300, min_freq=2)
    b = BPETokenizer().train(CORPUS, vocab_size=300, min_freq=2)
    assert a.merges == b.merges
    assert a.vocab == b.vocab
    assert a.encode("the cat sat") == b.encode("the cat sat")


def test_bpe_errors():
    try:
        BPETokenizer().train(CORPUS, vocab_size=100)
    except ValueError:
        pass
    else:
        raise AssertionError("vocab_size < 256 should raise")
    try:
        BPETokenizer().train(CORPUS, min_freq=0)
    except ValueError:
        pass
    else:
        raise AssertionError("min_freq < 1 should raise")
    tok = BPETokenizer()
    try:
        tok.encode(123)
    except TypeError:
        pass
    else:
        raise AssertionError("encode(non-str) should raise")
    try:
        tok.decode([999])
    except ValueError:
        pass
    else:
        raise AssertionError("decode(unknown id) should raise")


def test_text_dataset_from_token_ids():
    tok = BPETokenizer().train(CORPUS, vocab_size=300, min_freq=2)
    ids = tok.encode("the cat sat on the mat " * 10)
    ds = TextDataset.from_token_ids(ids, seq_len=8)
    assert ds.vocab_size == max(ids) + 1
    assert len(ds) == len(ids) - 8
    x, y = ds[0]
    assert len(x) == 8 and isinstance(y, int)
    assert x == ids[:8] and y == ids[8]
    # Fits the existing DataLoader: (batch, seq_len) x, (batch, 1) y.
    loader = DataLoader(ds, batch_size=4)
    xb, yb = next(iter(loader))
    assert xb.shape == (4, 8) and yb.shape == (4, 1)
    # decode/encode are the tokenizer's job in this mode.
    try:
        ds.decode([1, 2])
    except ValueError:
        pass
    else:
        raise AssertionError("token-id decode should raise")
    # Errors: too few ids, bad seq_len / stride.
    for bad in (dict(ids=[1, 2], seq_len=8),
                dict(ids=ids, seq_len=0),
                dict(ids=ids, seq_len=8, stride=0)):
        try:
            TextDataset.from_token_ids(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"from_token_ids({bad}) should raise")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All Day 6 tests passed.")
