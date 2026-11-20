"""Day 5 tests — plain asserts, runnable with pytest or directly:
    python3 tests/test_day5.py
    python3 -m pytest tests/test_day5.py -q

Covers: split_dataset, Dataset.map, Normalize, TextDataset.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darklearn import (Tensor, TensorDataset, split_dataset, Normalize,
                       TextDataset)


def test_text_dataset():
    ds = TextDataset("abca", seq_len=2)
    assert ds.vocab == ["a", "b", "c"]
    assert ds.stoi == {"a": 0, "b": 1, "c": 2}
    assert len(ds) == 2  # starts 0, 1 ("ab"->"c", "bc"->"a")
    assert ds[0] == ([0, 1], 2)
    assert ds[1] == ([1, 2], 0)
    # encode/decode round-trip
    assert ds.encode("cab") == [2, 0, 1]
    assert ds.decode([2, 0, 1]) == "cab"
    assert ds.decode(ds.encode("abca")) == "abca"
    # stride skips starts
    ds2 = TextDataset("abcdefgh", seq_len=2, stride=2)
    assert len(ds2) == 3  # starts 0, 2, 4
    assert ds2[2] == (ds2.encode("ef"), ds2.stoi["g"])
    # errors
    for bad in (dict(text="", seq_len=2), dict(text="ab", seq_len=2),
                dict(text="abc", seq_len=0), dict(text="abc", seq_len=2,
                                                  stride=0)):
        try:
            TextDataset(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"TextDataset({bad}) should raise")


def test_normalize():
    norm = Normalize(mean=[0.0, 10.0], std=[1.0, 5.0])
    assert norm([1.0, 15.0]) == [1.0, 1.0]
    assert norm([0.0, 10.0]) == [0.0, 0.0]
    assert norm([-1.0, 0.0]) == [-1.0, -2.0]
    # scalar mean/std broadcasts to every feature
    assert Normalize(2.0, 2.0)([4.0, 6.0]) == [1.0, 2.0]
    # Tensor in -> Tensor out, same shape
    t = Normalize([1.0, 2.0], [2.0, 2.0])(Tensor([[3.0, 4.0], [5.0, 6.0]]))
    assert isinstance(t, Tensor) and t.shape == (2, 2)
    assert t.tolist() == [[1.0, 1.0], [2.0, 2.0]]
    # from_dataset: mean 2, population std sqrt(8/3) on [[0],[2],[4]]
    ds = TensorDataset([[0.0], [2.0], [4.0]], [0.0, 0.0, 0.0])
    fitted = Normalize.from_dataset(ds)
    assert abs(fitted.mean[0] - 2.0) < 1e-9
    assert abs(fitted.std[0] - (8.0 / 3.0) ** 0.5) < 1e-6
    out = fitted([2.0])
    assert abs(out[0]) < 1e-9
    # zero-variance feature does not divide by zero
    flat = TensorDataset([[5.0, 1.0], [5.0, 3.0]], [0.0, 0.0])
    fitted2 = Normalize.from_dataset(flat)
    assert fitted2([5.0, 2.0]) == [0.0, 0.0]
    # errors
    for bad in (dict(mean=[0.0], std=[0.0]),
                dict(mean=[0.0, 0.0], std=[1.0])):
        try:
            Normalize(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Normalize({bad}) should raise")
    try:
        Normalize([0.0, 0.0], [1.0, 1.0])([1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("wrong row length should raise")


def test_split_dataset():
    ds = TensorDataset([[i, i + 1] for i in range(10)], list(range(10)))
    train, val = split_dataset(ds, val_fraction=0.2, seed=0)
    assert len(train) == 8 and len(val) == 2
    # disjoint and covering: every index appears exactly once
    assert sorted(train.indices + val.indices) == list(range(10))
    # same seed -> same split; different seed -> (almost surely) different
    train2, val2 = split_dataset(ds, val_fraction=0.2, seed=0)
    assert train2.indices == train.indices
    train3, _ = split_dataset(ds, val_fraction=0.2, seed=1)
    assert train3.indices != train.indices
    # lengths form
    t, v = split_dataset(ds, lengths=(6, 4), seed=2)
    assert len(t) == 6 and len(v) == 4
    assert sorted(t.indices + v.indices) == list(range(10))
    # val_fraction rounds and clamps: 10 * 0.05 -> 1, not 0
    _, v_small = split_dataset(ds, val_fraction=0.05, seed=0)
    assert len(v_small) == 1
    # errors
    for bad in (dict(lengths=(5, 4)), dict(lengths=(5,)),
                dict(val_fraction=0.0), dict(val_fraction=1.0),
                dict(val_fraction=-0.1), dict()):
        try:
            split_dataset(ds, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"split_dataset({bad}) should raise")


def test_dataset_map():
    ds = TensorDataset([[1.0, 2.0], [3.0, 4.0]], [10.0, 20.0])
    doubled = ds.map(lambda item: ([2 * x for x in item[0]], item[1]))
    assert len(doubled) == 2
    assert doubled[0] == ([2.0, 4.0], 10.0)
    assert doubled[1] == ([6.0, 8.0], 20.0)
    # original untouched
    assert ds[0] == ([1.0, 2.0], 10.0)
    # map composes: map of a map
    shifted = doubled.map(lambda item: ([x + 1 for x in item[0]], item[1]))
    assert shifted[0] == ([3.0, 5.0], 10.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All Day 5 tests passed.")
