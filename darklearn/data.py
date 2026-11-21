"""darklearn.data — Dataset / DataLoader (pure stdlib, seeded shuffle)."""

from __future__ import annotations

import random

from .tensor import Tensor


class Dataset:
    def __len__(self) -> int:  # pragma: no cover
        raise NotImplementedError

    def __getitem__(self, idx):  # pragma: no cover
        raise NotImplementedError

    def map(self, fn):
        """Return a view of this dataset with ``fn`` applied to each item.

        ``fn`` receives the item ``__getitem__`` returns and its return
        value becomes the mapped item. The original dataset is untouched.
        """
        return MappedDataset(self, fn)


class MappedDataset(Dataset):
    """Dataset view applying a transform ``fn`` to each item."""

    def __init__(self, dataset: Dataset, fn):
        self.dataset = dataset
        self.fn = fn

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.fn(self.dataset[idx])


class Normalize:
    """Per-feature standardisation: ``(x - mean) / std``.

    ``mean`` / ``std`` are scalars or per-feature sequences (one value
    per feature). Works on a feature row (list of floats) and returns a
    list, or on a Tensor and returns a Tensor of the same shape.
    This is a data transform, not a graph op — no grads flow through it.
    """

    def __init__(self, mean, std):
        self.mean = ([float(mean)] if isinstance(mean, (int, float))
                     else [float(m) for m in mean])
        self.std = ([float(std)] if isinstance(std, (int, float))
                    else [float(s) for s in std])
        if any(s == 0.0 for s in self.std):
            raise ValueError("Normalize std must be non-zero")
        if len(self.mean) != len(self.std):
            raise ValueError("Normalize mean/std length mismatch")

    def __call__(self, x):
        if isinstance(x, Tensor):
            if len(x.shape) == 2:
                rows = x.tolist()
                flat = [v for row in rows for v in self._normalize_list(row)]
            else:
                flat = self._normalize_list([float(v) for v in x._data])
            return Tensor(flat, shape=x.shape)
        return self._normalize_list([float(v) for v in x])

    def _normalize_list(self, row):
        if len(self.mean) == 1:
            m, s = self.mean[0], self.std[0]
            return [(v - m) / s for v in row]
        if len(row) != len(self.mean):
            raise ValueError(
                f"row length {len(row)} != mean length {len(self.mean)}")
        return [(v - m) / s for v, m, s in zip(row, self.mean, self.std)]

    @classmethod
    def from_dataset(cls, dataset, eps=1e-8):
        """Build a Normalize from a dataset's per-feature mean/std.

        Expects items ``(feature_row, target)`` where the row is a list
        of floats (as ``TensorDataset`` returns). Uses the population
        std; a zero-variance feature gets std 1.0 (plus ``eps``) so it
        maps to 0 instead of dividing by zero.
        """
        n = len(dataset)
        if n == 0:
            raise ValueError("Normalize.from_dataset needs a non-empty dataset")
        rows = [dataset[i][0] for i in range(n)]
        width = len(rows[0])
        means = [sum(r[j] for r in rows) / n for j in range(width)]
        stds = []
        for j in range(width):
            var = sum((r[j] - means[j]) ** 2 for r in rows) / n
            stds.append(max(var ** 0.5, eps))
        return cls(means, stds)


class TextDataset(Dataset):
    """Character-level text dataset, the input for the tiny LM (Month 4).

    Vocabulary = sorted unique characters of ``text``. Item ``i`` is
    ``(input_ids, next_id)``: the ``seq_len`` characters starting at
    ``i * stride`` and the single character that follows them — i.e.
    next-character prediction, which fits the existing ``DataLoader``
    (x batch is (batch, seq_len), y batch is (batch, 1)).
    """

    def __init__(self, text: str, seq_len: int, stride: int = 1):
        if seq_len < 1:
            raise ValueError("TextDataset needs seq_len >= 1")
        if stride < 1:
            raise ValueError("TextDataset needs stride >= 1")
        if not text:
            raise ValueError("TextDataset needs non-empty text")
        if len(text) <= seq_len:
            raise ValueError(
                f"text length {len(text)} must exceed seq_len {seq_len}")
        self.text = text
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.vocab = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = list(self.vocab)
        self.ids = [self.stoi[ch] for ch in text]

    def __len__(self) -> int:
        # valid starts: 0, stride, ... while start <= len(ids) - seq_len - 1
        return (len(self.ids) - self.seq_len - 1) // self.stride + 1

    def __getitem__(self, idx):
        start = idx * self.stride
        return (self.ids[start:start + self.seq_len],
                self.ids[start + self.seq_len])

    @classmethod
    def from_token_ids(cls, ids, seq_len: int, stride: int = 1):
        """Build a TextDataset from pre-tokenized ids (e.g. BPE output).

        Same ``(input_ids, next_id)`` items as the character mode, but
        the ids come from a tokenizer instead of a character vocab.
        ``decode``/``encode`` are not available in this mode — decode
        with the tokenizer that produced the ids. ``vocab_size`` is
        ``max(ids) + 1`` (the tokenizer's vocabulary size when ids
        cover it), which is what an embedding layer needs.
        """
        if seq_len < 1:
            raise ValueError("TextDataset needs seq_len >= 1")
        if stride < 1:
            raise ValueError("TextDataset needs stride >= 1")
        ids = [int(i) for i in ids]
        if len(ids) <= seq_len:
            raise ValueError(
                f"{len(ids)} token ids must exceed seq_len {seq_len}")
        self = cls.__new__(cls)
        self.text = None
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.vocab = None
        self.stoi = None
        self.itos = None
        self.ids = ids
        self.vocab_size = max(ids) + 1
        return self

    def encode(self, text: str):
        if self.stoi is None:
            raise ValueError(
                "token-id TextDataset has no character vocab — "
                "encode with the tokenizer that produced the ids")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        if self.itos is None:
            raise ValueError(
                "token-id TextDataset has no character vocab — "
                "decode with the tokenizer that produced the ids")
        return "".join(self.itos[i] for i in ids)


class Subset(Dataset):
    """Dataset view over ``indices`` of another dataset."""

    def __init__(self, dataset: Dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def split_dataset(dataset, lengths=None, val_fraction=None, seed=None):
    """Split ``dataset`` into (train, val) Subsets with a seeded shuffle.

    Give either ``lengths=(train_len, val_len)`` (must sum to
    ``len(dataset)``) or ``val_fraction`` in (0, 1) — the val size is
    rounded, clamped to at least 1 and at most ``n - 1``. The same
    ``seed`` always gives the same split; the two subsets are disjoint
    and together cover the whole dataset.
    """
    n = len(dataset)
    if lengths is not None:
        if len(lengths) != 2:
            raise ValueError("lengths must be (train_len, val_len)")
        train_len, val_len = int(lengths[0]), int(lengths[1])
        if train_len < 0 or val_len < 0 or train_len + val_len != n:
            raise ValueError(
                f"lengths {lengths} must be non-negative and sum to {n}")
    elif val_fraction is not None:
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1)")
        val_len = int(round(n * val_fraction))
        val_len = max(1, min(n - 1, val_len))
        train_len = n - val_len
    else:
        raise ValueError("pass lengths=(train_len, val_len) "
                         "or val_fraction=...")
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    return (Subset(dataset, indices[:train_len]),
            Subset(dataset, indices[train_len:]))


class TensorDataset(Dataset):
    """Dataset holding (features, targets) as lists or Tensors.

    Features are stored as a list of rows; targets as a flat list.
    ``__getitem__`` returns (feature_row_list, target_value).
    """

    def __init__(self, features, targets):
        if isinstance(features, Tensor):
            if len(features.shape) == 2:
                self.features = features.tolist()
            elif len(features.shape) == 1:
                self.features = [[float(x)] for x in features._data]
            else:
                raise ValueError("TensorDataset needs 1-d or 2-d features")
        else:
            self.features = [[float(x) for x in row] if isinstance(row, (list, tuple))
                             else [float(row)] for row in features]
        if isinstance(targets, Tensor):
            self.targets = [float(x) for x in targets._data]
        else:
            self.targets = [float(x) for x in targets]
        if len(self.features) != len(self.targets):
            raise ValueError("features/targets length mismatch")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class DataLoader:
    """Yield (x_batch, y_batch) Tensors of size ``batch_size``.

    ``shuffle=True`` uses a seeded RNG so runs are reproducible.
    ``drop_last=False`` keeps a final smaller batch.
    """

    def __init__(self, dataset: Dataset, batch_size: int = 32,
                 shuffle: bool = False, seed: int | None = None,
                 drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = seed
        self.drop_last = bool(drop_last)

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed).shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            if self.drop_last and len(batch_idx) < self.batch_size:
                break
            rows = [self.dataset[i][0] for i in batch_idx]
            ys = [self.dataset[i][1] for i in batch_idx]
            x = Tensor(rows, shape=(len(rows), len(rows[0])))
            y = Tensor(ys, shape=(len(ys), 1))
            yield x, y


__all__ = ["Dataset", "MappedDataset", "Subset", "TensorDataset",
           "DataLoader", "split_dataset", "Normalize", "TextDataset"]
