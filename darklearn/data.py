"""darklearn.data — Dataset / DataLoader (pure stdlib, seeded shuffle)."""

from __future__ import annotations

import random

from .tensor import Tensor


class Dataset:
    def __len__(self) -> int:  # pragma: no cover
        raise NotImplementedError

    def __getitem__(self, idx):  # pragma: no cover
        raise NotImplementedError


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


__all__ = ["Dataset", "TensorDataset", "DataLoader"]
