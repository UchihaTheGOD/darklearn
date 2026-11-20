"""Train/val split demo: train the MLP on 80%, report val loss.

Same synthetic task as examples/train_mlp.py (y = x0 + 2*x1), but the
data is split with darklearn.split_dataset and the model is evaluated
on the held-out 20% every 50 steps. Val loss should stay close to the
train loss — the split is random, so both come from the same task.
Run:  python3 examples/train_val_split.py   (from repo root)
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darklearn import (Tensor, Sequential, Linear, ReLU, Adam, MSELoss,
                       TensorDataset, split_dataset)


def _to_tensors(ds):
    rows = [ds[i][0] for i in range(len(ds))]
    ys = [ds[i][1] for i in range(len(ds))]
    return (Tensor(rows, shape=(len(rows), len(rows[0]))),
            Tensor(ys, shape=(len(ys), 1)))


def main():
    rng = random.Random(0)
    xs = [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(64)]
    ys = [x[0] + 2.0 * x[1] for x in xs]
    train_ds, val_ds = split_dataset(
        TensorDataset(xs, ys), val_fraction=0.25, seed=0)
    print(f"train={len(train_ds)} val={len(val_ds)}")
    xtr, ytr = _to_tensors(train_ds)
    xva, yva = _to_tensors(val_ds)

    model = Sequential(Linear(2, 8, seed=1), ReLU(), Linear(8, 1, seed=2))
    opt = Adam(model.parameters(), lr=0.05)
    loss_fn = MSELoss()

    def eval_loss(x, y):
        return loss_fn(model(x), y).item()

    first_tr = eval_loss(xtr, ytr)
    first_va = eval_loss(xva, yva)
    print(f"step 0    train={first_tr:.4f} val={first_va:.4f}")
    for step in range(1, 201):
        opt.zero_grad()
        loss = loss_fn(model(xtr), ytr)
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"step {step:<4} train={eval_loss(xtr, ytr):.4f} "
                  f"val={eval_loss(xva, yva):.4f}")
    final_tr = eval_loss(xtr, ytr)
    final_va = eval_loss(xva, yva)
    print(f"final train={final_tr:.4f} (started {first_tr:.4f}) "
          f"val={final_va:.4f} (started {first_va:.4f})")
    assert final_tr < first_tr, "train loss did not decrease"
    assert final_tr < 0.05, f"final train loss too high: {final_tr}"
    assert final_va < 0.05, f"final val loss too high: {final_va}"
    print("OK — train and val loss both decreased.")
    return final_tr, final_va


if __name__ == "__main__":
    main()
