"""Train a tiny MLP (2 -> 8 -> 1) on synthetic data with darklearn.

Target: y = x0 + 2*x1. Full-batch Adam, 200 steps. Loss must go down.
Run:  python3 examples/train_mlp.py   (from repo root)
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darklearn import Tensor, Sequential, Linear, ReLU, Adam, MSELoss


def main() -> float:
    rng = random.Random(0)
    xs = [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(64)]
    ys = [x[0] + 2.0 * x[1] for x in xs]
    xt = Tensor(xs, shape=(64, 2))
    yt = Tensor(ys, shape=(64, 1))

    model = Sequential(Linear(2, 8, seed=1), ReLU(), Linear(8, 1, seed=2))
    opt = Adam(model.parameters(), lr=0.05)
    loss_fn = MSELoss()

    first = loss_fn(model(xt), yt).item()
    print(f"step 0    loss={first:.4f}")
    for step in range(1, 201):
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"step {step:<4} loss={loss.item():.4f}")
    final = loss_fn(model(xt), yt).item()
    print(f"final loss={final:.4f} (started {first:.4f})")
    assert final < first, "loss did not decrease"
    assert final < 0.05, f"final loss too high: {final}"
    print("OK — loss decreased.")
    return final


if __name__ == "__main__":
    main()
