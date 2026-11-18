"""darklearn — pure-stdlib Python library for training ML and LLM models.

No external dependencies. Tensor + autograd + nn + optim + data.
Day 2: 0.1.0-day2. See docs/ROADMAP.md for the 6–7 month plan.
"""

from .tensor import Tensor
from .autograd import Value, backward, gradcheck
from .nn import (Module, Linear, ReLU, Tanh, Sigmoid, GELU, Sequential,
                 MSELoss, mse_loss, CrossEntropyLoss)
from .optim import Optimizer, SGD, Adam, StepLR, CosineAnnealingLR, clip_grad_norm_
from .data import Dataset, TensorDataset, DataLoader

__version__ = "0.1.0-day2"

__all__ = [
    "__version__",
    "Tensor", "Value", "backward", "gradcheck",
    "Module", "Linear", "ReLU", "Tanh", "Sigmoid", "GELU", "Sequential",
    "MSELoss", "mse_loss", "CrossEntropyLoss",
    "Optimizer", "SGD", "Adam", "StepLR", "CosineAnnealingLR", "clip_grad_norm_",
    "Dataset", "TensorDataset", "DataLoader",
]
