"""darklearn — pure-stdlib Python library for training ML and LLM models.

No external dependencies. Tensor + autograd + nn + optim + data, plus
DarkCore (darkcore.py): our own tiled/fused/tape engine with an
optional C core (built by scripts/build_c.py, never required).
Day 3: 0.1.0-day3. See docs/ROADMAP.md for the 6–7 month plan.
"""

from .tensor import Tensor
from .autograd import Value, backward, gradcheck
from .nn import (Module, Linear, ReLU, Tanh, Sigmoid, GELU, Sequential,
                 MSELoss, mse_loss, CrossEntropyLoss)
from .optim import Optimizer, SGD, Adam, StepLR, CosineAnnealingLR, clip_grad_norm_
from .data import Dataset, TensorDataset, DataLoader
from .darkcore import (Tape, CoreTensor, BufferPool, matmul_tiled,
                       matmul_rows, matmul, fused_linear_relu,
                       fused_sgd_step, C_AVAILABLE, c_available)

__version__ = "0.1.0-day3"

__all__ = [
    "__version__",
    "Tensor", "Value", "backward", "gradcheck",
    "Module", "Linear", "ReLU", "Tanh", "Sigmoid", "GELU", "Sequential",
    "MSELoss", "mse_loss", "CrossEntropyLoss",
    "Optimizer", "SGD", "Adam", "StepLR", "CosineAnnealingLR", "clip_grad_norm_",
    "Dataset", "TensorDataset", "DataLoader",
    "Tape", "CoreTensor", "BufferPool", "matmul_tiled", "matmul_rows",
    "matmul", "fused_linear_relu", "fused_sgd_step",
    "C_AVAILABLE", "c_available",
]
