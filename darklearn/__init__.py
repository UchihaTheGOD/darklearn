"""darklearn — pure-stdlib Python library for training ML and LLM models.

No external dependencies. Tensor + autograd + nn + optim + data, plus
DarkCore (darkcore.py): our own tiled/fused/tape engine with an
optional C core (built by scripts/build_c.py, never required).
Day 3: 0.1.0-day3. Day 4: 0.1.0-day4. Day 5: 0.1.0-day5. Day 6: 0.1.0-day6. Day 7: 0.1.0-day7. Day 8: 0.1.0-day8. Day 9: 0.1.0-day9. Day 10: 0.1.0-day10. Day 11: 0.1.0-day11. See docs/ROADMAP.md for the 6–7 month plan.
"""

from .tensor import Tensor
from .autograd import Value, backward, gradcheck
from .generate import beam_search, generate
from .nn import (Module, Linear, ReLU, Tanh, Sigmoid, GELU, Dropout,
                 Embedding, LayerNorm, sinusoidal_positional_encoding,
                 scaled_dot_product_attention, MultiHeadAttention,
                 TransformerBlock,
                 Sequential, MSELoss, mse_loss, CrossEntropyLoss, BCELoss)
from .optim import (Optimizer, SGD, Adam, AdamW, StepLR, CosineAnnealingLR,
                    clip_grad_norm_)
from .data import (Dataset, MappedDataset, Subset, TensorDataset,
                 DataLoader, split_dataset, Normalize, TextDataset)
from .tokenizer import BPETokenizer
from .trainer import Trainer, EarlyStopping
from .checkpoint import (load_checkpoint, load_state_dict,
                         load_tokenizer_state, save_checkpoint,
                         state_dict, tokenizer_state)
from .darkcore import (Tape, CoreTensor, BufferPool, matmul_tiled,
                       matmul_rows, matmul, fused_linear_relu,
                       fused_sgd_step, C_AVAILABLE, c_available)

__version__ = "0.1.0-day11"

__all__ = [
    "__version__",
    "Tensor", "Value", "backward", "gradcheck", "generate", "beam_search",
    "Module", "Linear", "ReLU", "Tanh", "Sigmoid", "GELU", "Dropout",
    "Embedding", "LayerNorm", "sinusoidal_positional_encoding",
    "scaled_dot_product_attention", "MultiHeadAttention",
    "TransformerBlock",
    "Sequential", "MSELoss", "mse_loss", "CrossEntropyLoss", "BCELoss",
    "Optimizer", "SGD", "Adam", "AdamW", "StepLR", "CosineAnnealingLR",
    "clip_grad_norm_",
    "Dataset", "MappedDataset", "Subset", "TensorDataset", "DataLoader",
    "split_dataset", "Normalize", "TextDataset",
    "BPETokenizer",
    "Trainer", "EarlyStopping",
    "state_dict", "load_state_dict", "tokenizer_state",
    "load_tokenizer_state", "save_checkpoint", "load_checkpoint",
    "Tape", "CoreTensor", "BufferPool", "matmul_tiled", "matmul_rows",
    "matmul", "fused_linear_relu", "fused_sgd_step",
    "C_AVAILABLE", "c_available",
]
