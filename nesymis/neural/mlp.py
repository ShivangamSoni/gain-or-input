"""
Path-A neural classifier.

Per the paper this is a lightweight MLP over the fused multimodal representation
h = [v; u] (1024 -> 5). With hidden=None it is a single linear layer, matching
the paper exactly; an optional hidden layer is supported for experimentation.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class NeuralClassifier(nn.Module):
    """`mu`/`sd` are non-trainable buffers holding train-split feature statistics
    (identity 0/1 unless the trainer fills them), mirroring Classifier so
    the modality baselines and Path-A are trained by the same recipe."""

    def __init__(self, input_dim: int, num_classes: int, hidden: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.register_buffer("mu", torch.zeros(input_dim))
        self.register_buffer("sd", torch.ones(input_dim))
        if hidden is None:
            self.net = nn.Linear(input_dim, num_classes)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mu) / self.sd)
