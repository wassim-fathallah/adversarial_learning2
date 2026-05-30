"""
Adversary neural network — Zhang et al. 2018.

Takes the classifier's scalar output Ŷ ∈ (0,1) as input (shape N×1)
and tries to predict all sensitive attributes simultaneously.
One logit output per sensitive attribute. Loss: BCEWithLogitsLoss.
"""

import torch
import torch.nn as nn


class Adversary(nn.Module):
    def __init__(self, n_sensitive: int, n_hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, n_hidden),       # input is the scalar prediction Ŷ
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_sensitive),  # one logit per sensitive attr
        )

    def forward(self, y_hat: torch.Tensor) -> torch.Tensor:
        """y_hat: (N, 1) prediction probabilities. Returns logits (N, n_sensitive)."""
        return self.net(y_hat)
