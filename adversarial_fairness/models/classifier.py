"""
Classifier neural network — Zhang et al. 2018 architecture.

The predictor outputs a scalar probability Ŷ = σ(f(X)).
The adversary then reads Ŷ directly (not any internal embedding),
which is the exact setup from Figure 1 of the paper.
"""

import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, n_features: int, n_hidden: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, n_hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(n_hidden, n_hidden),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(n_hidden, n_hidden),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns scalar prediction probability Ŷ ∈ (0,1), shape (N,1)."""
        return torch.sigmoid(self.network(x))
