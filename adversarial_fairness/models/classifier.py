"""
Classifier neural network — Zhang et al. 2018 architecture.

The predictor outputs a scalar probability Ŷ = σ(f(X)).
The adversary then reads Ŷ directly (not any internal embedding),
which is the exact setup from Figure 1 of the paper.
"""

import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, n_features: int, n_hidden: int = 256, n_layers: int = 2):
        super().__init__()
        # `n_layers` hidden layers of `n_hidden` neurons each. The default (2 × 256)
        # is the FFB-matched tabular MLP (Appendix C) and reproduces every prior run
        # byte-for-byte; the interface's classifier picker can request other depths
        # (e.g. 3 × 256, 2 × 128) without touching the default path.
        n_layers = max(1, int(n_layers))
        layers, in_dim = [], n_features
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, n_hidden), nn.ReLU(), nn.Dropout(0.2)]
            in_dim = n_hidden
        layers += [nn.Linear(in_dim, 1)]
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns scalar prediction probability Ŷ ∈ (0,1), shape (N,1)."""
        return torch.sigmoid(self.network(x))
