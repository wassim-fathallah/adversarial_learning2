"""
Image classifier — ResNet-18 backbone.

Drop-in replacement for the tabular Classifier when the dataset is an image.
forward(x) returns the scalar prediction Ŷ = σ(f(X)) with shape (N, 1), exactly
like models/classifier.py. The adversary reads only Ŷ; the training loop is unchanged.

Accepts flat (N, C*H*W) or shaped (N, C, H, W) input.
Adapts conv1 for any channel count (e.g. grayscale UTKFace: 1 channel, 48x48).
"""

import torch
import torch.nn as nn


class ImageClassifier(nn.Module):
    def __init__(self, image_shape, n_hidden: int = 128):
        """image_shape: (C, H, W) of one image (used to reshape flat input)."""
        super().__init__()
        self.image_shape = tuple(image_shape)
        in_channels = self.image_shape[0]

        try:
            from torchvision.models import resnet18
        except ImportError:
            raise ImportError(
                "torchvision is required for image datasets. "
                "Activate .venv_gpu and fix the paging file, then retry."
            )
        backbone = resnet18(weights=None)
        if in_channels != 3:
            # ResNet18 default conv1 expects 3 channels — replace for grayscale etc.
            backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                                       padding=3, bias=False)
        backbone.fc = nn.Linear(512, 1)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: flat (N, C*H*W) or shaped (N, C, H, W). Returns Ŷ ∈ (0,1), shape (N,1)."""
        if x.dim() == 2:
            x = x.view(x.size(0), *self.image_shape)
        return torch.sigmoid(self.backbone(x))
