"""The two candidate classifiers DiffEx will explain.

B: ResNet18 on single-cell phase crops (pixels in -> class logits).
C: MLP head on frozen CellDINO embeddings of the same crops.
Both 2-class (gene-vs-rest); extends cleanly to the full N-way softmax later.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_resnet(in_channels: int = 1, n_classes: int = 2) -> nn.Module:
    """ResNet18 with a 1-channel stem (phase) and a small head."""
    m = models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


class MLPHead(nn.Module):
    """Small MLP over CellDINO embeddings (option C)."""

    def __init__(self, in_dim: int, hidden: int = 256, n_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)
