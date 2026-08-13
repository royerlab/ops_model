"""K direction MLPs. Each maps a CellDINO embedding z -> a unit direction d_k(z).
An edit is z_new = z + alpha * d_k(z)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionBank(nn.Module):
    def __init__(self, dim: int, K: int = 10, hidden: int = 512):
        super().__init__()
        self.K = K
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, dim))
            for _ in range(K)
        ])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z (B,D) -> (K,B,D) unit directions."""
        return torch.stack([F.normalize(m(z), dim=-1) for m in self.mlps], dim=0)

    def direction(self, z: torch.Tensor, k: int) -> torch.Tensor:
        """Unit direction for one MLP: (B,D)."""
        return F.normalize(self.mlps[k](z), dim=-1)
