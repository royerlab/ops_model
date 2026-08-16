"""Unsupervised direction-discovery losses (Alex's recipe).

InfoNCE: each direction MLP's outputs cluster together (consistent axis) and apart
from other MLPs' — so the K directions are distinct, consistent axes of variation.
Decorrelation: push the K mean directions toward orthogonal (VICReg-style covariance
term) so MLPs don't collapse onto the same axis.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def infonce_directions(dirs: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """dirs (K,B,D) unit vectors. Item (k,i); positives = same direction k."""
    K, B, D = dirs.shape
    x = dirs.reshape(K * B, D)
    n = K * B
    sim = (x @ x.t()) / tau
    eye = torch.eye(n, device=x.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)
    labels = torch.arange(K, device=x.device).repeat_interleave(B)
    pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye
    lse_all = torch.logsumexp(sim, dim=1)
    lse_pos = torch.logsumexp(sim.masked_fill(~pos, -1e9), dim=1)
    return -(lse_pos - lse_all).mean()


def decorrelation(dirs: torch.Tensor) -> torch.Tensor:
    """Penalize off-diagonal cosine similarity of the K mean directions."""
    m = F.normalize(dirs.mean(dim=1), dim=-1)  # (K,D)
    g = m @ m.t()                              # (K,K)
    K = g.shape[0]
    off = g - torch.diag(torch.diag(g))
    return (off ** 2).sum() / (K * (K - 1))
