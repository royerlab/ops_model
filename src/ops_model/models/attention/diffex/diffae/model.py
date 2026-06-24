"""Conditional diffusion decoder (Alex's DiffEx design).

The UNet generates phase crops conditioned on the FROZEN CellDINO embedding —
NOT a jointly-trained encoder. The embedding is injected into the UNet's time
embedding (diffusers `class_embed_type="identity"`), which propagates FiLM-style
through the resnet blocks. Directions therefore live in CellDINO space, the same
space as the option-C classifier and the SetTransformer.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from diffusers import UNet2DModel


class DiffAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_blocks = len(cfg.block_out_channels)
        down = tuple("AttnDownBlock2D" if i >= n_blocks - 2 else "DownBlock2D"
                     for i in range(n_blocks))
        up = tuple("AttnUpBlock2D" if i < 2 else "UpBlock2D" for i in range(n_blocks))
        self.unet = UNet2DModel(
            sample_size=cfg.crop_size, in_channels=1, out_channels=1,
            block_out_channels=cfg.block_out_channels,
            layers_per_block=cfg.layers_per_block,
            down_block_types=down, up_block_types=up,
            class_embed_type="identity",
        )
        time_embed_dim = cfg.block_out_channels[0] * 4
        # project the frozen CellDINO embedding to the time-embedding dim. Deeper
        # MLP (not a single Linear) so the conditioning can actually be used.
        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cond_dim, time_embed_dim), nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        # learned null embedding for conditioning dropout / classifier-free guidance
        self.null_emb = nn.Parameter(torch.zeros(cfg.cond_dim))

    def cond(self, emb: torch.Tensor) -> torch.Tensor:
        """Frozen CellDINO embedding (B, cond_dim) -> time-embedding conditioning."""
        return self.cond_proj(emb)

    def null(self, n: int, device) -> torch.Tensor:
        return self.null_emb[None].expand(n, -1).to(device)

    def denoise(self, noisy, t, c) -> torch.Tensor:
        return self.unet(noisy, t, class_labels=c).sample

    def forward(self, noisy, t, emb):
        return self.denoise(noisy, t, self.cond(emb))
