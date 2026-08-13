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
        # attention at the two lowest-resolution stages (kept — this arch produces the
        # promising morphs; speed comes from multi-GPU, not from shrinking the model)
        down = tuple("AttnDownBlock2D" if i >= n_blocks - 2 else "DownBlock2D"
                     for i in range(n_blocks))
        up = tuple("AttnUpBlock2D" if i < 2 else "UpBlock2D" for i in range(n_blocks))
        # spatial conditioning: concat the cond image into the input (noisy target + cond = 2 ch)
        self.spatial_cond = getattr(cfg, "spatial_cond", False)
        in_channels = 2 if self.spatial_cond else 1
        self.unet = UNet2DModel(
            sample_size=cfg.crop_size, in_channels=in_channels, out_channels=1,
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
        # multi-marker: learned per-marker embedding added to the conditioning (which channel to render)
        self.n_markers = getattr(cfg, "n_markers", 0)
        if self.n_markers:
            self.marker_emb = nn.Embedding(self.n_markers, cfg.cond_dim)

    def cond(self, emb: torch.Tensor, marker_id=None) -> torch.Tensor:
        """Frozen CellDINO embedding (B, cond_dim) -> time-embedding conditioning.
        marker_id (B,) long: add the learned marker embedding (multi-marker virtual staining)."""
        if self.n_markers and marker_id is not None:
            emb = emb + self.marker_emb(marker_id)
        return self.cond_proj(emb)

    def null(self, n: int, device) -> torch.Tensor:
        return self.null_emb[None].expand(n, -1).to(device)

    def denoise(self, noisy, t, c, cond_img=None) -> torch.Tensor:
        x = torch.cat([noisy, cond_img], dim=1) if self.spatial_cond else noisy   # dense phase concat
        return self.unet(x, t, class_labels=c).sample

    def forward(self, noisy, t, emb, cond_img=None, marker_id=None):
        return self.denoise(noisy, t, self.cond(emb, marker_id), cond_img)
