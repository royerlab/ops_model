"""Config for the DiffAE (diffusion autoencoder) — DiffEx generator stage.

Semantic encoder -> z_sem; conditional UNet decoder denoises conditioned on z_sem
(injected into the time embedding). Trained jointly on a broad sample of phase
single-cell crops. Gate: round-trip (DDIM-invert -> reverse) reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..classifier.config import DEFAULT_OUT_ROOT, PMA_PHASE_GENEKO  # noqa: F401


@dataclass
class DiffAEConfig:
    # ---- data (locked 2026-06-16: broad sample, all classes incl NTC, all ranks) ----
    pma_parquet: str = PMA_PHASE_GENEKO
    n_crops: int = 50_000
    crop_size: int = 160
    channel: str = "Phase2D"
    mask_cell: bool = False
    seed: int = 0

    # ---- model ----
    cond_dim: int = 1024       # FROZEN CellDINO (ViT-L) embedding dim — the conditioning
    block_out_channels: tuple = (128, 256, 256, 512)
    layers_per_block: int = 2

    # ---- diffusion ----
    train_timesteps: int = 1000

    # ---- training: proper conditional-diffusion recipe ----
    # The v1 run had ~dead conditioning (emb/noise ratio 0.008). Fix = conditioning
    # dropout (forces the model to USE the embedding) + EMA + much longer training.
    epochs: int = 120              # resume across jobs to reach this
    batch_size: int = 48           # single-GPU (DataParallel breaks diffusers; DDP TODO)
    lr: float = 1e-4
    weight_decay: float = 0.0
    cond_dropout: float = 0.15     # replace embedding with learned null this often (enables CFG)
    ema_decay: float = 0.9995      # EMA weights used for sampling/eval
    resume: bool = True            # continue from saved train state if present
    # dihedral augmentation: randomly rot90×flip the TARGET image while keeping the
    # canonical CellDINO embedding fixed → teaches the model orientation is NOT
    # embedding-determined, so traversals stop spuriously rotating the cell.
    augment_dihedral: bool = True
    # affine augmentation: continuous rotation (±180°) + scale + flip with reflection
    # padding (no black corners), matching the contrastive data_loader recipe. Covers
    # arbitrary orientations (not just 90° steps); takes precedence over dihedral when set.
    augment_affine: bool = False
    affine_scale: float = 0.15     # ±fraction scale jitter
    init_ckpt: str | None = None   # warm-start: load these weights into the fresh model
    device: str = "cuda"
    num_workers: int = 0

    # ---- reconstruction gate ----
    recon_every: int = 5       # epochs between recon montages
    n_recon: int = 16          # cells in the montage
    ddim_steps: int = 50
