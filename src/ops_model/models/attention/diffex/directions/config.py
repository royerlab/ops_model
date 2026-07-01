"""Config for Stage 3 — contrastive direction discovery + ranking + traversal.

Pipeline (Alex's recipe, our CellDINO space):
  2a  train K direction MLPs UNSUPERVISED on CellDINO embeddings (InfoNCE + decorrelation)
  2b  rank directions post-hoc by how much each shifts a control-vs-target classifier
  3   traverse the selected direction (α∈[-3,3]), DDIM-sample an image per step, verify score
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..classifier.config import DEFAULT_OUT_ROOT, GRAINS, PMA_PHASE_GENEKO  # noqa: F401


@dataclass
class DirConfig:
    # ---- target / data ----
    grain: str = "geneKO"          # geneKO | complex
    target: str = "HSPA5"          # the KD class to explain
    control: str = "NTC"           # control class
    n_per_class: int = 1000        # top-attention cells per class
    crop_size: int = 160
    channel: str = "Phase2D"
    mask_cell: bool = False
    seed: int = 0

    # ---- direction method (plan C) ----
    # 'mean_diff' | 'lr_weight' = deterministic supervised control→KD direction (PRIMARY,
    # reproducible). 'unsupervised' = the paper's InfoNCE direction bank (secondary track,
    # NOT reproducible run-to-run — GPU/seed sensitive + best_k flips).
    direction_method: str = "mean_diff"
    deterministic: bool = True     # seed + cuDNN-deterministic so runs are repeatable

    # ---- 2a: unsupervised direction discovery (only when direction_method='unsupervised') ----
    cond_dim: int = 1024           # CellDINO ViT-L embedding dim
    K: int = 10                    # number of candidate directions
    hidden: int = 512
    dir_epochs: int = 100
    dir_lr: float = 1e-3
    tau: float = 0.1               # InfoNCE temperature
    decorr_weight: float = 1.0     # push the K mean-directions orthogonal

    # ---- 2b: ranking ----
    rank_alpha: float = 1.0        # ± shift magnitude when measuring classifier score change

    # ---- 3: traversal ----
    # alphas are MULTIPLES of the control→KD embedding gap ‖μ_KD−μ_ctrl‖ when
    # scale_alpha_to_gap=True (α=+1 ≈ a full control→KD traversal); else raw units.
    alphas: tuple = (-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
    scale_alpha_to_gap: bool = True
    orient_sign: bool = True       # orient so +α=toward KO (False = raw MLP sign, pre-orientation)
    n_traverse: int = 8            # source (control) cells to traverse
    ddim_steps: int = 50
    # edit-guidance: ε̃ = ε(z0) + w·(ε(z0+αd) − ε(z0)). w=1 = normal; w>1 amplifies the
    # embedding edit's effect on the image (the DiffAE under-uses the embedding otherwise).
    guidance_scales: tuple = (1.0, 3.0, 5.0)

    # ---- DiffAE decoder (must match the trained checkpoint) ----
    diffae_ckpt: str = f"{DEFAULT_OUT_ROOT}/diffae/phase_v1/diffae_best.pt"
    block_out_channels: tuple = (128, 256, 256, 512)
    layers_per_block: int = 2
    train_timesteps: int = 1000

    # ---- run ----
    device: str = "cuda"
    batch_size: int = 64
    num_workers: int = 0
