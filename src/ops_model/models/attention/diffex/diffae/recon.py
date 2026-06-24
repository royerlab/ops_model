"""Reconstruction gate: DDIM-invert each cell to x_T then reverse, both
conditioned on its z_sem. Faithful encode->decode round-trip → PSNR + montage.
If the DiffAE can't reconstruct cells, learned directions would be meaningless.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler


@torch.no_grad()
def reconstruct(model, x0: torch.Tensor, cell_emb: torch.Tensor, cfg, dev) -> torch.Tensor:
    """x0: (B,1,H,W) in [-1,1]; cell_emb: (B,cond_dim) frozen CellDINO embedding.
    Returns pixel reconstruction (B,1,H,W)."""
    model.eval()
    emb = model.cond(cell_emb.to(dev))

    inv = DDIMInverseScheduler(num_train_timesteps=cfg.train_timesteps)
    inv.set_timesteps(cfg.ddim_steps)
    x = x0.to(dev)
    for t in inv.timesteps:
        eps = model.denoise(x, t, emb)
        x = inv.step(eps, t, x).prev_sample

    fwd = DDIMScheduler(num_train_timesteps=cfg.train_timesteps)
    fwd.set_timesteps(cfg.ddim_steps)
    for t in fwd.timesteps:
        eps = model.denoise(x, t, emb)
        x = fwd.step(eps, t, x).prev_sample
    return x


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(4.0 / mse))  # data range 2 ([-1,1]) -> 2^2=4


@torch.no_grad()
def recon_report(model, images_norm: np.ndarray, embs: np.ndarray, cfg, dev,
                 out_png: Path, tag: str = "") -> float:
    """Reconstruct cfg.n_recon cells, write an original-vs-recon montage, return PSNR."""
    n = min(cfg.n_recon, len(images_norm))
    x0 = torch.as_tensor(images_norm[:n])
    cell_emb = torch.as_tensor(embs[:n])
    rec = reconstruct(model, x0, cell_emb, cfg, dev).cpu().numpy()
    orig = x0.numpy()
    psnr = _psnr(orig, rec)

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, n, figsize=(1.4 * n, 3.0), squeeze=False)
    for i in range(n):
        axes[0, i].imshow(orig[i, 0], cmap="gray", vmin=-1, vmax=1); axes[0, i].axis("off")
        axes[1, i].imshow(rec[i, 0], cmap="gray", vmin=-1, vmax=1); axes[1, i].axis("off")
    axes[0, 0].set_ylabel("orig", rotation=90); axes[1, 0].set_ylabel("recon", rotation=90)
    fig.suptitle(f"DiffAE reconstruction {tag}  PSNR={psnr:.2f} dB", fontsize=10)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[recon] {tag} PSNR={psnr:.2f} dB -> {out_png}")
    return psnr
