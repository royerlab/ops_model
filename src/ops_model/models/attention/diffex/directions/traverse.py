"""Stage 3: traverse the selected direction and DDIM-sample a counterfactual strip.

For each control cell: DDIM-invert to x_T (conditioned on its embedding z0), then for
each alpha reverse x_T conditioned on (z0 + alpha*d) -> image. Verify by RE-ENCODING
each generated image through CellDINO and scoring with the ranking classifier — the
score should move monotonically with alpha (the faithfulness check)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler

from ..classifier.celldino_features import embed_crops
from ..classifier.config import slugify
from ..diffae.config import DiffAEConfig
from ..diffae.model import DiffAE


def load_diffae(cfg, dev):
    dcfg = DiffAEConfig(
        crop_size=cfg.crop_size, cond_dim=cfg.cond_dim,
        block_out_channels=cfg.block_out_channels,
        layers_per_block=cfg.layers_per_block, train_timesteps=cfg.train_timesteps,
    )
    m = DiffAE(dcfg)
    m.load_state_dict(torch.load(cfg.diffae_ckpt, map_location="cpu"))
    return m.to(dev).eval()


@torch.no_grad()
def _ddim(diffae, x, emb, cfg, inverse: bool):
    sched = (DDIMInverseScheduler if inverse else DDIMScheduler)(
        num_train_timesteps=cfg.train_timesteps)
    sched.set_timesteps(cfg.ddim_steps)
    c = diffae.cond(emb)
    for t in sched.timesteps:
        x = sched.step(diffae.denoise(x, t, c), t, x).prev_sample
    return x


@torch.no_grad()
def _sample_guided(diffae, xT, emb, emb_base, w, cfg):
    """DDIM sample from xT. Edit-guidance: ε̃ = ε(base) + w·(ε(emb) − ε(base)).
    w=1 → plain ε(emb); w>1 amplifies the embedding edit's effect on the image."""
    fwd = DDIMScheduler(num_train_timesteps=cfg.train_timesteps)
    fwd.set_timesteps(cfg.ddim_steps)
    c, c0 = diffae.cond(emb), diffae.cond(emb_base)
    x = xT
    for t in fwd.timesteps:
        e = diffae.denoise(x, t, c)
        if w != 1.0:
            e = diffae.denoise(x, t, c0) + w * (e - diffae.denoise(x, t, c0))
        x = fwd.step(e, t, x).prev_sample
    return x


@torch.no_grad()
def traverse(diffae, bank, best_k, src_imgs_norm, src_embs, lr_w, lr_b, cfg, dev, out_dir,
             gap: float = 1.0, w: float = 1.0):
    """src_imgs_norm (M,1,H,W) in [-1,1]; src_embs (M,cond_dim).
    gap = ‖μ_KD−μ_ctrl‖ (α scaled by it); w = edit-guidance scale."""
    out_dir = Path(out_dir)
    alphas = list(cfg.alphas)
    H = cfg.crop_size
    # true classifier-free guidance: baseline = the learned NULL embedding, so
    # ε̃ = ε(∅) + w·(ε(z0+αd) − ε(∅)). w=1 = plain conditional sampling.
    null_base = diffae.null_emb.detach()[None].to(dev)
    gen_imgs = []   # (M, n_alpha) generated patches
    for i in range(len(src_imgs_norm)):
        z0 = torch.as_tensor(src_embs[i:i + 1], dtype=torch.float32).to(dev)
        d = bank.direction(z0, best_k)                       # (1,D), fixed at source
        # Fixed random noise per cell (constant across α): only the embedding edit
        # changes along a row. α=0 = DDPM recon of z0 (Alex's design), not the real image.
        g = torch.Generator(device=dev).manual_seed(1234 + i)
        xT = torch.randn(1, 1, H, H, generator=g, device=dev)
        row = []
        for a in alphas:
            img = _sample_guided(diffae, xT.clone(), z0 + (a * gap) * d, null_base, w, cfg)
            row.append(img.cpu().numpy()[0, 0])
        gen_imgs.append(row)
    gen = np.array(gen_imgs)  # (M, A, H, W)

    # verify: re-encode generated images -> CellDINO -> LR score
    flat = gen.reshape(-1, 1, gen.shape[-2], gen.shape[-1]).astype(np.float32)
    gen_embs = embed_crops(flat, cfg, cache_path=None)        # (M*A, cond_dim)
    scores = (gen_embs @ lr_w + lr_b).reshape(len(src_imgs_norm), len(alphas))

    _plot(gen, scores, alphas, cfg, out_dir, w)
    return scores


def _plot(gen, scores, alphas, cfg, out_dir, w=1.0):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    M, A = gen.shape[0], gen.shape[1]
    c = alphas.index(0.0) if 0.0 in alphas else A // 2   # α=0 reconstruction = baseline
    # traversal montage: rows = source cells, cols = alpha. Each off-center tile
    # overlays a diverging heatmap of the pixel change vs the α=0 frame
    # (red = intensity gained, blue = lost) — "what pixels swap as we move out".
    fig, axes = plt.subplots(M, A, figsize=(1.3 * A, 1.3 * M), squeeze=False)
    for i in range(M):
        vmax = float(max(np.abs(gen[i] - gen[i, c]).max(), 1e-3))  # per-cell scale
        for j in range(A):
            axes[i, j].imshow(gen[i, j], cmap="gray", vmin=-1, vmax=1)
            if j != c:
                diff = gen[i, j] - gen[i, c]
                amask = np.clip(np.abs(diff) / vmax, 0, 1) * 0.7  # transparent where unchanged
                axes[i, j].imshow(diff, cmap="bwr", vmin=-vmax, vmax=vmax, alpha=amask)
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(f"{alphas[j]:+.2g}×gap", fontsize=8)
    fig.suptitle(f"{cfg.target}: counterfactual traversal (control → KD), guidance w={w:g}; "
                 f"overlay = Δpixels vs α=0 (red +, blue −)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"traversal_{slugify(cfg.target)}_w{w:g}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # score progression
    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(M):
        ax.plot(alphas, scores[i], marker="o", alpha=0.6)
    ax.plot(alphas, scores.mean(0), color="black", lw=2.5, label="mean")
    ax.set_xlabel("α"); ax.set_ylabel("classifier logit (re-encoded gen image)")
    ax.set_title(f"{cfg.target}: score progression (monotonic ⇒ direction valid)")
    ax.axhline(0, color="gray", ls=":"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"scores_{slugify(cfg.target)}_w{w:g}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[traverse] wrote traversal/scores for {cfg.target} (w={w:g})")
