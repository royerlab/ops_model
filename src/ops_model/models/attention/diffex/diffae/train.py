"""DiffAE training — proper conditional-diffusion recipe.

Fixes the v1 failure (embedding had ~0.8% control). Essential components:
- **Conditioning dropout**: replace the embedding with a learned null ~cond_dropout of
  the time, forcing the model to actually USE the embedding (and enabling CFG).
- **EMA** weights for sampling/eval (near-essential for diffusion quality).
- **Resume** across 12h jobs (long training to convergence).
- **Conditioning gate** (not recon PSNR!): generate from fixed noise under two different
  embeddings and report emb-vs-noise MSE ratio — this is the metric that must climb.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from diffusers import DDIMScheduler, DDPMScheduler
from torch.utils.data import DataLoader, TensorDataset


def _device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[device] cuda unavailable -> cpu")
        return torch.device("cpu")
    return torch.device(name)


def _augment(x: torch.Tensor, scale_jit: float = 0.15) -> torch.Tensor:
    """Per-sample continuous rotation (±180°) + scale + horizontal flip, with REFLECTION
    padding so arbitrary angles introduce no black corners. Matches the contrastive
    data_loader recipe (RandAffine ±π, scale) and — unlike the discrete dihedral — makes
    orientation a nuisance across ALL angles, not just 90° steps. Embedding is NOT
    recomputed. x: (B,1,H,W)."""
    import math
    B = x.shape[0]; dev = x.device
    fl = torch.rand(B, device=dev) < 0.5
    x = torch.where(fl[:, None, None, None], torch.flip(x, dims=(-1,)), x)
    ang = (torch.rand(B, device=dev) * 2 - 1) * math.pi
    s = 1.0 / (1.0 + (torch.rand(B, device=dev) * 2 - 1) * scale_jit)   # inverse for sampling grid
    cos, sin = torch.cos(ang) * s, torch.sin(ang) * s
    theta = torch.zeros(B, 2, 3, device=dev, dtype=x.dtype)
    theta[:, 0, 0], theta[:, 0, 1] = cos, -sin
    theta[:, 1, 0], theta[:, 1, 1] = sin, cos
    grid = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)
    return torch.nn.functional.grid_sample(x, grid, mode="bilinear",
                                           padding_mode="reflection", align_corners=False)


def _dihedral(x: torch.Tensor) -> torch.Tensor:
    """Per-sample random dihedral transform (4 rot90 × flip = 8 orientations, incl.
    transpose). x: (B,1,H,W). The conditioning embedding is NOT recomputed — orientation
    is made a nuisance the model must push into the noise latent, not the embedding."""
    ks = torch.randint(0, 4, (x.shape[0],))
    fl = torch.rand(x.shape[0]) < 0.5
    out = torch.empty_like(x)
    for i in range(x.shape[0]):
        xi = torch.rot90(x[i], int(ks[i]), dims=(-2, -1))
        out[i] = torch.flip(xi, dims=(-1,)) if fl[i] else xi
    return out


@torch.no_grad()
def _sample(model, xT, emb, cfg):
    fwd = DDIMScheduler(num_train_timesteps=cfg.train_timesteps)
    fwd.set_timesteps(cfg.ddim_steps)
    c = model.cond(emb)
    x = xT
    for t in fwd.timesteps:
        x = fwd.step(model.denoise(x, t, c), t, x).prev_sample
    return x


@torch.no_grad()
def conditioning_gate(model, probe_embs, cfg, dev, out_png, tag="") -> float:
    """emb-vs-noise MSE ratio: how much the embedding controls generation. >~0.3 = real
    control; the v1 model was 0.008. Generates from fixed noise under different embeddings."""
    model.eval()
    H = cfg.crop_size
    k = max(1, min(4, len(probe_embs) // 2))
    ea = torch.as_tensor(probe_embs[:k], dtype=torch.float32, device=dev)
    eb = torch.as_tensor(probe_embs[k:2 * k], dtype=torch.float32, device=dev)
    A, B, A2 = [], [], []
    for i in range(k):
        g = torch.Generator(device=dev).manual_seed(7 + i)
        xT = torch.randn(1, 1, H, H, generator=g, device=dev)
        A.append(_sample(model, xT.clone(), ea[i:i + 1], cfg).cpu().numpy()[0, 0])
        B.append(_sample(model, xT.clone(), eb[i:i + 1], cfg).cpu().numpy()[0, 0])
        g2 = torch.Generator(device=dev).manual_seed(999 + i)
        xT2 = torch.randn(1, 1, H, H, generator=g2, device=dev)
        A2.append(_sample(model, xT2.clone(), ea[i:i + 1], cfg).cpu().numpy()[0, 0])
    A, B, A2 = np.array(A), np.array(B), np.array(A2)
    emb_mse = float(np.mean((A - B) ** 2))      # same noise, different embedding
    noise_mse = float(np.mean((A - A2) ** 2))   # different noise, same embedding
    ratio = emb_mse / (noise_mse + 1e-9)

    import matplotlib
    matplotlib.use("Agg"); matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(k, 3, figsize=(5.5, 1.7 * k), squeeze=False)
    for i in range(k):
        ax[i, 0].imshow(A[i], cmap="gray", vmin=-1, vmax=1)
        ax[i, 1].imshow(B[i], cmap="gray", vmin=-1, vmax=1)
        d = np.abs(A[i] - B[i]); ax[i, 2].imshow(d, cmap="hot", vmin=0, vmax=max(d.max(), 1e-3))
        for j in range(3):
            ax[i, j].axis("off")
        if i == 0:
            for j, t in enumerate(["emb A", "emb B", "|A−B|"]):
                ax[i, j].set_title(t, fontsize=9)
    fig.suptitle(f"conditioning gate {tag}: emb/noise ratio={ratio:.3f}", fontsize=10)
    fig.tight_layout(); fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[gate] {tag} emb/noise ratio={ratio:.3f}  (emb_mse={emb_mse:.4f} noise_mse={noise_mse:.4f})")
    return ratio


def train_diffae(model, images_norm: np.ndarray, embs: np.ndarray, cfg, out_dir: Path) -> dict:
    dev = _device(cfg.device)
    model = model.to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    sched = DDPMScheduler(num_train_timesteps=cfg.train_timesteps)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = nn.MSELoss()

    n_probe = min(8, len(images_norm) // 10 or 8)
    probe_emb = embs[-n_probe:]
    train_x, train_e = images_norm[:-n_probe], embs[:-n_probe]
    loader = DataLoader(
        TensorDataset(torch.as_tensor(train_x), torch.as_tensor(train_e)),
        batch_size=cfg.batch_size, shuffle=True, drop_last=True,
    )
    use_amp = dev.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    out_dir = Path(out_dir)
    (out_dir / "gate").mkdir(parents=True, exist_ok=True)

    # NOTE: nn.DataParallel is incompatible with the diffusers UNet (its `.dtype`
    # property breaks on DP replicas). Multi-GPU would need DDP. Single-GPU here.
    train_model = model

    @torch.no_grad()
    def ema_update():
        for e, p in zip(ema.parameters(), model.parameters()):
            e.mul_(cfg.ema_decay).add_(p.detach(), alpha=1 - cfg.ema_decay)
        for eb, pb in zip(ema.buffers(), model.buffers()):
            eb.copy_(pb)

    # resume
    state_path = out_dir / "diffae_train_state.pt"
    start_epoch, history, best_ratio = 0, [], -1.0
    if cfg.resume and state_path.exists():
        st = torch.load(state_path, map_location=dev)
        model.load_state_dict(st["model"]); ema.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"]); start_epoch = st["epoch"] + 1
        history = st.get("history", []); best_ratio = st.get("best_ratio", -1.0)
        print(f"[resume] from epoch {start_epoch} (best ratio {best_ratio:.3f})")

    for ep in range(start_epoch, cfg.epochs):
        model.train()
        ep_loss = 0.0
        for x, e in loader:
            x, e = x.to(dev), e.to(dev)
            if getattr(cfg, "augment_affine", False):       # continuous rotation+scale+flip
                x = _augment(x, getattr(cfg, "affine_scale", 0.15))
            elif getattr(cfg, "augment_dihedral", False):    # discrete 90°×flip
                x = _dihedral(x)
            if cfg.cond_dropout > 0:                       # conditioning dropout
                drop = torch.rand(e.shape[0], device=dev) < cfg.cond_dropout
                if drop.any():
                    e = torch.where(drop[:, None], model.null_emb[None].to(e.dtype), e)
            noise = torch.randn_like(x)
            t = torch.randint(0, cfg.train_timesteps, (x.shape[0],), device=dev).long()
            noisy = sched.add_noise(x, noise, t)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = crit(train_model(noisy, t, e), noise)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            ema_update()
            ep_loss += float(loss) * x.shape[0]
        ep_loss /= len(loader.dataset)

        rec = {"epoch": ep, "loss": ep_loss}
        if (ep + 1) % cfg.recon_every == 0 or ep == cfg.epochs - 1:
            try:
                ratio = conditioning_gate(ema, probe_emb, cfg, dev,
                                          out_dir / "gate" / f"gate_ep{ep:03d}.png", tag=f"ep{ep}")
                rec["cond_ratio"] = ratio
                if ratio > best_ratio:
                    best_ratio = ratio
                    torch.save(ema.state_dict(), out_dir / "diffae_best.pt")  # EMA = model to use
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] gate at epoch {ep} failed (continuing): {exc}")
        history.append(rec)
        # save resume state EVERY epoch (preemption resilience on contended GPUs)
        try:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                        "opt": opt.state_dict(), "epoch": ep, "history": history,
                        "best_ratio": best_ratio}, state_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] state save at epoch {ep} failed: {exc}")
        print(f"epoch {ep:03d}: loss={ep_loss:.4f}"
              + (f"  cond_ratio={rec['cond_ratio']:.3f}" if "cond_ratio" in rec else ""))

    torch.save(ema.state_dict(), out_dir / "diffae_ema_last.pt")
    return {"best_ratio": best_ratio, "history": history}
