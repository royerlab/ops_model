"""Orchestrator for the DiffAE generator stage.

    python -m ops_model.models.interpretability.diffae.generator.run

Steps: sample broad phase crops (cached) -> normalize -> train DiffAE (joint
encoder + conditional UNet) -> periodic + final reconstruction gate. Writes
diffae_best.pt / diffae_last.pt, recon montages, metrics.json under <out-dir>.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..classifier.config import DEFAULT_OUT_ROOT
from .config import DiffAEConfig
from .data import load_diffae_crops
from .model import DiffAE
from .train import train_diffae


def run_diffae(cfg: DiffAEConfig, out_dir: str) -> dict:
    out = Path(out_dir)
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    cond_ch = getattr(cfg, "cond_channel", None)
    spatial = getattr(cfg, "spatial_cond", False)
    base = f"{cfg.n_crops}_{cfg.crop_size}"                 # cache key
    cdtag = "" if getattr(cfg, "celldino_z_score", True) else "_rawcd"   # non-z-scored CellDINO → distinct emb cache
    loaded = load_diffae_crops(
        cfg,
        crops_cache=str(cache / f"diffae_crops_{base}.npz"),
        emb_cache=str(cache / f"diffae_celldino_{base}{cdtag}.npz"),
        cond_cache=(str(cache / f"diffae_cond_{cond_ch}_{base}.npz")
                    if cond_ch else None),
        return_cond_images=spatial,
    )
    cond_imgs = None
    if spatial:
        images, embs, cond_imgs = loaded          # phase images concatenated into the UNet input
    else:
        images, embs = loaded
    model = DiffAE(cfg)
    if getattr(cfg, "init_ckpt", None):               # warm-start from an existing model
        import torch
        model.load_state_dict(torch.load(cfg.init_ckpt, map_location="cpu"))
        print(f"warm-started weights from {cfg.init_ckpt}")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"DiffAE: {n_params:.1f}M params; cond_dim={embs.shape[1]}; "
          f"training on {len(images)} crops")

    result = train_diffae(model, images, embs, cfg, out, cond_images=cond_imgs)
    metrics = {
        "n_crops": int(len(images)), "crop_size": cfg.crop_size,
        "cond_dim": int(embs.shape[1]), "epochs": cfg.epochs,
        "n_params_M": round(n_params, 1),
        "best_cond_ratio": result["best_ratio"],  # emb/noise control (target ≫ v1's 0.008)
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "history.json").write_text(json.dumps(result["history"], indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Train the DiffAE (DiffEx generator)")
    ap.add_argument("--n-crops", type=int, default=50_000)
    ap.add_argument("--crop-size", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--out-dir", default=f"{DEFAULT_OUT_ROOT}/diffae/phase_v1")
    args = ap.parse_args()

    cfg = DiffAEConfig(
        n_crops=args.n_crops, crop_size=args.crop_size, epochs=args.epochs,
        batch_size=args.batch_size, device=args.device, num_workers=args.num_workers,
    )
    run_diffae(cfg, args.out_dir)


if __name__ == "__main__":
    main()
