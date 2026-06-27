"""Orchestrator for Stage 3 (directions → ranking → traversal).

    python -m ops_model.models.attention.diffex.directions.run --target HSPA5
    python -m ops_model.models.attention.diffex.directions.run --grain complex \
        --target "Chaperonin-containing T-complex"

Steps: gather target+control crops/embeddings → train K direction MLPs (unsupervised)
→ rank by control-vs-target LR score shift → traverse the selected direction on control
cells, DDIM-sample, verify monotonic re-encoded score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..classifier.config import DEFAULT_OUT_ROOT, GRAINS, slugify
from ..diffae.data import normalize
from .config import DirConfig
from .data import gather
from .model import DirectionBank
from .rank import rank_directions
from .traverse import load_diffae, traverse
from .train_directions import train_directions


def run_directions(cfg: DirConfig, out_dir: str) -> dict:
    dev = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    out = Path(out_dir); cache = out / "cache"; cache.mkdir(parents=True, exist_ok=True)
    tag = f"{slugify(cfg.target)}_{cfg.crop_size}"

    # gather
    images, embs, labels = gather(
        cfg, str(cache / f"crops_{tag}.npz"), str(cache / f"celldino_{tag}.npz"))

    # 2a: directions (unsupervised, all embeddings)
    bank = DirectionBank(cfg.cond_dim, cfg.K, cfg.hidden)
    train_directions(bank, embs, cfg, dev)
    torch.save(bank.state_dict(), out / "direction_bank.pt")

    # 2b: rank
    best_k, shifts, lr_w, lr_b, lr_acc = rank_directions(bank, embs, labels, cfg, dev)

    # 3: traverse control cells. Scale α to the control→KD embedding gap so
    # α=+1 ≈ a full traversal (unit directions × small α barely move otherwise).
    gap = 1.0
    if cfg.scale_alpha_to_gap:
        gap = float(np.linalg.norm(embs[labels == 1].mean(0) - embs[labels == 0].mean(0)))
        print(f"[traverse] control→KD gap ‖μ_KD−μ_ctrl‖ = {gap:.2f}; α scaled by it")
    ctrl_idx = np.flatnonzero(labels == 0)[: cfg.n_traverse]
    src_imgs = normalize(images[ctrl_idx])
    src_embs = embs[ctrl_idx]
    kd_idx = np.flatnonzero(labels == 1)[: cfg.n_traverse]   # REAL KD cells for reference column
    kd_imgs = normalize(images[kd_idx]) if len(kd_idx) else src_imgs
    diffae = load_diffae(cfg, dev)

    # sweep guidance scale w (w=1 = plain conditional; w>1 amplifies the embedding edit)
    sweep = {}
    for w in cfg.guidance_scales:
        sc = traverse(diffae, bank, best_k, src_imgs, src_embs, lr_w, lr_b, cfg, dev, out,
                      gap=gap, w=w, real_kd=kd_imgs)
        mono = float(np.mean([np.all(np.diff(s) > 0) or np.all(np.diff(s) < 0) for s in sc]))
        sweep[f"w{w:g}"] = {"mean_score_delta": float((sc[:, -1] - sc[:, 0]).mean()),
                            "frac_monotonic": mono}
        print(f"[w={w:g}] mean_score_delta={sweep[f'w{w:g}']['mean_score_delta']:.2f}  "
              f"frac_monotonic={mono:.2f}")
    metrics = {
        "target": cfg.target, "grain": cfg.grain, "K": cfg.K, "best_direction": best_k,
        "lr_acc": lr_acc, "score_shifts": shifts, "gap": gap,
        "guidance_sweep": sweep, "n_traverse": int(len(ctrl_idx)),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    ap = argparse.ArgumentParser(description="DiffEx Stage 3: directions + traversal")
    ap.add_argument("--grain", choices=list(GRAINS), default="geneKO")
    ap.add_argument("--target", default="HSPA5")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--dir-epochs", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--diffae-ckpt", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = DirConfig(grain=args.grain, target=args.target, K=args.K,
                    dir_epochs=args.dir_epochs, device=args.device)
    if args.diffae_ckpt:
        cfg.diffae_ckpt = args.diffae_ckpt
    out = args.out_dir or f"{DEFAULT_OUT_ROOT}/directions/{args.grain}/{slugify(args.target)}"
    run_directions(cfg, out)


if __name__ == "__main__":
    main()
