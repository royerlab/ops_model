"""Evaluate a virtual-staining DiffAE: predict the fluor marker from the phase CellDINO
embedding on a HELD-OUT set of cells (fresh seed → disjoint from training), then report
Pearson(pred, real) and save a `phase | predicted | real` montage.

    python -m ops_model.interpretability.diffae.generator.virtstain_eval \
        --out-dir "$OPS_BASE_PATH/analysis/virtual_staining/chromalive561_from_phase" \
        --marker-channel "mitochondria_ChromaLIVE 561 excitation" --channel mCherry --cond-channel Phase2D
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from .config import DiffAEConfig
from .data import load_diffae_crops
from .model import DiffAE
from .train import _sample


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12
    return float((a * b).sum() / d)


@torch.no_grad()
def evaluate(cfg: DiffAEConfig, out_dir: str, ckpt: str, n_eval: int, eval_seed: int):
    out = Path(out_dir)
    dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = DiffAE(cfg).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev)); model.eval()

    # fresh held-out cells (disjoint seed) — target marker + conditioning phase crops
    ecfg = dataclasses.replace(cfg, n_crops=n_eval, seed=eval_seed)
    cache = out / "cache_eval"; cache.mkdir(parents=True, exist_ok=True)
    real, embs, phase = load_diffae_crops(
        ecfg,
        crops_cache=str(cache / f"marker_{n_eval}_{cfg.crop_size}_s{eval_seed}.npz"),
        emb_cache=str(cache / f"phasecelldino_{n_eval}_{cfg.crop_size}_s{eval_seed}.npz"),
        cond_cache=str(cache / f"phase_{n_eval}_{cfg.crop_size}_s{eval_seed}.npz"),
        return_cond_images=True,
    )
    H = cfg.crop_size
    spatial = getattr(cfg, "spatial_cond", False)
    corrs, preds = [], []
    for i in range(len(real)):
        g = torch.Generator(device=dev).manual_seed(1000 + i)
        xT = torch.randn(1, 1, H, H, generator=g, device=dev)
        e = torch.as_tensor(embs[i:i + 1], dtype=torch.float32, device=dev)
        ci = torch.as_tensor(phase[i:i + 1], dtype=torch.float32, device=dev) if spatial else None
        pred = _sample(model, xT, e, cfg, cond_img=ci).cpu().numpy()[0, 0]
        preds.append(pred)
        corrs.append(_pearson(pred, real[i, 0]))
    corrs = np.array(corrs)
    # trivial baseline: does the phase image itself already correlate with the marker?
    base = np.array([_pearson(phase[i, 0], real[i, 0]) for i in range(len(real))])
    metrics = {"n_eval": int(len(real)), "eval_seed": eval_seed,
               "pearson_pred_vs_real_mean": round(float(corrs.mean()), 4),
               "pearson_pred_vs_real_std": round(float(corrs.std()), 4),
               "pearson_phase_vs_real_mean": round(float(base.mean()), 4),
               "marker_channel": cfg.marker_channel, "channel": cfg.channel,
               "cond_channel": cfg.cond_channel, "ckpt": ckpt}
    (out / "eval").mkdir(parents=True, exist_ok=True)
    (out / "eval" / "virtstain_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    # montage: top cells by Pearson (best-case read), phase | pred | real
    import matplotlib
    matplotlib.use("Agg"); matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    order = np.argsort(-corrs)[:12]
    rows = [("phase (input)", "gray", lambda i: phase[i, 0]),
            ("predicted", "magma", lambda i: preds[i]),
            ("real", "magma", lambda i: real[i, 0])]
    fig, ax = plt.subplots(3, len(order), figsize=(1.4 * len(order), 4.6), squeeze=False)
    for c, i in enumerate(order):
        for r, (_, cm, get) in enumerate(rows):
            ax[r, c].imshow(get(i), cmap=cm, vmin=-1, vmax=1)
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            ax[r, c].text(0.04, 0.96, f"r={corrs[i]:.2f}", transform=ax[r, c].transAxes,
                          fontsize=6.5, color="white", va="top", ha="left",
                          path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])
    for r, (label, _, _) in enumerate(rows):
        ax[r, 0].set_ylabel(label, fontsize=10)
    fig.suptitle(f"virtual staining {cfg.marker_channel}  |  Pearson {corrs.mean():.3f}±{corrs.std():.3f} "
                 f"(phase-baseline {base.mean():.3f})", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "eval" / "virtstain_montage.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] wrote {out/'eval'/'virtstain_montage.png'}")
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Evaluate virtual-staining DiffAE (Pearson + montage)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ckpt", default=None, help="default <out-dir>/diffae_best.pt")
    ap.add_argument("--marker-channel", required=True)
    ap.add_argument("--channel", default="mCherry")
    ap.add_argument("--cond-channel", default="Phase2D")
    ap.add_argument("--spatial-cond", action="store_true", help="model uses image-concat conditioning")
    ap.add_argument("--crop-size", type=int, default=160)
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--eval-seed", type=int, default=12345)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--submit", action="store_true", help="run on SLURM GPU instead of locally")
    ap.add_argument("--after", default=None, help="SLURM job id to gate on (afterany)")
    args = ap.parse_args()
    cfg = DiffAEConfig(crop_size=args.crop_size, channel=args.channel, cond_channel=args.cond_channel,
                       spatial_cond=args.spatial_cond, marker_channel=args.marker_channel, device=args.device)
    ckpt = args.ckpt or str(Path(args.out_dir) / "diffae_best.pt")
    if args.submit:
        from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
        sp = {"slurm_partition": "gpu", "gpus_per_node": 1, "cpus_per_task": 8, "mem_gb": 96,
              "timeout_min": 60, "slurm_constraint": "[a100_80|h100|h200|6000_blackwell]"}
        if args.after:
            sp["slurm_additional_parameters"] = {"dependency": f"afterany:{args.after}"}
        submit_parallel_jobs(jobs_to_submit=[{"name": "virtstain_eval", "func": evaluate,
            "kwargs": {"cfg": cfg, "out_dir": args.out_dir, "ckpt": ckpt,
                       "n_eval": args.n_eval, "eval_seed": args.eval_seed}}],
            experiment="diffae", slurm_params=sp, log_dir="diffae", wait_for_completion=False)
    else:
        evaluate(cfg, args.out_dir, ckpt, args.n_eval, args.eval_seed)


if __name__ == "__main__":
    main()
