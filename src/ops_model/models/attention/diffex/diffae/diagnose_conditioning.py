"""Diagnostic: does the trained DiffAE's embedding actually control generation?

From a SINGLE fixed noise, generate conditioned on (null / control-centroid /
KD-centroid). If conditioning is strong, control vs KD should look visibly
different. Key metric: MSE(ctrl-img, kd-img) at fixed noise, compared to
MSE between two DIFFERENT noises — if embedding-driven change ≪ noise-driven
change, the embedding has weak control (the bug we suspect).

    python -m ops_model.models.attention.diffex.diffae.diagnose_conditioning
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler

from ..classifier.config import DEFAULT_OUT_ROOT
from .config import DiffAEConfig
from .model import DiffAE


def run_diagnose(ckpt: str, emb_npz: str, crops_npz: str, out_dir: str,
                 n_noise: int = 6, ddim_steps: int = 50, device: str = "cuda") -> dict:
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    cfg = DiffAEConfig()
    model = DiffAE(cfg)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(dev).eval()

    embs = np.load(emb_npz)["features"]
    labels = np.load(crops_npz, allow_pickle=True)["labels"]
    conds = {
        "null": np.zeros(embs.shape[1], np.float32),
        "ctrl": embs[labels == 0].mean(0).astype(np.float32),
        "kd": embs[labels == 1].mean(0).astype(np.float32),
    }
    print(f"‖μ_kd−μ_ctrl‖ = {np.linalg.norm(conds['kd']-conds['ctrl']):.2f}")

    H = cfg.crop_size

    @torch.no_grad()
    def sample(xT, emb):
        fwd = DDIMScheduler(num_train_timesteps=cfg.train_timesteps)
        fwd.set_timesteps(ddim_steps)
        c = model.cond(torch.as_tensor(emb, device=dev)[None])
        x = xT
        for t in fwd.timesteps:
            x = fwd.step(model.denoise(x, t, c), t, x).prev_sample
        return x.cpu().numpy()[0, 0]

    imgs = {k: [] for k in conds}
    for i in range(n_noise):
        g = torch.Generator(device=dev).manual_seed(100 + i)
        xT = torch.randn(1, 1, H, H, generator=g, device=dev)
        for k, e in conds.items():
            imgs[k].append(sample(xT.clone(), e))
    arr = {k: np.array(v) for k, v in imgs.items()}

    # embedding-driven change (same noise, ctrl vs kd) vs noise-driven change
    mse_ctrl_kd = float(np.mean((arr["ctrl"] - arr["kd"]) ** 2))
    mse_null_ctrl = float(np.mean((arr["null"] - arr["ctrl"]) ** 2))
    mse_noise = float(np.mean([(arr["ctrl"][i] - arr["ctrl"][j]) ** 2
                               for i in range(n_noise) for j in range(i + 1, n_noise)]))
    ratio = mse_ctrl_kd / (mse_noise + 1e-9)
    metrics = {
        "mse_ctrl_vs_kd_same_noise": mse_ctrl_kd,
        "mse_null_vs_ctrl": mse_null_ctrl,
        "mse_noise_vs_noise": mse_noise,
        "embedding_vs_noise_ratio": ratio,
    }
    print(json.dumps(metrics, indent=2))
    print(f"INTERPRETATION: embedding/noise ratio = {ratio:.3f}. "
          f"≪1 ⇒ embedding has WEAK control (noise dominates); ~1+ ⇒ strong control.")

    # montage: rows = noise; cols = null | ctrl | kd | |ctrl−kd|
    import matplotlib
    matplotlib.use("Agg"); matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cols = ["null", "ctrl", "kd", "|ctrl−kd|"]
    fig, ax = plt.subplots(n_noise, 4, figsize=(7, 1.7 * n_noise), squeeze=False)
    for i in range(n_noise):
        ax[i, 0].imshow(arr["null"][i], cmap="gray", vmin=-1, vmax=1)
        ax[i, 1].imshow(arr["ctrl"][i], cmap="gray", vmin=-1, vmax=1)
        ax[i, 2].imshow(arr["kd"][i], cmap="gray", vmin=-1, vmax=1)
        d = np.abs(arr["ctrl"][i] - arr["kd"][i])
        ax[i, 3].imshow(d, cmap="hot", vmin=0, vmax=max(d.max(), 1e-3))
        for j in range(4):
            ax[i, j].axis("off")
            if i == 0:
                ax[i, j].set_title(cols[j], fontsize=9)
    fig.suptitle(f"DiffAE conditioning test — emb/noise MSE ratio={ratio:.3f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "conditioning_test.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    (out / "diagnose_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"wrote {out}/conditioning_test.png")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    base = f"{DEFAULT_OUT_ROOT}/diffae/phase_v1"
    dirs = f"{DEFAULT_OUT_ROOT}/directions/geneKO/HSPA5/cache"
    ap.add_argument("--ckpt", default=f"{base}/diffae_best.pt")
    ap.add_argument("--emb-npz", default=f"{dirs}/celldino_HSPA5_160.npz")
    ap.add_argument("--crops-npz", default=f"{dirs}/crops_HSPA5_160.npz")
    ap.add_argument("--out-dir", default=f"{base}/diagnose")
    ap.add_argument("--local", action="store_true", help="run here instead of SLURM")
    args = ap.parse_args()

    kwargs = dict(ckpt=args.ckpt, emb_npz=args.emb_npz, crops_npz=args.crops_npz,
                  out_dir=args.out_dir)
    if args.local:
        run_diagnose(device="cpu", **kwargs)
        return
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    submit_parallel_jobs(
        jobs_to_submit=[{"name": "diffae_diagnose", "func": run_diagnose,
                         "kwargs": kwargs, "metadata": {"stage": "diagnose"}}],
        experiment="diffae_diagnose",
        slurm_params={"slurm_partition": "gpu", "slurm_gres": "gpu:1",
                      "cpus_per_task": 4, "mem_gb": 32, "timeout_min": 30},
        log_dir="diffae_diagnose", wait_for_completion=False,
    )


if __name__ == "__main__":
    main()
