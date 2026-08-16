"""PC latent-walk morphs: for a marker (or phase), decode a DiffAE traversal along each principal
component of that marker's CellDINO feature space, showing what each unsupervised PC axis looks like
as a cell morph — the generative analogue of the PC-strip tab (which bins real cells).

Per marker: refit PCA on its per-cell CellDINO features (same fit as build_pcs_marker), take one base
control cell's CellDINO embedding z0 (from the cached anchor embeddings) with a fixed diffusion seed, and
for each PC p decode z0 + (α·σ_p)·(v_p ⊙ sd) across α ∈ [−n_std, +n_std] with the marker's DiffAE. σ_p is
the PC score std; v_p is the (z-scored) eigenvector, mapped back to raw CellDINO space by the mean per-exp
sd. Output: one composite figure per marker (rows = PCs, cols = α).

  python -m ops_model.models.interpretability.diffae.viewer.build_pc_walks --markers "Mitochondria_TOMM20"
  python -m ops_model.models.interpretability.diffae.viewer.build_pc_walks --all          # SLURM, every marker
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

OUT_DIR = "/hpc/projects/icd.fast.ops/analysis/figure4_pc_walks"


@torch.no_grad()
def pc_walks_marker(marker_channel, channel, ckpt, out_root, n_pcs=20, n_cells=10,
                    w=2.0, device="cuda", batch=48, upsize=256, force=False):
    """Write PC-walk traversals into the viewer cache as a `pc` grain: one target per PC
    (viewer_assets/<modality>/pc/PC##/cell<c>/frame_<i>.webp + meta.json), so the viewer treats each PC
    like any other perturbation (α scrub, cell selection, pinning). α uses the same VIEWER_ALPHAS grid
    (−5…+5) as every other traversal, here in units of the PC score's σ."""
    import json
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor
    from sklearn.decomposition import PCA
    from .build_pcs_marker import _marker_meta, _fp, _load, FIT_N
    from .precompute import DirConfig, load_diffae, _sample_guided, _save_webp, VIEWER_ALPHAS
    from ..classifier.config import slugify

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    modality = slugify(marker_channel) if marker_channel else "phase"

    # 1) refit PCA on the CellDINO features (per-experiment z-score, pooled) — as build_pcs_marker
    if marker_channel:
        leaf, reporter, chan, exps, fdir = _marker_meta(marker_channel)
        fps = [_fp(e, reporter, fdir) for e in exps]
        channel = channel or chan
    else:                                                    # phase: pooled features_processed_Phase (cell_dino v2)
        import glob as _g
        fps = sorted(_g.glob("/hpc/projects/icd.fast.ops/*/3-assembly/cell_dino_features_v2/"
                             "anndata_objects/features_processed_Phase.h5ad"))[:12]
        channel = channel or "Phase2D"
    rng = np.random.RandomState(0)
    zparts, sds = [], []
    for fp in fps:
        if not fp or not os.path.exists(fp):
            continue
        X, _ = _load(fp, [], FIT_N, rng)
        mu, sd = X.mean(0), X.std(0) + 1e-8
        zparts.append((X - mu) / sd); sds.append(sd)
    pca = PCA(n_components=n_pcs, svd_solver="randomized", random_state=0).fit(np.vstack(zparts))
    sd_mean = np.mean(sds, 0)
    evr = (pca.explained_variance_ratio_ * 100)
    std_p = np.sqrt(pca.explained_variance_)                 # PC score std (walk magnitude unit)
    print(f"[pcw] {marker_channel}: PCA {n_pcs} PCs; PC1-5 var {evr[:5].round(2)}")

    # 2) base control cells (cached anchor CellDINO embs) + fixed diffusion seeds
    cfg = DirConfig(grain="geneKO", target="NTC", control="NTC", device=device)
    if ckpt:
        cfg.diffae_ckpt = ckpt
    if marker_channel:
        cfg.marker_channel = marker_channel
    if channel:
        cfg.channel = channel
    H = cfg.crop_size
    ctrl = np.load(f"{out_root}/viewer_assets/{modality}/_anchors/NTC/ctrl.npz")["ctrl_embs"]
    ncell = min(n_cells, len(ctrl))
    z0 = torch.as_tensor(ctrl[:ncell], dtype=torch.float32, device=dev)
    xT = torch.stack([torch.randn(1, H, H, generator=torch.Generator(device=dev).manual_seed(1234 + c), device=dev)
                      for c in range(ncell)])
    diffae = load_diffae(cfg, dev)
    null = diffae.null_emb.detach()[None].to(dev)
    alphas = list(VIEWER_ALPHAS)                              # same −5…+5 grid as every other traversal (σ units)
    n_frames = len(alphas)

    # 3) per PC: decode each base cell across the α (σ) sweep, write frames + meta (viewer `pc` grain)
    for p in range(n_pcs):
        slug = f"PC{p + 1:02d}"
        adir = Path(out_root) / "viewer_assets" / modality / "pc" / slug
        if (adir / "meta.json").exists() and not force:
            continue
        vp = torch.as_tensor(pca.components_[p] * sd_mean, dtype=torch.float32, device=dev)[None]
        conds, keys = [], []
        for c in range(ncell):
            for i, a in enumerate(alphas):
                conds.append(z0[c:c + 1] + (a * std_p[p]) * vp); keys.append((c, i))
        gen = np.empty((ncell, n_frames, H, H), np.float32)
        for i0 in range(0, len(conds), batch):
            cb = torch.cat(conds[i0:i0 + batch], 0)
            xb = torch.cat([xT[c:c + 1] for c, _ in keys[i0:i0 + batch]], 0)
            outb = _sample_guided(diffae, xb, cb, null.expand(cb.shape[0], -1), w, cfg).cpu().numpy()[:, 0]
            for k, (c, i) in enumerate(keys[i0:i0 + batch]):
                gen[c, i] = outb[k]
        fp2 = ThreadPoolExecutor(max_workers=8)
        for c in range(ncell):
            (adir / f"cell{c}").mkdir(parents=True, exist_ok=True)
            for i in range(n_frames):
                fp2.submit(_save_webp, adir / f"cell{c}" / f"frame_{i:02d}.webp", gen[c, i], upsize)
        fp2.shutdown(wait=True)
        (adir / "meta.json").write_text(json.dumps({
            "grain": "pc", "target": slug, "modality": modality, "control": None,
            "marker_channel": marker_channel, "channel": channel, "slug": slug, "w": w,
            "alphas": alphas, "n_cells": ncell, "has_scores": False, "has_real": True,
            "real_dir": f"{modality}/_anchors/NTC", "asset_dir": f"{modality}/pc/{slug}",
            "explained_variance": round(float(evr[p]), 2)}))
        print(f"[pcw] {modality}/{slug}: {ncell}×{n_frames} ({evr[p]:.1f}% var)")
    print(f"[pc-walks] {modality}: {n_pcs} PC targets written")
    return f"{out_root}/viewer_assets/{modality}/pc"


def _marker_jobs(n_pcs, n_cells, force):
    from . import catalog as C
    from ..classifier.config import slugify
    jobs = []
    for d, mc, ch in C.complete_markers():
        jobs.append({"name": f"pcw_{slugify(mc)[:18]}", "func": pc_walks_marker,
                     "kwargs": dict(marker_channel=mc, channel=ch, ckpt=f"{C.DD}/{d}/diffae_best.pt",
                                    out_root=C.OUT, n_pcs=n_pcs, n_cells=n_cells, force=force)})
    return jobs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--markers", nargs="+", help="marker_channel(s) to run locally (needs a GPU)")
    ap.add_argument("--all", action="store_true", help="submit every complete marker via SLURM (GPU)")
    ap.add_argument("--n-pcs", type=int, default=20)
    ap.add_argument("--n-cells", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.all:
        from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
        from . import catalog as C
        jobs = _marker_jobs(a.n_pcs, a.n_cells, a.force)
        jobs.append({"name": "pcw_phase", "func": pc_walks_marker,                   # phase embedding too
                     "kwargs": dict(marker_channel=None, channel="Phase2D",
                                    ckpt=f"{C.DD}/phase_v1/diffae_best.pt", out_root=C.OUT,
                                    n_pcs=a.n_pcs, n_cells=a.n_cells, force=a.force)})
        print(f"[pc-walks] submitting {len(jobs)} jobs (markers + phase)")
        submit_parallel_jobs(jobs_to_submit=jobs, experiment="diffex_pc_walks",
                             slurm_params={"slurm_partition": "gpu", "gpus_per_node": 1, "cpus_per_task": 12,
                                           "mem_gb": 64, "timeout_min": 1000},   # ~200 PCs @ ~3.4min/PC
                             log_dir="diffex_pc_walks", wait_for_completion=False)
    elif a.markers:
        from . import catalog as C
        by_mc = {mc: (d, ch) for d, mc, ch in C.complete_markers()}
        for mc in a.markers:
            d, ch = by_mc[mc]
            pc_walks_marker(mc, ch, f"{C.DD}/{d}/diffae_best.pt", C.OUT,
                            n_pcs=a.n_pcs, n_cells=a.n_cells, force=a.force)
    else:
        ap.error("pass --markers <name> or --all")
