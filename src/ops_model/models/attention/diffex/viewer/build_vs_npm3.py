"""VS-stain NPM3 (nucleolus-GC) from the POLR1B *phase* gen frames with the multi-marker spatial-cond DiffAE,
then save the VS-NPM3 intensity as frames_f32.npz in a new modality so the org-seg pipeline can segment the
nucleoli directly on the virtual NPM3 image (instead of MO-thresholding phase inside the VS-nucleus).

Sharded per-cell across GPUs. Reads phase/geneKO/POLR1B/cell{c}/frames_f32.npz (native 160, [-1,1]) →
writes vs_npm3_from_phase/geneKO/POLR1B/cell{c}/frames_f32.npz.

Run: python -m ops_model.models.attention.diffex.viewer.build_vs_npm3
"""
import json
import os
from pathlib import Path

import numpy as np

from . import catalog as C

_ASSETS = "viewer_assets_v5"
VOUT = "/hpc/projects/icd.fast.ops/analysis/virtual_staining/multi_marker"
SRC_MOD, GRAIN, TARGET = "phase", "geneKO", "POLR1B"
OUT_MOD = "vs_npm3_from_phase"
MARKER = "nucleolus-GC_NPM3"
NCELL = 100
CHUNK = 5


def _vs_job(cells):
    import torch
    from diffusers import DDIMScheduler
    from ..diffae.config import DiffAEConfig
    from ..diffae.model import DiffAE
    from ..classifier.celldino_features import embed_crops
    dev = torch.device("cuda")
    markers = json.load(open(f"{VOUT}/markers.json")); mi = markers.index(MARKER)
    cfg = DiffAEConfig(spatial_cond=True, n_markers=len(markers), device="cuda"); Hg = cfg.crop_size
    model = DiffAE(cfg).to(dev).eval(); model.load_state_dict(torch.load(f"{VOUT}/diffae_best.pt", map_location=dev))
    src = f"{C.OUT}/{_ASSETS}/{SRC_MOD}/{GRAIN}/{TARGET}"
    outd = Path(C.OUT) / _ASSETS / OUT_MOD / GRAIN / TARGET

    @torch.no_grad()
    def vs_batch(P):                                             # P: (N,1,Hg,Hg) phase in [-1,1] → VS-NPM3 in [-1,1]
        fwd = DDIMScheduler(num_train_timesteps=cfg.train_timesteps); fwd.set_timesteps(cfg.ddim_steps)
        ci = torch.as_tensor(P, device=dev)
        emb = torch.as_tensor(embed_crops(P, cfg), device=dev)
        mk = torch.full((P.shape[0],), mi, dtype=torch.long, device=dev)
        c = model.cond(emb, mk); x = torch.randn(P.shape[0], 1, Hg, Hg, device=dev)
        for t in fwd.timesteps:
            x = fwd.step(model.denoise(x, t, c, ci), t, x).prev_sample
        return x.cpu().numpy()[:, 0]

    for cidx in cells:
        f = f"{src}/cell{cidx}/frames_f32.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f); g = z["gen"].astype(np.float32); al = z["alphas"]   # (A,160,160) phase, [-1,1]
        P = g[:, None]                                                       # (A,1,160,160)
        vs = vs_batch(P)                                                     # (A,160,160) VS-NPM3, [-1,1]
        cd = outd / f"cell{cidx}"; cd.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cd / "frames_f32.npz", gen=vs.astype(np.float32), alphas=al)
        (cd / "meta.json").write_text(json.dumps({"alphas": [float(a) for a in al]}))   # build_mini_zarr reads cell0/meta.json
    print(f"[vs-npm3] wrote {len(cells)} cells -> {outd}", flush=True)


def real_gen_panel_job(n=6):
    """VS-NPM3 is just the MEASUREMENT: apply it to BOTH real PHASE and gen PHASE cells, seg the virtual nucleoli,
    and show the PHASE image under that seg. Real = top-n phase NTC/KO cells; gen = gen phase traversal frames."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import torch
    from diffusers import DDIMScheduler
    from iohub import open_ome_zarr
    from skimage.segmentation import find_boundaries
    from .build_pc_crops_masked import BASE, _crop, _zarr_patch
    from .morpho_pipeline import _seg_masked_object, MO_PARAMS
    from ..diffae.config import DiffAEConfig
    from ..diffae.model import DiffAE
    from ..classifier.celldino_features import embed_crops
    import zarr
    dev = torch.device("cuda"); HALF = 80
    OUT = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals_violin/_native"
    tp = {**MO_PARAMS, "min_object_size": 25, "mo_object_min_area_px": 25}
    _zarr_patch()
    markers = json.load(open(f"{VOUT}/markers.json")); mi = markers.index(MARKER)
    cfg = DiffAEConfig(spatial_cond=True, n_markers=len(markers), device="cuda"); Hg = cfg.crop_size
    model = DiffAE(cfg).to(dev).eval(); model.load_state_dict(torch.load(f"{VOUT}/diffae_best.pt", map_location=dev))

    @torch.no_grad()
    def vs(P):                                                          # (N,1,Hg,Hg) phase [-1,1] → VS-NPM3 [-1,1]
        fwd = DDIMScheduler(num_train_timesteps=cfg.train_timesteps); fwd.set_timesteps(cfg.ddim_steps)
        ci = torch.as_tensor(P, device=dev); emb = torch.as_tensor(embed_crops(P, cfg), device=dev)
        mk = torch.full((P.shape[0],), mi, dtype=torch.long, device=dev)
        c = model.cond(emb, mk); x = torch.randn(P.shape[0], 1, Hg, Hg, device=dev)
        for t in fwd.timesteps:
            x = fwd.step(model.denoise(x, t, c, ci), t, x).prev_sample
        return x.cpu().numpy()[:, 0]

    def _pos(exp, well):
        return f"{BASE}/{exp}/3-assembly/phenotyping_v3.zarr/{well[0]}/{well[1:]}/0"

    def phase_crop(exp, well, x, y):
        p = _pos(exp, well); pos = open_ome_zarr(p, mode="r"); names = list(pos.channel_names)
        idx = next((i for i, nm in enumerate(names) if "phase2d" in nm.lower() or "phase" in nm.lower()), 0)
        return _crop(zarr.open(f"{p}/0", mode="r"), idx, x, y, HALF).astype(np.float32)

    def anchor_row():
        """real NTC = the EXACT anchor cells gen α0 reconstructs (phase ctrl.npz anchor_imgs, index-matched to
        gen cell c) — NOT a fresh top-ranked NTC pull, which would compare unrelated cells."""
        anc = np.load(f"{C.OUT}/{_ASSETS}/phase/_anchors/NTC/ctrl.npz")["anchor_imgs"]   # (>=100,1,160,160), [-1,1], lossless
        crops = [np.clip((anc[c, 0] + 1) / 2, 0, 1).astype(np.float32) for c in range(n)]
        P = (np.stack(crops)[:, None] * 2 - 1).astype(np.float32)
        vsimg = vs(P)
        rk = pd.read_parquet(f"{C.OUT}/{_ASSETS}/_rankings/pma_shap_phase_geneKO.parquet")
        rk = rk[(rk["gene"].astype(str) == "NTC") & (rk.get("rank_type", "top") == "top")].sort_values("rank")
        out = []
        for c in range(n):
            r = rk.iloc[c]                                              # anchor cell c = rank-(c+1) NTC (how ctrl.npz was built)
            nz = zarr.open(f"{_pos(r['experiment'], str(r['well']))}/labels/nuclear_seg/0", mode="r")
            nucm = _crop(nz, None, int(round(r["x_pheno"])), int(round(r["y_pheno"])), HALF) > 0
            V = np.clip((vsimg[c] + 1) / 2, 0, 1); lo, hi = np.percentile(V, [1, 99.5])
            Vs = np.clip((V - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
            lab = _seg_masked_object(Vs, tp=tp, nucleus=True, nucleus_override=nucm, override_erode=4)
            out.append((crops[c], lab))
        return out

    def real_row(gene):                                                 # KO has no anchor correspondence → top-ranked (not cell-matched to gen)
        df = pd.read_parquet(f"{C.OUT}/{_ASSETS}/_rankings/pma_shap_phase_geneKO.parquet")
        df = df[(df["gene"].astype(str) == gene) & (df.get("rank_type", "top") == "top")].sort_values("rank").head(n * 4)
        crops, coords = [], []
        for _, r in df.iterrows():
            try:
                pc = phase_crop(r["experiment"], str(r["well"]), int(round(r["x_pheno"])), int(round(r["y_pheno"])))
            except Exception:
                continue
            lo, hi = np.percentile(pc, [1, 99.5]); crops.append(np.clip((pc - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32))
            coords.append((r["experiment"], str(r["well"]), int(round(r["x_pheno"])), int(round(r["y_pheno"]))))
            if len(crops) >= n:
                break
        P = (np.stack(crops)[:, None] * 2 - 1).astype(np.float32)
        vsimg = vs(P)                                                   # VS-NPM3 of real phase
        out = []
        for i, (exp, well, x, y) in enumerate(coords):
            nz = zarr.open(f"{_pos(exp, well)}/labels/nuclear_seg/0", mode="r")
            nucm = _crop(nz, None, x, y, HALF) > 0                      # real nucleus
            V = np.clip((vsimg[i] + 1) / 2, 0, 1); lo, hi = np.percentile(V, [1, 99.5])
            Vs = np.clip((V - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
            lab = _seg_masked_object(Vs, tp=tp, nucleus=True, nucleus_override=nucm, override_erode=4)
            out.append((crops[i], lab))                                 # show PHASE + VS-derived seg
        return out

    vsdir = f"{C.OUT}/{_ASSETS}/{OUT_MOD}/{GRAIN}/{TARGET}"; phdir = f"{C.OUT}/{_ASSETS}/phase/{GRAIN}/{TARGET}"
    nuc = np.load(f"{OUT}/polr1b_100/vs_nucleus.npz")["masks"]           # VS-H2B nucleus for gen
    Z = {0: 8, 1: 10, 3: 14}
    gen = {}
    for a, ai in Z.items():
        row = []
        for c in range(n):
            V = np.clip((np.load(f"{vsdir}/cell{c}/frames_f32.npz")["gen"][ai] + 1) / 2, 0, 1)
            lo, hi = np.percentile(V, [1, 99.5]); Vs = np.clip((V - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
            lab = _seg_masked_object(Vs, tp=tp, nucleus=True, nucleus_override=nuc[c, ai] > 0, override_erode=4)
            ph = np.clip((np.load(f"{phdir}/cell{c}/frames_f32.npz")["gen"][ai] + 1) / 2, 0, 1)
            row.append((ph, lab))
        gen[a] = row
    rows = [("real KO", real_row("POLR1B")), ("real NTC (anchor)", anchor_row()),   # NTC = the exact cell-matched anchor gen α0 reconstructs
            ("gen α0", gen[0]), ("gen α1", gen[1]), ("gen α3", gen[3])]
    fig, axes = plt.subplots(5, n, figsize=(n * 2.0, 10.0), facecolor="white")
    for ri, (label, cells) in enumerate(rows):
        for ci in range(n):
            ax = axes[ri, ci]; ax.axis("off")
            if ri == 0:
                ax.set_title(f"cell {ci}", fontsize=9)
            if ci == 0:
                ax.text(-0.18, 0.5, label, transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=11, fontweight="bold")
            if ci >= len(cells):
                continue
            img, lab = cells[ci]; im = (img - img.min()) / (np.ptp(img) + 1e-9)
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            b = find_boundaries(np.asarray(lab) > 0, mode="outer"); ov = np.zeros((*b.shape, 4)); ov[b] = [1, 0.3, 0, 1]; ax.imshow(ov)
    fig.suptitle("VS-NPM3 measurement — real PHASE vs gen PHASE, both segged via VS-NPM3 nucleoli (orange)", fontsize=12)
    fig.tight_layout(rect=[0.02, 0, 1, 0.97])
    o = f"{OUT}/polr1b_vsnpm3_100/DEBUG_seg_panel.png"; fig.savefig(o, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"saved {o}", flush=True)


def submit_panel():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    sp = {"slurm_partition": "gpu", "gpus_per_node": 1, "cpus_per_task": 8, "mem_gb": 96, "timeout_min": 30,
          "slurm_constraint": "[a100_80|h100|h200|6000_blackwell]",
          "slurm_setup": ["export OPS_DIFFEX_ASSETS=viewer_assets_v5", "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]}
    submit_parallel_jobs(jobs_to_submit=[{"name": "vsnpm3_panel", "func": real_gen_panel_job, "kwargs": {"n": 6}}],
                         experiment="diffex_vspanel", slurm_params=sp, log_dir="diffex_vspanel", wait_for_completion=False)


def submit(n=NCELL, chunk=CHUNK):
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    jobs = [{"name": f"vsnpm3_{lo}", "func": _vs_job, "kwargs": {"cells": list(range(lo, min(lo + chunk, n)))}}
            for lo in range(0, n, chunk)]
    sp = {"slurm_partition": "gpu", "gpus_per_node": 1, "cpus_per_task": 8, "mem_gb": 96, "timeout_min": 60,
          "slurm_constraint": "[a100_80|h100|h200|6000_blackwell]",
          "slurm_setup": ["export OPS_DIFFEX_ASSETS=viewer_assets_v5", "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]}
    print(f"[vs-npm3] {len(jobs)} shards → {OUT_MOD}/{GRAIN}/{TARGET}")
    submit_parallel_jobs(jobs_to_submit=jobs, experiment="diffex_vsnpm3", slurm_params=sp, log_dir="diffex_vsnpm3", wait_for_completion=False)


if __name__ == "__main__":
    submit()
