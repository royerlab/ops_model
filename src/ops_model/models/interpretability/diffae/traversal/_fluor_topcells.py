"""Per-marker fluorescence Top Cells (top-30 by v5 accuracy) — crops the actual MARKER channel
(reuses materialize_crops, name-based out_channels=[cfg.channel], exactly like the fluor traversals),
PLUS a tiny transparent-inside / blue-outside seg overlay per cell so the viewer can toggle the mask
client-side (no doubled crop cache).

Output: viewer_assets_v5/top_cells/markers/<channel-slug>/{crops/*.webp, overlays/*.png, index.json}
"""
import os, json
import numpy as np
import pandas as pd
import zarr
from PIL import Image
from . import catalog as C
from ..classifier.config import slugify, GRAINS
from ..classifier.data import make_labels_df, materialize_crops
from ..directions.config import DirConfig
from ..generator.data import normalize
from .precompute import _save_webp
from .build_pc_crops_masked import BASE, CROP_SIZE, MASK_DILATION, OVERLAY_RGB, OVERLAY_ALPHA, _crop, _zarr_patch

ASSETS = "viewer_assets_v5"
RANKDIR = f"{C.OUT}/{ASSETS}/_rankings/fluor_shap/geneKO"     # NEW shap_screen per-channel rankings (same cells as traversals)
OUT = f"{C.OUT}/{ASSETS}/top_cells/markers"
TOP_N = 40


def _overlay_rgba(seg, half):
    """Transparent inside the (dilated) center cell, blue+alpha outside → RGBA uint8 (the toggleable mask layer)."""
    from scipy.ndimage import binary_dilation
    center = seg[half, half]
    if center == 0:
        c = seg[half - 12:half + 12, half - 12:half + 12]; nz = c[c > 0]
        center = np.bincount(nz).argmax() if nz.size else 0
    rgba = np.zeros((*seg.shape, 4), np.uint8)
    if center != 0:
        inv = ~binary_dilation(seg == center, iterations=MASK_DILATION)
        rgba[inv, 0], rgba[inv, 1], rgba[inv, 2] = [int(v * 255) for v in OVERLAY_RGB]
        rgba[inv, 3] = int(OVERLAY_ALPHA * 255)
    return rgba


def crop_marker_shard(mc, ch, top_n=TOP_N, rankdir=RANKDIR, block="genes"):
    """Crop top-N accuracy cells per class for one marker channel → per-marker crops/ + overlays/ + index.json.
    block="genes" (geneKO) or "complexes" (EBI); complex crops MERGE into the marker's existing index.json."""
    _zarr_patch()
    df = pd.read_parquet(f"{rankdir}/{slugify(mc)}.parquet")
    df = df[df["rank"] <= top_n]                              # include NTC (its top-N controls) — pinned by default in the tab
    recs = df.rename(columns={"gene": "cls", "pma_attention": "score"}).copy()
    recs["label"] = 0
    cfg = DirConfig(grain="geneKO", target=recs["cls"].iloc[0], device="cpu")
    cfg.marker_channel = mc; cfg.channel = ch; cfg.num_workers = 8
    raw, _, exps = materialize_crops(make_labels_df(recs, cfg), cfg, cache_path=None)   # marker channel (raw intensity); drops failed-store experiments
    recs = recs[recs["experiment"].isin(set(exps))].reset_index(drop=True)   # realign to surviving cells (materialize drops whole failed experiments, in order)
    pc = normalize(raw)                                       # per-cell z-score (current default)
    lo, hi = np.percentile(raw, (1, 99))                      # marker-global intensity window (over ALL this marker's cells)
    if hi - lo < 1e-6:
        hi = lo + 1
    out = f"{OUT}/{slugify(mc)}"; cdir, ndir, odir = f"{out}/crops", f"{out}/crops_norm", f"{out}/overlays"
    for dd in (cdir, ndir, odir):
        os.makedirs(dd, exist_ok=True)
    half = CROP_SIZE // 2
    segcache, genes = {}, {}
    for i, r in recs.iterrows():
        if i >= len(raw):
            break
        key = f"{r['experiment']}_{r['well']}_{int(round(r['x_pheno']))}_{int(round(r['y_pheno']))}".replace("/", "-")
        _save_webp(f"{cdir}/{key}.webp", pc[i, 0], 256)                                        # per-cell normalized
        gnorm = np.clip((raw[i, 0] - lo) / (hi - lo), 0, 1)                                    # marker-global normalized (intensity comparable)
        Image.fromarray((gnorm * 255).astype(np.uint8)).resize((256, 256), Image.BILINEAR).save(f"{ndir}/{key}.webp")
        # seg overlay (same store/coords → aligned with the marker crop)
        ek = (r["experiment"], r["well"])
        if ek not in segcache:
            pos = f"{BASE}/{r['experiment']}/3-assembly/phenotyping_v3.zarr/{str(r['well'])[0]}/{str(r['well'])[1:]}/0"
            try:
                segcache[ek] = zarr.open(f"{pos}/labels/cell_seg/0", mode="r")
            except Exception:
                segcache[ek] = None
        if segcache[ek] is not None:
            try:
                seg = _crop(segcache[ek], None, int(round(r["x_pheno"])), int(round(r["y_pheno"])), half)
                ov = Image.fromarray(_overlay_rgba(seg, half)).resize((256, 256), Image.NEAREST)
                ov.save(f"{odir}/{key}.png")
            except Exception:
                pass
        genes.setdefault(str(r["cls"]), []).append(
            {"img": f"{key}.webp", "ov": f"{key}.png", "exp": r["experiment"], "well": str(r["well"]),
             "x": int(round(r["x_pheno"])), "y": int(round(r["y_pheno"])),
             "rank": int(r["rank"]), "conf": round(float(r.get("score", 0) or 0), 5)})
    for g in genes:
        genes[g] = sorted(genes[g], key=lambda c: c["rank"])[:top_n]
    ipath = f"{out}/index.json"
    prev = json.load(open(ipath)) if os.path.exists(ipath) else {}
    idx = {k: prev[k] for k in ("genes", "complexes", "top_n") if k in prev}   # merge; drop any stray keys
    idx[block] = {g: {"attention": [], "accuracy": genes[g]} for g in sorted(genes)}
    idx["top_n"] = top_n
    json.dump(idx, open(ipath, "w"))
    return {"marker": mc, block: len(genes), "cells": sum(len(v) for v in genes.values())}


def main():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    cm = {mc: (d, ch) for d, mc, ch in C.complete_markers()}
    jobs = []
    for mc, (d, ch) in cm.items():
        if os.path.exists(f"{RANKDIR}/{slugify(mc)}.parquet"):
            jobs.append({"name": f"ftc_{slugify(mc)[:20]}", "func": crop_marker_shard, "kwargs": {"mc": mc, "ch": ch}})
    print(f"[fluor-topcells] {len(jobs)} marker crop jobs (top {TOP_N} + overlays)")
    submit_parallel_jobs(jobs, experiment="diffex_fluor_topcells",
                         slurm_params={"slurm_partition": "cpu", "cpus_per_task": 8, "mem_gb": 32, "timeout_min": 150},
                         log_dir="diffex_fluor_topcells", wait_for_completion=False)


if __name__ == "__main__":
    main()
