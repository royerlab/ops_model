"""Render Kevin's CellDINO attention-head pixel-attribution npz into static WebP tiles the viewer
overlays (inferno) on the real phenotype cells. The npz cells ARE the phase·geneKO top-20 phenotype
cells (`phenotype_cells.py`); each gene has its top-6 attention heads, one attribution map per cell.

Input  (per gene): {AH}/pixel_attribution_cache/<GENE>.npz
    maps (n_cells, n_heads, 128, 128) f16  · crops (n_cells, 128, 128) f32 (z-scored)
    heads (n_heads, 2) int32 = (layer, head) · patch_masks (n_cells, 196) bool
  + {AH}/head_rankings_per_gene.json  (per-gene ranked heads + metrics, order matches `heads`).

Output (webapp-ready, dependency-free — inferno LUT + normalization applied LIVE in-browser):
    {AH}/<GENE>/cell<c>/crop.webp        grayscale crop, per-crop robust min-max
    {AH}/<GENE>/cell<c>/head<h>.webp     grayscale attribution, scaled by that gene's max (full 8-bit)
    {AH}/<GENE>/heads.json               {gene, n_cells, gene_max, heads:[{layer,head,feature,...}]}
    {AH}/index.json                      {global_max, genes:[...]}  (availability + fixed-norm scale)

Normalization is deferred to the client so Display can switch per-map / per-gene / fixed live:
per-gene encoding preserves each gene's full 8-bit precision; `gene_max`/`global_max` let JS rescale.
"""
from __future__ import annotations

import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from . import catalog as C

# Source = Kevin's persistent analysis dump; rendered WebP tiles go to a separate `phase/` output the
# webapp reads (keep source & derived split so wiping the rendered dir never loses the inputs).
SRC = f"{C.OUT}/viewer_assets/attention_heads/celldino_attention_head_analysis"
CACHE = f"{SRC}/pixel_attribution_cache"
RANKINGS = f"{SRC}/head_rankings_per_gene.json"
AH = f"{C.OUT}/viewer_assets/attention_heads/phase"


def _save_gray(path, u8, upsize):
    im = Image.fromarray(u8)
    if upsize:
        im = im.resize((upsize, upsize), Image.BILINEAR)
    im.save(path, quality=90, method=6)


def _crop_u8(crop):
    """z-scored crop → uint8 via robust (1–99 pct) min-max, matching typical grayscale display."""
    lo, hi = np.percentile(crop, (1, 99))
    if hi <= lo:
        hi = lo + 1e-6
    return (np.clip((crop - lo) / (hi - lo), 0, 1) * 255).astype("uint8")


def build(genes=None, upsize=256, n_workers=8, out=AH):
    rankings = json.load(open(RANKINGS))
    paths = sorted(glob.glob(f"{CACHE}/*.npz"))
    if genes:
        want = set(genes)
        paths = [p for p in paths if Path(p).stem in want]
    global_max, done, bad = 0.0, [], []
    for p in paths:
        gene = Path(p).stem
        try:
            d = np.load(p, allow_pickle=True)
            maps, crops, heads, pmasks = d["maps"].astype(np.float32), d["crops"], d["heads"], d["patch_masks"]
        except Exception as e:                       # truncated/still-writing npz → skip loudly, don't abort
            bad.append(gene)
            print(f"[attn] SKIP {gene}: unreadable npz ({type(e).__name__})")
            continue
        n_cells, n_heads = maps.shape[:2]
        # Match Ritvik's overlay pipeline: Gaussian-smooth each map (σ=2), then mask to the CELL
        # (patch_masks → 14×14 → 128 nearest); outside the cell → 0. Bake both here; clim + alpha stay live.
        H = maps.shape[-1]
        grid = int(round(pmasks.shape[1] ** 0.5))
        idx = np.minimum((np.arange(H) * grid // H), grid - 1)   # nearest upscale 14→128
        cell_masks = np.empty((n_cells, H, H), bool)
        pmaps = np.empty_like(maps)
        for c in range(n_cells):
            cm = pmasks[c].reshape(grid, grid)[np.ix_(idx, idx)]
            cell_masks[c] = cm
            for h in range(n_heads):
                pmaps[c, h] = np.where(cm, gaussian_filter(maps[c, h], sigma=2.0), 0.0)
        gene_max = float(pmaps.max())
        global_max = max(global_max, gene_max)
        scale = 255.0 / gene_max if gene_max > 0 else 0.0
        gdir = Path(out) / gene
        pool = ThreadPoolExecutor(max_workers=n_workers)
        for c in range(n_cells):
            cdir = gdir / f"cell{c}"
            cdir.mkdir(parents=True, exist_ok=True)
            pool.submit(_save_gray, cdir / "crop.webp", _crop_u8(crops[c]), upsize)
            pool.submit(_save_gray, cdir / "mask.webp", (cell_masks[c] * 255).astype("uint8"), upsize)
            for h in range(n_heads):
                u8 = np.clip(pmaps[c, h] * scale, 0, 255).astype("uint8")
                pool.submit(_save_gray, cdir / f"head{h}.webp", u8, upsize)
        pool.shutdown(wait=True)
        # per-head metrics: match npz head index -> (layer,head) -> ranking metrics (order-independent)
        rk = {(r["layer"], r["head"]): r for r in rankings.get(gene, [])}
        head_meta = []
        for h in range(n_heads):
            layer, head = int(heads[h][0]), int(heads[h][1])
            m = rk.get((layer, head), {})
            head_meta.append({"layer": layer, "head": head, "feature": m.get("feature"),
                              "spec_p10": m.get("spec_p10"), "spec_min": m.get("spec_min"),
                              "auroc_vs_ntc": m.get("auroc_vs_ntc")})
        (gdir / "heads.json").write_text(json.dumps(
            {"gene": gene, "n_cells": n_cells, "gene_max": gene_max, "heads": head_meta}))
        done.append(gene)
        if len(done) % 50 == 0:
            print(f"[attn] {len(done)}/{len(paths)} genes rendered")
    (Path(out) / "index.json").write_text(json.dumps(
        {"global_max": global_max, "n_heads_max": 6, "genes": sorted(done)}))
    print(f"[attn] done: {len(done)} genes, global_max={global_max:.4f} -> {out}/index.json")
    if bad:
        print(f"[attn] {len(bad)} unreadable npz skipped: {', '.join(bad)}")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--genes", nargs="*", help="subset of genes (default: all in cache)")
    ap.add_argument("--upsize", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    build(genes=args.genes, upsize=args.upsize, n_workers=args.workers)
