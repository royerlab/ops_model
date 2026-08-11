"""Morphometrics on GENERATED cells: run each marker's organelle segmentation (org_seg_params) on
the generated α-frames (whole crop, no cell mask), measure classical org-profiler features, and
compare their α-trajectory to the REAL NTC→KO shift (op_cp_features) to show the generated images
AMPLIFY the real morphometric change.

`preview()` renders the design: org-seg masks + per-organelle feature colormaps across α, so we can
eyeball that segmentation + feature measurement work before scaling the full compute/cache.
"""
from __future__ import annotations

import numpy as np
import yaml

ORG_SEG_YAML = "/hpc/projects/intracellular_dashboard/fast_ops/configs/org_seg_params.yaml"
CACHE = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets"
PIXEL_UM = 0.325                                   # phenotype native pixel size


_DEFAULT_FRANGI = {"method": "frangi", "frangi": {"pixel_size_um": 0.1, "min_object_size": 10, "postprocess": True}}


def seg_config(marker_key):
    """org_seg_params block for a marker key (e.g. 'LAMP1') → (method, params dict). Markers with no
    config (e.g. ChromaLIVE) fall back to a generic frangi (tubular/network-ish)."""
    y = yaml.safe_load(open(ORG_SEG_YAML))
    for blk in y.get(marker_key, []):
        if isinstance(blk, dict) and "segmentation_config" in blk:
            sc = blk["segmentation_config"]
            return sc.get("method", "frangi"), sc
    return "frangi", _DEFAULT_FRANGI


def seg_organelles(crop, marker_key, pixel_um=PIXEL_UM):
    """Generated crop (H,W float) → labeled organelle mask via the marker's org_seg_params method.
    blob → LoG (discrete vesicles); frangi/tubular → vesselness ridge → connected components (fragments,
    so `count` = network fragmentation for mito/MICOS)."""
    method, sc = seg_config(marker_key)
    if method == "blob":
        from organelle_profiler.organelle_seg.blob_detection import _segment_blob_log
        return _segment_blob_log(crop.astype(np.float32), pixel_um, sc["blob"])
    from skimage.filters import frangi, threshold_otsu
    from skimage.measure import label
    from skimage.morphology import remove_small_objects
    v = frangi(crop.astype(np.float32), black_ridges=False)
    if not (v > 0).any():
        return np.zeros(crop.shape, np.int32)
    m = v > threshold_otsu(v[v > 0])
    m = remove_small_objects(m, min_size=8)
    return label(m).astype(np.int32)


def _org_feats(labels, intensity):
    """Per-organelle regionprops → dict of {label: {area, mean_int, eccentricity}} + aggregates."""
    from skimage.measure import regionprops
    rp = regionprops(labels, intensity_image=intensity)
    feats = {r.label: {"area": r.area, "mean_int": r.intensity_mean,
                       "ecc": r.eccentricity if r.area >= 5 else 0.0} for r in rp}
    agg = {"count": len(rp), "total_area": sum(f["area"] for f in feats.values()),
           "mean_int": float(np.mean([f["mean_int"] for f in feats.values()])) if feats else 0.0}
    return feats, agg


def _paint(labels, feats, key, cmap):
    """Color each organelle by feats[label][key] → RGB image (background black)."""
    import matplotlib.cm as cm
    vals = np.array([f[key] for f in feats.values()]) if feats else np.array([0.0])
    lo, hi = float(vals.min()), float(vals.max() + 1e-9)
    rgb = np.zeros((*labels.shape, 3), np.float32)
    mp = cm.get_cmap(cmap)
    for lab, f in feats.items():
        rgb[labels == lab] = mp((f[key] - lo) / (hi - lo))[:3]
    return rgb


AGG_KEYS = ["count", "total_area", "mean_area", "mean_int", "mean_ecc"]


def _aggregate(feats):
    if not feats:
        return {k: 0.0 for k in AGG_KEYS}
    A = [f["area"] for f in feats.values()]; I = [f["mean_int"] for f in feats.values()]; E = [f["ecc"] for f in feats.values()]
    return {"count": len(feats), "total_area": float(np.sum(A)), "mean_area": float(np.mean(A)),
            "mean_int": float(np.mean(I)), "mean_ecc": float(np.mean(E))}


def compute_target(marker_key, marker_dir, target, grain="geneKO", n_cells=6):
    """Cache morphometrics for one traversal: per (cell, α) the org-seg label mask (PNG) + per-organelle
    features (JSON) + per-α bag aggregates (for the plot). Generated side only (real NTC/KO ref added
    once process_single_cell matches the op_cp_features schema). → viewer_assets/_morphometrics/<dir>/<grain>/<target>/"""
    import json
    import os
    from PIL import Image
    base = f"{CACHE}/{marker_dir}/{grain}/{target}"
    mp = f"{base}/cell0/meta.json" if os.path.exists(f"{base}/cell0/meta.json") else f"{base}/meta.json"
    alphas = json.load(open(mp))["alphas"]
    out = f"{CACHE}/_morphometrics/{marker_dir}/{grain}/{target}"
    os.makedirs(out, exist_ok=True)
    agg_per_cell = []                                  # [cell][alpha] = agg dict
    for c in range(n_cells):
        cdir = f"{out}/cell{c}"; os.makedirs(cdir, exist_ok=True)
        cell_feats, cell_agg = {}, []
        for ai in range(len(alphas)):
            f = f"{base}/cell{c}/frame_{ai:02d}.webp"
            if not os.path.exists(f):
                cell_agg.append({k: 0.0 for k in AGG_KEYS}); continue
            im = np.asarray(Image.open(f).convert("L"), np.float32) / 255.0
            labels = seg_organelles(im, marker_key)
            feats, _ = _org_feats(labels, im)
            Image.fromarray(labels.astype(np.uint16)).save(f"{cdir}/a{ai:02d}_labels.png")
            cell_feats[ai] = {str(k): {kk: float(vv) for kk, vv in v.items()} for k, v in feats.items()}
            cell_agg.append(_aggregate(feats))
        json.dump(cell_feats, open(f"{cdir}/feats.json", "w"))
        agg_per_cell.append(cell_agg)
    # bag mean per α across cells
    agg = {k: [float(np.mean([agg_per_cell[c][ai][k] for c in range(n_cells)])) for ai in range(len(alphas))] for k in AGG_KEYS}
    json.dump({"marker_key": marker_key, "marker_dir": marker_dir, "target": target, "grain": grain,
               "alphas": alphas, "n_cells": n_cells, "agg": agg, "features": AGG_KEYS},
              open(f"{out}/morpho.json", "w"))
    print(f"[morpho] {marker_dir}/{grain}/{target}: {n_cells} cells × {len(alphas)} α cached -> {out}")
    print(f"  count α-series: {[round(v,1) for v in agg['count']]}")
    print(f"  total_area α-series: {[round(v,0) for v in agg['total_area']]}")
    return out


def preview(marker_key="LAMP1", marker_dir="lysosome_LAMP1", target="ABCE1", cell=0,
            alpha_idxs=(0, 6, 8, 12, 16), out="/hpc/projects/icd.fast.ops/models/diffex/morpho_preview.png"):
    """Design preview: rows = α; cols = [raw crop | org-seg outlines | organelles colored by area |
    by mean-intensity | by eccentricity]. Title per row = aggregate (count, total area)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from skimage.segmentation import find_boundaries
    plt.rcParams["pdf.fonttype"] = 42

    import json
    base = f"{CACHE}/{marker_dir}/geneKO/{target}"
    meta = json.load(open(f"{base}/cell{cell}/meta.json")) if __import__("os").path.exists(f"{base}/cell{cell}/meta.json") \
        else json.load(open(f"{base}/meta.json"))
    alphas = meta["alphas"]
    cols = ["raw crop", "org-seg", "→ area", "→ mean intensity", "→ eccentricity"]
    fig, ax = plt.subplots(len(alpha_idxs), len(cols), figsize=(len(cols) * 2.4, len(alpha_idxs) * 2.4))
    for r, ai in enumerate(alpha_idxs):
        im = np.asarray(Image.open(f"{base}/cell{cell}/frame_{ai:02d}.webp").convert("L"), np.float32) / 255.0
        labels = seg_organelles(im, marker_key)
        feats, agg = _org_feats(labels, im)
        panels = [(im, "gray", None), (find_boundaries(labels), "gray", None),
                  (_paint(labels, feats, "area", "viridis"), None, None),
                  (_paint(labels, feats, "mean_int", "inferno"), None, None),
                  (_paint(labels, feats, "ecc", "plasma"), None, None)]
        for c, (img, cmap, _) in enumerate(panels):
            a = ax[r, c]
            if c == 1:
                a.imshow(im, cmap="gray"); a.imshow(np.ma.masked_where(~img, img), cmap="autumn", alpha=0.9)
            else:
                a.imshow(img, cmap=cmap)
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(cols[c], fontsize=9)
            if c == 0:
                a.set_ylabel(f"α={alphas[ai]:+.0f}\n{agg['count']} org\narea {int(agg['total_area'])}", fontsize=8)
    fig.suptitle(f"Morphometrics preview — {marker_key} / {target} / cell{cell}: org-seg + per-organelle feature colormaps across α",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[morpho-preview] {out}")


if __name__ == "__main__":
    preview()
