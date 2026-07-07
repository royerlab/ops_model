"""latent-lens multiscale UMAP montage: place each gene's ALREADY-GENERATED traversal frame at its
gene-UMAP coordinate, so panning the embedding shows one cell morphed toward each neighborhood.

Harvests the traversal CACHE (`viewer_assets/<modality>/<grain>/<slug>/cell<c>/frame_<i>.webp`) — the
frames were decoded with the correct top-attention directions, so no re-embedding / no gene_bulked
centroids (those are all-cell means → ~13× too weak → collapsed morphs). Layout = the phase gene UMAP
`X_umap`. One α (frame index) per montage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import anndata as ad
from PIL import Image

from latent_lens import MontageConfig, build_montage

from ..classifier.config import slugify
from .precompute import VIEWER_ALPHAS

OUT = "/hpc/projects/icd.fast.ops/models/diffex"


def _embed_coords(ann, embedding, span=12.0):
    """obsm X_<embedding> → coords auto-oriented so NTC sits bottom-left, rescaled to a common `span`
    (so UMAP and PHATE montages are comparable size and share px_per_umap regardless of native scale)."""
    c = np.asarray(ann.obsm[f"X_{embedding}"]).astype(float).copy()
    pert = ann.obs["perturbation"].astype(str).values
    ntc = np.array([p.startswith("NTC") for p in pert])
    lo, hi = c.min(0), c.max(0); mid = (lo + hi) / 2
    if ntc.any():
        nc = c[ntc].mean(0)
        if nc[0] > mid[0]: c[:, 0] = lo[0] + hi[0] - c[:, 0]   # NTC → left (small x)
        if nc[1] < mid[1]: c[:, 1] = lo[1] + hi[1] - c[:, 1]   # NTC → large y (bottom, since y maps top→bottom)
    lo = c.min(0); s = span / max((c.max(0) - lo).max(), 1e-9)
    return (c - lo) * s


def montage_from_cache(h5ad, out_zarr, cell=0, alpha=2.0, modality="phase", grain="geneKO",
                       tile=256, px_per_umap=5600, embedding="umap"):
    """Build the montage from the traversal cache: each gene tile = its cell-`cell`, α=`alpha` frame,
    placed at the gene's position in `embedding` (obsm X_<embedding>, e.g. umap or phate).
    Crops kept GRAYSCALE (white category tint). crop_size=256 sharp; px_per_umap≈22×crop fills canvas."""
    al = list(VIEWER_ALPHAS)
    ai = int(np.argmin([abs(a - alpha) for a in al]))     # frame index for the requested α
    a0 = int(np.argmin([abs(a) for a in al]))             # α=0 frame index (shared anchor recon)
    ann = ad.read_h5ad(h5ad)
    coords_all = _embed_coords(ann, embedding)
    gc = {str(g): coords_all[i] for i, g in enumerate(ann.obs["perturbation"])}
    va = f"{OUT}/viewer_assets/{modality}/{grain}"

    genes, coords, srcs, ntc = [], [], [], []
    for g, xy in gc.items():
        if str(g).startswith("NTC"):                      # NTC is split into ~50 NTC_grp* embedding nodes
            ntc.append((g, xy)); continue
        p = f"{va}/{slugify(g)}/cell{cell}/frame_{ai:02d}.webp"
        try:
            Image.open(p)   # exists + readable
        except Exception:
            continue
        genes.append(g); coords.append(xy); srcs.append((slugify(g), cell, ai))
    ntc_ref = srcs[0][0] if srcs else None
    if ntc_ref:                                           # NTC nodes = the SAME cell `cell` at α=0 (base NTC recon,
        for g, xy in ntc:                                 # no morph): NTC→NTC_group direction is ~0, so α=0 is exact
            genes.append(g); coords.append(xy); srcs.append((ntc_ref, cell, a0))
    coords = np.asarray(coords, dtype=np.float32)
    print(f"[montage] {len(srcs) - len(ntc)} genes + {len(ntc)} NTC nodes (cell {cell}, α={al[ai]:g}, {embedding})")

    def crops(i):
        slug, c, fi = srcs[i]
        return np.asarray(Image.open(f"{va}/{slug}/cell{c}/frame_{fi:02d}.webp").convert("L"))

    build_montage(umap_coords=coords, crops=crops, categories=np.array(["geneKO"] * len(genes)),
                  category_colors={"geneKO": (1.0, 1.0, 1.0)}, output_path=out_zarr,   # white = no tint → grayscale
                  labels=np.array(genes), config=MontageConfig(crop_size=tile, px_per_umap=px_per_umap))
    print(f"[montage] wrote {out_zarr}: {len(genes)} genes")
    return out_zarr, genes


def build_montage_web(h5ad, out_zarr, cell=0, alpha=2.0, embedding="umap"):
    """One step for the viewer: harvest the cache → montage zarr → PNG tiles + labels (in `embedding`)."""
    _, placed = montage_from_cache(h5ad, out_zarr, cell=cell, alpha=alpha, embedding=embedding)
    return montage_to_tiles(out_zarr, h5ad, out_dir=str(out_zarr)[:-5] + "_tiles", placed=set(placed), embedding=embedding)


def montage_to_tiles(zarr_path, h5ad, out_dir=None, tile=512, placed=None,
                     embedding="umap", color_fields=None):
    """Transcode the montage OME-Zarr RGB pyramid → PNG tiles + tiles.json + labels.json for the
    web viewer (OpenSeadragon). Avoids blosc-in-browser; served same-origin from viewer_assets."""
    import json
    import zarr
    zg = zarr.open(str(zarr_path), mode="r")
    at = dict(zg.attrs)
    ext = at["umap_extent"]
    levels = sorted((d["path"] for d in at["multiscales"][0]["datasets"]), key=int)
    import shutil
    out = Path(out_dir or (str(zarr_path)[:-5] + "_tiles"))
    if out.exists():
        shutil.rmtree(out)                          # clear stale tiles from prior builds (else orphan crops persist)
    out.mkdir(parents=True, exist_ok=True)
    W0 = H0 = 0
    for k in levels:
        a = zg[k]                                     # (3, Y, X) float32 in [0,1]
        _, Y, X = a.shape
        if k == "0":
            W0, H0 = X, Y
        ld = out / f"L{k}"; ld.mkdir(exist_ok=True)
        for row in range(0, Y, tile):
            for col in range(0, X, tile):
                blk = np.asarray(a[:, row:row + tile, col:col + tile])
                if not blk.any():
                    continue                          # skip background (sparse montage) → OSD treats as transparent
                img = (np.clip(blk.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
                Image.fromarray(img).save(ld / f"{col // tile}_{row // tile}.png")
    # color-by fields: ALL usable categorical obs columns (auto-detected) — non-numeric, 2–300 distinct,
    # not list-like, not free-text. Covers complexes, all leiden/ontology resolutions, GO/Reactome/KEGG, etc.
    ann = ad.read_h5ad(h5ad); obs = ann.obs
    if color_fields is None:
        color_fields = []
        for c in obs.columns:
            s = obs[c]
            if s.dtype.kind in "fiu":
                continue
            v = s.astype(str); n = v.nunique()
            if 2 <= n <= 300 and v.str.startswith("[").mean() <= 0.3 and v.str.len().mean() <= 60:
                color_fields.append(c)
    fields = [c for c in color_fields if c in obs.columns]
    (out / "tiles.json").write_text(json.dumps(
        {"width": W0, "height": H0, "tileSize": tile, "levels": [int(k) for k in levels],
         "embedding": f"phase gene {embedding.upper()} (CellDINO)", "color_fields": fields}))

    # gene labels: umap coord → level-0 pixel (y flipped to match the montage build), normalized by width.
    # Each carries has-crop (else the viewer draws a dot) + categorical fields for color-by overlays.
    coords = _embed_coords(ann, embedding)              # same auto-orient + rescale as the build
    placed = placed or set()
    pu = ext["px_per_umap"]
    def _cat(v):
        s = str(v)
        return "" if s in ("nan", "NaN", "None", "") else s
    labels = []
    for i, g in enumerate(obs["perturbation"]):
        g = str(g)                                  # NTC included (its α=0 anchor tile is placed)
        rec = {"g": g, "nx": float((coords[i, 0] - ext["xmin"]) * pu / W0),
               "ny": float((coords[i, 1] - ext["ymin"]) * pu / W0), "crop": g in placed}
        for c in fields:
            rec[c] = _cat(obs[c].iloc[i]) if c in obs else ""
        labels.append(rec)
    (out / "labels.json").write_text(json.dumps(labels))
    print(f"[tiles] {out}: {len(levels)} levels, {len(labels)} labels "
          f"({sum(l['crop'] for l in labels)} with crops), canvas {W0}x{H0}")
    return str(out)
