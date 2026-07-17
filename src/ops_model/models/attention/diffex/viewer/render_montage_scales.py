"""Stitch a DeepZoom montage level into a single composite PNG (white bg) with a bottom-left
embedding legend (leiden_r4, big dots, NTC as a dark labelled circle). The montage image and its
baked viewer-style gene names come straight from the built tiles — finer levels give crisper text.

  python -m ops_model.models.attention.diffex.viewer.render_montage_scales --alphas 1-5 --levels 3,4

Each level of `_montage/phase_geneKO_phate_cell1_a<A>_tiles/L<lvl>/` is a level-of-detail montage
(coarse levels show a decimated non-overlapping subset; finer levels fill in more cells at higher res).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Circle, Rectangle
from PIL import Image

plt.rcParams["pdf.fonttype"] = 42
Image.MAX_IMAGE_PIXELS = None

VA = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets"
OUT_DIR = "/hpc/projects/icd.fast.ops/analysis/figure4_embedding/montage_scales"
HIRES_ROOT = "/hpc/projects/icd.fast.ops/analysis/figure4_embedding/hires_tiles"
COMPOSED_DIR = "/hpc/projects/icd.fast.ops/analysis/figure4_embedding/montage_composed"
UMAP_H5AD = ("/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3/cell_dino/"
             "zscore_per_exp/paper_v2/phase_only/fixed_80%/cosine/gene_embedding_pca_optimized.h5ad")
LEIDEN = "leiden_r4"
VIEWER_ALPHAS = [-5.0, -4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
# EBI complexes to box in the montage: label -> (complex name in labels.json, color)
MARKS = {"40S": ("40S cytosolic small ribosomal subunit", "#0b6b73"),   # dark teal / dark orange for contrast
         "60S": ("60S cytosolic large ribosomal subunit", "#b35900")}
CPLX_TRAVERSAL = "40S_cytosolic_small_ribosomal_subunit__to__60S_cytosolic_large_ribosomal_subunit"
MARK_GENE = {"40S": "RPS25", "60S": "RPL37A"}   # specific member to box (verified in the complex)


def _frame(alpha):
    return max(0, min(16, int(round(8 + alpha / 5.0 * 8))))   # 17 frames span alpha -5..+5; 8 = base


def _stretch(im, lo_p=3.0, hi_p=97.0):
    """Per-crop percentile contrast stretch (latent_lens _normalize_crop uses 1/99; tighter = punchier).
    Input/return uint8."""
    a = im.astype(np.float32)
    lo, hi = np.percentile(a, lo_p), np.percentile(a, hi_p)
    if hi <= lo:
        return im
    return (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def _composite(tile_dir, level, width, height, ts, bg=255):
    lw = max(1, width >> level); lh = max(1, height >> level)
    canv = np.full((lh, lw), bg, np.uint8)
    d = f"{tile_dir}/L{level}"
    n = 0; x0 = y0 = 1 << 30; x1 = y1 = 0
    for f in os.listdir(d):
        if not f.endswith(".png"):
            continue
        col, row = (int(x) for x in f[:-4].split("_"))
        t = np.asarray(Image.open(f"{d}/{f}").convert("L"))
        py, px = row * ts, col * ts; th, tw = t.shape
        canv[py:py + th, px:px + tw] = t[: lh - py, : lw - px]
        n += 1
        x0 = min(x0, px); y0 = min(y0, py); x1 = max(x1, px + tw); y1 = max(y1, py + th)
    m = ts // 4          # canvas white; tiles keep their black interior (bright points render on black)
    bbox = (max(0, x0 - m), max(0, y0 - m), min(lw, x1 + m), min(lh, y1 + m))
    return canv, n, bbox


def _leiden_colors(labels):
    clusters = sorted({g.get(LEIDEN, "") for g in labels if g.get(LEIDEN, "") != ""},
                      key=lambda s: (len(s), s))
    cmap = plt.get_cmap("hsv")   # all bright/saturated, no dark clusters (dark is reserved for NTC)
    return {c: cmap(i / max(1, len(clusters) - 1)) for i, c in enumerate(clusters)}


def _legend(ax, labels, cw, dpi, box=(0.008, 0.008, 0.26, 0.26), dark=False):
    lut = _leiden_colors(labels)
    ex = np.array([g["nx"] for g in labels]); ey = np.array([g["ny"] for g in labels])
    ec = np.array([lut.get(g.get(LEIDEN, ""), (0.7, 0.7, 0.7, 1)) for g in labels])
    is_ntc = np.array([str(g["g"]).startswith("NTC") for g in labels])
    tcol = "white" if dark else "#111"
    iax = ax.inset_axes(list(box))
    if dark:
        iax.set_facecolor("white")
    dot = (cw / dpi) * 6
    iax.scatter(ex[~is_ntc], ey[~is_ntc], c=ec[~is_ntc], s=dot, edgecolors="none", zorder=2)
    if is_ntc.any():
        nx, ny = ex[is_ntc], ey[is_ntc]
        iax.scatter(nx, ny, c="#111", s=dot * 1.2, edgecolors="none", zorder=4)
        cx, cy = nx.mean(), ny.mean()
        rr = 1.7 * np.percentile(np.hypot(nx - cx, ny - cy), 90) + 0.01
        iax.add_patch(Circle((cx, cy), rr, fill=False, ls="--", ec="#111", lw=max(1.2, cw / dpi * 0.15), zorder=5))
        iax.annotate("NTC", (cx, cy - rr * 1.5), ha="center", va="bottom",
                     fontsize=max(10, cw / dpi * 0.95), weight="bold", color="#111")
    iax.set_aspect("equal"); iax.invert_yaxis(); iax.set_xticks([]); iax.set_yticks([])
    for s in iax.spines.values():
        s.set_visible(False)
    iax.set_title(f"embedding · {LEIDEN}", fontsize=max(11, cw / dpi * 1.1), weight="bold", color=tcol)


def _mark(ax, tile_dir, labels, W, lvl, ts, x0c, y0c, cw, dpi):
    """Box the montage tile nearest each marked complex's embedding centroid + label it."""
    lw = W >> lvl
    occ = [tuple(int(v) for v in f[:-4].split("_"))
           for f in os.listdir(f"{tile_dir}/L{lvl}") if f.endswith(".png")]
    for lab, (cname, color) in MARKS.items():
        pts = [(g["nx"] * lw, g["ny"] * lw) for g in labels if g.get("ebi_complex") == cname]
        if not pts:
            continue
        cx = np.mean([p[0] for p in pts]); cy = np.mean([p[1] for p in pts])
        col, row = min(occ, key=lambda t: ((t[0] + 0.5) * ts - cx) ** 2 + ((t[1] + 0.5) * ts - cy) ** 2)
        rx, ry = col * ts - x0c, row * ts - y0c
        ax.add_patch(Rectangle((rx, ry), ts, ts, fill=False, ec=color, lw=max(3, cw / dpi * 0.5), zorder=6))
        ax.text(rx + ts / 2, ry - 6, lab, ha="center", va="bottom", color=color,
                fontsize=max(13, cw / dpi * 1.3), weight="bold", zorder=7)


def build_hires(alphas, cell=1, crop=512, embedding="phate", border_field=LEIDEN):
    """Rebuild the phase geneKO montage at a larger crop_size (crisper baked text, 64px font at crop 512)
    with per-cell leiden borders, to a SEPARATE dir (viewer cache untouched)."""
    from .build_umap_montage import montage_from_cache, montage_to_tiles, ZARR_SCRATCH
    import shutil
    os.makedirs(HIRES_ROOT, exist_ok=True)
    for a in alphas:
        stem = f"phase_geneKO_{embedding}_cell{cell}_a{a:g}"
        scratch = f"{ZARR_SCRATCH}/{stem}_hires.zarr"
        _, placed = montage_from_cache(UMAP_H5AD, scratch, cell=cell, alpha=a, modality="phase", grain="geneKO",
                                       tile=crop, px_per_umap=int(round(crop * 5600 / 256)),
                                       embedding=embedding, border_field=border_field)
        montage_to_tiles(scratch, UMAP_H5AD, out_dir=f"{HIRES_ROOT}/{stem}_tiles",
                         placed=set(placed), embedding=embedding)
        shutil.rmtree(scratch, ignore_errors=True)
        print(f"[hires] built {stem}")


def render(alpha, levels, out_dir=None, tiles_root=None):
    out_dir = out_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    tile_dir = f"{tiles_root or (VA + '/_montage')}/phase_geneKO_phate_cell1_a{alpha}_tiles"
    meta = json.load(open(f"{tile_dir}/tiles.json"))
    W, H, ts = meta["width"], meta["height"], meta["tileSize"]
    labels = json.load(open(f"{tile_dir}/labels.json"))
    for lvl in levels:
        if not os.path.isdir(f"{tile_dir}/L{lvl}"):
            print(f"[montage] a{alpha} L{lvl}: missing, skip"); continue
        canv, n, (x0, y0, x1, y1) = _composite(tile_dir, lvl, W, H, ts)
        sub = canv[y0:y1, x0:x1]
        ch, cw = sub.shape
        dpi = 150
        fig = plt.figure(figsize=(cw / dpi, ch / dpi), dpi=dpi, facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(sub, cmap="gray", vmin=0, vmax=255, aspect="auto"); ax.axis("off")
        _mark(ax, tile_dir, labels, W, lvl, ts, x0, y0, cw, dpi)
        _legend(ax, labels, cw, dpi)
        out = f"{out_dir}/montage_a{alpha}_L{lvl}_{cw}x{ch}.png"
        fig.savefig(out, dpi=dpi, facecolor="white")
        plt.close(fig)
        print(f"[montage] a{alpha} L{lvl}: {cw}x{ch} ({n} tiles) -> {out}")


def render_traversal(cells=(0, 1, 2, 3, 4, 5), alphas=(-3, 0, 1, 3), out_dir=None):
    """One PNG per cell of the 40S->60S complex-level DiffAE traversal (tight, no inter-image gaps)."""
    out_dir = out_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    fis = [int(np.argmin([abs(v - a) for v in VIEWER_ALPHAS])) for a in alphas]
    for cell in cells:
        d = f"{VA}/phase/complex/{CPLX_TRAVERSAL}/cell{cell}"
        if not os.path.isdir(d):
            continue
        n = len(alphas)
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.35), gridspec_kw={"wspace": 0.03})
        for ax, a, fi in zip(np.atleast_1d(axes), alphas, fis):
            p = f"{d}/frame_{fi:02d}.webp"
            if os.path.exists(p):
                ax.imshow(np.asarray(Image.open(p).convert("L")), cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"α = {a:g}", fontsize=13, weight="bold")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        axes[0].set_ylabel("40S", fontsize=15, weight="bold", color=MARKS["40S"][1])
        axes[-1].yaxis.set_label_position("right")
        axes[-1].set_ylabel("60S", fontsize=15, weight="bold", color=MARKS["60S"][1], rotation=270, labelpad=18)
        fig.suptitle(f"40S → 60S complex traversal · cell {cell}", fontsize=15, weight="bold", y=1.04)
        out = f"{out_dir}/traversal_40S_to_60S_cell{cell}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"[traversal] 40S->60S cells {list(cells)} alphas {list(alphas)} -> {out_dir}/traversal_40S_to_60S_cell*.png")


def render_ntc_pair(cell=1, ref_gene="FANCC", out_dir=None):
    """Separate PNG: the original NTC cell (real) next to its generative reconstruction (α=0, frame_08)."""
    out_dir = out_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    real = f"{VA}/phase/_anchors/NTC/cell{cell}/real.webp"
    gen = f"{VA}/phase/geneKO/{ref_gene}/cell{cell}/frame_08.webp"   # α=0 = unmorphed base reconstruction
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.6), gridspec_kw={"wspace": 0.04})
    for ax, p, t in [(axes[0], real, "original"), (axes[1], gen, "generative")]:
        if os.path.exists(p):
            im = np.asarray(Image.open(p).convert("L"))
            if t == "original":
                im = im[::-1, ::-1]            # flip H+V so the original matches the generative orientation
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(t, fontsize=14, weight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("NTC cell", fontsize=15, weight="bold", y=1.03)
    out = f"{out_dir}/ntc_original_vs_generative_cell{cell}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[ntc] cell{cell} original vs generative -> {out}")


def render_composed(alphas, cell=1, level=4, crop=256, ppu=5600, embedding="phate", out_dir=None,
                    bg="white", stretch=True, marks=True):
    """EXACTLY reproduce the viewer montage layout (same placed-gene set/order, same compute_priority +
    compute_canvas_size + assign_cells_to_grid grid-snapped placement, crop 256 / ppu 5600), then overlay
    the added features: per-cell leiden borders, crisp gene labels, and one correct 40S/60S member box.
    Native 256px crops (dark cells) on white. -> COMPOSED_DIR."""
    import anndata as ad
    from latent_lens.grid import assign_cells_to_grid, compute_priority, compute_canvas_size
    from .build_umap_montage import _embed_coords, OUT as MOUT, VIEWER_ALPHAS
    from ..classifier.config import slugify

    out_dir = out_dir or COMPOSED_DIR
    os.makedirs(out_dir, exist_ok=True)
    ann = ad.read_h5ad(UMAP_H5AD)
    coords_all = _embed_coords(ann, embedding).astype(np.float32)
    perts = ann.obs["perturbation"].astype(str).values
    leiden = (ann.obs[LEIDEN].astype(str).values if LEIDEN in ann.obs else np.array([""] * len(perts)))
    ebi = (ann.obs["ebi_complex"].astype(str).values if "ebi_complex" in ann.obs else np.array([""] * len(perts)))
    va = f"{MOUT}/viewer_assets/phase/geneKO"
    al = list(VIEWER_ALPHAS); a0 = int(np.argmin([abs(x) for x in al]))

    # placed-gene set/order EXACTLY as montage_from_cache: real genes with cache, then NTC nodes
    real = [i for i, g in enumerate(perts)
            if not g.startswith("NTC") and os.path.exists(f"{va}/{slugify(g)}/cell{cell}/frame_{a0:02d}.webp")]
    ntc = [i for i, g in enumerate(perts) if g.startswith("NTC")]
    order = np.array(real + ntc)
    ntc_ref = slugify(perts[real[0]]) if real else None
    gcoords = coords_all[order]
    priority = compute_priority(gcoords)
    canvas_w, canvas_h, _umin, umap_px_0 = compute_canvas_size(gcoords, ppu, crop, 8)
    ds = 2 ** level
    gh = (canvas_h // ds) // crop; gw = (canvas_w // ds) // crop
    selected, positions = assign_cells_to_grid(umap_px_0 / ds, crop, gh, gw, priority)
    if len(selected) == 0:
        print("[composed] no cells at this level"); return

    clusters = sorted({v for v in leiden if v not in ("", "nan", "None")}, key=lambda s: (len(s), s))
    cmap = plt.get_cmap("hsv")
    lut = {c: cmap(i / max(1, len(clusters) - 1)) for i, c in enumerate(clusters)}
    axn, ayn = np.ptp(coords_all[:, 0]) or 1, np.ptp(coords_all[:, 1]) or 1
    labels_all = [{"g": perts[i], "nx": float((coords_all[i, 0] - coords_all[:, 0].min()) / axn),
                   "ny": float((coords_all[i, 1] - coords_all[:, 1].min()) / ayn), LEIDEN: leiden[i]}
                  for i in range(len(perts))]

    grs = positions[:, 0]; gcs = positions[:, 1]; r0, c0 = grs.min(), gcs.min()
    Hc, Wc = int((grs.max() - r0 + 1) * crop), int((gcs.max() - c0 + 1) * crop)
    cen = gcoords.mean(0)                                          # embedding centroid (for "arm end" pick)
    thin = max(1, crop // 130); mlw = max(9, crop // 16); dpi = 150
    fill = 0 if bg == "black" else 255
    for a in alphas:
        ai = int(np.argmin([abs(x - a) for x in al]))
        canv = np.full((Hc, Wc), fill, np.uint8)
        info = []
        for sidx, (gr, gc) in zip(selected, positions):
            gi = order[sidx]; g = perts[gi]; is_ntc = g.startswith("NTC")
            p = f"{va}/{ntc_ref if is_ntc else slugify(g)}/cell{cell}/frame_{(a0 if is_ntc else ai):02d}.webp"
            if not os.path.exists(p):
                continue
            im = np.asarray(Image.open(p).convert("L"))[:crop, :crop]
            if stretch:
                im = _stretch(im)                                 # match the built montage's punchy contrast
            yy, xx = int((gr - r0) * crop), int((gc - c0) * crop)
            canv[yy:yy + crop, xx:xx + crop] = im
            info.append((xx, yy, "NTC" if is_ntc else g, leiden[gi], ebi[gi], is_ntc, gi))

        fig = plt.figure(figsize=(Wc / dpi, Hc / dpi), dpi=dpi, facecolor=bg)
        ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(canv, cmap="gray", vmin=0, vmax=255, aspect="auto"); ax.axis("off")
        fs = crop * 0.135 / dpi * 72
        for xx, yy, gname, lv, _e, is_ntc, _gi in info:           # thin per-cell leiden border + matching label
            col = "#111" if is_ntc else lut.get(lv, (0.7, 0.7, 0.7, 1))
            ax.add_patch(Rectangle((xx, yy), crop, crop, fill=False, ec=col, lw=thin, alpha=0.5, zorder=4))
            tcol = "white" if is_ntc else tuple(0.5 + 0.5 * c for c in col[:3])   # lighten label for legibility
            ax.text(xx + crop * 0.04, yy + crop * 0.04, gname, fontsize=fs, color=tcol, weight="bold",
                    family="monospace", va="top", ha="left", zorder=5,
                    path_effects=[pe.withStroke(linewidth=fs * 0.2, foreground="black")])
        for lab, (cname, mcol) in (MARKS.items() if marks else []):   # pinned (or farthest-out) member, bold box
            m = [c for c in info if c[4] == cname]
            if not m:
                continue
            pinned = [c for c in m if c[2] == MARK_GENE.get(lab)]
            it = pinned[0] if pinned else max(m, key=lambda c: float(np.linalg.norm(coords_all[c[6]] - cen)))
            xx, yy = it[0], it[1]                              # border sits just OUTSIDE the tile edge (inner edge at boundary)
            ax.add_patch(Rectangle((xx - mlw / 2, yy - mlw / 2), crop + mlw, crop + mlw, fill=False,
                                   ec=mcol, lw=mlw, zorder=6))
            ax.text(xx + crop / 2, yy - mlw, f"ribo{lab}", ha="center", va="bottom",
                    color=mcol, fontsize=fs * 1.2, weight="bold", zorder=7)
        _legend(ax, labels_all, Wc, dpi, box=(0.004, 0.004, 0.2, 0.2), dark=(bg == "black"))
        out = f"{out_dir}/composed_a{a:g}_L{level}_n{len(info)}_{Wc}x{Hc}.png"
        fig.savefig(out, dpi=dpi, facecolor=bg); plt.close(fig)
        print(f"[composed] a{a:g} L{level}: {len(info)} cells, {Wc}x{Hc} -> {out}")


def _parse_range(s):
    if "-" in s:
        lo, hi = (int(x) for x in s.split("-")); return list(range(lo, hi + 1))
    return [int(x) for x in s.split(",")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="1-5")
    ap.add_argument("--levels", default="3,4")
    ap.add_argument("--out", default=None)
    ap.add_argument("--traversal", action="store_true", help="also render the 40S->60S complex traversal panels")
    ap.add_argument("--ntc-pair", dest="ntc_pair", action="store_true", help="render the NTC original-vs-generative panel")
    ap.add_argument("--cells", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5], help="cells for the traversal panels")
    ap.add_argument("--montage", dest="montage", action="store_true", default=True)
    ap.add_argument("--no-montage", dest="montage", action="store_false", help="skip montage; only traversal")
    ap.add_argument("--hires", action="store_true", help="rebuild montage at crop 512 (crisp text + leiden borders) first")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--composed", action="store_true", help="reproduce viewer montage layout + overlaid features")
    ap.add_argument("--no-stretch", dest="stretch", action="store_false", help="skip per-crop percentile contrast stretch")
    ap.add_argument("--no-marks", dest="marks", action="store_false", help="omit the 40S/60S highlight boxes")
    ap.add_argument("--bg", default="white", choices=["white", "black"])
    a = ap.parse_args()
    alphas = _parse_range(a.alphas)
    if a.composed:
        for lv in _parse_range(a.levels):
            render_composed(alphas, level=lv, out_dir=a.out, bg=a.bg, stretch=a.stretch, marks=a.marks)
    tiles_root = None
    if a.hires:
        build_hires(alphas, crop=a.crop)
        tiles_root = HIRES_ROOT
    if a.montage and not a.composed:
        for al in alphas:
            render(al, _parse_range(a.levels), a.out, tiles_root=tiles_root)
    if a.traversal:
        render_traversal(cells=a.cells, out_dir=a.out)
    if a.ntc_pair:
        for c in a.cells:
            render_ntc_pair(cell=c, out_dir=a.out)
