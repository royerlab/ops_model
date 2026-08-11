"""Nuclear/cytoplasmic proteasome-intensity ratio (PSMB7) — real cells + DiffEx traversal, replacing the
tubular seg (which over-fragments as signal shifts nucleus<->cytoplasm).

Masks are proteasome-INDEPENDENT and framed IDENTICALLY to the generated frames (same materialize_crops
pipeline the traversal build used): nucleus = the aligned `nuclei_prediction` DNN channel; cytoplasm =
cell foreground minus nucleus. Metric = mean GFP(nucleus) / mean GFP(cytoplasm) — scale-invariant, so the
8-bit display frames are valid. Cell index == rank order of the generation ranking, so each generated cell
reuses its own anchor's nucleus.

Outputs a full panel (real | α0 | α1 | α3 image row + nucleus/cytoplasm overlay row) + the N/C violin.
Run (SLURM): python nc_ratio.py --submit
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import binary_closing, disk

from _setacc_common import _materialize

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

VA = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5"
OUT = "/hpc/projects/icd.fast.ops/analysis/figure4_nc_ratio"
MARKER_DIR = "proteasome_PSMB7"
MC = "proteasome_PSMB7"          # marker_channel (DirConfig); channel to READ is passed separately
TARGET = "PSMB6"
GRAIN = "geneKO"
RANK = f"{VA}/_rankings/fluor/geneKO/proteasome_PSMB7.parquet"   # the ranking the traversal was generated from
N_REAL = 300
COLORS = {"real": "#999999", "KO": "#2e8b57", "α=0": "#c6dbef", "α=1": "#6baed6", "α=3": "#08519c"}
EXAMPLE_CELL = 0


def _rank_df(gene, n=None):
    d = pd.read_parquet(RANK)
    d = d[d["gene"].astype(str) == gene]
    if "rank_type" in d.columns:
        d = d[d["rank_type"] == "top"]
    d = d.sort_values("rank").reset_index(drop=True)
    return d.head(n) if n else d


def _central(binmask):
    lab, n = ndi.label(binmask)
    if n == 0:
        return binmask
    cy, cx = np.array(binmask.shape) // 2
    cen = lab[cy, cx] or (1 + int(np.argmax(np.bincount(lab.ravel())[1:])))
    return lab == cen


def _nucleus(pred):
    """Nucleus mask from the aligned nuclei_prediction crop: blurred Otsu, fill, central component."""
    g = gaussian(pred, 2); v = g[g > 0]
    if v.size == 0:
        return np.zeros(pred.shape, bool)
    m = ndi.binary_fill_holes(binary_closing(g > threshold_otsu(v), disk(3)))
    return _central(m)


def _foreground(gray):
    """Cell foreground (drops background so 'all other pixels' don't include empty corners)."""
    g = gaussian(gray, 2); v = g[g > 0]
    if v.size == 0:
        return np.ones(gray.shape, bool)
    fg = ndi.binary_fill_holes(binary_closing(g > np.percentile(g, 55), disk(3)))
    return _central(fg) if fg.any() else fg


def _ratio(gray, nucleus, cyto):
    if nucleus.sum() < 20 or cyto.sum() < 20:
        return np.nan
    return float(gray[nucleus].mean() / (gray[cyto].mean() + 1e-6))


def _crops(df, channel):
    """materialize_crops the given channel for these cells — identical framing to the generated frames."""
    raw, recs = _materialize(df.assign(gene=TARGET), MC, channel, TARGET)
    return raw[:, 0], recs                                        # (N, H, W)


def real_nc(gene, n):
    df = _rank_df(gene, n * 2)
    gfp, recs = _crops(df, "GFP")
    nuc, _ = _crops(df, "nuclei_prediction")
    out = []
    for i in range(min(len(gfp), len(nuc))):
        nm = _nucleus(nuc[i]); fg = _foreground(gfp[i]); cy = fg & ~nm
        v = _ratio(gfp[i], nm, cy)
        if np.isfinite(v):
            out.append(v)
        if len(out) >= n:
            break
    return np.array(out)


def gen_nc(alphas_show=(0, 1, 3)):
    md = f"{VA}/{MARKER_DIR}/{GRAIN}/{TARGET}"
    al = np.array(json.load(open(f"{md}/meta.json"))["alphas"])
    idxs = {a: int(np.argmin(np.abs(al - a))) for a in alphas_show}
    ncell = len([d for d in os.listdir(md) if d.startswith("cell")])
    anch = _rank_df(TARGET, ncell)
    nuc, _ = _crops(anch, "nuclei_prediction")                   # anchor nucleus per cell, aligned to frames
    per = {a: [] for a in alphas_show}
    for c in range(min(ncell, len(nuc))):
        nm = _nucleus(nuc[c])
        for a, i in idxs.items():
            fp = f"{md}/cell{c}/frame_{i:02d}.webp"
            if not os.path.exists(fp):
                continue
            gray = np.asarray(Image.open(fp).convert("L"), np.float32)
            nmr = nm if nm.shape == gray.shape else _rs(nm, gray.shape)
            fg = _foreground(gray); cy = fg & ~nmr
            v = _ratio(gray, nmr, cy)
            if np.isfinite(v):
                per[a].append(v)
    return {a: np.array(v) for a, v in per.items()}, idxs, md, nuc


def _rs(mask, shape):
    from skimage.transform import resize
    return resize(mask.astype(float), shape) > 0.5


def _ov(gray, nm, cy):
    rgb = np.stack([gray] * 3, -1) / max(gray.max(), 1e-6)
    rgb[nm] = 0.5 * rgb[nm] + 0.5 * np.array([0.15, 0.55, 1.0])
    rgb[cy] = 0.75 * rgb[cy] + 0.25 * np.array([1.0, 0.5, 0.1])
    return np.clip(rgb, 0, 1)


def panel(gen, idxs, md, nuc, real_gfp_ex, real_nuc_ex):
    """Full panel: real | α0/α1/α3 image row + nucleus(blue)/cytoplasm(orange) overlay row, violin at right."""
    from skimage.transform import resize
    c = EXAMPLE_CELL

    def _fit(pred, shape):
        return pred if pred.shape == shape else resize(pred, shape, preserve_range=True)
    cols = [("real", real_gfp_ex, real_nuc_ex)]
    for a, i in idxs.items():
        g = np.asarray(Image.open(f"{md}/cell{c}/frame_{i:02d}.webp").convert("L"), np.float32)
        cols.append((f"α={a}", g, _fit(nuc[c], g.shape)))
    nc = len(cols)
    fig = plt.figure(figsize=(nc * 2.0 + 4.2, 4.4), facecolor="white")
    gs = fig.add_gridspec(2, nc + 2, width_ratios=[1] * nc + [0.25, 2.4], hspace=0.06, wspace=0.06,
                          left=0.02, right=0.985, top=0.9, bottom=0.06)
    for j, (t, g, pred) in enumerate(cols):
        nm = _nucleus(pred); fg = _foreground(g); cy = fg & ~nm
        ax = fig.add_subplot(gs[0, j]); ax.imshow(g, cmap="gray"); ax.set_title(t, fontsize=13); ax.axis("off")
        ax2 = fig.add_subplot(gs[1, j]); ax2.imshow(_ov(g, nm, cy)); ax2.axis("off")
    axv = fig.add_subplot(gs[:, nc + 1])
    data = [gen.get(0, []), gen.get(1, []), gen.get(3, [])]
    _violin_into(axv, PANEL_REAL, data)
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}/PSMB7_nc_ratio_panel.{ext}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {OUT}/PSMB7_nc_ratio_panel", flush=True)


PANEL_REAL = {}


def _violin_into(ax, real, gen_data):
    data = [real.get("NTC", []), real.get("KO", []), *gen_data]
    labels = ["real", "KO", "α=0", "α=1", "α=3"]
    keep = [i for i, d in enumerate(data) if len(d)]
    parts = ax.violinplot([np.asarray(data[i]) for i in keep], positions=keep, showextrema=False, widths=0.82)
    for pc, i in zip(parts["bodies"], keep):
        pc.set_facecolor(COLORS[labels[i]]); pc.set_alpha(0.6); pc.set_edgecolor(COLORS[labels[i]]); pc.set_linewidth(1.5)
    for i in keep:
        ax.hlines(np.mean(data[i]), i - 0.34, i + 0.34, color="#222", lw=3, zorder=5)
    ax.axhline(1.0, color="#999", lw=2, ls="--")
    pooled = np.concatenate([np.asarray(data[i]) for i in keep])
    ylo, yhi = np.percentile(pooled, (1, 98)); pad = 0.05 * (yhi - ylo + 1e-9)
    ax.set_ylim(ylo - pad, yhi + pad)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=20)
    ax.set_ylabel("Nuclear / cytoplasmic\nproteasome intensity", fontsize=18)
    ax.tick_params(axis="y", labelsize=18, width=2, length=7); ax.tick_params(axis="x", length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def build():
    ntc = real_nc("NTC", N_REAL); ko = real_nc(TARGET, N_REAL)
    PANEL_REAL["NTC"] = ntc; PANEL_REAL["KO"] = ko
    gen, idxs, md, nuc = gen_nc()
    # example real crop for the panel's "real" column (top KO cell)
    ex = _rank_df(TARGET, 1)
    rg, _ = _crops(ex, "GFP"); rn, _ = _crops(ex, "nuclei_prediction")
    m = lambda d: round(float(np.mean(d)), 3) if len(d) else None
    print(f"  real NTC {m(ntc)} (n{len(ntc)}) / KO {m(ko)} (n{len(ko)}) | gen α0 {m(gen.get(0,[]))} "
          f"α1 {m(gen.get(1,[]))} α3 {m(gen.get(3,[]))}", flush=True)
    panel(gen, idxs, md, nuc, rg[0], rn[0])


def submit():
    import pathlib
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    figdir = str(pathlib.Path(__file__).resolve().parent)
    os.environ["PYTHONPATH"] = figdir + os.pathsep + os.environ.get("PYTHONPATH", "")
    submit_parallel_jobs([{"name": "psmb7_nc", "func": build, "kwargs": {}}], experiment="diffex_nc",
                         slurm_params={"slurm_partition": "cpu", "cpus_per_task": 8, "mem_gb": 64, "timeout_min": 120},
                         log_dir="diffex_nc", wait_for_completion=False)


if __name__ == "__main__":
    submit() if (len(sys.argv) > 1 and sys.argv[1] == "--submit") else build()
