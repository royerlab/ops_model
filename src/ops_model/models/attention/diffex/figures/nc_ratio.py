"""Nuclear/cytoplasmic proteasome-intensity ratio (PSMB7) — real cells + DiffEx traversal, replacing the
tubular seg (which over-fragments as signal shifts nucleus<->cytoplasm).

Nucleus = the cell's real DNA seg (`nuclei_prediction` channel, cropped via the SAME materialize_crops
pipeline the traversal frames use → aligned + proteasome-independent). Cytoplasm = everything that is not
the nucleus. Each generated cell's α=0 reconstructs its NTC anchor (real_dir=_anchors/NTC); anchor coords
aren't stored, so we match each anchor's real.webp to the NTC ranking (GFP corr) to recover them.
Metric = mean GFP(nucleus) / mean GFP(cytoplasm), scale-invariant.

Output format REUSES the morpho-violin figures: per-cell `<stem>_images_cell{N}.{png,svg}` (grayscale +
nucleus-overlay rows via render_images) + `<stem>_violin.{png,svg}` (same styling as figure4_morpho_violin).
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
from figure4_morpho_traversal import render_images                 # reuse the 2-row image block (grayscale + overlay)
from figure4_morpho_violin import COLORS, VIEW_PCT                  # reuse the violin palette + view-clip

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

VA = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5"
OUT = "/hpc/projects/icd.fast.ops/analysis/figure4_nc_ratio"
MARKER_DIR = "proteasome_PSMB7"
MC = "proteasome_PSMB7"
TARGET = "PSMB6"
GRAIN = "geneKO"
RANK = f"{VA}/_rankings/fluor_shap/geneKO/proteasome_PSMB7.parquet"
N_REAL = 300
ALPHAS_SHOW = (0, 1, 3)
N_IMG_CELLS = 10                                                   # example cells to render (like the morpho examples)
OUT_STEM = "PSMB7_nc"


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
    g = gaussian(pred, 2); v = g[g > 0]
    if v.size == 0:
        return np.zeros(pred.shape, bool)
    return _central(ndi.binary_fill_holes(binary_closing(g > threshold_otsu(v), disk(3))))


def _mets(gray, nm):
    """Both scale-invariant readouts: N/C ratio (nucleus/cytoplasm) and nuclear enrichment (nucleus/whole-crop)."""
    if nm.sum() < 20 or (~nm).sum() < 20:
        return None
    nmean = gray[nm].mean()
    return {"nc": float(nmean / (gray[~nm].mean() + 1e-6)),
            "nuc": float(nmean / (gray.mean() + 1e-6))}


def _crops(df, channel):
    raw, recs = _materialize(df.assign(gene=TARGET), MC, channel, TARGET)   # identical framing to the generated frames
    return raw[:, 0], recs


def real_nc(gene, n):
    df = _rank_df(gene, n * 2)
    gfp, _ = _crops(df, "GFP"); nuc, _ = _crops(df, "nuclei_prediction")
    out = {"nc": [], "nuc": []}
    for i in range(min(len(gfp), len(nuc))):
        mm = _mets(gfp[i], _nucleus(nuc[i]))
        if mm:
            out["nc"].append(mm["nc"]); out["nuc"].append(mm["nuc"])
        if len(out["nc"]) >= n:
            break
    return {k: np.array(v) for k, v in out.items()}


def _match_anchors(ncell, K=2500):
    """Recover each NTC anchor's coords by matching its real.webp to the top-K NTC ranking (GFP corr), then
    crop that cell's nuclei_prediction. Returns {cell: (nucleus_mask, corr)}."""
    from skimage.transform import resize
    topK = _rank_df("NTC", K)
    gfp, _ = _crops(topK, "GFP"); nucp, _ = _crops(topK, "nuclei_prediction")
    zc = lambda a: (a - a.mean()) / (a.std() + 1e-6)
    G = np.stack([zc(g).ravel() for g in gfp])
    res = {}
    for c in range(ncell):
        ap = f"{VA}/{MARKER_DIR}/_anchors/NTC/cell{c}/real.webp"
        if not os.path.exists(ap):
            continue
        ar = np.asarray(Image.open(ap).convert("L"), np.float32)
        if ar.shape != gfp[0].shape:
            ar = resize(ar, gfp[0].shape, preserve_range=True)
        cc = G @ zc(ar).ravel() / G.shape[1]
        b = int(cc.argmax())
        res[c] = (_nucleus(nucp[b]), float(cc[b]))
    mc = np.median([v[1] for v in res.values()]) if res else 0
    print(f"  [match] {len(res)} anchors matched to NTC ranking, median corr={mc:.3f}", flush=True)
    return res


def gen_nc(match, md, idxs):
    from skimage.transform import resize
    per = {a: {"nc": [], "nuc": []} for a in idxs}
    for c, (nm, cc) in match.items():
        if cc < 0.6:
            continue
        for a, i in idxs.items():
            fp = f"{md}/cell{c}/frame_{i:02d}.webp"
            if not os.path.exists(fp):
                continue
            gray = np.asarray(Image.open(fp).convert("L"), np.float32)
            nmr = nm if nm.shape == gray.shape else (resize(nm.astype(float), gray.shape) > 0.5)
            mm = _mets(gray, nmr)
            if mm:
                per[a]["nc"].append(mm["nc"]); per[a]["nuc"].append(mm["nuc"])
    return {a: {k: np.array(v) for k, v in d.items()} for a, d in per.items()}


def _panels(md, c, nucleus, idxs):
    """render_images panels: (title, grayscale, label-mask, per-object feats). Nucleus = label 1 (colored via
    the inferno overlay path); cytoplasm stays grayscale — same 2-row block as the morpho image panels."""
    from skimage.transform import resize
    out = []

    def add(title, gray):
        nm = nucleus if nucleus.shape == gray.shape else (resize(nucleus.astype(float), gray.shape) > 0.5)
        out.append((title, gray, nm.astype(np.int32), {"1": {"comp": 1.0}}))
    ar = f"{VA}/{MARKER_DIR}/_anchors/NTC/cell{c}/real.webp"
    if os.path.exists(ar):
        add("real (NTC)", np.asarray(Image.open(ar).convert("L"), np.float32))
    for a, i in idxs.items():
        add(f"α={a:+.0f}", np.asarray(Image.open(f"{md}/cell{c}/frame_{i:02d}.webp").convert("L"), np.float32))
    return out


def _images(subdir, match, md, idxs):
    os.makedirs(subdir, exist_ok=True)
    for c in list(match)[:N_IMG_CELLS]:
        panels = _panels(md, c, match[c][0], idxs)
        figi = plt.figure(figsize=(len(panels) * 2.3, 5.0), facecolor="white")
        render_images(figi, figi.add_gridspec(1, 1)[0], panels, "comp", 0.0, 1.0, title_fs=22, cbar=False, hspace=0.05)
        for ext in ("png", "svg"):
            figi.savefig(f"{subdir}/{OUT_STEM}_images_cell{c}.{ext}", dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(figi)


def _violin(subdir, data, labels, ylabel):
    keep = [i for i, d in enumerate(data) if len(d)]
    figv = plt.figure(figsize=(4.8, 5.6), facecolor="white"); ax = figv.add_subplot(111); ax.set_facecolor("white")
    pooled = np.concatenate([data[i] for i in keep]); ylo, yhi = np.percentile(pooled, VIEW_PCT); pad = 0.05 * (yhi - ylo + 1e-9)
    clamp = lambda d: np.clip(d, ylo, yhi)                     # trimmed tail accumulates as a bulge (not cut off)
    parts = ax.violinplot([clamp(data[i]) for i in keep], positions=keep, showmeans=False, showextrema=False, showmedians=False, widths=0.82)
    for pc, i in zip(parts["bodies"], keep):
        pc.set_facecolor(COLORS[labels[i]]); pc.set_alpha(0.6); pc.set_edgecolor(COLORS[labels[i]]); pc.set_linewidth(1.5)
    for i in keep:
        ax.hlines(np.mean(data[i]), i - 0.34, i + 0.34, color="#222", lw=3, zorder=5)   # mean = full data
    ax.axhline(0, color="#999", lw=2)
    ax.set_ylim(ylo - pad, yhi + pad)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=30)
    ax.set_ylabel(ylabel, fontsize=30 if len(ylabel) <= 18 else (24 if len(ylabel) <= 26 else 19))   # single line
    ax.tick_params(axis="y", labelsize=26, width=2.5, length=9); ax.tick_params(axis="x", length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(2.5)
    os.makedirs(subdir, exist_ok=True)
    for ext in ("png", "svg"):
        figv.savefig(f"{subdir}/{OUT_STEM}_violin.{ext}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figv)


def make_figures(ntc, ko, per, match, md, idxs):
    labels = ["real", "KO", "α=0", "α=1", "α=3"]
    m = lambda d: round(float(np.mean(d)), 3) if len(d) else None
    for sub, key, ylabel in (("nc_ratio", "nc", "Nuclear / cytoplasmic proteasome"),
                             ("nuclear_intensity", "nuc", "Nuclear proteasome intensity")):
        subdir = f"{OUT}/{sub}"
        _images(subdir, match, md, idxs)
        nbase = float(np.nanmean(ntc[key])) or 1e-9                 # real bars: baseline = real-NTC mean
        g0 = per.get(0, {}).get(key, np.array([]))
        gbase = float(np.nanmean(g0)) or 1e-9                        # gen: self-referenced to α=0 (uniform)
        pctr = lambda v: (np.asarray(v, float) - nbase) / abs(nbase) * 100   # real vs real-NTC
        pctg = lambda v: (np.asarray(v, float) - gbase) / abs(gbase) * 100   # gen vs its α=0 reconstruction
        data = [pctr(ntc[key]), pctr(ko[key]), pctg(g0),
                pctg(per.get(1, {}).get(key, np.array([]))), pctg(per.get(3, {}).get(key, np.array([])))]
        _violin(subdir, data, labels, ylabel)
        print(f"  [{sub}] real NTC {m(data[0])} / KO {m(data[1])} | α0 {m(data[2])} α1 {m(data[3])} α3 {m(data[4])} -> {subdir}", flush=True)


def build():
    md = f"{VA}/{MARKER_DIR}/{GRAIN}/{TARGET}"
    al = np.array(json.load(open(f"{md}/meta.json"))["alphas"])
    idxs = {a: int(np.argmin(np.abs(al - a))) for a in ALPHAS_SHOW}
    ncell = len([d for d in os.listdir(md) if d.startswith("cell")])
    ntc = real_nc("NTC", N_REAL); ko = real_nc(TARGET, N_REAL)
    match = _match_anchors(ncell)
    per = gen_nc(match, md, idxs)
    make_figures(ntc, ko, per, match, md, idxs)


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
