"""Figure 4 morpho — VIOLIN variant (separate subdir). The traversal image panel is IDENTICAL to the
line-graph figure (reuses image_panels + render_images: original-NTC anchor + generated α frames, both
the grayscale and feature-colored seg-overlay rows). To the RIGHT of the images (not below) a VIOLIN plot
shows the per-cell distribution (variance over the 100 cells) of the feature for real KO, generated α=0,
and generated α=1 — all as %-change vs NTC (real KO vs real-NTC mean; generated vs its own α=0 baseline).

Run: OPS_DIFFEX_ASSETS=viewer_assets_v5 python figure4_morpho_violin.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure4_morpho_traversal import FIGURES, VA, image_panels, render_images
from ops_model.models.interpretability.diffae.traversal.morpho_pipeline import MORPHO_TARGETS, real_percell
from ops_model.models.interpretability.diffae.classifier.config import slugify
from ops_model.paths import BASE_PATH

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

OUT = f"{BASE_PATH}/analysis/figure4_traversals_violin"
COLORS = {"real": "#999999", "KO": "#2e8b57", "α=0": "#c6dbef", "α=1": "#6baed6", "α=3": "#08519c"}
ALPHAS_SHOW = [0, 1, 3]                           # image panel columns (α=3 = exaggeration, not α=5)
CELLS = list(range(30))                           # render 30 example cells to pick from
VIEW_PCT = (1, 98)                                # y-axis view window (percentile of pooled data) — clips the
                                                  # VIEW only; KDE + median stay on the full data (never moves)


def _store_cfg(marker_dir, target, grain):
    for v in MORPHO_TARGETS.values():
        if v["marker_dir"] == marker_dir and slugify(v["target"]) == slugify(target) and v.get("grain", "geneKO") == grain:
            return v["store_marker"], v.get("store_channel")
    return None, None


def _pct(vals, base):
    b = abs(base) or 1e-9
    return (np.asarray(vals, float) - base) / b * 100.0


def make_violin(dir_, feature, simple, out_stem):
    marker_dir, grain, target = dir_.split("/", 2)
    md = f"{VA}/_morphometrics/{dir_}"
    ff = json.load(open(f"{md}/full_features.json"))
    alphas = ff["alphas"]
    ai = lambda a: min(range(len(alphas)), key=lambda i: abs(alphas[i] - a))
    z, i1, i3 = ai(0), ai(1.0), ai(3.0)

    df = pd.read_parquet(f"{md}/full_features.parquet")
    gen = {a: df.loc[df["alpha_idx"] == idx, feature].dropna().values for a, idx in ((0, z), (1, i1), (3, i3))}
    base = float(np.mean(gen[0]))

    store_marker, store_channel = _store_cfg(marker_dir, target, grain)
    rr = (real_percell(marker_dir, target, grain, store_marker, [feature], store_channel=store_channel) or {}).get(feature) if store_marker else None
    if rr is not None and len(rr["ntc"]) and len(rr["ko"]):
        nbase = float(np.nanmean(rr["ntc"]))                     # real %-change baseline = real-NTC mean
        real_ntc = _pct(rr["ntc"][~np.isnan(rr["ntc"])], nbase)
        real_ko = _pct(rr["ko"][~np.isnan(rr["ko"])], nbase)
    else:
        real_ntc = real_ko = np.array([]); print(f"  {out_stem}: no real per-cell for {feature}")

    data = [real_ntc, real_ko, _pct(gen[0], base), _pct(gen[1], base), _pct(gen[3], base)]
    labels = ["real", "KO", "α=0", "α=1", "α=3"]
    os.makedirs(os.path.dirname(f"{OUT}/{out_stem}"), exist_ok=True)

    # ---- (1) image panel — identical to the line-graph figure — one per example cell ----
    for cell in CELLS:
        try:
            panels, okey, lo, hi = image_panels(md, dir_, dir_, feature, cell, ALPHAS_SHOW)
        except (FileNotFoundError, IndexError):
            continue
        nc = len(panels)
        figi = plt.figure(figsize=(nc * 2.3, 5.0), facecolor="white")
        render_images(figi, figi.add_gridspec(1, 1)[0], panels, okey, lo, hi, title_fs=22, hspace=0.05)
        for ext in ("png", "svg"):
            figi.savefig(f"{OUT}/{out_stem}_images_cell{cell}.{ext}", dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(figi)

    # ---- (2) violin — separate file; full (unclipped) per-cell distributions, median line from the data ----
    keep = [i for i, d in enumerate(data) if len(d)]
    figv = plt.figure(figsize=(4.8, 5.6), facecolor="white")
    ax = figv.add_subplot(111); ax.set_facecolor("white")
    parts = ax.violinplot([data[i] for i in keep], positions=keep, showmeans=False, showextrema=False, showmedians=False, widths=0.82)
    for pc, i in zip(parts["bodies"], keep):
        pc.set_facecolor(COLORS[labels[i]]); pc.set_alpha(0.6); pc.set_edgecolor(COLORS[labels[i]]); pc.set_linewidth(1.5)
    for i in keep:
        ax.hlines(np.mean(data[i]), i - 0.34, i + 0.34, color="#222", lw=3, zorder=5)   # mean line (full data)
    ax.axhline(0, color="#999", lw=2)
    pooled = np.concatenate([data[i] for i in keep])          # view-clip: robust ylim, data/median untouched
    ylo, yhi = np.percentile(pooled, VIEW_PCT); pad = 0.05 * (yhi - ylo + 1e-9)
    ax.set_ylim(ylo - pad, yhi + pad)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=30)
    import textwrap
    ylab = "\n".join(textwrap.wrap(simple, 14)) if len(simple) > 14 else simple   # wrap long labels so they don't clip the canvas
    ax.set_ylabel(ylab, fontsize=26 if len(simple) > 18 else 30)
    ax.tick_params(axis="y", labelsize=26, width=2.5, length=9)
    ax.tick_params(axis="x", length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(2.5)
    for ext in ("png", "svg"):
        figv.savefig(f"{OUT}/{out_stem}_violin.{ext}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figv)
    km = float(np.median(real_ko)) if len(real_ko) else None
    print(f"saved {OUT}/{out_stem}_violin + {len(CELLS)} cell images  (real KO med {None if km is None else round(km)}%, "
          f"gen α1 med {round(float(np.median(_pct(gen[1], base))))}%, α3 med {round(float(np.median(_pct(gen[3], base))))}%)")


def _job(fig):
    import os
    os.environ.setdefault("OPS_DIFFEX_ASSETS", "viewer_assets_v5")   # worker reads v5 assets
    make_violin(fig["dir"], fig["feature"], fig.get("simple", fig["label"]), f"{fig['group']}/{fig['out_stem']}")


def submit(figs=None):
    """One SLURM job per target (parallel) — each reads the op_cp store + renders violin + 10 cell images."""
    import pathlib
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    figdir = str(pathlib.Path(__file__).resolve().parent)     # workers must import the loose figures scripts
    os.environ["PYTHONPATH"] = figdir + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ.setdefault("OPS_DIFFEX_ASSETS", "viewer_assets_v5")
    figs = figs or FIGURES
    jobs = [{"name": f"violin_{f['group'][:18]}", "func": _job, "kwargs": {"fig": f}} for f in figs]
    print(f"[violin] {len(jobs)} target jobs in parallel")
    submit_parallel_jobs(jobs, experiment="diffex_violin",
                         slurm_params={"slurm_partition": "cpu", "cpus_per_task": 8, "mem_gb": 64, "timeout_min": 120},
                         log_dir="diffex_violin", wait_for_completion=False)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "local":       # serial (debug)
        for f in FIGURES:
            try:
                make_violin(f["dir"], f["feature"], f.get("simple", f["label"]), f"{f['group']}/{f['out_stem']}")
            except Exception as e:
                print(f"skip {f['out_stem']}: {type(e).__name__}: {e}")
    else:
        submit()
