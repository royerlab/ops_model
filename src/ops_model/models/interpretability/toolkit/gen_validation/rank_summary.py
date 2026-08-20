"""Population summary of the v5 SetTransformer target-rank across the DiffAE traversal.

For every geneKO/complex traversal, rank_target[α] = the 1-indexed position of the true class in the
classifier's ranking of a bag-20 set of the α-frames. Aggregate mean & median rank across classes at
each α, for both anchor pools (accpool = 25 hand-picked accuracy NTCs = deployed default; attn = top-attention).
Rank 1 = perfect recovery; lower is better.

Two figures:
  rank_summary.png          — all classes
  rank_summary_realfilt.png — only classes whose REAL cells are distinguishable (real top1_acc > 0.5 @ bag20),
                              i.e. the feasible ceiling; measures recovery where a phenotype actually exists.
"""
import json, glob, os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({                                    # figure-ready: ~2x default text
    "font.size": 19, "axes.titlesize": 23, "axes.labelsize": 21,
    "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 14,
    "figure.titlesize": 25, "axes.linewidth": 1.4,
    "xtick.major.size": 7, "ytick.major.size": 7, "xtick.major.width": 1.4, "ytick.major.width": 1.4,
})
XTICKS = list(range(-5, 6))                               # integer α ticks
BASE = "/hpc/projects/icd.fast.ops/models/diffex"
OUTDIR = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals/rank_summary"
os.makedirs(OUTDIR, exist_ok=True)
ASSETS = os.environ.get("RANK_ASSETS", "viewer_assets_v5_accpool")   # default accpool; new multibag traversals = viewer_assets_v5
OVERLAY = os.environ.get("RANK_OVERLAY", "") or None                 # optional dashed overlay tree (empty = none)
SUF = os.environ.get("RANK_SUF", "")                                 # filename suffix (e.g. _v5new)
GRAINS = {"geneKO": ("geneKO", 1000), "complex": ("complex", 99)}
RED = "#c0392b"
REAL = json.load(open(f"{BASE}/viewer_assets_v5/real_acc20.json"))   # real top1_acc@bag20 by asset_dir


def collect(assets, sub, keep=None):
    """→ (alphas, matrix[n_class × n_alpha] of rank_target). keep: optional set of names to include."""
    rows, alphas = [], None
    for d in sorted(glob.glob(f"{BASE}/{assets}/phase/{sub}/*")):
        name = os.path.basename(d)
        if not os.path.isdir(d) or "__to__" in name:
            continue
        if keep is not None and name not in keep:
            continue
        sf = f"{d}/scores_v5.json"
        if not os.path.exists(sf):
            continue
        sc = json.load(open(sf))
        rk = sc.get("rank_target")
        if not rk:
            continue
        alphas = sc["alphas"]
        rows.append([np.nan if v is None else float(v) for v in rk])
    return np.array(alphas), np.array(rows) if rows else np.empty((0, 0))


def real_keep(sub):
    """dir-basenames whose real cells clear top1_acc>0.5 @bag20. Keyed by asset_dir 'phase/{sub}/{name}'."""
    pre = f"phase/{sub}/"
    return {k[len(pre):] for k, v in REAL.items() if k.startswith(pre) and v > 0.5}


def make_fig(title, fname, errstyle):
    """errstyle: 'mean_sem' (mean line + SEM band) or 'median_iqr' (median line + IQR band). Accpool only.
    Overlays all classes (gray) vs real-distinguishable (real top1>0.5 @bag20, red) on the same axes."""
    clip = lambda a: np.maximum(a, 0.9)   # keep bands positive on the log axis
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.2), constrained_layout=True)
    for ax, (gname, (sub, nclass)) in zip(axes, GRAINS.items()):
        ns = {}
        for keep, ftag, color in [(None, "all classes", "#7f8c8d"), (real_keep(sub), "real top1 > 0.5", RED)]:
            al, M = collect(ASSETS, sub, keep)
            if M.size == 0:
                continue
            n = int(np.isfinite(M[:, len(al) // 2]).sum()); ns[ftag] = n
            if errstyle == "mean_sem":
                c = np.nanmean(M, axis=0); e = np.nanstd(M, axis=0) / np.sqrt(np.isfinite(M).sum(axis=0))
                lo, hi = clip(c - e), c + e
                lab = f"{ftag} — mean ± SEM"
            else:   # median_iqr
                c = np.nanmedian(M, axis=0)
                lo, hi = clip(np.nanpercentile(M, 25, axis=0)), np.nanpercentile(M, 75, axis=0)
                lab = f"{ftag} — median (IQR)"
            ax.plot(al, c, "-", color=color, lw=3.4, label=lab)
            ax.fill_between(al, lo, hi, color=color, alpha=0.22, lw=0)
            alo, Mo = (collect(OVERLAY, sub, keep) if OVERLAY else (None, np.empty((0,0))))                          # dashed = 200-cell bag (geneKO only)
            if Mo.size:
                co = np.nanmean(Mo, 0) if errstyle == "mean_sem" else np.nanmedian(Mo, 0)
                ax.plot(alo, co, "--", color=color, lw=2.8, label=f"{ftag} — 200-cell bag")
        ax.axvline(0, color="#bbb", lw=1, zorder=0)
        ax.axhline(1, color="#27ae60", lw=1, ls=":", zorder=0)
        ax.set_yscale("log")
        ax.set_xlabel("traversal α")
        ax.set_ylabel("target rank")
        ax.set_title(f"{gname}  ·  {ns.get('all classes', nclass)} classes ({ns.get('real top1 > 0.5', '?')} real-distinguishable)")
        ax.set_xticks(XTICKS); ax.set_xlim(-5, 5)
        ax.grid(alpha=0.25, which="both")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=15, framealpha=0.9)
    fig.suptitle(title, fontweight="bold")
    out = f"{OUTDIR}/{fname}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def make_top5_fig(fname):
    """% of classes whose target lands in the top-5 ranking at each α — all classes vs real-distinguishable
    (real top1-acc > 0.5 @ bag20). Accpool only. top-5 hit = rank_target <= 5."""
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.2), constrained_layout=True)
    for ax, (gname, (sub, nclass)) in zip(axes, GRAINS.items()):
        ns = {}
        for keep, ftag, color in [(None, "all classes", "#7f8c8d"), (real_keep(sub), "real top1 > 0.5", RED)]:
            al, M = collect(ASSETS, sub, keep)
            if M.size == 0:
                continue
            valid = np.isfinite(M)
            f5 = 100 * (valid & (M <= 5)).sum(axis=0) / valid.sum(axis=0)   # target in top-5
            f1 = 100 * (valid & (M == 1)).sum(axis=0) / valid.sum(axis=0)   # target is #1 (top-1)
            ns[ftag] = int(valid[:, len(al) // 2].sum())
            ax.plot(al, f5, "-", color=color, lw=3.4, label=f"{ftag} — top-5")
            ax.plot(al, f1, ":", color=color, lw=3.4, label=f"{ftag} — top-1")
            b5, b1 = int(np.argmax(f5)), int(np.argmax(f1))
            print(f"  {gname:8s} | {ftag:16s} | peak top-5 = {f5[b5]:.0f}% @α={al[b5]:+.1f} | peak top-1 = {f1[b1]:.0f}% @α={al[b1]:+.1f}")
            alo, Mo = (collect(OVERLAY, sub, keep) if OVERLAY else (None, np.empty((0,0))))                          # dashed = 200-cell bag top-5 (geneKO only)
            if Mo.size:
                vo = np.isfinite(Mo); f5o = 100 * (vo & (Mo <= 5)).sum(0) / vo.sum(0)
                ax.plot(alo, f5o, "--", color=color, lw=2.8, label=f"{ftag} — 200-cell bag top-5")
        ax.axvline(0, color="#bbb", lw=1, zorder=0)
        ax.set_ylim(0, 100)
        ax.set_xlabel("traversal α")
        ax.set_ylabel("% recovered")
        ax.set_title(f"{gname}  ·  {ns.get('all classes', nclass)} classes ({ns.get('real top1 > 0.5', '?')} real-distinguishable)")
        ax.set_xticks(XTICKS); ax.set_xlim(-5, 5)
        ax.grid(alpha=0.25)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=15, framealpha=0.9)
    fig.suptitle("Top-5 (solid) / top-1 (dotted) recovery vs α  (bag=20)", fontweight="bold")
    out = f"{OUTDIR}/{fname}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); fig.savefig(out.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig); print("saved", out)


TITLE = "Target rank along the traversal (bag=20, accuracy anchors)"
make_fig(TITLE, "rank_summary_mean_sem"+SUF, "mean_sem")
make_fig(TITLE, "rank_summary_median_iqr"+SUF, "median_iqr")
make_top5_fig("top5_recovery_summary"+SUF)
