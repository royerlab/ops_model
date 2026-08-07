"""Real-vs-Generated DISTINCTIVENESS mAP (within-domain), across all α.

Per grain (geneKO = gene-level distinctiveness, complex = EBI): take the top-K accuracy cells per class and measure
the standard copairs distinctiveness mAP (each class's cells retrieve each other vs all other classes) on the REAL
cells (once) and on the GENERATED cells at EVERY α. Reuses gen_real_map_cache embeddings; no re-embedding.

Within-domain, so the DiffAE domain offset (common to all generated cells) cancels — no standardization needed.
Outputs: {grain}_real.json + {grain}_gen_a{ai}.json → plot: gen distinctiveness vs α + real reference, and a
Real-vs-Generated violin at the α where generated distinctiveness peaks.
"""
import os, json, glob
import numpy as np
import pandas as pd

CACHE = os.environ.get("GRD_CACHE", "/hpc/projects/icd.fast.ops/analysis/figure4_traversals/gen_real_map_cache")
OUT = os.environ.get("GRD_OUT", "/hpc/projects/icd.fast.ops/analysis/figure4_traversals/gen_real_distinct")
K = int(os.environ.get("GRD_K", 20))                   # top-K cells/class (200 for the valid200 bag)
NA = int(os.environ.get("GRD_NA", 17))                 # α grid points (7 for valid200)


def _distinct(feats, labels):
    """copairs distinctiveness: per class, do its cells retrieve each other above all other-class cells → {class: mAP}."""
    from copairs import map as cm
    meta = pd.DataFrame({"gene": labels})
    ap = cm.average_precision(meta, np.ascontiguousarray(feats, np.float32), pos_sameby=["gene"], pos_diffby=[],
                              neg_sameby=[], neg_diffby=["gene"], distance="cosine")
    m = cm.mean_average_precision(ap, sameby=["gene"], null_size=200, threshold=0.05, seed=0)
    return dict(zip(m["gene"], m["mean_average_precision"]))


def _load(grain):
    return sorted(glob.glob(f"{CACHE}/{grain}/*.npz"))


def compute_real(grain):
    rf, rl = [], []
    for c in _load(grain):
        d = np.load(c, allow_pickle=True); g = str(d["gene"])
        r = np.asarray(d["real"], np.float32)[:K]; rf.append(r); rl += [g] * len(r)
    os.makedirs(OUT, exist_ok=True)
    real = _distinct(np.concatenate(rf), rl)
    json.dump(real, open(f"{OUT}/{grain}_real.json", "w"))
    return {"grain": grain, "n": len(real), "median": float(np.median(list(real.values())))}


def compute_gen(grain, ai):
    gf, gl, alpha = [], [], None
    for c in _load(grain):
        d = np.load(c, allow_pickle=True); g = str(d["gene"]); alpha = float(d["alphas"][ai])
        gv = d["gen"][ai]
        if gv is not None and len(gv):
            gv = np.asarray(gv, np.float32)[:K]; gf.append(gv); gl += [g] * len(gv)
    os.makedirs(OUT, exist_ok=True)
    gen = _distinct(np.concatenate(gf), gl)
    json.dump({"alpha": alpha, "gen": gen}, open(f"{OUT}/{grain}_gen_a{ai}.json", "w"))
    return {"grain": grain, "ai": ai, "alpha": alpha, "n": len(gen), "median": float(np.median(list(gen.values())))}


_ALLGRAINS = [("geneKO", "Gene-level"), ("complex", "Protein\ncomplex")]
_KEYS = os.environ.get("GRD_GRAINS", "geneKO,complex").split(",")      # restrict grains (valid200 = geneKO only)
GRAINS = [g for g in _ALLGRAINS if g[0] in _KEYS]


def _series(grain, base=None):
    """→ (alphas, DataFrame[class × alpha] gen mAP, real dict)."""
    base = base or OUT
    real = json.load(open(f"{base}/{grain}_real.json"))
    al, cols = [], {}
    for f in sorted(glob.glob(f"{base}/{grain}_gen_a*.json"), key=lambda p: int(p.split("_a")[-1][:-5])):
        d = json.load(open(f)); al.append(d["alpha"]); cols[d["alpha"]] = d["gen"]
    return sorted(al), pd.DataFrame(cols), real


def plot(overlay=None, overlay_label="200-cell bag", overlay_K=50, fname="distinct_vs_alpha"):
    """overlay: {grain: distinct_dir} → dashed second median line (e.g. the 200-cell bag) on that grain's axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp
    plt.rcParams["pdf.fonttype"] = 42
    REAL_C, GEN_C, OV_C = "#8fa9c9", "#7fbf9a", "#2e8b57"

    # (1) curve: generated distinctiveness (median, IQR) vs α + real median reference
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    peak_ai = {}
    for ax, (grain, lbl) in zip(axes, GRAINS):
        al, df, real = _series(grain)
        med = df[al].median(0); q1 = df[al].quantile(.25); q3 = df[al].quantile(.75)
        gtag = "" if not (overlay and grain in overlay) else f" (top-{K})"
        ax.plot(al, med, "-", color=GEN_C, lw=2.6, label=f"Generated (median, IQR){gtag}")
        ax.fill_between(al, q1, q3, color=GEN_C, alpha=.22, lw=0)
        rm = np.median(list(real.values()))
        ax.axhline(rm, color=REAL_C, ls="--", lw=2.2, label=f"Real top-{K} ({rm:.2f})")
        if overlay and grain in overlay:                                 # dashed = overlay bag (its own α grid + real)
            alo, dfo, realo = _series(grain, base=overlay[grain])
            medo = dfo[alo].median(0)
            ax.plot(alo, medo, "--", color=OV_C, lw=2.4, label=f"Generated — {overlay_label} (top-{overlay_K})")
            ax.axhline(np.median(list(realo.values())), color=OV_C, ls=":", lw=1.6,
                       label=f"Real top-{overlay_K} ({np.median(list(realo.values())):.2f})")
        ax.axvline(0, color="#ccc", lw=1); ax.axvline(1, color="#27ae60", lw=1, ls=":")
        ax.set_title(lbl.replace("\n", " ")); ax.set_xlabel("traversal α"); ax.grid(alpha=.25)
        peak_ai[grain] = int(np.argmax(med.values))
    axes[0].set_ylabel("distinctiveness / EBI mAP"); axes[0].set_ylim(-0.02, 1.02); axes[0].legend(fontsize=9)
    fig.suptitle(f"Generated distinctiveness vs α (within-domain copairs, top-{K} cells/class)", fontweight="bold")
    fig.tight_layout()
    for e in ("png", "svg"):
        fig.savefig(f"{OUT}/{fname}.{e}", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"saved {fname}")

    # (2) Real-vs-Generated violin at the peak-α of each grain
    fig, ax = plt.subplots(figsize=(7, 5.5))
    xt, xl = [], []
    for gi, (grain, lbl) in enumerate(GRAINS):
        al, df, real = _series(grain)
        ga = al[peak_ai[grain]]
        pairs = [("Real", np.array(list(real.values())), REAL_C), ("Gen", df[ga].dropna().values, GEN_C)]
        base = gi * 3
        for j, (_, vals, col) in enumerate(pairs):
            pos = base + j
            vp = ax.violinplot([vals], positions=[pos], widths=0.85, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(col); b.set_edgecolor("none"); b.set_alpha(0.9)
            ax.hlines(np.median(vals), pos - 0.42, pos + 0.42, color="k", lw=3, zorder=5)
        xt.append(base + 0.5); xl.append(f"{lbl}\n(gen α={ga:.0f})")
    ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=13)
    ax.set_ylabel("mAP score (distinctiveness / EBI)", fontsize=15); ax.set_ylim(-0.02, 1.02); ax.tick_params(labelsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[mp.Patch(color=REAL_C, label=f"Real (top-{K} acc cells)"),
                       mp.Patch(color=GEN_C, label="Generated (peak α)"),
                       plt.Line2D([0], [0], color="k", lw=3, label="Median")],
              loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=12, frameon=False)
    fig.tight_layout()
    for e in ("png", "svg"):
        fig.savefig(f"{OUT}/distinct_violin.{e}", dpi=150, bbox_inches="tight")
    plt.close(fig); print("saved distinct_violin")

    # (3) per-class scatter: real vs generated distinctiveness at peak α (paired by class)
    from scipy.stats import spearmanr, pearsonr
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (grain, lbl) in zip(axes, GRAINS):
        al, df, real = _series(grain)
        ga = al[peak_ai[grain]]
        gen = df[ga].to_dict()
        cls = [c for c in real if c in gen and not np.isnan(gen[c])]
        x = np.array([real[c] for c in cls]); y = np.array([gen[c] for c in cls])
        ax.scatter(x, y, s=12, alpha=.5, color="#555")
        lim = max(x.max(), y.max()) * 1.05
        ax.plot([0, lim], [0, lim], "--", color="#c0392b", lw=1.5, label="y = x")
        rho = spearmanr(x, y)[0]; r = pearsonr(x, y)[0]
        ax.set_title(f"{lbl.replace(chr(10),' ')} (gen α={ga:.0f})\nSpearman ρ={rho:.2f}, Pearson r={r:.2f}, n={len(cls)}")
        ax.set_xlabel("Real distinctiveness mAP"); ax.set_ylabel("Generated distinctiveness mAP")
        ax.set_xlim(-0.02, lim); ax.set_ylim(-0.02, lim); ax.grid(alpha=.25); ax.legend(fontsize=10)
    fig.suptitle(f"Per-class distinctiveness: Real vs Generated (top-{K} cells/class)", fontweight="bold")
    fig.tight_layout()
    for e in ("png", "svg"):
        fig.savefig(f"{OUT}/distinct_scatter.{e}", dpi=150, bbox_inches="tight")
    plt.close(fig); print("saved distinct_scatter")


def main():
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    keys = [g[0] for g in GRAINS]
    mem = int(os.environ.get("GRD_MEM", 240))              # bump for high-K geneKO (1000-class copairs OOMs >240)
    tmin = int(os.environ.get("GRD_TIME", 180))
    gen_only = os.environ.get("GRD_GEN_ONLY") == "1"       # skip real (already computed; uses only 30 cached cells)
    ais = [int(x) for x in os.environ["GRD_AIS"].split(",")] if os.environ.get("GRD_AIS") else list(range(NA))  # resubmit specific α
    jobs = [] if gen_only else [{"name": f"grd_real_{g}", "func": compute_real, "kwargs": {"grain": g}} for g in keys]
    for g in keys:
        jobs += [{"name": f"grd_{g}_{ai}", "func": compute_gen, "kwargs": {"grain": g, "ai": ai}} for ai in ais]
    print(f"[gen-real-distinct] {len(jobs)} jobs · mem={mem}GB · grains={keys}")
    submit_parallel_jobs(jobs, experiment="gen_real_distinct",
                         slurm_params={"slurm_partition": "cpu", "cpus_per_task": 16, "mem_gb": mem, "timeout_min": tmin},
                         log_dir="gen_real_distinct", wait_for_completion=False)


if __name__ == "__main__":
    import sys
    if "--plot" in sys.argv:
        plot()
    else:
        main()
