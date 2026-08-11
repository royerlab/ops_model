"""Compare the 200-cell validation bag (valid200) vs the 45-cell bag on the Cell-DINO mAP measures, both grains.

Centroid recovery (gen cells -> nearest faithful 1000-real centroid; same centroids for both bags, only the
generated bag differs) and within-domain distinctiveness. Rows = grain (geneKO, complex); cols = centroid
mAP / top-1 / top-5 / distinctiveness. geneKO has both bags; complex only exists at the 45-cell bag (no
200-cell complex generation). thr!=None restricts to the SetTransformer-distinguishable subset: classes whose
REAL cells clear top1_acc > thr @ bag20 (real_acc20.json) — a FIXED set (same classes for both bags). Reads
gen_real_centroid.score + gen_real_distinct JSONs per cache.
"""
import json, glob, os
import numpy as np
from ops_model.models.interpretability.diffae.classifier.config import slugify

CV = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals"
OUT = f"{CV}/valid200_metrics"
BIG, SM = "#c0392b", "#888"
REAL_ACC = json.load(open("/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5/real_acc20.json"))

# (label, color, ls, centroid_dir, distinct_dir) per grain; distinct na inferred from glob
BAGS = {
    "geneKO": [("200-cell bag", BIG, "-", f"{CV}/gen_real_centroid_valid200", f"{CV}/gen_real_distinct_valid200"),
               ("45-cell bag", SM, "--", f"{CV}/gen_real_centroid", f"{CV}/gen_real_distinct")],
    "complex": [("45-cell bag", SM, "--", f"{CV}/gen_real_centroid", f"{CV}/gen_real_distinct")],
}
GRAIN_LBL = {"geneKO": "Gene-level (distinctiveness)", "complex": "Protein complex (EBI)"}


def _keep(grain, thr):
    """SetTransformer-distinguishable subset: real cells top1_acc > thr @ bag20. Slug-keyed (matches complex
    names with spaces). None = all classes."""
    if thr is None:
        return None
    pre = f"phase/{grain}/"
    return {k[len(pre):] for k, v in REAL_ACC.items() if k.startswith(pre) and v > thr}


def _in(c, keep):
    return keep is None or slugify(c) in keep


def _cent(centdir, grain, keep):
    d = json.load(open(f"{centdir}/{grain}_scored.json")); al = d["alphas"]; by = d["by_alpha"]
    def f(k):
        return np.array([np.mean([v for c, v in by[str(a)][k].items() if _in(c, keep)]) for a in al])
    n = sum(_in(c, keep) for c in by[str(al[0])]["map"])
    return np.array(al), f("map"), f("top1"), f("top5"), n


def _dist(distdir, grain, keep):
    na = len(glob.glob(f"{distdir}/{grain}_gen_a*.json"))
    real = json.load(open(f"{distdir}/{grain}_real.json"))
    rv = [v for c, v in real.items() if _in(c, keep)]
    rm = float(np.median(rv)) if rv else np.nan
    al, med = [], []
    for ai in range(na):
        g = json.load(open(f"{distdir}/{grain}_gen_a{ai}.json"))
        vv = [v for c, v in g["gen"].items() if _in(c, keep)]
        al.append(g["alpha"]); med.append(np.median(vv) if vv else np.nan)
    return np.array(al), np.array(med), rm


def plot(thr=None, suffix=""):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["pdf.fonttype"] = 42
    os.makedirs(OUT, exist_ok=True)
    grains = list(BAGS)
    fig, axes = plt.subplots(len(grains), 4, figsize=(21, 5 * len(grains)), squeeze=False)
    for gi, grain in enumerate(grains):
        ax = axes[gi]
        for lbl, c, ls, cd, dd in BAGS[grain]:
            keep = _keep(grain, thr)
            al, mp, t1, t5, n = _cent(cd, grain, keep)
            tag = f"{lbl} (n={n})"
            ax[0].plot(al, mp, ls, color=c, lw=2.4, label=tag)
            ax[1].plot(al, t1 * 100, ls, color=c, lw=2.4, label=tag)
            ax[2].plot(al, t5 * 100, ls, color=c, lw=2.4, label=tag)
            ald, med, rm = _dist(dd, grain, keep)
            ax[3].plot(ald, med, ls, color=c, lw=2.4, label=lbl)
            ax[3].axhline(rm, color=c, ls=":", lw=1.5, label=f"{lbl} real ({rm:.3f})")
        ax[0].set_ylabel(f"{GRAIN_LBL[grain]}\n\nmAP (1/rank true centroid)")
        ax[0].set_title("centroid-recovery mAP vs α"); ax[1].set_title("centroid top-1 vs α")
        ax[2].set_title("centroid top-5 vs α"); ax[3].set_title("distinctiveness median mAP vs α")
        ax[1].set_ylabel("% cells nearest true centroid"); ax[2].set_ylabel("% cells within top-5")
        ax[3].set_ylabel("median distinctiveness / EBI mAP")
        for a in ax:
            a.set_xlabel("traversal α"); a.grid(alpha=.25); a.axvline(0, color="#ccc", lw=1); a.legend(fontsize=9)
    ttl = "Cell-DINO mAP: 200-cell vs 45-cell bag (faithful real centroids)"
    if thr is not None:
        ttl += f"  —  SetTransformer-distinguishable subset (real top1_acc > {thr} @ bag20)"
    fig.suptitle(ttl, fontweight="bold", fontsize=15)
    fig.tight_layout()
    for e in ("png", "svg"):
        fig.savefig(f"{OUT}/valid200_map_compare{suffix}.{e}", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"saved valid200_map_compare{suffix}")
    for grain in grains:
        for lbl, c, ls, cd, dd in BAGS[grain]:
            keep = _keep(grain, thr)
            al, mp, t1, t5, n = _cent(cd, grain, keep); k = int(np.argmax(mp))
            print(f"  {grain:8s} {lbl:12s} n={n:4d}: peak α={al[k]:+.1f}  mAP={mp[k]:.3f}  top1={t1[k]:.1%}  top5={t5[k]:.1%}")


if __name__ == "__main__":
    plot()                          # all classes
    plot(thr=0.5, suffix="_acc50")   # SetTransformer-distinguishable subset (real top1_acc>0.5 @bag20)
