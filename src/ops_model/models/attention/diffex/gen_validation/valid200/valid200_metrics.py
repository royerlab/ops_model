"""SetTransformer rank / set-accuracy on the 200-cell validation bag (viewer_assets_valid200), vs the small bag.

valid200 = 1000 phase geneKO traversals with 200 generated cells/class (bigger bag → more evidence for the
set classifier). scores_v5.json is already computed inline (P(target), rank_target, top1/top5_target per α).
This aggregates them across all classes and overlays the small-bag reference (viewer_assets_v5, ~45-cell bag).

mAP is separate (needs Cell-DINO embeddings of the 200-cell bags → gen_real_map); this file is the classifier
rank / set-accuracy only (no GPU — pure JSON aggregation).
"""
import json, glob, os
import numpy as np

BASE = "/hpc/projects/icd.fast.ops/models/diffex"
BIG = f"{BASE}/viewer_assets_valid200/phase/geneKO"      # 200-cell bag
SMALL = f"{BASE}/viewer_assets_v5/phase/geneKO"          # ~45-cell bag (reference)
OUT = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals/valid200_metrics"


def _collect(d):
    """→ (alphas, P, RK, T1, T5) each (n_class, n_alpha) from scores_v5.json under dir d."""
    al, P, RK, T1, T5, genes = None, [], [], [], [], []
    for f in sorted(glob.glob(f"{d}/*/scores_v5.json")):
        s = json.load(open(f)); al = s["alphas"]
        P.append(s["p_target"]); RK.append(s["rank_target"]); T1.append(s["top1_target"]); T5.append(s["top5_target"])
        genes.append(os.path.basename(os.path.dirname(f)))
    return np.array(al), np.array(P), np.array(RK, float), np.array(T1, float), np.array(T5, float), genes


def summary():
    os.makedirs(OUT, exist_ok=True)
    al, P, RK, T1, T5, genes = _collect(BIG)
    bag = json.load(open(f"{BIG}/{genes[0]}/scores_v5.json")).get("bag")
    pk = int(np.argmax(P.mean(0)))          # peak α by mean P(target)
    print(f"[valid200] {len(genes)} geneKO, bag={bag}, α={list(np.round(al,1))}")
    print(f"  peak α={al[pk]:+.1f}:  mean P(target)={P[:,pk].mean():.2f}  median rank={np.median(RK[:,pk]):.0f}  "
          f"top1={T1[:,pk].mean():.0%}  top5={T5[:,pk].mean():.0%}")
    # real-distinguishable-ish subset: classes that ever reach top-1 (min rank == 1)
    dist = RK.min(1) == 1
    print(f"  among classes reaching top-1 at some α (n={dist.sum()}): peak top5={T5[dist, pk].mean():.0%}")
    return al, P, RK, T1, T5, bag


def plot():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["pdf.fonttype"] = 42
    os.makedirs(OUT, exist_ok=True)
    alB, PB, RKB, T1B, T5B, gB = _collect(BIG); bagB = json.load(open(f"{BIG}/{gB[0]}/scores_v5.json")).get("bag")
    alS, PS, RKS, T1S, T5S, gS = _collect(SMALL); bagS = json.load(open(f"{SMALL}/{gS[0]}/scores_v5.json")).get("bag")
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for al, P, RK, T1, T5, bag, c, ls in [(alB, PB, RKB, T1B, T5B, bagB, "#c0392b", "-"),
                                          (alS, PS, RKS, T1S, T5S, bagS, "#888", "--")]:
        lbl = f"bag={bag}"
        m = P.mean(0); q1, q3 = np.percentile(P, [25, 75], 0)
        ax[0].plot(al, m, ls, color=c, lw=2.4, label=lbl); ax[0].fill_between(al, q1, q3, color=c, alpha=.12, lw=0)
        ax[1].plot(al, np.median(RK, 0), ls, color=c, lw=2.4, label=lbl)
        ax[2].plot(al, (T5).mean(0) * 100, ls, color=c, lw=2.4, label=f"{lbl} top-5")
        ax[2].plot(al, (T1).mean(0) * 100, ls, color=c, lw=1.4, alpha=.7, label=f"{lbl} top-1")
    ax[0].set_title("P(target class) vs α (mean, IQR)"); ax[0].set_ylabel("P(target)"); ax[0].set_ylim(-.02, 1.02)
    ax[1].set_title("median target rank vs α"); ax[1].set_ylabel("rank (1 = top pick)"); ax[1].set_yscale("log"); ax[1].axhline(1, color="#ccc", lw=1)
    ax[2].set_title("% recovered vs α"); ax[2].set_ylabel("% of geneKOs"); ax[2].set_ylim(-2, 102)
    for a in ax:
        a.set_xlabel("traversal α"); a.grid(alpha=.25); a.legend(fontsize=8)
    fig.suptitle(f"v5 SetTransformer on the 200-cell bag (n={len(gB)} geneKO) vs the {bagS}-cell bag", fontweight="bold")
    fig.tight_layout()
    for e in ("png", "svg"):
        fig.savefig(f"{OUT}/valid200_setacc.{e}", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"saved {OUT}/valid200_setacc")


if __name__ == "__main__":
    summary(); plot()
