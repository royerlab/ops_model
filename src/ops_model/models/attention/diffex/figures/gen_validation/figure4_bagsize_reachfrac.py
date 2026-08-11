"""What fraction of *distinguishable* classes have a GENERATED traversal that reaches real-cell accuracy,
at each bag size? At each bag we keep only classes whose REAL top1_acc is meaningful (>= REAL_THR) — i.e.
there is a real signal to reach — so the denominator n differs per bag (real accuracy climbs with bag size).
A class "reaches" if generated top1_acc is within a lenient margin of real (gen >= real - MARGIN).
Plot % reaching vs bag size, one line per grain; each point annotated with its per-bag n. pdf.fonttype 42."""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
BAGT = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5_bagtest"
OUT = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals"
BAGS = [20, 50, 100, 200, 500]
MARGIN = 0.1                                             # lenient: generated within 0.1 below real counts as reaching
REAL_THR = 0.5                                           # keep only classes whose real cells are distinguishable at that bag


def _load(grain):
    """[(real[bag], gen[bag]) per class] for a grain, aligned to BAGS (nan where missing)."""
    out = []
    for f in sorted(glob.glob(f"{BAGT}/_bagexp_*.json")):
        d = json.load(open(f))
        if d.get("grain") != grain:
            continue
        re = {int(k): v for k, v in d["real_expectation"].items()}
        real = np.array([re.get(b, np.nan) for b in BAGS])
        gen = np.array([d["bag"].get(str(b), {}).get("top1_acc", np.nan) for b in BAGS])
        out.append((real, gen))
    return out


def main():
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for grain, col in [("geneKO", "#1f77b4"), ("complex", "#d62728")]:
        cls = _load(grain)
        if not cls:
            continue
        frac, ns = [], []
        for i, b in enumerate(BAGS):
            reach, tot = 0, 0
            for real, gen in cls:
                if not (np.isfinite(real[i]) and np.isfinite(gen[i])) or real[i] < REAL_THR:
                    continue                              # drop classes with no real signal at this bag
                tot += 1
                reach += gen[i] >= real[i] - MARGIN
            frac.append(100 * reach / tot if tot else np.nan); ns.append(tot)
        ax.plot(BAGS, frac, "-o", color=col, lw=2.5, ms=7, label=f"{grain}")
        for x, y, n in zip(BAGS, frac, ns):
            ax.annotate(f"{y:.0f}%\nn={n}", (x, y), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=7.5, color=col)
    ax.set_xscale("log"); ax.set_xticks(BAGS); ax.set_xticklabels(BAGS)
    ax.set_xlabel("bag size (# cells)   —   n per point = classes with real top1_acc ≥ %.1f at that bag" % REAL_THR)
    ax.set_ylabel(f"% of distinguishable classes reaching real accuracy\n(generated ≥ real − {MARGIN})")
    ax.set_ylim(0, 105)
    ax.set_title("Among perturbations distinguishable by real cells at each bag size,\n"
                 "what fraction do generated traversals recapitulate?")
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}/v5_bagsize_reachfrac.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {OUT}/v5_bagsize_reachfrac.png / .svg")


if __name__ == "__main__":
    main()
