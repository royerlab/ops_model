"""Bag-size sweep plots for the new multibag v5 traversals. Three measures, one line per bag size:
  (1) SetTransformer P(target) / rank / top-k vs α       (bags 20,50,100,200,400)
  (2) centroid-recovery mAP / top-1 / top-5 vs α          (bags 20,50,100,200,400)
  (3) within-domain distinctiveness mAP vs α              (K 20,50,100 — copairs cap)
Split geneKO/complex and (for 1) all-classes vs real-distinguishable (real top1_acc>0.5 @bag20).
"""
import os, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({                                    # figure-ready: big, readable text + heavy elements
    "font.size": 24, "axes.titlesize": 32, "axes.labelsize": 30,
    "xtick.labelsize": 24, "ytick.labelsize": 24, "legend.fontsize": 22,
    "figure.titlesize": 36, "axes.linewidth": 2.0,
    "xtick.major.size": 10, "ytick.major.size": 10, "xtick.major.width": 2.0, "ytick.major.width": 2.0,
    "lines.linewidth": 3.6, "legend.title_fontsize": 22,
})
LW, LWD, LEG = 5.5, 4.0, 22                              # solid / dotted line widths; legend fontsize
CV = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals"
OUT = f"{CV}/bag_sweep_plots_v5new"; os.makedirs(OUT, exist_ok=True)
B = "/hpc/projects/icd.fast.ops/models/diffex"
BAGS = [20, 50, 100, 200]          # 400 dropped: its cells 200-399 are the weak strict-multibag anchors (see anchor_halves)
KS = [20, 50, 100, 200]
CENT_STD = os.environ.get("CENT_STD", "perbag")   # centroid/pooled plots read per-bag α=0 std (domain-honest, matches score_embs_v5); "global" for old panel-mu
CENT_SUF = "_perbag" if CENT_STD == "perbag" else ""
COL = {b: cm.viridis(i / (len(BAGS) - 1)) for i, b in enumerate(BAGS)}
COLK = {k: cm.viridis(i / (len(KS) - 1)) for i, k in enumerate(KS)}
from ops_model.models.interpretability.diffex.classifier.config import slugify
REAL = json.load(open(f"{B}/viewer_assets_v5/real_acc20.json"))


def _keep(grain):
    pre = f"phase/{grain}/"
    return {k[len(pre):] for k, v in REAL.items() if k.startswith(pre) and v > 0.5}


# ---------- (1) SetTransformer bag-sweep ----------
def _agg_st(grain):
    agg = {b: {"al": None, "P": [], "RK": [], "T5": [], "T1": [], "names": []} for b in BAGS}
    for f in glob.glob(f"{CV}/bag_sweep_v5new/{grain}/*.json"):
        d = json.load(open(f)); g = d["gene"]
        for b in BAGS:
            s = d["by_bag"].get(str(b)) or d["by_bag"].get(b)
            if not s or s.get("p_target") is None:
                continue
            agg[b]["al"] = s["alphas"]
            agg[b]["P"].append([np.nan if v is None else v for v in s["p_target"]])
            agg[b]["RK"].append([np.nan if v is None else v for v in s["rank_target"]])
            agg[b]["T5"].append([np.nan if v is None else v for v in s["top5_target"]])
            agg[b]["T1"].append([np.nan if v is None else v for v in s["top1_target"]])
            agg[b]["names"].append(g)
    return agg


def plot_settransformer():
    A = {g: _agg_st(g) for g in ["geneKO", "complex"]}
    for subset, tag in [(None, "all"), (True, "realdist")]:
        fig, axes = plt.subplots(2, 4, figsize=(30, 14))                     # 2 rows = geneKO/complex; 4 cols
        for row, grain in enumerate(["geneKO", "complex"]):
            ax = axes[row]; agg = A[grain]; keep = _keep(grain) if subset else None
            n = 0
            for b in BAGS:
                a = agg[b]
                if a["al"] is None:
                    continue
                al = a["al"]
                msk = np.array([True] * len(a["names"])) if keep is None else np.array([slugify(x) in keep or x in keep for x in a["names"]])
                if not msk.any():
                    continue
                n = int(msk.sum())
                P = np.array(a["P"])[msk]; RK = np.array(a["RK"])[msk]; T5 = np.array(a["T5"])[msk]; T1 = np.array(a["T1"])[msk]
                ax[0].plot(al, np.nanmean(P, 0), "-", color=COL[b], lw=LW, label=f"bag {b}")
                ax[1].plot(al, np.nanmedian(RK, 0), "-", color=COL[b], lw=LW, label=f"bag {b}")
                ax[2].plot(al, np.nanmean(RK, 0), "-", color=COL[b], lw=LW, label=f"bag {b}")
                ax[3].plot(al, np.nanmean(T5, 0) * 100, "-", color=COL[b], lw=LW, label=f"bag {b}")
                ax[3].plot(al, np.nanmean(T1, 0) * 100, ":", color=COL[b], lw=LWD)
            ax[0].set_ylabel(f"{grain} (n={n})\n\nP(target)"); ax[0].set_ylim(-.02, 1.02)
            ax[1].set_ylabel("median target rank"); ax[1].set_yscale("log"); ax[1].axhline(1, color="#ccc")
            ax[2].set_ylabel("mean target rank"); ax[2].set_yscale("log"); ax[2].axhline(1, color="#ccc")
            ax[3].set_ylabel("% recovered"); ax[3].set_ylim(-2, 102)
            if row == 0:
                ax[0].set_title("P(target)"); ax[1].set_title("median target rank"); ax[2].set_title("mean target rank"); ax[3].set_title("% top-5 (solid) / top-1 (dotted)")
            for a_ in ax:
                a_.set_xlabel("traversal α"); a_.set_xticks(range(-5, 6)); a_.grid(alpha=.25); a_.axvline(0, color="#ccc", lw=1); a_.legend(fontsize=LEG)
        ttl = ("all classes" if tag == "all" else
               "real-distinguishable classes only (genes whose REAL cells score top-1 accuracy > 0.5 @ bag-20)")
        fig.suptitle(f"SetTransformer bag-sweep — {ttl}   ·   new multibag v5", fontweight="bold")
        fig.tight_layout()
        for e in ("png", "svg"):
            fig.savefig(f"{OUT}/settransformer_bagsweep_{tag}.{e}", dpi=150, bbox_inches="tight")
        plt.close(fig); print(f"saved settransformer_bagsweep_{tag}")


# ---------- (2) centroid recovery bag-sweep ----------
def plot_centroid():
    fig, axes = plt.subplots(2, 3, figsize=(23, 14))                        # 2 rows = geneKO / complex
    for row, grain in enumerate(["geneKO", "complex"]):
        ax = axes[row]
        p = f"{CV}/centroid_bagsweep_v5new{CENT_SUF}/{grain}_bagsweep.json"
        if not os.path.exists(p):
            print(f"no centroid bagsweep for {grain}"); continue
        d = json.load(open(p)); by = d["by_bag"]
        for b in BAGS:
            bb = by.get(str(b)) or by.get(b)
            if not bb:
                continue
            al = sorted(float(a) for a in bb)
            mp = [np.mean(list(bb[str(a) if str(a) in bb else a]["map"].values())) for a in al]
            t1 = [np.mean(list(bb[str(a) if str(a) in bb else a]["top1"].values())) for a in al]
            t5 = [np.mean(list(bb[str(a) if str(a) in bb else a]["top5"].values())) for a in al]
            ax[0].plot(al, mp, "-", color=COL[b], lw=LW, label=f"bag {b}")
            ax[1].plot(al, np.array(t1) * 100, "-", color=COL[b], lw=LW, label=f"bag {b}")
            ax[2].plot(al, np.array(t5) * 100, "-", color=COL[b], lw=LW, label=f"bag {b}")
        cf = f"{CV}/centroid_bagsweep_v5new/{grain}_ceiling.json"                  # real-cell ceiling (per-cell → bag-independent)
        if os.path.exists(cf):
            c = json.load(open(cf))
            ax[0].axhline(c["map"], ls=":", color="#c0392b", lw=LWD, label="real cells")
            ax[1].axhline(c["top1"] * 100, ls=":", color="#c0392b", lw=LWD, label="real cells")
            ax[2].axhline(c["top5"] * 100, ls=":", color="#c0392b", lw=LWD, label="real cells")
        ax[0].set_ylabel(f"{grain} (n={d['n_classes']})\n\nmAP")
        ax[1].set_ylabel("% of cells (top-1)")
        ax[2].set_ylabel("% of cells (top-5)")
        if row == 0:
            ax[0].set_title("centroid-recovery mAP"); ax[1].set_title("top-1"); ax[2].set_title("top-5")
        for a_ in ax:
            a_.set_xlabel("traversal α"); a_.set_xticks(range(-5, 6)); a_.grid(alpha=.25); a_.axvline(0, color="#ccc", lw=1); a_.legend(fontsize=LEG)
    fig.suptitle(f"Centroid-recovery bag-sweep — multibag v5 ({CENT_STD} α=0 standardization)", fontweight="bold")
    fig.tight_layout()
    gtag = "" if CENT_STD == "perbag" else "_global"
    for e in ("png", "svg"):
        fig.savefig(f"{OUT}/centroid_bagsweep{gtag}.{e}", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"saved centroid_bagsweep{gtag} (2-row)")


# ---------- (2b) pooled bag-level centroid recovery (per-bag real ceiling) ----------
def plot_centroid_pooled():
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))                         # geneKO | complex
    for ax, grain in zip(axes, ["geneKO", "complex"]):
        p = f"{CV}/centroid_pooled_bagsweep_v5new{CENT_SUF}/{grain}_pooled.json"
        if not os.path.exists(p):
            print(f"no pooled centroid for {grain}"); continue
        d = json.load(open(p)); ng = 0
        for b in BAGS:
            bb = d["gen"].get(str(b))
            if not bb:
                continue
            al = sorted(float(a) for a in bb)
            t1 = [np.mean(list(bb[str(a) if str(a) in bb else a]["top1"].values())) for a in al]
            ax.plot(al, np.array(t1) * 100, "-", color=COL[b], lw=LW, label=f"gen bag {b}")
            rc = d["real"].get(str(b))
            if rc:
                ng = len(rc); ax.axhline(np.mean(list(rc.values())) * 100, ls=":", color=COL[b], lw=LWD)
        ax.set_xlabel("traversal α"); ax.set_xticks(range(-5, 6)); ax.grid(alpha=.25); ax.axvline(0, color="#ccc", lw=1)
        ax.set_ylim(-2, 102); ax.set_title(f"{grain} (n={ng})")
    axes[0].set_ylabel("% classes recovering\ntrue real centroid (top-1)")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=COL[b], lw=LW, label=f"gen bag {b}") for b in BAGS]
    handles.append(Line2D([0], [0], color="k", ls=":", lw=LWD, label="real ceiling (per bag)"))
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.87, 0.5), frameon=False, fontsize=LEG)
    fig.suptitle(f"Pooled bag-level centroid recovery  ·  multibag v5 ({CENT_STD} α=0 std)\nsolid = generated · dotted = real-cell ceiling (per bag)",
                 fontweight="bold", fontsize=24)
    fig.tight_layout(rect=[0.02, 0, 0.86, 0.96])
    gtag = "" if CENT_STD == "perbag" else "_global"
    for e in ("png", "svg"):
        fig.savefig(f"{OUT}/centroid_pooled_bagsweep{gtag}.{e}", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"saved centroid_pooled_bagsweep{gtag}")


# ---------- (2c) cross-domain retrieval mAP (proper1k, full real gallery) ----------
def plot_proper1k():
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, grain in zip(axes, ["geneKO", "complex"]):
        p = f"{CV}/gen_real_centroid_v5new/{grain}_propermap1k.json"
        if not os.path.exists(p):
            print(f"no proper1k for {grain}"); continue
        d = json.load(open(p)); al = sorted(float(a) for a in d["gen"]); by = d["gen"]
        mp = [np.mean(list(by[str(a) if str(a) in by else a].values())) for a in al]
        ax.plot(al, mp, "-", color="#2e8b57", lw=LW, label="generated → real")
        if d.get("ceiling"):
            ax.axhline(np.mean(list(d["ceiling"].values())), ls=":", color="#c0392b", lw=LWD, label="real self-consistency")
        ax.set_xlabel("traversal α"); ax.set_xticks(range(-5, 6)); ax.grid(alpha=.25); ax.axvline(0, color="#ccc", lw=1)
        ax.set_ylim(-.02, 1.02); ax.set_title(f"{grain} (n={d['n_classes']})"); ax.legend(fontsize=LEG)
    axes[0].set_ylabel("cross-domain retrieval mAP\n(generated cell → real class cells)")
    fig.suptitle("Cross-domain retrieval mAP — generated → full 1000-cell real gallery  ·  multibag v5", fontweight="bold", fontsize=26)
    fig.tight_layout()
    for e in ("png", "svg"):
        fig.savefig(f"{OUT}/retrieval_map_proper1k.{e}", dpi=150, bbox_inches="tight")
    plt.close(fig); print("saved retrieval_map_proper1k")


# ---------- (2d) first-200 vs second-200 anchor halves ----------
def plot_halves():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, grain in zip(axes, ["geneKO", "complex"]):
        p = f"{CV}/centroid_halves_v5new/{grain}_halves.json"
        if not os.path.exists(p):
            print(f"no halves for {grain}"); continue
        d = json.load(open(p)); n = 0
        for key, col in [("first", "#1f77b4"), ("second", "#d62728")]:
            al = sorted(float(a) for a in d[key])
            t1 = [np.mean(list(d[key][str(a) if str(a) in d[key] else a].values())) * 100 for a in al]
            n = len(d[key][str(al[np.argmax(t1)])])
            ax.plot(al, t1, "-", color=col, lw=LW)
        ax.set_xlabel("traversal α"); ax.set_xticks(range(-5, 6)); ax.grid(alpha=.25); ax.axvline(0, color="#ccc", lw=1)
        ax.set_ylim(-2, 102); ax.set_title(f"{grain} (n={n})")
    axes[0].set_ylabel("% classes recovering\ntrue real centroid (top-1)")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0], [0], color="#1f77b4", lw=LW, label="hand-picked anchors (curated, cells 0–199)"),
                        Line2D([0], [0], color="#d62728", lw=LW, label="strict multibag top-200 NTC (cells 200–399)")],
               loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=2, frameon=False, fontsize=LEG)
    fig.suptitle("Anchor selection drives recovery: hand-picked (first 200) vs strict multibag top-NTC (second 200)\nsame directions/traversals — only the anchor cells differ  ·  multibag v5",
                 fontweight="bold", fontsize=22, y=0.99)
    fig.subplots_adjust(top=0.84, bottom=0.28, left=0.08, right=0.97, wspace=0.16)   # extra bottom room: legend clear of x-axis
    for e in ("png", "svg"):
        fig.savefig(f"{OUT}/anchor_halves.{e}", dpi=150, bbox_inches="tight")
    plt.close(fig); print("saved anchor_halves")


# ---------- (2e) SetTransformer: first-200 vs second-200 anchor halves ----------
def plot_st_halves():
    fig, axes = plt.subplots(2, 4, figsize=(25, 11))
    for row, grain in enumerate(["geneKO", "complex"]):
        ax = axes[row]; files = glob.glob(f"{CV}/st_halves_v5new/{grain}/*.json")
        agg = {h: {"al": None, "P": [], "RK": [], "T5": [], "T1": []} for h in ("first", "second")}
        for f in files:
            d = json.load(open(f))
            for h in ("first", "second"):
                s = d.get(h)
                if not s or s.get("p_target") is None:
                    continue
                agg[h]["al"] = s["alphas"]
                for k, kk in [("P", "p_target"), ("RK", "rank_target"), ("T5", "top5_target"), ("T1", "top1_target")]:
                    agg[h][k].append([np.nan if v is None else v for v in s[kk]])
        n = len(agg["first"]["P"])
        for h, col, lab in [("first", "#1f77b4", "hand-picked (0–199)"), ("second", "#d62728", "strict multibag NTC (200–399)")]:
            a = agg[h]
            if a["al"] is None:
                continue
            al = a["al"]
            ax[0].plot(al, np.nanmean(a["P"], 0), "-", color=col, lw=LW, label=lab)
            ax[1].plot(al, np.nanmedian(a["RK"], 0), "-", color=col, lw=LW, label=lab)
            ax[2].plot(al, np.nanmean(a["RK"], 0), "-", color=col, lw=LW, label=lab)
            ax[3].plot(al, np.nanmean(a["T5"], 0) * 100, "-", color=col, lw=LW, label=lab)
            ax[3].plot(al, np.nanmean(a["T1"], 0) * 100, ":", color=col, lw=LWD)
        ax[0].set_ylabel(f"{grain} (n={n})\n\nP(target)"); ax[0].set_ylim(-.02, 1.02)
        ax[1].set_ylabel("median target rank"); ax[1].set_yscale("log"); ax[1].axhline(1, color="#ccc")
        ax[2].set_ylabel("mean target rank"); ax[2].set_yscale("log"); ax[2].axhline(1, color="#ccc")
        ax[3].set_ylabel("% recovered"); ax[3].set_ylim(-2, 102)
        if row == 0:
            ax[0].set_title("P(target)"); ax[1].set_title("median target rank")
            ax[2].set_title("mean target rank"); ax[3].set_title("% top-5 (solid) / top-1 (dotted)")
        for a_ in ax:
            a_.set_xlabel("traversal α"); a_.set_xticks(range(-5, 6)); a_.grid(alpha=.25); a_.axvline(0, color="#ccc", lw=1)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(0.995, 0.5), fontsize=LEG, frameon=False)
    fig.suptitle("SetTransformer: hand-picked (first 200) vs strict multibag top-NTC (second 200) anchors  ·  bag=200 each  ·  multibag v5",
                 fontweight="bold", fontsize=22)
    fig.tight_layout(rect=(0, 0, 0.995, 1))
    for e in ("png", "svg"):
        fig.savefig(f"{OUT}/st_anchor_halves.{e}", dpi=150, bbox_inches="tight")
    plt.close(fig); print("saved st_anchor_halves")


# ---------- (3) distinctiveness sweep ----------
def plot_distinct(stat="median"):
    agg = np.mean if stat == "mean" else np.median
    suf = "_mean" if stat == "mean" else "_median"
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    for ax, grain in zip(axes, ["geneKO", "complex"]):
        for k in KS:
            d = f"{CV}/gen_real_distinct_v5new_K{k}"
            rf = f"{d}/{grain}_real.json"
            if not os.path.exists(rf):
                continue
            al, v = [], []
            for f in sorted(glob.glob(f"{d}/{grain}_gen_a*.json"), key=lambda p: int(p.split("_a")[-1][:-5])):
                g = json.load(open(f)); al.append(g["alpha"]); v.append(agg(list(g["gen"].values())))
            if not al:                                       # gen OOM'd at this K (e.g. geneKO K≥100) → skip, no phantom ceiling
                continue
            order = np.argsort(al); al = np.array(al)[order]; v = np.array(v)[order]
            ax.plot(al, v, "-", color=COLK[k], lw=LW, label=f"top-{k}")
            ax.axhline(agg(list(json.load(open(rf)).values())), color=COLK[k], ls=":", lw=LWD)
        ax.set_xlabel("traversal α"); ax.set_xticks(range(-5, 6)); ax.grid(alpha=.25); ax.axvline(0, color="#ccc", lw=1)
        ax.set_title(grain); ax.legend(fontsize=LEG, title="cells/class")
    axes[0].set_ylabel(f"{stat} distinctiveness mAP")
    fig.suptitle(f"Distinctiveness sweep ({stat}) — new multibag v5", fontweight="bold")
    fig.tight_layout()
    for e in ("png", "svg"):
        fig.savefig(f"{OUT}/distinct_sweep{suf}.{e}", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"saved distinct_sweep{suf}")


def plot_distinct_violin(k=50):
    """Per-class distinctiveness distribution (generated vs real) at each grain's peak α, K cells/class."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
    for ax, grain in zip(axes, ["geneKO", "complex"]):
        d = f"{CV}/gen_real_distinct_v5new_K{k}"
        rf = f"{d}/{grain}_real.json"
        if not os.path.exists(rf):
            print(f"no distinct K{k} for {grain}"); continue
        gens = {}
        for f in glob.glob(f"{d}/{grain}_gen_a*.json"):
            g = json.load(open(f)); gens[g["alpha"]] = g["gen"]
        al = sorted(gens); med = [np.median(list(gens[a].values())) for a in al]
        pa = al[int(np.argmax(med))]                                         # peak α by median gen mAP
        gv = np.array(list(gens[pa].values())); rv = np.array(list(json.load(open(rf)).values()))
        parts = ax.violinplot([rv, gv], positions=[0, 1], widths=0.8, showextrema=False, showmedians=False)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(["#8fa9c9", "#2e8b57"][i]); pc.set_alpha(.85); pc.set_edgecolor("none")
        for pos, vals in [(0, rv), (1, gv)]:                                 # thick black median bar only (no box/extrema)
            ax.hlines(np.median(vals), pos - 0.34, pos + 0.34, color="k", lw=6, zorder=5)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["real", f"generated\n(α={pa:+g})"])
        ax.set_title(f"{grain} (n={len(gv)})"); ax.grid(alpha=.25, axis="y")
    axes[0].set_ylabel("distinctiveness / EBI\nmAP score")
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    axes[-1].legend(handles=[Patch(facecolor="#8fa9c9", label="Real"), Patch(facecolor="#2e8b57", label="Generated (peak α)"),
                             Line2D([0], [0], color="k", lw=6, label="Median")],
                    loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=LEG)
    fig.suptitle(f"Per-class distinctiveness: real vs generated (peak α, top-{k} cells/class)  ·  multibag v5", fontweight="bold", fontsize=26)
    fig.tight_layout()
    for e in ("png", "svg"):
        fig.savefig(f"{OUT}/distinct_violin.{e}", dpi=150, bbox_inches="tight")
    plt.close(fig); print("saved distinct_violin")


def plot_control():
    """Control: the first/second centroid-recovery gap is a standardization artifact. Peak-α top-1 under GLOBAL
    (panel α=0 mean, current metric) vs PER-BAG (each half's own α=0, as score_embs_v5 does). The gap collapses
    under per-bag → the 76/13 is metric standardization, not phenotype loss."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    schemes = [("global", "global α=0\n(panel mean)"), ("perbag", "per-bag α=0\n(own mean)")]
    for ax, grain in zip(axes, ["geneKO", "complex"]):
        c = json.load(open(f"{CV}/control_halves_v5new/{grain}_control.json"))
        vals = {s: {} for s, _ in schemes}
        for s, _ in schemes:
            for h in ("first", "second"):
                byA = {float(a): float(np.mean(list(r.values()))) for a, r in c[s][h].items() if r}
                vals[s][h] = max(byA.values()) if byA else 0.0
        x = np.arange(len(schemes)); w = 0.36
        ax.bar(x - w / 2, [vals[s]["first"] * 100 for s, _ in schemes], w, color="#1f77b4", label="hand-picked (0–199)")
        ax.bar(x + w / 2, [vals[s]["second"] * 100 for s, _ in schemes], w, color="#d62728", label="strict multibag NTC (200–399)")
        for i, (s, _) in enumerate(schemes):
            for dx, h in [(-w / 2, "first"), (w / 2, "second")]:
                ax.text(i + dx, vals[s][h] * 100 + 1.5, f"{vals[s][h]*100:.0f}", ha="center", va="bottom", fontsize=20)
        ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in schemes]); ax.set_ylim(0, 100)
        ax.set_title(grain); ax.grid(alpha=.25, axis="y")
    axes[0].set_ylabel("peak-α centroid top-1 (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.055), ncol=2, fontsize=LEG, frameon=False)
    fig.suptitle("Centroid-recovery gap is a standardization artifact: per-bag α=0 collapses it", fontweight="bold", fontsize=24)
    fig.subplots_adjust(top=0.85, bottom=0.24, left=0.09, right=0.97, wspace=0.14)
    for e in ("png", "svg"):
        fig.savefig(f"{OUT}/control_halves_zscore.{e}", dpi=150, bbox_inches="tight")
    plt.close(fig); print("saved control_halves_zscore")


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "control":
        plot_control(); sys.exit()
    if which == "violin":
        plot_distinct_violin(); sys.exit()
    if which == "pooled":
        plot_centroid_pooled(); sys.exit()
    if which == "proper1k":
        plot_proper1k(); sys.exit()
    if which == "halves":
        plot_halves(); sys.exit()
    if which == "sthalves":
        plot_st_halves(); sys.exit()
    if which in ("all", "st"):
        plot_settransformer()
    if which in ("all", "cent"):
        plot_centroid()
    if which in ("all", "dist"):
        plot_distinct("median"); plot_distinct("mean")
