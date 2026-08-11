"""Figure 4 summary: v5 SetTransformer accuracy of GENERATED traversals vs α, aggregated across the
1K geneKO set and the EBI-complex set. For each α: mean P(target) + IQR (Q1–Q3) band over all traversals.
Reads scores_v5.json (per-α P(target)) from viewer_assets_v5. White bg, pdf.fonttype 42."""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
V5 = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_v5/phase"
OUT = "/hpc/projects/icd.fast.ops/analysis/figure4_traversals"


EVAL = "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/phase"


def _real_acc20(csv):
    """Alex real-cell top1_acc at bag=20, keyed by gene/label (from the eval CSV)."""
    import csv as _csv
    return {r["gene_name"]: float(r["top1_acc"]) for r in _csv.DictReader(open(f"{EVAL}/{csv}")) if int(r["n_cells"]) == 20}


def _geneKO_allow(thr=0.9):
    """geneKO dir-names whose REAL-cell top1_acc@bag20 > thr (Alex gene eval)."""
    return {g for g, a in _real_acc20("eval_phase_e200_pergene_val.csv").items() if a > thr}


def _complex_allow(thr=0.9):
    """Complex dir-slugs whose MEAN member-gene real top1_acc@bag20 > thr — grouped by Alex's own
    label_name in the ebionly eval CSV (his reported members), then mean of those member accuracies."""
    import csv as _csv
    from collections import defaultdict
    from ops_model.models.attention.diffex.classifier.config import slugify
    by = defaultdict(list)
    for r in _csv.DictReader(open(f"{EVAL}/eval_phase_ebionly_e200_pergene_val.csv")):
        if int(r["n_cells"]) == 20:
            by[r["label_name"]].append(float(r["top1_acc"]))
    return {slugify(lbl) for lbl, accs in by.items() if np.mean(accs) > thr}


def _real_map(sub):
    """{dir-name: real-cell top1_acc @bag20}. geneKO = gene eval; complex = mean of Alex's members (by slug)."""
    if sub == "geneKO":
        return _real_acc20("eval_phase_e200_pergene_val.csv")
    import csv as _csv
    from collections import defaultdict
    from ops_model.models.attention.diffex.classifier.config import slugify
    by = defaultdict(list)
    for r in _csv.DictReader(open(f"{EVAL}/eval_phase_ebionly_e200_pergene_val.csv")):
        if int(r["n_cells"]) == 20:
            by[r["label_name"]].append(float(r["top1_acc"]))
    return {slugify(l): float(np.mean(v)) for l, v in by.items()}


VALID200 = "/hpc/projects/icd.fast.ops/models/diffex/viewer_assets_valid200/phase"   # 200-cell bag (geneKO only)


def _collect(sub, allow=None, base=None):
    """Stack p_target across NTC-anchored traversals in a subdir → (alphas, matrix, names). allow: dir-name set."""
    base = base or V5
    rows, names, alphas = [], [], None
    for f in sorted(glob.glob(f"{base}/{sub}/*/scores_v5.json")):
        if "__to__" in f:                       # NTC-anchored only (skip alt-anchor A→B)
            continue
        name = os.path.basename(os.path.dirname(f))
        if allow is not None and name not in allow:
            continue
        d = json.load(open(f))
        alphas = d["alphas"]
        rows.append([np.nan if v is None else v for v in d["p_target"]]); names.append(name)
    return np.array(alphas, float), np.array(rows, float), names


def _render(series, out_stem, title, overlays=None):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for sub, allow, label, col in series:
        al, M, names = _collect(sub, allow)
        if not len(M):
            print(f"no scores for {sub}"); continue
        mean = np.nanmean(M, 0); nfin = np.sum(np.isfinite(M), 0)
        sem = np.nanstd(M, 0, ddof=1) / np.sqrt(np.maximum(nfin, 1))               # generated SEM per α
        ax.plot(al, mean, "-", color=col, lw=2.5, label=f"{label} generated (n={M.shape[0]})", zorder=5)
        ax.fill_between(al, mean - sem, mean + sem, color=col, alpha=0.2, lw=0)     # mean ± SEM
        rm = _real_map(sub); reals = np.array([rm[n] for n in names if n in rm])   # real-cell ceiling @bag20
        if len(reals):
            rmean = reals.mean(); rsem = reals.std(ddof=1) / np.sqrt(len(reals))
            ax.axhline(rmean, color=col, ls=":", lw=1.8, alpha=0.95, zorder=6, label=f"{label} real @bag20 ({rmean:.2f})")
            ax.axhspan(rmean - rsem, rmean + rsem, color=col, alpha=0.1, lw=0)      # real mean ± SEM band
        print(f"[{out_stem}] {sub}: n={M.shape[0]} peak mean={np.nanmax(mean):.3f} @α={al[np.nanargmax(mean)]:+g} real mean@bag20={reals.mean():.3f}")
    for sub, allow, label, col, base in (overlays or []):                          # dashed = 200-cell bag (same real ceiling)
        al, M, names = _collect(sub, allow, base)
        if not len(M):
            print(f"no overlay scores for {sub}"); continue
        mean = np.nanmean(M, 0); sem = np.nanstd(M, 0, ddof=1) / np.sqrt(np.maximum(np.sum(np.isfinite(M), 0), 1))
        ax.plot(al, mean, "--", color=col, lw=2.5, label=f"{label} (n={M.shape[0]})", zorder=5)
        ax.fill_between(al, mean - sem, mean + sem, color=col, alpha=0.15, lw=0)
        print(f"[{out_stem}] OVERLAY {sub}: n={M.shape[0]} peak mean={np.nanmax(mean):.3f} @α={al[np.nanargmax(mean)]:+g}")
    ax.axvline(0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("α  (traversal strength; ±1 ≈ control→KD gap)")
    ax.set_ylabel("P(target class)  —  v5 SetTransformer")
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1), framealpha=0.9)   # outside, to the right
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}/{out_stem}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {OUT}/{out_stem}.png / .svg")


def main():
    suf = os.environ.get("SUMMARY_SUFFIX", "")   # e.g. "_corrected" → new files, don't overwrite the old plot
    ov_full = None if os.environ.get("NO_OVERLAY") else [("geneKO", None, "geneKO 200-cell bag", "#1f77b4", VALID200),
                                                          ("complex", None, "complex 200-cell bag", "#d62728", VALID200)]
    # full: all geneKO + all complexes
    _render([("geneKO", None, "geneKO (1K)", "#1f77b4"), ("complex", None, "EBI complexes", "#d62728")],
            "v5_accuracy_vs_alpha_summary" + suf,
            "Generated-traversal accuracy vs α\n(generated mean ± SEM; dotted = real-cell mean top1_acc @bag20)",
            overlays=ov_full)
    # filtered: high real-accuracy classes (real acc>thr @bag20). Dotted line = real-cell ceiling for that set.
    for thr, stem in [(0.8, "v5_accuracy_vs_alpha_realacc80" + suf), (0.9, "v5_accuracy_vs_alpha_realacc90" + suf)]:
        gk, cx = _geneKO_allow(thr), _complex_allow(thr)
        print(f"filtered allowlists (>{thr}): {len(gk)} geneKO, {len(cx)} complex")
        ov = None if os.environ.get("NO_OVERLAY") else [("geneKO", gk, f"geneKO 200-cell bag (real acc>{thr})", "#1f77b4", VALID200),
                                                        ("complex", cx, f"complex 200-cell bag (real acc>{thr})", "#d62728", VALID200)]
        _render([("geneKO", gk, f"geneKO (real acc>{thr} @bag20)", "#1f77b4"),
                 ("complex", cx, f"EBI complex (mean member real acc>{thr})", "#d62728")],
                stem,
                f"Generated accuracy vs α — high real-accuracy classes only\n(geneKO real acc>{thr}; complex mean-member real acc>{thr}; @bag20)",
                overlays=ov)


if __name__ == "__main__":
    main()
