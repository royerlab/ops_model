"""Violin plot comparing per-perturbation/per-complex mAP distributions across
the v4 attention-weighted and sister-weighted v3-pipeline runs vs the
mean-aggregation baseline.

Looks under ``paper_v1/{attention,sister}/<strategy>/phase_only/...`` for each
strategy and ``paper_v1/phase_only/...`` for the baseline. Skips strategies
whose metric CSVs aren't on disk yet, so this is safe to re-run as more
strategies finish.

Bruno paths
-----------
* Script: this file
* Baseline:           <root>/phase_only/fixed_80%/cosine/metrics/*.csv
* Attention strats:   <root>/attention/<strategy>/phase_only/fixed_80%/cosine/metrics/*.csv
* Sister strats:      <root>/sister/<strategy>/phase_only/fixed_80%/cosine/metrics/*.csv
* Output:             <root>/attention/comparison/v4_attn_<n>metric_violin.{png,pdf,svg}

where ``<root>`` is
/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3/cell_dino/zscore_per_exp/paper_v1
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(
    "/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3/"
    "cell_dino/zscore_per_exp/paper_v1"
)

# Order: baseline (gray) → single-head attention → multi-head combos →
# softmax sweep → acc_select → ebi_then_geneko fallback → sister-coherence
# strategies (sister-only first, then attention×sister compounds).
# 4th tuple field: "baseline" | "attention" | "sister" (= which parent subdir).
STRATEGIES = [
    ("phase_only",                    "baseline (mean agg)",        "#888888",  "baseline"),
    ("geneko",                        "geneKO",                     "#ff7f0e",  "attention"),
    ("ebi",                           "EBI",                        "#1f77b4",  "attention"),
    ("max",                           "max(EBI,geneKO)",            "#2ca02c",  "attention"),
    ("min",                           "min(EBI,geneKO)",            "#9467bd",  "attention"),
    ("product",                       "EBI × geneKO",               "#8c564b",  "attention"),
    ("concordance_50",                "concordance top-50%",        "#e377c2",  "attention"),
    ("softmax_K100",                  "softmax K=100",              "#bcbd22",  "attention"),
    ("softmax_K1k",                   "softmax K=1k",               "#17becf",  "attention"),
    ("softmax_K10k",                  "softmax K=10k",              "#d62728",  "attention"),
    ("acc_select_geneko_raw",         "acc-sel geneKO raw",         "#7f7f7f",  "attention"),
    ("acc_select_geneko_weighted",    "acc-sel geneKO wtd",         "#1a9850",  "attention"),
    ("acc_select_chad_raw",           "acc-sel CHAD raw",           "#3690c0",  "attention"),
    ("acc_select_chad_weighted",      "acc-sel CHAD wtd",           "#d94801",  "attention"),
    ("ebi_then_geneko",               "EBI then geneKO",            "#5e3c99",  "attention"),
    # sister-coherence strategies (orange/red family to visually group them)
    ("sister",                        "sister",                     "#fdae6b",  "sister"),
    ("sister_pow2",                   "sister²",                    "#fd8d3c",  "sister"),
    ("sister_pow4",                   "sister⁴",                    "#f16913",  "sister"),
    ("sister_floored_01",             "max(sister,0.1)",            "#fee08b",  "sister"),
    ("sister_smoothed_01",            "sister+0.1",                 "#fdb863",  "sister"),
    ("sister_filter_gt0_nankeep",     "sister>0 (NaN-keep)",        "#fff5b8",  "sister"),
    ("sister_filter_gt0",             "sister>0",                   "#ffd92f",  "sister"),
    ("sister_filter_005",             "sister≥0.05",                "#fee391",  "sister"),
    ("sister_filter_010",             "sister≥0.10",                "#fec44f",  "sister"),
    ("sister_filter_015",             "sister≥0.15",                "#fe9929",  "sister"),
    ("sister_filter_020",             "sister≥0.20",                "#ec7014",  "sister"),
    ("sister_filter_03",              "sister≥0.3",                 "#a63603",  "sister"),
    ("sister_filter_05",              "sister≥0.5",                 "#7f2704",  "sister"),
    ("sister_filter_08",              "sister≥0.8",                 "#400d00",  "sister"),
    ("attn_ebi_x_sister",             "EBI × sister",               "#3182bd",  "sister"),
    ("attn_geneko_x_sister",          "geneKO × sister",            "#08519c",  "sister"),
    ("attn_ebi_plus_sister",          "EBI + sister",               "#9ecae1",  "sister"),
    # misalignment-based (continuous; green family to distinguish from sister-ratio family)
    ("misalign_w_a30",                "1/(1+ns/30)",                "#74c476",  "sister"),
    ("misalign_w_a100",               "1/(1+ns/100)",               "#a1d99b",  "sister"),
    ("misalign_exp_s30",              "exp(-ns/30)",                "#41ab5d",  "sister"),
    ("misalign_w_a30_x_ebi",          "(1/(1+ns/30))·EBI",          "#238b45",  "sister"),
    # region-homogeneity miscall-score (purple family)
    ("region_miscall_filter_03",      "miscall ≤ 0.3",              "#9e9ac8",  "sister"),
    ("region_miscall_filter_05",      "miscall ≤ 0.5",              "#bcbddc",  "sister"),
    ("region_miscall_w_a03",          "1/(1+ms/0.3)",               "#807dba",  "sister"),
    ("region_miscall_w_a10",          "1/(1+ms/1.0)",               "#6a51a3",  "sister"),
    ("attn_ebi_x_region_w_a03",       "(1/(1+ms/0.3))·EBI",         "#4a1486",  "sister"),
]

METRICS = [
    ("phenotypic_activity.csv",        "Activity"),
    ("phenotypic_distinctiveness.csv", "Distinctiveness"),
    ("phenotypic_consistency_ebi.csv", "EBI consistency"),
    ("phenotypic_consistency_manual.csv", "Manual-annotation consistency"),
]


def _metric_path(name: str, csv_name: str, parent: str) -> Path:
    if parent == "baseline":
        return ROOT / "phase_only/fixed_80%/cosine/metrics" / csv_name
    if parent == "attention":
        return ROOT / "attention" / name / "phase_only/fixed_80%/cosine/metrics" / csv_name
    if parent == "sister":
        return ROOT / "sister" / name / "phase_only/fixed_80%/cosine/metrics" / csv_name
    raise ValueError(f"unknown parent: {parent!r}")


def _load_values(name: str, csv_name: str, parent: str) -> np.ndarray:
    f = _metric_path(name, csv_name, parent)
    if not f.exists():
        return np.array([])
    df = pd.read_csv(f)
    return df["mean_average_precision"].dropna().to_numpy()


def main() -> None:
    # Filter to strategies whose metric CSVs exist (skip in-flight ones).
    available = []
    for name, label, color, parent in STRATEGIES:
        any_metric = any(_metric_path(name, m[0], parent).exists() for m in METRICS)
        if any_metric:
            available.append((name, label, color, parent))
        else:
            print(f"skipping {name}: metric CSVs not on disk yet")

    n_metr = len(METRICS)
    n_strat = len(available)
    # Layout: 2x2 for 4 metrics, otherwise single row. Width scales with n_strat
    # so individual violins stay readable when 20+ strategies are compared.
    if n_metr == 4:
        n_rows, n_cols = 2, 2
        fig, axes_grid = plt.subplots(n_rows, n_cols,
                                       figsize=(max(14, 0.55 * n_strat), 11),
                                       sharey=False)
        axes = axes_grid.flatten().tolist()
    else:
        n_rows, n_cols = 1, n_metr
        fig, axes = plt.subplots(1, n_metr,
                                  figsize=(max(11, 0.55 * n_strat * n_metr), 5.5),
                                  sharey=False)
        if n_metr == 1:
            axes = [axes]
        else:
            axes = list(axes)

    for j, (csv_name, mlabel) in enumerate(METRICS):
        ax = axes[j]
        data = [_load_values(name, csv_name, parent) for name, _, _, parent in available]
        # Keep only positions with data for this metric.
        keep = [k for k, v in enumerate(data) if len(v) > 0]
        positions = np.array([k + 1 for k in keep])
        kept_data = [data[k] for k in keep]

        parts = ax.violinplot(
            kept_data, positions=positions, showmeans=False, showmedians=True,
            showextrema=True, widths=0.75,
        )
        for ki, body in enumerate(parts["bodies"]):
            body.set_facecolor(available[keep[ki]][2])
            body.set_edgecolor("black")
            body.set_alpha(0.65)
        for key in ("cbars", "cmins", "cmaxes", "cmedians"):
            if key in parts:
                parts[key].set_color("black")
                parts[key].set_linewidth(0.8)

        for ki, vals in zip(keep, kept_data):
            m = float(vals.mean())
            ax.scatter([ki + 1], [m], marker="D", s=28, color="black", zorder=5)
            ax.text(ki + 1, m, f"  {m:.3f}", va="center", ha="left",
                    fontsize=7, color="black")

        # Red dashed baseline reference line
        base_vals = _load_values("phase_only", csv_name, "baseline")
        if base_vals.size:
            base_mean = float(base_vals.mean())
            ax.axhline(base_mean, color="red", linestyle="--", linewidth=1.2,
                       label=f"baseline = {base_mean:.3f}")

        ax.set_xticks(np.arange(n_strat) + 1)
        ax.set_xticklabels([s[1] for s in available], rotation=35, ha="right",
                           fontsize=7)
        ax.set_title(mlabel, fontsize=11)
        ax.set_ylabel("mAP" if j == 0 else "")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower right", fontsize=7, frameon=False)

    fig.suptitle(
        "Attention-weighted cell→guide aggregation vs mean baseline\n"
        "(v3 pca_optimization on Alex Lin's v4 PMA-trained cell-dino features)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    out_dir = ROOT / "attention" / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{n_metr}metric"
    for ext in ("png", "pdf", "svg"):
        out = out_dir / f"v4_attn_{suffix}_violin.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
