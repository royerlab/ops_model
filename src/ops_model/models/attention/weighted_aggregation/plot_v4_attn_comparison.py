"""Violin plot comparing per-perturbation/per-complex mAP distributions across
the v4 attention-weighted v3-pipeline runs vs the mean-aggregation baseline.

Looks under ``paper_v1/attention/<strategy>/phase_only/...`` for each strategy
and ``paper_v1/phase_only/...`` for the baseline. Skips strategies whose
metric CSVs aren't on disk yet, so this is safe to re-run as more strategies
finish.

Bruno paths
-----------
* Script: this file
* Baseline:    <root>/phase_only/fixed_80%/cosine/metrics/*.csv
* Strategies:  <root>/attention/<strategy>/phase_only/fixed_80%/cosine/metrics/*.csv
* Output:      <root>/attention/comparison/v4_attn_<n>metric_violin.{png,pdf,svg}

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

# Order: baseline (gray) → single-head → multi-head combos → softmax sweep →
# acc_select (variable cell-cutoff per gene/complex from v3 cdino accuracy).
STRATEGIES = [
    ("phase_only",                    "baseline (mean agg)",        "#888888",  True),
    ("geneko",                        "geneKO",                     "#ff7f0e",  False),
    ("ebi",                           "EBI",                        "#1f77b4",  False),
    ("max",                           "max(EBI,geneKO)",            "#2ca02c",  False),
    ("min",                           "min(EBI,geneKO)",            "#9467bd",  False),
    ("product",                       "EBI × geneKO",               "#8c564b",  False),
    ("concordance_50",                "concordance top-50%",        "#e377c2",  False),
    ("softmax_K100",                  "softmax K=100",              "#bcbd22",  False),
    ("softmax_K1k",                   "softmax K=1k",               "#17becf",  False),
    ("softmax_K10k",                  "softmax K=10k",              "#d62728",  False),
    ("acc_select_geneko_raw",         "acc-sel geneKO raw",         "#7f7f7f",  False),
    ("acc_select_geneko_weighted",    "acc-sel geneKO wtd",         "#1a9850",  False),
    ("acc_select_chad_raw",           "acc-sel CHAD raw",           "#3690c0",  False),
    ("acc_select_chad_weighted",      "acc-sel CHAD wtd",           "#d94801",  False),
    ("ebi_then_geneko",               "EBI then geneKO",            "#5e3c99",  False),
]

METRICS = [
    ("phenotypic_activity.csv",        "Activity"),
    ("phenotypic_distinctiveness.csv", "Distinctiveness"),
    ("phenotypic_consistency_ebi.csv", "EBI consistency"),
]


def _metric_path(name: str, csv_name: str, is_baseline: bool) -> Path:
    if is_baseline:
        return ROOT / "phase_only/fixed_80%/cosine/metrics" / csv_name
    return ROOT / "attention" / name / "phase_only/fixed_80%/cosine/metrics" / csv_name


def _load_values(name: str, csv_name: str, is_baseline: bool) -> np.ndarray:
    f = _metric_path(name, csv_name, is_baseline)
    if not f.exists():
        return np.array([])
    df = pd.read_csv(f)
    return df["mean_average_precision"].dropna().to_numpy()


def main() -> None:
    # Filter to strategies whose metric CSVs exist (skip in-flight ones).
    available = []
    for name, label, color, is_base in STRATEGIES:
        any_metric = any(_metric_path(name, m[0], is_base).exists() for m in METRICS)
        if any_metric:
            available.append((name, label, color, is_base))
        else:
            print(f"skipping {name}: metric CSVs not on disk yet")

    n_metr = len(METRICS)
    n_strat = len(available)
    fig, axes = plt.subplots(1, n_metr, figsize=(max(11, 1.1 * n_strat * n_metr), 5.5),
                             sharey=False)
    if n_metr == 1:
        axes = [axes]

    for j, (csv_name, mlabel) in enumerate(METRICS):
        ax = axes[j]
        data = [_load_values(name, csv_name, is_base) for name, _, _, is_base in available]
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
        base_vals = _load_values("phase_only", csv_name, True)
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
