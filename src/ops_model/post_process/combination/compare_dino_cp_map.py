"""Compare DINO vs CellProfiler mAP scores across all 4 phenotypic metrics.

Metrics compared:
  - Activity       (phenotypic_activity.csv)
  - Distinctiveness(phenotypic_distinctiveness.csv)
  - Consistency    (phenotypic_consistency_corum.csv)
  - CHAD           (phenotypic_consistency_manual.csv)

For each metric + mode (full / downsampled) produces:
  1. Scatterplot DINO vs CP with linear regression + discordant labels
  2. Per-gene delta bar chart (DINO - CP), sorted, top outliers labelled
  3. Summary violin grid: delta distributions for all 4 metrics side-by-side
     with Wilcoxon signed-rank test p-value to detect systematic bias

Usage
-----
  python -m ops_model.post_process.combination.compare_dino_cp_map \\
      -o /hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized

Expects outputs from pca_optimization.py at:
  <output_dir>/dino/[downsampled/]metrics/phenotypic_*.csv
  <output_dir>/cellprofiler/[downsampled/]metrics/phenotypic_*.csv

Output
------
  <output_dir>/comparison/
    {mode}_{metric}_scatter.png
    {mode}_{metric}_delta_bar.png
    {mode}_all_metrics_delta_violin.png
    {mode}_{metric}.csv
    comparison_summary.csv
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

DISCORDANCE_THRESHOLD = 0.15
TOP_N_LABELS = 12

METRICS = {
    # key: (csv_filename, display_label, id_column)
    "activity":        ("phenotypic_activity.csv",           "Activity",             "perturbation"),
    "distinctiveness": ("phenotypic_distinctiveness.csv",    "Distinctiveness",      "perturbation"),
    "corum":           ("phenotypic_consistency_corum.csv",  "Consistency (CORUM)",  "complex_id"),
    "chad":            ("phenotypic_consistency_manual.csv", "Consistency (CHAD)",   "complex_num"),
}

# Palette
_COL_SIG_BOTH   = "#2196F3"
_COL_SIG_EITHER = "#90CAF9"
_COL_NONSIG     = "#BBBBBB"
_COL_DISCORD    = "crimson"
_COL_DINO       = "#E65100"   # orange
_COL_CP         = "#1565C0"   # blue


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------

def _load_metric_csv(path: Path, id_col: str) -> Optional[pd.DataFrame]:
    """Load a phenotypic metric CSV, drop NTC rows (gene-level only), return relevant columns."""
    if not path.exists():
        logger.warning(f"  Not found: {path}")
        return None
    df = pd.read_csv(path)
    if id_col not in df.columns:
        logger.warning(f"  {path.name}: expected id column '{id_col}', got {list(df.columns)}")
        return None
    # Drop NTC only for gene/perturbation-level metrics
    if id_col == "perturbation":
        df = df[~df[id_col].str.contains("NTC|non-targeting", case=False, na=False)]
    cols = [c for c in [id_col, "mean_average_precision",
                         "corrected_p_value", "below_corrected_p"] if c in df.columns]
    return df[cols].rename(columns={id_col: "entity"}).copy()


def _merge_pair(dino_df: pd.DataFrame, cp_df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join on entity, add delta columns."""
    merged = dino_df.merge(cp_df, on="entity", suffixes=("_dino", "_cp"))
    merged["delta"] = merged["mean_average_precision_dino"] - merged["mean_average_precision_cp"]
    merged["delta_abs"] = merged["delta"].abs()
    if "below_corrected_p_dino" in merged.columns and "below_corrected_p_cp" in merged.columns:
        merged["sig_dino"]   = merged["below_corrected_p_dino"].astype(bool)
        merged["sig_cp"]     = merged["below_corrected_p_cp"].astype(bool)
        merged["sig_both"]   = merged["sig_dino"] & merged["sig_cp"]
        merged["sig_either"] = merged["sig_dino"] | merged["sig_cp"]
    else:
        for c in ["sig_dino", "sig_cp", "sig_both", "sig_either"]:
            merged[c] = False
    merged["discordant"] = merged["delta_abs"] > DISCORDANCE_THRESHOLD
    return merged


# ----------------------------------------------------------------------------
# Plot 1: Scatter with regression
# ----------------------------------------------------------------------------

def _plot_scatter(ax: plt.Axes, merged: pd.DataFrame, title: str) -> dict:
    x = merged["mean_average_precision_dino"].values
    y = merged["mean_average_precision_cp"].values

    colours = np.where(
        merged["discordant"], _COL_DISCORD,
        np.where(merged["sig_both"], _COL_SIG_BOTH,
        np.where(merged["sig_either"], _COL_SIG_EITHER, _COL_NONSIG)),
    )
    ax.scatter(x, y, c=colours, s=28, alpha=0.75, linewidths=0)

    slope, intercept, r, p_val, _ = stats.linregress(x, y)
    x_line = np.array([x.min(), x.max()])
    ax.plot(x_line, slope * x_line + intercept, "k--", lw=1.4, zorder=5)

    lim = (min(x.min(), y.min()) - 0.02, max(x.max(), y.max()) + 0.02)
    ax.plot(lim, lim, color="grey", lw=0.8, ls=":", alpha=0.5)
    ax.set_xlim(*lim); ax.set_ylim(*lim)

    # Label top discordant
    for _, row in merged[merged["discordant"]].nlargest(TOP_N_LABELS, "delta_abs").iterrows():
        ax.annotate(row["entity"],
                    (row["mean_average_precision_dino"], row["mean_average_precision_cp"]),
                    fontsize=5.5, xytext=(3, 2), textcoords="offset points",
                    color=_COL_DISCORD, clip_on=True)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], marker="o", color="w", markerfacecolor=_COL_SIG_BOTH,   ms=7, label="sig both"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=_COL_SIG_EITHER, ms=7, label="sig either"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=_COL_NONSIG,     ms=7, label="not sig"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=_COL_DISCORD,    ms=7,
               label=f"discordant (Δ>{DISCORDANCE_THRESHOLD})"),
    ], fontsize=7, loc="upper left")

    n, n_disc = len(merged), int(merged["discordant"].sum())
    ax.text(0.98, 0.02, f"n={n}  discordant={n_disc}  r={r:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="dimgrey")
    ax.set_xlabel("mAP — DINO", fontsize=10)
    ax.set_ylabel("mAP — CellProfiler", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")

    return {"r": r, "slope": slope, "p_regression": p_val,
            "n": n, "n_discordant": n_disc}


# ----------------------------------------------------------------------------
# Plot 2: Per-gene delta bar chart
# ----------------------------------------------------------------------------

def _plot_delta_bar(ax: plt.Axes, merged: pd.DataFrame, title: str) -> None:
    """Sorted bar chart of per-gene DINO − CP delta, top outliers labelled."""
    df = merged[["entity", "delta"]].sort_values("delta", ascending=False).reset_index(drop=True)
    colours = [_COL_DINO if d > 0 else _COL_CP for d in df["delta"]]

    ax.bar(range(len(df)), df["delta"], color=colours, width=1.0, linewidth=0, alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)

    # Label extremes
    n_label = min(TOP_N_LABELS // 2, len(df) // 4)
    for idx in list(range(n_label)) + list(range(len(df) - n_label, len(df))):
        row = df.iloc[idx]
        va = "bottom" if row["delta"] >= 0 else "top"
        offset = 0.005 if row["delta"] >= 0 else -0.005
        ax.text(idx, row["delta"] + offset, row["entity"],
                fontsize=5, ha="center", va=va, rotation=90, clip_on=True)

    # Mean delta line + annotation
    mean_d = float(df["delta"].mean())
    ax.axhline(mean_d, color="black", lw=1.2, ls="--", alpha=0.6)
    ax.text(len(df) * 0.98, mean_d, f"mean Δ={mean_d:+.3f}",
            ha="right", va="bottom" if mean_d >= 0 else "top",
            fontsize=8, color="black", alpha=0.7)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=_COL_DINO, label="DINO higher"),
        Patch(facecolor=_COL_CP,   label="CP higher"),
    ], fontsize=8, loc="lower right")

    ax.set_xlabel("Genes (sorted by Δ)", fontsize=10)
    ax.set_ylabel("DINO − CP mAP", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(-1, len(df))
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)


# ----------------------------------------------------------------------------
# Plot 3: Delta violin summary (all 4 metrics together)
# ----------------------------------------------------------------------------

def _plot_delta_violin(
    ax: plt.Axes,
    metric_deltas: Dict[str, np.ndarray],
    title: str,
) -> None:
    """Violin + strip of DINO−CP delta for each metric. Wilcoxon p-value annotated."""
    labels = []
    data = []
    pvals = []
    for key, (_, label, _id) in METRICS.items():
        if key not in metric_deltas:
            continue
        d = metric_deltas[key]
        labels.append(label)
        data.append(d)
        if len(d) >= 5:
            _, p = stats.wilcoxon(d, alternative="two-sided")
        else:
            p = float("nan")
        pvals.append(p)

    if not data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    positions = np.arange(1, len(data) + 1)

    vp = ax.violinplot(data, positions=positions, showmedians=True,
                        showextrema=True, widths=0.6)
    for body in vp["bodies"]:
        body.set_facecolor("#90CAF9")
        body.set_alpha(0.6)
    vp["cmedians"].set_color("black")
    vp["cmedians"].set_linewidth(1.8)

    # Strip of individual points
    rng = np.random.RandomState(42)
    for i, d in enumerate(data):
        jitter = rng.uniform(-0.12, 0.12, size=len(d))
        colours = [_COL_DINO if v > 0 else _COL_CP for v in d]
        ax.scatter(positions[i] + jitter, d, c=colours, s=12, alpha=0.5, linewidths=0, zorder=3)

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)

    # Wilcoxon p-value annotations
    y_max = max(np.max(np.abs(d)) for d in data) * 1.15
    for i, (p, d) in enumerate(zip(pvals, data)):
        mean_d = float(np.mean(d))
        p_str = (f"p={p:.2e}" if not np.isnan(p) else "n/a")
        sig_star = ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns") if not np.isnan(p) else ""
        ax.text(positions[i], y_max * 0.97,
                f"{sig_star}\n{p_str}\nmean{mean_d:+.3f}",
                ha="center", va="top", fontsize=7, color="dimgrey")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("DINO − CP mAP", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=_COL_DINO, label="DINO higher"),
        Patch(facecolor=_COL_CP,   label="CP higher"),
    ], fontsize=8)


# ----------------------------------------------------------------------------
# Core comparison for one mode (full or downsampled)
# ----------------------------------------------------------------------------

def _compare_mode(
    mode: str,
    dino_metrics_dir: Path,
    cp_metrics_dir: Path,
    comp_dir: Path,
) -> list:
    """Run all comparisons for one mode. Returns summary rows."""
    summary_rows = []
    metric_deltas: Dict[str, np.ndarray] = {}

    for metric_key, (csv_name, metric_label, id_col) in METRICS.items():
        dino_path = dino_metrics_dir / csv_name
        cp_path   = cp_metrics_dir   / csv_name

        dino_df = _load_metric_csv(dino_path, id_col)
        cp_df   = _load_metric_csv(cp_path, id_col)
        if dino_df is None or cp_df is None:
            logger.info(f"  Skipping {metric_label} ({mode}): one or both files missing")
            continue

        merged = _merge_pair(dino_df, cp_df)
        if len(merged) < 3:
            logger.info(f"  Skipping {metric_label} ({mode}): too few common genes ({len(merged)})")
            continue

        logger.info(f"  {metric_label} ({mode}): {len(merged)} genes, "
                    f"mean Δ={merged['delta'].mean():+.3f}, "
                    f"{int(merged['discordant'].sum())} discordant")

        metric_deltas[metric_key] = merged["delta"].values
        merged.to_csv(comp_dir / f"{mode}_{metric_key}.csv", index=False)

        # --- Scatter plot ---
        fig, ax = plt.subplots(figsize=(7, 6.5))
        stats_row = _plot_scatter(ax, merged, f"DINO vs CP — {metric_label} ({mode})")
        fig.tight_layout()
        fig.savefig(comp_dir / f"{mode}_{metric_key}_scatter.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        # --- Delta bar chart ---
        fig, ax = plt.subplots(figsize=(max(8, len(merged) * 0.12), 5))
        _plot_delta_bar(ax, merged, f"Per-gene DINO − CP: {metric_label} ({mode})")
        fig.tight_layout()
        fig.savefig(comp_dir / f"{mode}_{metric_key}_delta_bar.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"    Saved {mode}_{metric_key}_scatter.png + delta_bar.png")

        stats_row.update({"mode": mode, "metric": metric_key,
                           "mean_delta": float(merged["delta"].mean()),
                           "median_delta": float(merged["delta"].median())})
        summary_rows.append(stats_row)

    # --- Summary violin (all metrics together) ---
    if metric_deltas:
        fig, ax = plt.subplots(figsize=(max(7, len(metric_deltas) * 2.2), 5.5))
        _plot_delta_violin(ax, metric_deltas,
                           f"DINO − CP mAP delta across all metrics ({mode})")
        fig.tight_layout()
        fig.savefig(comp_dir / f"{mode}_all_metrics_delta_violin.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved {mode}_all_metrics_delta_violin.png")

    return summary_rows


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def compare_maps(output_dir: str) -> None:
    output_dir = Path(output_dir)
    comp_dir = output_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    modes = {
        "full": (
            output_dir / "dino"        / "metrics",
            output_dir / "cellprofiler"/ "metrics",
        ),
        "downsampled": (
            output_dir / "dino"        / "downsampled" / "metrics",
            output_dir / "cellprofiler"/ "downsampled" / "metrics",
        ),
    }

    all_summary = []
    ran_any = False

    for mode, (dino_dir, cp_dir) in modes.items():
        # Skip mode if neither metrics directory exists
        if not dino_dir.exists() and not cp_dir.exists():
            continue
        logger.info(f"\n=== Mode: {mode} ===")
        rows = _compare_mode(mode, dino_dir, cp_dir, comp_dir)
        all_summary.extend(rows)
        ran_any = True

    if not ran_any:
        logger.error(
            "No metrics directories found. Run pca_optimization.py for both "
            "DINO and CellProfiler features first."
        )
        return

    if all_summary:
        summary_df = pd.DataFrame(all_summary)
        summary_df.to_csv(comp_dir / "comparison_summary.csv", index=False)
        logger.info(f"\nSummary saved to comparison/comparison_summary.csv")
        logger.info(f"\n{'mode':<12} {'metric':<16} {'n':>5} {'mean_Δ':>8} {'r':>7} {'n_disc':>7}")
        logger.info("-" * 60)
        for _, r in summary_df.iterrows():
            logger.info(f"{r['mode']:<12} {r['metric']:<16} {r['n']:>5} "
                        f"{r['mean_delta']:>+8.3f} {r['r']:>7.3f} {r['n_discordant']:>7}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare DINO vs CellProfiler across all 4 phenotypic mAP metrics"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str,
        default="/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized",
        help="Root output directory (same as used for pca_optimization.py)",
    )
    args = parser.parse_args()
    compare_maps(args.output_dir)


if __name__ == "__main__":
    main()
