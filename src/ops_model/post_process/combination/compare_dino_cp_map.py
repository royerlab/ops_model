"""Compare DINO vs CellProfiler mAP scores across full and downsampled outputs.

Loads phenotypic_activity.csv from both feature types, merges on perturbation,
and produces scatterplots with linear regression. Gene KOs with large mAP
discordance are highlighted.

Usage
-----
  python -m ops_model.post_process.combination.compare_dino_cp_map \\
      -o /hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized

Expects outputs from pca_optimization.py at:
  <output_dir>/dino/[downsampled/]metrics/phenotypic_activity.csv
  <output_dir>/cellprofiler/[downsampled/]metrics/phenotypic_activity.csv

Output
------
  <output_dir>/comparison/
    dino_vs_cp_full.png
    dino_vs_cp_downsampled.png
    dino_vs_cp_comparison.csv
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Genes flagged as discordant if |DINO_mAP - CP_mAP| > this threshold
DISCORDANCE_THRESHOLD = 0.15
# Top N most discordant genes to label
TOP_N_LABELS = 12


def _load_activity_csv(path: Path) -> pd.DataFrame | None:
    """Load phenotypic_activity.csv, returning None if missing."""
    if not path.exists():
        logger.warning(f"  Not found: {path}")
        return None
    df = pd.read_csv(path)
    # Drop NTC rows
    df = df[~df["perturbation"].str.contains("NTC|non-targeting", case=False, na=False)]
    return df[["perturbation", "mean_average_precision", "corrected_p_value", "below_corrected_p"]].copy()


def _merge_dino_cp(dino_df: pd.DataFrame, cp_df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join on perturbation, suffix columns by feature type."""
    merged = dino_df.merge(cp_df, on="perturbation", suffixes=("_dino", "_cp"))
    merged["map_delta"] = merged["mean_average_precision_dino"] - merged["mean_average_precision_cp"]
    merged["map_delta_abs"] = merged["map_delta"].abs()
    merged["sig_dino"] = merged["below_corrected_p_dino"]
    merged["sig_cp"] = merged["below_corrected_p_cp"]
    merged["sig_both"] = merged["sig_dino"] & merged["sig_cp"]
    merged["sig_either"] = merged["sig_dino"] | merged["sig_cp"]
    merged["discordant"] = merged["map_delta_abs"] > DISCORDANCE_THRESHOLD
    return merged


def _scatter_with_regression(
    ax: plt.Axes,
    merged: pd.DataFrame,
    title: str,
) -> dict:
    """Draw mAP scatterplot with regression line + discordant highlights."""
    x = merged["mean_average_precision_dino"].values
    y = merged["mean_average_precision_cp"].values

    # Colour coding
    colours = np.where(
        merged["discordant"],
        "crimson",
        np.where(merged["sig_both"], "#2196F3", np.where(merged["sig_either"], "#90CAF9", "#BBBBBB")),
    )

    ax.scatter(x, y, c=colours, s=30, alpha=0.7, linewidths=0)

    # Linear regression
    slope, intercept, r, p_val, se = stats.linregress(x, y)
    x_line = np.array([x.min(), x.max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, "k--", lw=1.5, label=f"r={r:.3f}  slope={slope:.2f}")

    # Diagonal reference
    lim_lo = min(x.min(), y.min()) - 0.02
    lim_hi = max(x.max(), y.max()) + 0.02
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="grey", lw=0.8, ls=":", alpha=0.6)
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)

    # Label top discordant genes
    discordant = merged[merged["discordant"]].nlargest(TOP_N_LABELS, "map_delta_abs")
    for _, row in discordant.iterrows():
        ax.annotate(
            row["perturbation"],
            (row["mean_average_precision_dino"], row["mean_average_precision_cp"]),
            fontsize=6, xytext=(4, 2), textcoords="offset points",
            color="crimson", clip_on=True,
        )

    ax.set_xlabel("mAP — DINO", fontsize=11)
    ax.set_ylabel("mAP — CellProfiler", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    # Colour legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3", markersize=7, label="sig both"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#90CAF9", markersize=7, label="sig either"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#BBBBBB", markersize=7, label="not sig"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="crimson", markersize=7,
               label=f"discordant (Δ>{DISCORDANCE_THRESHOLD})"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left")

    n = len(merged)
    n_disc = int(merged["discordant"].sum())
    ax.text(
        0.98, 0.02, f"n={n}  discordant={n_disc}  r={r:.3f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="dimgrey",
    )

    return {"r": r, "slope": slope, "p": p_val, "n": n, "n_discordant": n_disc}


def compare_maps(output_dir: str) -> None:
    output_dir = Path(output_dir)
    comp_dir = output_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Paths for each mode (dino/ and cellprofiler/ subdirs as written by pca_optimization.py)
    modes = {
        "full": {
            "dino": output_dir / "dino" / "metrics" / "phenotypic_activity.csv",
            "cp":   output_dir / "cellprofiler" / "metrics" / "phenotypic_activity.csv",
            "out":  comp_dir / "dino_vs_cp_full.png",
        },
        "downsampled": {
            "dino": output_dir / "dino" / "downsampled" / "metrics" / "phenotypic_activity.csv",
            "cp":   output_dir / "cellprofiler" / "downsampled" / "metrics" / "phenotypic_activity.csv",
            "out":  comp_dir / "dino_vs_cp_downsampled.png",
        },
    }

    all_rows = []
    available_modes = {k: v for k, v in modes.items()
                       if v["dino"].exists() and v["cp"].exists()}

    if not available_modes:
        logger.error("No matching DINO + CellProfiler activity CSVs found. "
                     "Run pca_optimization.py for both feature types first.")
        return

    n_modes = len(available_modes)
    fig, axes = plt.subplots(1, n_modes, figsize=(7 * n_modes, 6.5), squeeze=False)

    for col, (mode, paths) in enumerate(available_modes.items()):
        logger.info(f"Loading {mode} outputs...")
        dino_df = _load_activity_csv(paths["dino"])
        cp_df   = _load_activity_csv(paths["cp"])
        if dino_df is None or cp_df is None:
            continue

        merged = _merge_dino_cp(dino_df, cp_df)
        logger.info(f"  {mode}: {len(merged)} common perturbations, "
                    f"{int(merged['discordant'].sum())} discordant")

        stats_row = _scatter_with_regression(
            axes[0, col], merged,
            title=f"DINO vs CellProfiler mAP — {mode}",
        )
        stats_row["mode"] = mode
        all_rows.append(stats_row)

        # Per-mode individual plot (full resolution)
        fig_single, ax_single = plt.subplots(figsize=(8, 7))
        _scatter_with_regression(ax_single, merged, title=f"DINO vs CellProfiler mAP — {mode}")
        fig_single.tight_layout()
        fig_single.savefig(paths["out"], dpi=150, bbox_inches="tight")
        plt.close(fig_single)
        logger.info(f"  Saved {paths['out'].name}")

        # Save per-mode CSV
        mode_csv = comp_dir / f"dino_vs_cp_{mode}.csv"
        merged.to_csv(mode_csv, index=False)

        # Print top discordant
        disc = merged[merged["discordant"]].sort_values("map_delta_abs", ascending=False)
        if len(disc):
            logger.info(f"  Top discordant genes ({mode}):")
            for _, row in disc.head(10).iterrows():
                logger.info(
                    f"    {row['perturbation']:<20}  DINO={row['mean_average_precision_dino']:.3f}  "
                    f"CP={row['mean_average_precision_cp']:.3f}  Δ={row['map_delta']:+.3f}"
                )

    # Combined figure
    fig.suptitle("DINO vs CellProfiler mAP Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    combined_out = comp_dir / "dino_vs_cp_combined.png"
    fig.savefig(combined_out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {combined_out}")

    # Summary stats CSV
    if all_rows:
        pd.DataFrame(all_rows).to_csv(comp_dir / "comparison_summary.csv", index=False)
        logger.info(f"Saved comparison_summary.csv")
        for row in all_rows:
            logger.info(f"  {row['mode']}: r={row['r']:.3f}, slope={row['slope']:.2f}, "
                        f"n={row['n']}, discordant={row['n_discordant']}")


def main():
    parser = argparse.ArgumentParser(description="Compare DINO vs CellProfiler mAP scores")
    parser.add_argument(
        "-o", "--output-dir", type=str,
        default="/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized",
        help="Root output directory (same as used for pca_optimization.py)",
    )
    args = parser.parse_args()
    compare_maps(args.output_dir)


if __name__ == "__main__":
    main()
