"""Compare DINO vs CellProfiler mAP scores across all 4 phenotypic metrics.

Metrics compared:
  - Activity            (phenotypic_activity.csv)
  - Distinctiveness     (phenotypic_distinctiveness.csv)
  - Consistency (CORUM) (phenotypic_consistency_corum.csv)
  - Consistency (CHAD)  (phenotypic_consistency_manual.csv)

Produces two figures per mode (full / downsampled):
  1. {mode}_panel.png        — 2×4 grid: row 1 = scatter + regression,
                               row 2 = slopegraph (paired DINO→CP lines per gene/complex)
  2. {mode}_delta_violin.png — delta distributions for all 4 metrics side-by-side
                               with Wilcoxon p-values

Usage
-----
  python -m ops_model.post_process.combination.compare_dino_cp_map \\
      -o /hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized

Expects:
  <output_dir>/dino/[downsampled/]metrics/phenotypic_*.csv
  <output_dir>/cellprofiler/[downsampled/]metrics/phenotypic_*.csv

Output → <output_dir>/comparison/
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
TOP_N_LABELS = 10
# Max non-discordant lines shown in slopegraph (subsampled if exceeded)
SLOPE_MAX_BG = 120

METRICS: Dict[str, Tuple[str, str, str]] = {
    # key: (csv_filename, display_label, id_column)
    "activity":        ("phenotypic_activity.csv",           "Activity",             "perturbation"),
    "distinctiveness": ("phenotypic_distinctiveness.csv",    "Distinctiveness",      "perturbation"),
    "corum":           ("phenotypic_consistency_corum.csv",  "Consistency\n(CORUM)", "complex_id"),
    "chad":            ("phenotypic_consistency_manual.csv", "Consistency\n(CHAD)",  "complex_num"),
}

_COL_DINO       = "#E65100"
_COL_CP         = "#1565C0"
_COL_SIG_BOTH   = "#2196F3"
_COL_SIG_EITHER = "#90CAF9"
_COL_NONSIG     = "#CCCCCC"
_COL_DISCORD    = "#D32F2F"


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------

def _load_metric_csv(path: Path, id_col: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if id_col not in df.columns:
        logger.warning(f"  {path.name}: id column '{id_col}' not found — cols: {list(df.columns)}")
        return None
    if id_col == "perturbation":
        df = df[~df[id_col].str.contains("NTC|non-targeting", case=False, na=False)]
    keep = [c for c in [id_col, "mean_average_precision",
                         "corrected_p_value", "below_corrected_p"] if c in df.columns]
    return df[keep].rename(columns={id_col: "entity"}).copy()


def _merge_pair(dino_df: pd.DataFrame, cp_df: pd.DataFrame) -> pd.DataFrame:
    merged = dino_df.merge(cp_df, on="entity", suffixes=("_dino", "_cp"))
    merged["delta"]     = merged["mean_average_precision_dino"] - merged["mean_average_precision_cp"]
    merged["delta_abs"] = merged["delta"].abs()
    for col in ["sig_dino", "sig_cp", "sig_both", "sig_either"]:
        merged[col] = False
    if "below_corrected_p_dino" in merged.columns:
        merged["sig_dino"]   = merged["below_corrected_p_dino"].astype(bool)
        merged["sig_cp"]     = merged["below_corrected_p_cp"].astype(bool)
        merged["sig_both"]   = merged["sig_dino"] & merged["sig_cp"]
        merged["sig_either"] = merged["sig_dino"] | merged["sig_cp"]
    merged["discordant"] = merged["delta_abs"] > DISCORDANCE_THRESHOLD
    return merged


# ----------------------------------------------------------------------------
# Axis: scatter + regression
# ----------------------------------------------------------------------------

def _ax_scatter(ax: plt.Axes, merged: pd.DataFrame, title: str) -> dict:
    x = merged["mean_average_precision_dino"].values
    y = merged["mean_average_precision_cp"].values

    c = np.where(merged["discordant"], _COL_DISCORD,
        np.where(merged["sig_both"], _COL_SIG_BOTH,
        np.where(merged["sig_either"], _COL_SIG_EITHER, _COL_NONSIG)))
    ax.scatter(x, y, c=c, s=22, alpha=0.8, linewidths=0, zorder=3)

    slope, intercept, r, p_val, _ = stats.linregress(x, y)
    xl = np.array([x.min(), x.max()])
    ax.plot(xl, slope * xl + intercept, "k--", lw=1.2, zorder=4,
            label=f"r={r:.3f}  slope={slope:.2f}")

    lim = (min(x.min(), y.min()) - 0.02, max(x.max(), y.max()) + 0.02)
    ax.plot(lim, lim, color="grey", lw=0.7, ls=":", alpha=0.4)
    ax.set_xlim(*lim); ax.set_ylim(*lim)

    for _, row in merged[merged["discordant"]].nlargest(TOP_N_LABELS, "delta_abs").iterrows():
        ax.annotate(row["entity"],
                    (row["mean_average_precision_dino"], row["mean_average_precision_cp"]),
                    fontsize=4.5, xytext=(3, 2), textcoords="offset points",
                    color=_COL_DISCORD, clip_on=True)

    n, nd = len(merged), int(merged["discordant"].sum())
    ax.text(0.97, 0.03, f"n={n}  disc={nd}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color="dimgrey")
    ax.set_xlabel("DINO mAP", fontsize=8)
    ax.set_ylabel("CP mAP", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    ax.legend(fontsize=6.5, loc="upper left", framealpha=0.7)
    ax.tick_params(labelsize=7)

    return {"r": r, "slope": slope, "p_regression": p_val,
            "n": n, "n_discordant": nd,
            "mean_delta": float(merged["delta"].mean()),
            "median_delta": float(merged["delta"].median())}


# ----------------------------------------------------------------------------
# Axis: slopegraph
# ----------------------------------------------------------------------------

def _ax_slope(ax: plt.Axes, merged: pd.DataFrame, title: str) -> None:
    """Paired slopegraph: two columns (DINO | CP), lines per gene coloured by direction."""
    x0, x1 = 0.0, 1.0
    rng = np.random.RandomState(42)

    disc_mask  = merged["discordant"].values
    bg_idx     = np.where(~disc_mask)[0]
    fore_idx   = np.where(disc_mask)[0]

    # Subsample background lines if too many
    if len(bg_idx) > SLOPE_MAX_BG:
        bg_idx = rng.choice(bg_idx, SLOPE_MAX_BG, replace=False)

    dino_vals = merged["mean_average_precision_dino"].values
    cp_vals   = merged["mean_average_precision_cp"].values

    # Background (non-discordant) — thin, transparent
    for i in bg_idx:
        colour = _COL_DINO if merged["delta"].iloc[i] > 0 else _COL_CP
        ax.plot([x0, x1], [dino_vals[i], cp_vals[i]],
                color=colour, alpha=0.18, lw=0.7, solid_capstyle="round")

    # Foreground (discordant) — thicker, opaque, labelled
    top_disc = merged[disc_mask].nlargest(TOP_N_LABELS, "delta_abs")
    for _, row in top_disc.iterrows():
        colour = _COL_DINO if row["delta"] > 0 else _COL_CP
        ax.plot([x0, x1], [row["mean_average_precision_dino"], row["mean_average_precision_cp"]],
                color=colour, alpha=0.85, lw=1.4, solid_capstyle="round", zorder=4)
        # Label on the side where the value is higher
        if row["delta"] > 0:
            ax.text(x0 - 0.03, row["mean_average_precision_dino"], row["entity"],
                    ha="right", va="center", fontsize=4.5, color=_COL_DINO, clip_on=True)
        else:
            ax.text(x1 + 0.03, row["mean_average_precision_cp"], row["entity"],
                    ha="left", va="center", fontsize=4.5, color=_COL_CP, clip_on=True)

    # Median lines
    ax.plot([x0, x1],
            [np.median(dino_vals), np.median(cp_vals)],
            color="black", lw=2.0, ls="--", alpha=0.6, zorder=5,
            label=f"median  DINO={np.median(dino_vals):.3f}  CP={np.median(cp_vals):.3f}")

    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([x0, x1])
    ax.set_xticklabels(["DINO", "CellProfiler"], fontsize=9, fontweight="bold")
    ax.set_ylabel("mAP", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=6, loc="lower right", framealpha=0.7)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], color=_COL_DINO, lw=1.5, label="DINO higher"),
        Line2D([0],[0], color=_COL_CP,   lw=1.5, label="CP higher"),
        Line2D([0],[0], color="black",   lw=1.5, ls="--", label="median"),
    ], fontsize=6, loc="lower right", framealpha=0.7)


# ----------------------------------------------------------------------------
# Axis: delta violin (summary panel)
# ----------------------------------------------------------------------------

def _ax_violin(ax: plt.Axes, metric_deltas: Dict[str, np.ndarray], title: str) -> None:
    labels, data, pvals = [], [], []
    for key, (_, label, _id) in METRICS.items():
        if key not in metric_deltas:
            continue
        d = metric_deltas[key]
        labels.append(label)
        data.append(d)
        pvals.append(stats.wilcoxon(d)[1] if len(d) >= 5 else float("nan"))

    if not data:
        return

    positions = np.arange(1, len(data) + 1)
    vp = ax.violinplot(data, positions=positions, showmedians=True,
                        showextrema=True, widths=0.55)
    for body in vp["bodies"]:
        body.set_facecolor("#B0C4DE")
        body.set_alpha(0.55)
    vp["cmedians"].set_color("black")
    vp["cmedians"].set_linewidth(2.0)

    rng = np.random.RandomState(42)
    for i, d in enumerate(data):
        jitter = rng.uniform(-0.1, 0.1, size=len(d))
        ax.scatter(positions[i] + jitter, d,
                   c=[_COL_DINO if v > 0 else _COL_CP for v in d],
                   s=10, alpha=0.45, linewidths=0, zorder=3)

    ax.axhline(0, color="black", lw=0.9, ls="--", alpha=0.45)

    y_top = max(np.max(np.abs(d)) for d in data) * 1.18
    for i, (p, d) in enumerate(zip(pvals, data)):
        sig = ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns") \
              if not np.isnan(p) else "n/a"
        p_str = f"p={p:.1e}" if not np.isnan(p) else ""
        ax.text(positions[i], y_top,
                f"{sig}\n{p_str}\nμΔ={np.mean(d):+.3f}",
                ha="center", va="top", fontsize=6.5, color="dimgrey")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("DINO − CP mAP", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=_COL_DINO, label="DINO higher"),
                       Patch(facecolor=_COL_CP,   label="CP higher")],
              fontsize=8, loc="lower right")


# ----------------------------------------------------------------------------
# One mode
# ----------------------------------------------------------------------------

def _compare_mode(
    mode: str,
    dino_dir: Path,
    cp_dir: Path,
    comp_dir: Path,
) -> List[dict]:
    available = {}
    for key, (csv_name, label, id_col) in METRICS.items():
        dino_df = _load_metric_csv(dino_dir / csv_name, id_col)
        cp_df   = _load_metric_csv(cp_dir   / csv_name, id_col)
        if dino_df is not None and cp_df is not None:
            merged = _merge_pair(dino_df, cp_df)
            if len(merged) >= 3:
                available[key] = (merged, label)
                merged.to_csv(comp_dir / f"{mode}_{key}.csv", index=False)
                logger.info(f"  {label:25s}: {len(merged):3d} entities, "
                            f"mean Δ={merged['delta'].mean():+.3f}, "
                            f"{int(merged['discordant'].sum())} discordant")

    if not available:
        logger.warning(f"  No metrics available for mode '{mode}'")
        return []

    n_metrics = len(available)

    # ----------------------------------------------------------------
    # Figure 1: 2-row × n_metrics panel (scatter | slopegraph)
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(2, n_metrics,
                              figsize=(4.5 * n_metrics, 9),
                              gridspec_kw={"hspace": 0.45, "wspace": 0.35})
    if n_metrics == 1:
        axes = axes.reshape(2, 1)

    summary_rows = []
    for col, (key, (merged, label)) in enumerate(available.items()):
        short_label = label.replace("\n", " ")
        stats_row = _ax_scatter(axes[0, col], merged,
                                 f"{short_label}")
        _ax_slope(axes[1, col], merged, f"{short_label}")
        stats_row.update({"mode": mode, "metric": key})
        summary_rows.append(stats_row)

    fig.suptitle(f"DINO vs CellProfiler — {mode}", fontsize=13, fontweight="bold", y=1.01)
    fig.savefig(comp_dir / f"{mode}_panel.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {mode}_panel.png")

    # ----------------------------------------------------------------
    # Figure 2: delta violin summary
    # ----------------------------------------------------------------
    metric_deltas = {k: m["delta"].values for k, (m, _) in available.items()}
    fig2, ax2 = plt.subplots(figsize=(max(6, n_metrics * 2.5), 5))
    _ax_violin(ax2, metric_deltas,
               f"DINO − CP mAP delta — {mode} (Wilcoxon signed-rank)")
    fig2.tight_layout()
    fig2.savefig(comp_dir / f"{mode}_delta_violin.png", dpi=180, bbox_inches="tight")
    plt.close(fig2)
    logger.info(f"  Saved {mode}_delta_violin.png")

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
            output_dir / "dino"         / "metrics",
            output_dir / "cellprofiler" / "metrics",
        ),
        "downsampled": (
            output_dir / "dino"         / "downsampled" / "metrics",
            output_dir / "cellprofiler" / "downsampled" / "metrics",
        ),
    }

    all_summary = []
    ran_any = False
    for mode, (dino_dir, cp_dir) in modes.items():
        if not dino_dir.exists() and not cp_dir.exists():
            continue
        logger.info(f"\n=== {mode} ===")
        all_summary.extend(_compare_mode(mode, dino_dir, cp_dir, comp_dir))
        ran_any = True

    if not ran_any:
        logger.error("No metrics directories found. Run pca_optimization.py for both "
                     "DINO and CellProfiler features first.")
        return

    if all_summary:
        df = pd.DataFrame(all_summary)
        df.to_csv(comp_dir / "comparison_summary.csv", index=False)
        logger.info(f"\n{'mode':<12} {'metric':<16} {'n':>5} {'mean_Δ':>8} {'r':>7} {'disc':>6}")
        logger.info("-" * 58)
        for _, r in df.iterrows():
            logger.info(f"{r['mode']:<12} {r['metric']:<16} {r['n']:>5} "
                        f"{r['mean_delta']:>+8.3f} {r['r']:>7.3f} {r['n_discordant']:>6}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare DINO vs CellProfiler across all phenotypic mAP metrics"
    )
    parser.add_argument("-o", "--output-dir", type=str,
                        default="/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized")
    args = parser.parse_args()
    compare_maps(args.output_dir)


if __name__ == "__main__":
    main()
