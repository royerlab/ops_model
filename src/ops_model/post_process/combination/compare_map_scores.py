"""Compare phenotypic mAP scores between two feature sets or channel subsets.

Supports two comparison modes:
  --comparison dino-vs-cp        DINO embeddings vs CellProfiler features
  --comparison phase-vs-nophase  Phase-only embedding vs no-Phase embedding
                                  (run separately per feature type via --feature-type)

Metrics compared:
  - Activity            (phenotypic_activity.csv)
  - Distinctiveness     (phenotypic_distinctiveness.csv)
  - Consistency (CORUM) (phenotypic_consistency_corum.csv)
  - Consistency (CHAD)  (phenotypic_consistency_manual.csv)

Produces two figures per mode (full / downsampled where applicable):
  1. {tag}_panel.png        — 2×4 grid: scatter + regression / slopegraph
  2. {tag}_delta_violin.png — delta distributions (Wilcoxon p-values)

Usage
-----
  # DINO vs CellProfiler
  python -m ops_model.post_process.combination.compare_map_scores \\
      -o /hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized

  # Phase-only vs No-phase (DINO features)
  python -m ops_model.post_process.combination.compare_map_scores \\
      -o /hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized \\
      --comparison phase-vs-nophase --feature-type dino

  # Phase-only vs No-phase (CellProfiler features)
  python -m ops_model.post_process.combination.compare_map_scores \\
      -o /hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized \\
      --comparison phase-vs-nophase --feature-type cellprofiler

Output → <output_dir>/comparison/{comparison_name}/
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
SLOPE_MAX_BG = 120

METRICS: Dict[str, Tuple[str, str, str]] = {
    # key: (csv_filename, display_label, id_column)
    "activity":        ("phenotypic_activity.csv",           "Activity",             "perturbation"),
    "distinctiveness": ("phenotypic_distinctiveness.csv",    "Distinctiveness",      "perturbation"),
    "corum":           ("phenotypic_consistency_corum.csv",  "Consistency\n(CORUM)", "complex_id"),
    "chad":            ("phenotypic_consistency_manual.csv", "Consistency\n(CHAD)",  "complex_num"),
}

_COL_SIG_BOTH   = "#2196F3"
_COL_SIG_EITHER = "#90CAF9"
_COL_NONSIG     = "#CCCCCC"
_COL_DISCORD    = "#D32F2F"

# Per-comparison colour pairs (label_a colour, label_b colour)
_COMPARISON_COLOURS = {
    "dino-vs-cp":        ("#E65100", "#1565C0"),   # orange, blue
    "phase-vs-nophase":  ("#2E7D32", "#6A1B9A"),   # green, purple
}


# ----------------------------------------------------------------------------
# Comparison configs
# ----------------------------------------------------------------------------

def _build_comparison_configs(output_dir: Path, comparison: str, feature_type: str) -> List[dict]:
    """Return list of comparison dicts: {tag, label_a, label_b, dir_a, dir_b, col_a, col_b}."""
    col_a, col_b = _COMPARISON_COLOURS.get(comparison, ("#E65100", "#1565C0"))

    if comparison == "dino-vs-cp":
        return [
            dict(tag="full",        label_a="DINO", label_b="CellProfiler", col_a=col_a, col_b=col_b,
                 dir_a=output_dir / "dino"         / "metrics",
                 dir_b=output_dir / "cellprofiler" / "metrics"),
            dict(tag="downsampled", label_a="DINO", label_b="CellProfiler", col_a=col_a, col_b=col_b,
                 dir_a=output_dir / "dino"         / "downsampled" / "metrics",
                 dir_b=output_dir / "cellprofiler" / "downsampled" / "metrics"),
        ]

    if comparison == "phase-vs-nophase":
        ft = feature_type
        return [
            dict(tag=f"{ft}_full",        label_a="Phase only", label_b="No phase", col_a=col_a, col_b=col_b,
                 dir_a=output_dir / ft / "phase_only" / "metrics",
                 dir_b=output_dir / ft / "no_phase"   / "metrics"),
        ]

    raise ValueError(f"Unknown comparison: {comparison!r}. Choose dino-vs-cp or phase-vs-nophase.")


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------

_CHAD_YAML = "/hpc/projects/icd.ops/configs/gene_clusters/chad_positive_controls_v4.yml"

def _chad_num_to_name() -> dict:
    """Load CHAD YAML and return {int_key: name} lookup."""
    try:
        import yaml
        with open(_CHAD_YAML) as f:
            d = yaml.safe_load(f)
        return {k: v["name"] for k, v in d.items() if isinstance(v, dict) and "name" in v}
    except Exception:
        return {}


def _load_metric_csv(path: Path, id_col: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if id_col not in df.columns:
        logger.warning(f"  {path.name}: id column '{id_col}' not found — cols: {list(df.columns)}")
        return None
    if id_col == "perturbation":
        df = df[~df[id_col].str.contains("NTC|non-targeting", case=False, na=False)]
    if id_col == "complex_num":
        name_map = _chad_num_to_name()
        if name_map:
            df[id_col] = df[id_col].map(lambda x: name_map.get(x, name_map.get(int(x), x)))
    keep = [c for c in [id_col, "mean_average_precision",
                         "corrected_p_value", "below_corrected_p"] if c in df.columns]
    return df[keep].rename(columns={id_col: "entity"}).copy()


def _merge_pair(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    merged = df_a.merge(df_b, on="entity", suffixes=("_a", "_b"))
    merged["delta"]     = merged["mean_average_precision_a"] - merged["mean_average_precision_b"]
    merged["delta_abs"] = merged["delta"].abs()
    for col in ["sig_a", "sig_b", "sig_both", "sig_either"]:
        merged[col] = False
    if "below_corrected_p_a" in merged.columns:
        merged["sig_a"]      = merged["below_corrected_p_a"].astype(bool)
        merged["sig_b"]      = merged["below_corrected_p_b"].astype(bool)
        merged["sig_both"]   = merged["sig_a"] & merged["sig_b"]
        merged["sig_either"] = merged["sig_a"] | merged["sig_b"]
    merged["discordant"] = merged["delta_abs"] > DISCORDANCE_THRESHOLD
    return merged


# ----------------------------------------------------------------------------
# Axis: scatter + regression
# ----------------------------------------------------------------------------

def _ax_scatter(ax: plt.Axes, merged: pd.DataFrame, title: str,
                label_a: str, label_b: str, col_a: str, col_b: str) -> dict:
    x = merged["mean_average_precision_a"].values
    y = merged["mean_average_precision_b"].values

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

    for _, row in merged[merged["discordant"]].iterrows():
        ax.annotate(row["entity"],
                    (row["mean_average_precision_a"], row["mean_average_precision_b"]),
                    fontsize=4.5, xytext=(3, 2), textcoords="offset points",
                    color=_COL_DISCORD, clip_on=True)

    n, nd = len(merged), int(merged["discordant"].sum())
    ax.text(0.97, 0.03, f"n={n}  disc={nd}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color="dimgrey")
    ax.set_xlabel(f"{label_a} mAP", fontsize=8)
    ax.set_ylabel(f"{label_b} mAP", fontsize=8)
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

def _ax_slope(ax: plt.Axes, merged: pd.DataFrame, title: str,
              label_a: str, label_b: str, col_a: str, col_b: str) -> None:
    x0, x1 = 0.0, 1.0
    rng = np.random.RandomState(42)

    disc_mask = merged["discordant"].values
    bg_idx    = np.where(~disc_mask)[0]
    if len(bg_idx) > SLOPE_MAX_BG:
        bg_idx = rng.choice(bg_idx, SLOPE_MAX_BG, replace=False)

    vals_a = merged["mean_average_precision_a"].values
    vals_b = merged["mean_average_precision_b"].values

    for i in bg_idx:
        colour = col_a if merged["delta"].iloc[i] > 0 else col_b
        ax.plot([x0, x1], [vals_a[i], vals_b[i]],
                color=colour, alpha=0.18, lw=0.7, solid_capstyle="round")

    for _, row in merged[disc_mask].nlargest(TOP_N_LABELS, "delta_abs").iterrows():
        colour = col_a if row["delta"] > 0 else col_b
        ax.plot([x0, x1], [row["mean_average_precision_a"], row["mean_average_precision_b"]],
                color=colour, alpha=0.85, lw=1.4, solid_capstyle="round", zorder=4)
        if row["delta"] > 0:
            ax.text(x0 - 0.03, row["mean_average_precision_a"], row["entity"],
                    ha="right", va="center", fontsize=4.5, color=col_a, clip_on=True)
        else:
            ax.text(x1 + 0.03, row["mean_average_precision_b"], row["entity"],
                    ha="left", va="center", fontsize=4.5, color=col_b, clip_on=True)

    ax.plot([x0, x1], [np.median(vals_a), np.median(vals_b)],
            color="black", lw=2.0, ls="--", alpha=0.6, zorder=5)

    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([x0, x1])
    ax.set_xticklabels([label_a, label_b], fontsize=9, fontweight="bold")
    ax.set_ylabel("mAP", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    ax.tick_params(axis="y", labelsize=7)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], color=col_a,   lw=1.5, label=f"{label_a} higher"),
        Line2D([0],[0], color=col_b,   lw=1.5, label=f"{label_b} higher"),
        Line2D([0],[0], color="black", lw=1.5, ls="--", label="median"),
    ], fontsize=6, loc="lower right", framealpha=0.7)


# ----------------------------------------------------------------------------
# Axis: delta violin (summary panel)
# ----------------------------------------------------------------------------

def _ax_violin(ax: plt.Axes, metric_deltas: Dict[str, np.ndarray], title: str,
               label_a: str, label_b: str, col_a: str, col_b: str) -> None:
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
                   c=[col_a if v > 0 else col_b for v in d],
                   s=10, alpha=0.45, linewidths=0, zorder=3)

    ax.axhline(0, color="black", lw=0.9, ls="--", alpha=0.45)

    y_top = max(np.max(np.abs(d)) for d in data) * 1.18
    for i, (p, d) in enumerate(zip(pvals, data)):
        sig = ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns") \
              if not np.isnan(p) else "n/a"
        p_str = f"p={p:.1e}" if not np.isnan(p) else ""
        ax.text(positions[i], y_top, f"{sig}\n{p_str}\nμΔ={np.mean(d):+.3f}",
                ha="center", va="top", fontsize=6.5, color="dimgrey")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(f"{label_a} − {label_b} mAP", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=col_a, label=f"{label_a} higher"),
                       Patch(facecolor=col_b, label=f"{label_b} higher")],
              fontsize=8, loc="lower right")


# ----------------------------------------------------------------------------
# Core: one comparison config
# ----------------------------------------------------------------------------

def _run_comparison(cfg: dict, comp_dir: Path) -> List[dict]:
    tag, label_a, label_b = cfg["tag"], cfg["label_a"], cfg["label_b"]
    col_a, col_b = cfg["col_a"], cfg["col_b"]
    dir_a, dir_b = cfg["dir_a"], cfg["dir_b"]

    available = {}
    for key, (csv_name, metric_label, id_col) in METRICS.items():
        df_a = _load_metric_csv(dir_a / csv_name, id_col)
        df_b = _load_metric_csv(dir_b / csv_name, id_col)
        if df_a is not None and df_b is not None:
            merged = _merge_pair(df_a, df_b)
            if len(merged) >= 3:
                available[key] = (merged, metric_label)
                merged.to_csv(comp_dir / f"{tag}_{key}.csv", index=False)
                logger.info(f"  {metric_label:25s}: {len(merged):3d} entities, "
                            f"mean Δ={merged['delta'].mean():+.3f}, "
                            f"{int(merged['discordant'].sum())} discordant")

    if not available:
        logger.warning(f"  No metrics found for '{tag}' — check that both dirs exist")
        return []

    n_metrics = len(available)

    fig, axes = plt.subplots(2, n_metrics, figsize=(4.5 * n_metrics, 9),
                              gridspec_kw={"hspace": 0.45, "wspace": 0.35})
    if n_metrics == 1:
        axes = axes.reshape(2, 1)

    summary_rows = []
    for col, (key, (merged, metric_label)) in enumerate(available.items()):
        short = metric_label.replace("\n", " ")
        stats_row = _ax_scatter(axes[0, col], merged, short, label_a, label_b, col_a, col_b)
        _ax_slope(axes[1, col], merged, short, label_a, label_b, col_a, col_b)
        stats_row.update({"tag": tag, "metric": key, "label_a": label_a, "label_b": label_b})
        summary_rows.append(stats_row)

    fig.suptitle(f"{label_a} vs {label_b} — {tag}", fontsize=13, fontweight="bold", y=1.01)
    fig.savefig(comp_dir / f"{tag}_panel.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {tag}_panel.png")

    metric_deltas = {k: m["delta"].values for k, (m, _) in available.items()}
    fig2, ax2 = plt.subplots(figsize=(max(6, n_metrics * 2.5), 5))
    _ax_violin(ax2, metric_deltas,
               f"{label_a} − {label_b} mAP delta — {tag} (Wilcoxon)",
               label_a, label_b, col_a, col_b)
    fig2.tight_layout()
    fig2.savefig(comp_dir / f"{tag}_delta_violin.png", dpi=180, bbox_inches="tight")
    plt.close(fig2)
    logger.info(f"  Saved {tag}_delta_violin.png")

    return summary_rows


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def compare_maps(output_dir: str, comparison: str = "dino-vs-cp",
                 feature_type: str = "dino") -> None:
    output_dir = Path(output_dir)
    comp_dir = output_dir / "comparison" / comparison.replace("-", "_")
    comp_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    configs = _build_comparison_configs(output_dir, comparison, feature_type)

    all_summary = []
    ran_any = False
    for cfg in configs:
        if not cfg["dir_a"].exists() and not cfg["dir_b"].exists():
            continue
        logger.info(f"\n=== {cfg['tag']}: {cfg['label_a']} vs {cfg['label_b']} ===")
        all_summary.extend(_run_comparison(cfg, comp_dir))
        ran_any = True

    if not ran_any:
        logger.error(f"No metrics directories found for comparison '{comparison}'. "
                     f"Expected:\n  A: {configs[0]['dir_a']}\n  B: {configs[0]['dir_b']}")
        return

    if all_summary:
        df = pd.DataFrame(all_summary)
        df.to_csv(comp_dir / "summary.csv", index=False)
        logger.info(f"\n{'tag':<20} {'metric':<16} {'n':>5} {'mean_Δ':>8} {'r':>7} {'disc':>6}")
        logger.info("-" * 62)
        for _, r in df.iterrows():
            logger.info(f"{r['tag']:<20} {r['metric']:<16} {r['n']:>5} "
                        f"{r['mean_delta']:>+8.3f} {r['r']:>7.3f} {r['n_discordant']:>6}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare phenotypic mAP scores between two feature sets or channel subsets"
    )
    parser.add_argument("-o", "--output-dir", type=str,
                        default="/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized")
    parser.add_argument("--comparison", type=str, default="dino-vs-cp",
                        choices=["dino-vs-cp", "phase-vs-nophase"],
                        help="Which comparison to run (default: dino-vs-cp)")
    parser.add_argument("--feature-type", type=str, default="dino",
                        choices=["dino", "cellprofiler"],
                        help="Feature type for phase-vs-nophase comparison (default: dino)")
    args = parser.parse_args()
    compare_maps(args.output_dir, args.comparison, args.feature_type)


if __name__ == "__main__":
    main()
