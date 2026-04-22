"""Per-reporter cell-count titration analysis.

Reads existing per-signal guide h5ads, repeatedly downsamples cells at each
guide (3/4 ratio per step, from all cells down to 50k), scores all 4 phenotypic
metrics at each titration point, and produces two summary plots per reporter
plus a combined overview:

  1. **% significant** — fraction of perturbations/complexes passing corrected
     p-value threshold for each metric vs cell count.
  2. **mean mAP** — average mAP score for each metric vs cell count.

Usage::

    # Run locally for a single variant (default -o matches pca_optimization.py)
    python -m ops_model.post_process.combination.pca_titration

    # Submit as SLURM jobs (one per reporter)
    python -m ops_model.post_process.combination.pca_titration --slurm

    # Include minibinder geneKO subset analysis with comparison overlay plots
    python -m ops_model.post_process.combination.pca_titration --minibinder-subset

    # Minibinder subset with SLURM submission
    python -m ops_model.post_process.combination.pca_titration --minibinder-subset --slurm

    # Override root (same as pca_optimization -o)
    python -m ops_model.post_process.combination.pca_titration \
        -o /hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3
"""

import argparse
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd

from ops_model.features.anndata_utils import (
    aggregate_to_level,
    normalize_guide_adata,
)
from ops_utils.analysis.map_scores import (
    compute_auc_score,
    phenotypic_activity_assesment,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOWNSAMPLE_RATIO = 0.75  # multiply cell count by this each step
MIN_CELLS = 5_000  # stop titrating below this
NULL_SIZE = 10_000  # smaller null for speed (per-reporter)
METRICS = ("activity", "distinctiveness", "corum", "chad")
SCALES = ("linear", "log2", "log10")  # x-axis scale variants to save

# Shared plot styling / labels (used by compare_pca_titration_versions and below)
TITRATION_METRIC_COLORS = {
    "activity": "steelblue",
    "distinctiveness": "mediumseagreen",
    "corum": "mediumpurple",
    "chad": "darkorange",
}
TITRATION_RATIO_LABELS = {
    "activity": "% Active",
    "distinctiveness": "% Distinctive",
    "corum": "% CORUM consistent",
    "chad": "% CHAD consistent",
}
TITRATION_MAP_LABELS = {
    "activity": "Activity mAP",
    "distinctiveness": "Distinctiveness mAP",
    "corum": "CORUM mAP",
    "chad": "CHAD mAP",
}
SCALE_LABEL_SHORT = {"linear": "linear", "log2": "log₂", "log10": "log₁₀"}

MINIBINDER_TARGETS_CSV = Path("/hpc/projects/icd.fast.ops/configs/library/minibinder_geneKO_targets.csv")

# Same default root as pca_optimization.py --output-dir
DEFAULT_PCA_OPT_ROOT = "/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3"


def _format_cell_count(n: int) -> str:
    """Format cell count as human-readable string: 1.2M, 500K, 50K, etc."""
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{v:.1f}M" if v != int(v) else f"{int(v)}M"
    elif n >= 1_000:
        v = n / 1_000
        return f"{v:.0f}K" if v >= 10 else f"{v:.1f}K"
    return str(n)


def _round_ticks(x_min: float, x_max: float, n: int = 7) -> list:
    """Return ~n round tick values spanning [x_min, x_max].

    Generates geometrically-spaced candidate positions then rounds each to
    1 significant figure, so ticks always land on human-readable values
    (e.g. 50K, 100K, 500K, 1M) regardless of where the raw data falls.
    """
    import math

    positions = np.geomspace(x_min, x_max, n)
    ticks = []
    seen = set()
    for v in positions:
        mag = 10 ** math.floor(math.log10(v))
        rounded = int(round(v / mag) * mag)
        if rounded > 0 and rounded not in seen and x_min * 0.5 <= rounded <= x_max * 2:
            ticks.append(rounded)
            seen.add(rounded)
    return sorted(ticks)


def _apply_x_scale(ax, x_values, scale: str, tick_fontsize: int = 12):
    """Apply x-axis scale with round human-readable tick labels.

    Ticks are rounded to 1 significant figure so they always show clean
    values (50K, 100K, 500K, 1M…) regardless of the raw 0.75-ratio data.
    The ``scale`` parameter controls only the visual spacing of the axis.

    linear  — uniform spacing
    log2    — log base-2 spacing
    log10   — log base-10 spacing
    """
    from matplotlib.ticker import FuncFormatter, NullFormatter

    x_min, x_max = min(x_values), max(x_values)
    ticks = _round_ticks(x_min, x_max)

    if scale == "linear":
        ax.set_xscale("linear")
    elif scale == "log2":
        ax.set_xscale("log", base=2)
    elif scale == "log10":
        ax.set_xscale("log", base=10)
    else:
        raise ValueError(f"Unknown scale: {scale!r}")

    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [_format_cell_count(t) for t in ticks],
        rotation=45,
        ha="right",
        fontsize=tick_fontsize,
    )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _format_cell_count(int(v))))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_logger():
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger("copairs").setLevel(logging.WARNING)
    return logging.getLogger(__name__)


def _prepare_for_copairs(adata: ad.AnnData) -> ad.AnnData:
    """Strip obs to copairs-required columns and cast X to float64."""
    if "n_cells" not in adata.obs.columns:
        adata.obs["n_cells"] = 1
    keep = [c for c in ["sgRNA", "perturbation", "n_cells"] if c in adata.obs.columns]
    adata.obs = adata.obs[keep].copy()
    for col in adata.obs.columns:
        if adata.obs[col].dtype.name == "category":
            adata.obs[col] = adata.obs[col].astype(str)
    adata.X = np.asarray(adata.X, dtype=np.float64)
    return adata


def _subsample_and_aggregate(
    adata_cells: ad.AnnData, target_n_cells: int, rng: np.random.RandomState,
    min_exp: bool = False,
) -> ad.AnnData:
    """Subsample real cells from the cell-level h5ad, then re-aggregate to guide level.

    If ``min_exp`` is True, selects the fewest experiments (ranked by cell count,
    descending) whose combined cells reach ``target_n_cells``, then randomly
    samples from just those. Otherwise samples uniformly across all cells.
    """
    n_total = adata_cells.n_obs
    if n_total <= target_n_cells:
        sub = adata_cells
    elif min_exp and "experiment" in adata_cells.obs.columns:
        # Pick fewest experiments to reach target, largest first
        exp_counts = adata_cells.obs["experiment"].value_counts()
        kept_exps = []
        running = 0
        for exp_id, count in exp_counts.items():
            kept_exps.append(exp_id)
            running += count
            if running >= target_n_cells:
                break
        mask = adata_cells.obs["experiment"].isin(kept_exps).values
        pool_idx = np.where(mask)[0]
        if len(pool_idx) <= target_n_cells:
            idx = pool_idx
        else:
            idx = rng.choice(pool_idx, target_n_cells, replace=False)
        idx.sort()
        sub = adata_cells[idx].copy()
    else:
        idx = rng.choice(n_total, target_n_cells, replace=False)
        idx.sort()
        sub = adata_cells[idx].copy()

    g = aggregate_to_level(
        sub,
        level="guide",
        method="mean",
        preserve_batch_info=False,
        subsample_controls=False,
    )
    return g


def _filter_map_to_targets(map_df, targets):
    """Filter a map DataFrame to only rows whose perturbation is in ``targets``."""
    if map_df is None or targets is None or "perturbation" not in map_df.columns:
        return map_df
    return map_df[map_df["perturbation"].isin(targets)]


def _ratio_and_mean_from_map(map_df):
    """Extract (ratio_significant, mean_mAP) from a map DataFrame."""
    if map_df is None or len(map_df) == 0:
        return math.nan, math.nan
    ratio = (
        float(map_df["below_corrected_p"].mean())
        if "below_corrected_p" in map_df.columns
        else math.nan
    )
    mean_map = (
        float(map_df["mean_average_precision"].mean())
        if "mean_average_precision" in map_df.columns
        else math.nan
    )
    return ratio, mean_map


def _score_all_metrics(
    g_norm: ad.AnnData, _logger, distance="cosine", subset_targets: Optional[set] = None
) -> dict:
    """Score all 4 metrics on a guide-level AnnData. Returns dict of ratios + mAPs.

    If ``subset_targets`` is provided, mAP is computed using ALL perturbations
    (so the full context is preserved) but ratios and means are reported only
    for the perturbations in ``subset_targets`` (Option B scoring).
    """
    from ops_utils.analysis.map_scores import (
        phenotypic_distinctivness,
        phenotypic_consistency_corum,
        phenotypic_consistency_manual_annotation,
    )

    result = {
        "activity_ratio": math.nan,
        "activity_map_mean": math.nan,
        "distinctiveness_ratio": math.nan,
        "distinctiveness_map_mean": math.nan,
        "corum_ratio": math.nan,
        "corum_map_mean": math.nan,
        "chad_ratio": math.nan,
        "chad_map_mean": math.nan,
    }

    try:
        g_copairs = _prepare_for_copairs(g_norm.copy())
        activity_map, active_ratio = phenotypic_activity_assesment(
            g_copairs,
            plot_results=False,
            null_size=NULL_SIZE,
            distance=distance,
        )
        if subset_targets is not None:
            filt = _filter_map_to_targets(activity_map, subset_targets)
            result["activity_ratio"], result["activity_map_mean"] = (
                _ratio_and_mean_from_map(filt)
            )
        else:
            result["activity_ratio"] = float(active_ratio)
            result["activity_map_mean"] = float(
                activity_map["mean_average_precision"].mean()
            )
    except Exception as exc:
        _logger.warning(f"    Activity scoring failed: {exc}")
        return result

    try:
        dist_map, dist_ratio = phenotypic_distinctivness(
            g_copairs,
            plot_results=False,
            null_size=NULL_SIZE,
            distance=distance,
        )
        if subset_targets is not None:
            filt = _filter_map_to_targets(dist_map, subset_targets)
            result["distinctiveness_ratio"], result["distinctiveness_map_mean"] = (
                _ratio_and_mean_from_map(filt)
            )
        else:
            result["distinctiveness_ratio"] = float(dist_ratio)
            if dist_map is not None and "mean_average_precision" in dist_map.columns:
                result["distinctiveness_map_mean"] = float(
                    dist_map["mean_average_precision"].mean()
                )
    except Exception as exc:
        _logger.warning(f"    Distinctiveness scoring failed: {exc}")

    try:
        e_norm = aggregate_to_level(
            g_copairs, "gene", preserve_batch_info=False, subsample_controls=False
        )
        e_copairs = _prepare_for_copairs(e_norm)

        corum_map, corum_ratio = phenotypic_consistency_corum(
            e_copairs,
            plot_results=False,
            null_size=NULL_SIZE,
            cache_similarity=True,
            distance=distance,
        )
        if subset_targets is not None:
            filt = _filter_map_to_targets(corum_map, subset_targets)
            result["corum_ratio"], result["corum_map_mean"] = _ratio_and_mean_from_map(
                filt
            )
        else:
            result["corum_ratio"] = float(corum_ratio)
            if corum_map is not None and "mean_average_precision" in corum_map.columns:
                result["corum_map_mean"] = float(
                    corum_map["mean_average_precision"].mean()
                )

        chad_map, chad_ratio = phenotypic_consistency_manual_annotation(
            e_copairs,
            plot_results=False,
            null_size=NULL_SIZE,
            cache_similarity=True,
            distance=distance,
        )
        if subset_targets is not None:
            filt = _filter_map_to_targets(chad_map, subset_targets)
            result["chad_ratio"], result["chad_map_mean"] = _ratio_and_mean_from_map(
                filt
            )
        else:
            result["chad_ratio"] = float(chad_ratio)
            if chad_map is not None and "mean_average_precision" in chad_map.columns:
                result["chad_map_mean"] = float(
                    chad_map["mean_average_precision"].mean()
                )
    except Exception as exc:
        _logger.warning(f"    Consistency scoring failed: {exc}")

    return result


# ---------------------------------------------------------------------------
# Core titration function (one per reporter — pickle-friendly for submitit)
# ---------------------------------------------------------------------------


def _load_minibinder_targets() -> set:
    """Load the 20 minibinder geneKO target names from the library CSV."""
    df = pd.read_csv(MINIBINDER_TARGETS_CSV)
    return set(df["gene_name"].dropna().unique())


def _subset_to_targets(adata: ad.AnnData, targets: set, _logger) -> ad.AnnData:
    """Subset cell-level adata to cells whose perturbation is in ``targets`` or is a control (NTC)."""
    pert_col = "perturbation" if "perturbation" in adata.obs.columns else "label_str"
    mask = adata.obs[pert_col].isin(targets) | adata.obs[pert_col].str.startswith("NTC")
    n_before = adata.n_obs
    adata_sub = adata[mask].copy()
    _logger.info(
        f"  Subset to {len(targets)} targets + NTC: {n_before:,} → {adata_sub.n_obs:,} cells"
    )
    return adata_sub


def _run_titration_points(
    adata_cells, cell_targets, norm_method, signal, rng, _logger, subset_targets=None,
    min_exp=False,
):
    """Score all titration points for an adata, returning a DataFrame of results.

    If ``subset_targets`` is provided, scores are computed using all perturbations
    but reported only for the subset (Option B).
    """
    # Drop signal col if present (not needed for scoring, can interfere with aggregation)
    if "signal" in adata_cells.obs.columns:
        adata_cells.obs = adata_cells.obs.drop(columns=["signal"])

    rows = []
    for target in cell_targets:
        _logger.info(f"  Scoring at {target:,} cells...")
        t_step = time.time()

        g_sub = _subsample_and_aggregate(adata_cells, target, rng, min_exp=min_exp)
        g_norm = normalize_guide_adata(g_sub, norm_method)
        scores = _score_all_metrics(g_norm, _logger, subset_targets=subset_targets)
        scores["n_cells"] = target
        scores["n_guides"] = g_sub.n_obs
        pert_col = (
            "perturbation" if "perturbation" in g_sub.obs.columns else "label_str"
        )
        n_perts = g_sub.obs[pert_col].nunique()
        scores["n_perturbations"] = n_perts
        scores["cells_per_perturbation"] = target / n_perts if n_perts > 0 else target
        scores["signal"] = signal
        rows.append(scores)

        _logger.info(
            f"    act={scores['activity_ratio']:.1%} dist={scores['distinctiveness_ratio']:.1%} "
            f"corum={scores['corum_ratio']:.1%} chad={scores['chad_ratio']:.1%} "
            f"({time.time() - t_step:.0f}s)"
        )
    return pd.DataFrame(rows)


def _build_titration_schedule(total_cells: int) -> list:
    """Build titration schedule: total, total*0.75, total*0.75^2, ... >= MIN_CELLS."""
    cell_targets = []
    n = total_cells
    while n >= MIN_CELLS:
        cell_targets.append(int(n))
        n = int(n * DOWNSAMPLE_RATIO)
    if not cell_targets:
        cell_targets = [total_cells]
    return cell_targets


def titrate_single_reporter(
    cells_h5ad_path: str,
    output_dir: str,
    norm_method: str = "ntc",
    random_seed: int = 42,
    minibinder_subset: bool = False,
    min_exp: bool = False,
) -> str:
    """Run cell-count titration for a single reporter.

    Loads the full cell-level PCA-reduced h5ad, subsamples real cells at each
    titration point, re-aggregates to guide level, and scores all 4 metrics.

    If ``minibinder_subset`` is True, also runs the titration on only the 20
    minibinder geneKO targets and produces a comparison overlay plot.

    Returns a status string.
    """
    _logger = _init_logger()
    t_start = time.time()
    cells_h5ad_path = Path(cells_h5ad_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(random_seed)

    adata_cells = ad.read_h5ad(cells_h5ad_path)
    signal = adata_cells.obs.get(
        "signal", pd.Series([cells_h5ad_path.stem.replace("_cells", "")])
    ).iloc[0]
    if isinstance(signal, float):
        signal = cells_h5ad_path.stem.replace("_cells", "")
    total_cells = adata_cells.n_obs

    _logger.info(f"Titrating {signal}: {total_cells:,} cells, {adata_cells.n_vars} PCs")

    cell_targets = _build_titration_schedule(total_cells)
    _logger.info(f"  Titration points: {cell_targets}")

    from ops_utils.data.feature_discovery import sanitize_signal_filename

    sig_safe = sanitize_signal_filename(signal)[:40]

    # Per-reporter subdir keeps CSVs and all scale/format variants together
    reporter_dir = output_dir / sig_safe
    reporter_dir.mkdir(parents=True, exist_ok=True)

    # --- Full titration ---
    df_full = _run_titration_points(
        adata_cells.copy(),
        cell_targets,
        norm_method,
        signal,
        np.random.RandomState(random_seed),
        _logger,
        min_exp=min_exp,
    )
    csv_path = reporter_dir / f"{sig_safe}_titration.csv"
    df_full.to_csv(csv_path, index=False)
    _logger.info(f"  Saved {csv_path}")

    # --- Minibinder subset titrations ---
    df_library = None  # Option A: subset cells, score subset perturbations
    df_scores = None  # Option B: all cells, score only subset perturbations
    minibinder_dir = None
    if minibinder_subset:
        targets = _load_minibinder_targets()
        minibinder_dir = output_dir / "minibinder" / sig_safe
        minibinder_dir.mkdir(parents=True, exist_ok=True)

        # Option A — subset library: filter cells to targets, titrate smaller pool
        _logger.info("  [Option A] Subset library titration...")
        adata_sub = _subset_to_targets(adata_cells, targets, _logger)
        sub_targets = _build_titration_schedule(adata_sub.n_obs)
        _logger.info(f"  Subset titration points: {sub_targets}")
        df_library = _run_titration_points(
            adata_sub,
            sub_targets,
            norm_method,
            signal,
            np.random.RandomState(random_seed),
            _logger,
        )
        df_library.to_csv(
            minibinder_dir / f"{sig_safe}_titration_library.csv", index=False
        )

        # Option B — subset scores: same full cells, but report only subset mAP/ratios
        _logger.info("  [Option B] Subset scores from full pool...")
        df_scores = _run_titration_points(
            adata_cells.copy(),
            cell_targets,
            norm_method,
            signal,
            np.random.RandomState(random_seed),
            _logger,
            subset_targets=targets,
        )
        df_scores.to_csv(
            minibinder_dir / f"{sig_safe}_titration_scores.csv", index=False
        )

    # Plot — PNG + SVG for each scale
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _plot_titration(df_full, signal, reporter_dir, sig_safe, plt)
        if minibinder_dir is not None:
            mb_metrics = ("activity", "distinctiveness")
            _plot_titration(
                df_library, signal, minibinder_dir, sig_safe, plt, metrics=mb_metrics
            )
            # Option A comparison: full vs subset library
            _plot_titration_comparison(
                df_full,
                df_library,
                signal,
                minibinder_dir,
                sig_safe,
                plt,
                label_a="All Perts",
                label_b="Minibinder Library",
                metrics=mb_metrics,
            )
            # Option B comparison: full scores vs subset scores (same cells)
            _plot_titration_comparison(
                df_full,
                df_scores,
                signal,
                minibinder_dir,
                sig_safe,
                plt,
                label_a="All Perts",
                label_b="Minibinder Scores",
                suffix="scores",
                metrics=mb_metrics,
            )
    except Exception as exc:
        _logger.warning(f"  Plotting failed: {exc}")

    elapsed = time.time() - t_start
    return f"SUCCESS: {signal} — {len(cell_targets)} titration points in {elapsed:.0f}s"


_X_AXIS_VARIANTS = [
    ("n_cells", "Total Cells", "totalcells"),
    ("cells_per_perturbation", "Cells / Perturbation", "perpert"),
]


def titration_x_axis_base_label(x_col: str) -> str:
    """Human-readable x-axis title segment for a titration CSV column."""
    for col, label, _ in _X_AXIS_VARIANTS:
        if col == x_col:
            return label
    return x_col


def _plot_titration(df, signal, reporter_dir: Path, sig_safe, plt, metrics=None):
    """Generate titration plots for one reporter across all scales and x-axis types.

    Saves PNG + SVG for each (scale × x-axis) combination into ``reporter_dir``.
    All text is sized at 1.5× the matplotlib default for legibility.
    """
    if metrics is None:
        metrics = METRICS
    reporter_dir = Path(reporter_dir)
    reporter_dir.mkdir(parents=True, exist_ok=True)

    colors = TITRATION_METRIC_COLORS
    ratio_labels = TITRATION_RATIO_LABELS
    map_labels = TITRATION_MAP_LABELS
    _scale_label = SCALE_LABEL_SHORT

    for x_col, x_label_base, x_suffix in _X_AXIS_VARIANTS:
        if x_col not in df.columns:
            continue
        x = df[x_col].values

        for scale in SCALES:
            fig, axes = plt.subplots(1, 2, figsize=(22, 9))
            xlabel = f"{x_label_base} ({_scale_label[scale]})"

            # Panel 1: % significant
            ax = axes[0]
            for metric in metrics:
                col = f"{metric}_ratio"
                if col in df.columns:
                    vals = df[col].values * 100
                    ax.plot(
                        x,
                        vals,
                        marker="o",
                        color=colors[metric],
                        label=ratio_labels[metric],
                        linewidth=3.5,
                        markersize=8,
                    )
            ax.set_xlabel(xlabel, fontsize=22)
            ax.set_ylabel("% Significant", fontsize=22)
            ax.set_title(f"{signal} — % Significant", fontsize=24)
            ax.tick_params(axis="y", labelsize=18)
            _apply_x_scale(ax, x, scale, tick_fontsize=18)

            # Panel 2: mean mAP
            ax = axes[1]
            for metric in metrics:
                col = f"{metric}_map_mean"
                if col in df.columns:
                    vals = df[col].values
                    ax.plot(
                        x,
                        vals,
                        marker="s",
                        color=colors[metric],
                        label=map_labels[metric],
                        linewidth=3.5,
                        markersize=8,
                    )
            ax.set_xlabel(xlabel, fontsize=22)
            ax.set_ylabel("Mean mAP", fontsize=22)
            ax.set_title(f"{signal} — Mean mAP", fontsize=24)
            ax.tick_params(axis="y", labelsize=18)
            _apply_x_scale(ax, x, scale, tick_fontsize=18)

            # Single legend below the plots
            handles, labels_list = axes[0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels_list,
                loc="lower center",
                ncol=4,
                fontsize=19,
                bbox_to_anchor=(0.5, -0.02),
            )

            fig.suptitle(
                f"Cell Count Titration — {signal}  [{scale}]",
                fontsize=31,
                fontweight="bold",
            )
            fig.tight_layout(rect=[0, 0.06, 1, 0.97])

            stem = reporter_dir / f"{sig_safe}_titration_{x_suffix}_{scale}"
            fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
            fig.savefig(f"{stem}.svg", bbox_inches="tight")
            plt.close(fig)


def _plot_titration_comparison(
    df_full,
    df_subset,
    signal,
    reporter_dir,
    sig_safe,
    plt,
    label_a="All Perts",
    label_b="Minibinder",
    suffix="library",
    metrics=("activity", "distinctiveness"),
):
    """Overlay two titration curves to show the shift.

    Produces one figure per (scale × x-axis) combination with 2 panels.
    Curve A shown as solid lines, curve B as dashed lines.
    Both total-cells and cells/perturbation x-axes are generated.
    ``suffix`` distinguishes Option A (library) vs Option B (scores) filenames.
    """
    reporter_dir = Path(reporter_dir)
    colors = TITRATION_METRIC_COLORS
    ratio_labels = TITRATION_RATIO_LABELS
    map_labels = TITRATION_MAP_LABELS
    _scale_label = SCALE_LABEL_SHORT

    for x_col, x_label_base, x_suffix in _X_AXIS_VARIANTS:
        if x_col not in df_full.columns or x_col not in df_subset.columns:
            continue
        x_full = df_full[x_col].values
        x_sub = df_subset[x_col].values
        x_all = np.concatenate([x_full, x_sub])

        for scale in SCALES:
            fig, axes = plt.subplots(1, 2, figsize=(22, 9))
            xlabel = f"{x_label_base} ({_scale_label[scale]})"

            # Panel 1: % significant
            ax = axes[0]
            for metric in metrics:
                col = f"{metric}_ratio"
                c = colors[metric]
                if col in df_full.columns:
                    ax.plot(
                        x_full,
                        df_full[col].values * 100,
                        marker="o",
                        color=c,
                        label=f"{ratio_labels[metric]} ({label_a})",
                        linewidth=3.5,
                        markersize=8,
                    )
                if col in df_subset.columns:
                    ax.plot(
                        x_sub,
                        df_subset[col].values * 100,
                        marker="^",
                        color=c,
                        label=f"{ratio_labels[metric]} ({label_b})",
                        linewidth=3.5,
                        markersize=8,
                        linestyle="--",
                        alpha=0.7,
                    )
            ax.set_xlabel(xlabel, fontsize=22)
            ax.set_ylabel("% Significant", fontsize=22)
            ax.set_title(
                f"{signal} — % Significant: {label_a} vs {label_b}", fontsize=22
            )
            ax.tick_params(axis="y", labelsize=18)
            _apply_x_scale(ax, x_all, scale, tick_fontsize=18)

            # Panel 2: mean mAP
            ax = axes[1]
            for metric in metrics:
                col = f"{metric}_map_mean"
                c = colors[metric]
                if col in df_full.columns:
                    ax.plot(
                        x_full,
                        df_full[col].values,
                        marker="s",
                        color=c,
                        label=f"{map_labels[metric]} ({label_a})",
                        linewidth=3.5,
                        markersize=8,
                    )
                if col in df_subset.columns:
                    ax.plot(
                        x_sub,
                        df_subset[col].values,
                        marker="D",
                        color=c,
                        label=f"{map_labels[metric]} ({label_b})",
                        linewidth=3.5,
                        markersize=8,
                        linestyle="--",
                        alpha=0.7,
                    )
            ax.set_xlabel(xlabel, fontsize=22)
            ax.set_ylabel("Mean mAP", fontsize=22)
            ax.set_title(f"{signal} — Mean mAP: {label_a} vs {label_b}", fontsize=22)
            ax.tick_params(axis="y", labelsize=18)
            _apply_x_scale(ax, x_all, scale, tick_fontsize=18)

            # Single legend below the plots
            handles, labels_list = axes[0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels_list,
                loc="lower center",
                ncol=4,
                fontsize=14,
                bbox_to_anchor=(0.5, -0.02),
            )

            fig.suptitle(
                f"Titration {suffix.title()} — {signal}  [{scale}]",
                fontsize=31,
                fontweight="bold",
            )
            fig.tight_layout(rect=[0, 0.06, 1, 0.97])

            stem = reporter_dir / f"{sig_safe}_comparison_{suffix}_{x_suffix}_{scale}"
            fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
            fig.savefig(f"{stem}.svg", bbox_inches="tight")
            plt.close(fig)


def _plot_combined_titration(
    output_dir,
    plt,
    csv_glob="**/*_titration.csv",
    title_suffix=None,
    filename_prefix="titration_combined",
):
    """Combine all per-reporter titration CSVs into one summary plot.

    Saves PNG + SVG for each scale (linear, log2, log10).
    CSVs are discovered recursively so reporter subdirs are supported.
    """
    csv_files = sorted(Path(output_dir).glob(csv_glob))
    if not csv_files:
        return

    all_dfs = [pd.read_csv(f) for f in csv_files]
    combined = pd.concat(all_dfs, ignore_index=True)

    signals = combined["signal"].unique()
    n_signals = len(signals)
    colors_cycle = plt.cm.tab20(np.linspace(0, 1, max(n_signals, 2)))

    metric_info = [
        ("activity", "% Active", "steelblue"),
        ("distinctiveness", "% Distinctive", "mediumseagreen"),
        ("corum", "% CORUM", "mediumpurple"),
        ("chad", "% CHAD", "darkorange"),
    ]

    _scale_label = {"linear": "linear", "log2": "log₂", "log10": "log₁₀"}

    for x_col, x_label_base, x_suffix in _X_AXIS_VARIANTS:
        if x_col not in combined.columns:
            continue
        x_all = combined[x_col].values
        x_min, x_max = float(x_all.min()), float(x_all.max())

        for scale in SCALES:
            fig, axes = plt.subplots(2, 4, figsize=(56, 18))
            xlabel = f"{x_label_base} ({_scale_label[scale]})"

            def _style_combined_axis(ax, _scale=scale, _xmin=x_min, _xmax=x_max):
                _apply_x_scale(ax, [_xmin, _xmax], _scale, tick_fontsize=19)
                ax.set_xlim(_xmin * 0.7, _xmax * 1.3)

            # Row 0: % significant per metric
            for col_idx, (metric, label, _) in enumerate(metric_info):
                ax = axes[0, col_idx]
                ratio_col = f"{metric}_ratio"
                for i, sig in enumerate(sorted(signals)):
                    sub = combined[combined["signal"] == sig].sort_values(x_col)
                    if ratio_col in sub.columns:
                        ax.plot(
                            sub[x_col],
                            sub[ratio_col] * 100,
                            marker="o",
                            color=colors_cycle[i % len(colors_cycle)],
                            label=sig[:25],
                            linewidth=3,
                            markersize=8,
                            alpha=0.8,
                        )
                ax.set_xlabel(xlabel, fontsize=24)
                ax.set_ylabel("% Significant", fontsize=24)
                ax.set_title(label, fontsize=26)
                ax.tick_params(axis="y", labelsize=19)
                _style_combined_axis(ax)

            # Row 1: mean mAP per metric
            for col_idx, (metric, label, _) in enumerate(metric_info):
                ax = axes[1, col_idx]
                map_col = f"{metric}_map_mean"
                for i, sig in enumerate(sorted(signals)):
                    sub = combined[combined["signal"] == sig].sort_values(x_col)
                    if map_col in sub.columns:
                        ax.plot(
                            sub[x_col],
                            sub[map_col],
                            marker="s",
                            color=colors_cycle[i % len(colors_cycle)],
                            label=sig[:25],
                            linewidth=3,
                            markersize=8,
                            alpha=0.8,
                        )
                ax.set_xlabel(xlabel, fontsize=24)
                ax.set_ylabel("Mean mAP", fontsize=24)
                ax.set_title(f"{label} mAP", fontsize=26)
                ax.tick_params(axis="y", labelsize=19)
                _style_combined_axis(ax)

            # Single legend at bottom
            handles, labels_list = axes[0, 0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels_list,
                loc="lower center",
                ncol=min(8, n_signals),
                fontsize=19,
                bbox_to_anchor=(0.5, -0.02),
            )

            title_tag = f" — {title_suffix}" if title_suffix else ""
            fig.suptitle(
                f"Cell Count Titration — All Reporters{title_tag}  [{scale}]",
                fontsize=34,
                fontweight="bold",
            )
            fig.tight_layout(rect=[0, 0.04, 1, 0.97])

            stem = Path(output_dir) / f"{filename_prefix}_{x_suffix}_{scale}"
            fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
            fig.savefig(f"{stem}.svg", bbox_inches="tight")
            plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Per-reporter cell-count titration analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default=DEFAULT_PCA_OPT_ROOT,
        help=f"Root output directory (same as pca_optimization -o; default: {DEFAULT_PCA_OPT_ROOT})",
    )
    parser.add_argument("--norm-method", type=str, default="ntc",
                        help="Normalization method (default: ntc)")
    parser.add_argument("--slurm", action="store_true",
                        help="Submit one SLURM job per reporter")
    parser.add_argument("--slurm-memory", type=str, default="200GB")
    parser.add_argument(
        "--slurm-time",
        type=int,
        default=30,
        help="SLURM time limit per job in minutes (default: 30)",
    )
    parser.add_argument("--slurm-cpus", type=int, default=8)
    parser.add_argument("--slurm-partition", type=str, default="cpu,gpu")
    parser.add_argument("--replot", action="store_true",
                        help="Regenerate all plots from existing CSVs without recomputing scores")
    parser.add_argument("--downsampled", action="store_true",
                        help="Look in downsampled/ subdir")
    parser.add_argument("--cell-profiler", action="store_true",
                        help="Look in cellprofiler/ subdir")
    parser.add_argument(
        "--include-cellpainting", action="store_true",
        help="Look under with_cellpainting/ (same as pca_optimization --include-cellpainting)",
    )
    parser.add_argument("--minibinder-subset", action="store_true",
                        help="Also run titration on the 20 minibinder geneKO targets "
                             "and produce comparison overlay plots")
    parser.add_argument(
        "--fixed-threshold", type=float, default=None,
        help="Match pca_optimization --fixed-threshold (uses fixed_<pct>/ not consensus_sweep/)",
    )
    parser.add_argument(
        "--distance", type=str, default="cosine", choices=["cosine", "euclidean"],
        help="Match pca_optimization --distance (default: cosine)",
    )
    parser.add_argument("--zscore-per-experiment", action="store_true",
                        help="Look in zscore_per_exp/ subdir")
    parser.add_argument("--min-exp-titration", action="store_true",
                        help="At each titration level, draw cells from the fewest "
                             "experiments needed (largest first) instead of sampling "
                             "across all experiments. Output → titration_min_exp/")
    phase_group = parser.add_mutually_exclusive_group()
    phase_group.add_argument("--phase-only", action="store_true")
    phase_group.add_argument("--no-phase", action="store_true")
    return parser


def _resolve_output_dir(args) -> Path:
    """Mirror pca_optimization.main() output nesting (non --direct)."""
    output_dir = Path(args.output_dir)
    if args.cell_profiler:
        output_dir = output_dir / "cellprofiler"
    else:
        output_dir = output_dir / "dino"

    if getattr(args, "zscore_per_experiment", False):
        output_dir = output_dir / "zscore_per_exp"

    if getattr(args, "include_cellpainting", False):
        output_dir = output_dir / "with_cellpainting"

    if getattr(args, "phase_only", False) and args.downsampled:
        output_dir = output_dir / "phase_only_downsampled"
    elif getattr(args, "no_phase", False) and args.downsampled:
        output_dir = output_dir / "no_phase_downsampled"
    elif getattr(args, "phase_only", False):
        output_dir = output_dir / "phase_only"
    elif getattr(args, "no_phase", False):
        output_dir = output_dir / "no_phase"
    elif args.downsampled:
        output_dir = output_dir / "downsampled"
    else:
        output_dir = output_dir / "all"

    ft = getattr(args, "fixed_threshold", None)
    if ft is not None:
        output_dir = output_dir / f"fixed_{ft:.0%}"
    else:
        output_dir = output_dir / "consensus_sweep"

    output_dir = output_dir / args.distance
    return output_dir


def resolve_titration_output_dir(args: argparse.Namespace) -> Path:
    """Where ``pca_titration`` writes per-reporter outputs: ``<variant>/titration``."""
    return _resolve_output_dir(args) / "titration"


def _replot_one(csv_path: Path, minibinder_subset: bool = False) -> str:
    """Plot a single reporter from its CSV; returns sig_safe for progress reporting."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    signal = (
        df["signal"].iloc[0]
        if "signal" in df.columns
        else csv_path.stem.replace("_titration", "")
    )
    reporter_dir = csv_path.parent
    sig_safe = reporter_dir.name
    _plot_titration(df, signal, reporter_dir, sig_safe, plt)

    # Replot minibinder subset plots if CSVs exist
    if minibinder_subset:
        titration_dir = reporter_dir.parent
        mb_dir = titration_dir / "minibinder" / sig_safe
        mb_library_csv = mb_dir / f"{sig_safe}_titration_library.csv"
        mb_scores_csv = mb_dir / f"{sig_safe}_titration_scores.csv"
        mb_metrics = ("activity", "distinctiveness")

        if mb_library_csv.exists():
            df_library = pd.read_csv(mb_library_csv)
            _plot_titration(
                df_library, signal, mb_dir, sig_safe, plt, metrics=mb_metrics
            )
            _plot_titration_comparison(
                df,
                df_library,
                signal,
                mb_dir,
                sig_safe,
                plt,
                label_a="All Perts",
                label_b="Minibinder Library",
                metrics=mb_metrics,
            )

        if mb_scores_csv.exists():
            df_scores = pd.read_csv(mb_scores_csv)
            _plot_titration_comparison(
                df,
                df_scores,
                signal,
                mb_dir,
                sig_safe,
                plt,
                label_a="All Perts",
                label_b="Minibinder Scores",
                suffix="scores",
                metrics=mb_metrics,
            )

    plt.close("all")
    return sig_safe


def _replot(titration_dir: Path, minibinder_subset: bool = False):
    """Regenerate all per-reporter and combined plots from existing CSVs, in parallel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from ops_utils.hpc.resource_manager import get_optimal_workers
    from tqdm import tqdm

    # Only glob top-level reporter CSVs, not minibinder subdirectory CSVs
    csv_files = sorted(
        p
        for p in titration_dir.glob("*/*_titration.csv")
        if "minibinder" not in p.parts
    )
    if not csv_files:
        print(f"No *_titration.csv files found under {titration_dir}")
        return

    # Plotting is CPU + light RAM bound; leave GPU out of the equation
    n_workers = get_optimal_workers(use_gpu=False, model_ram_gb=0.05, data_ram_gb=0.2)
    label = "reporters" + (" + minibinder" if minibinder_subset else "")
    print(f"Replotting {len(csv_files)} {label} with {n_workers} workers...")

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_replot_one, csv_path, minibinder_subset): csv_path
            for csv_path in csv_files
        }
        with tqdm(total=len(futures), unit="reporter") as pbar:
            for fut in as_completed(futures):
                try:
                    pbar.set_postfix_str(fut.result())
                except Exception as exc:
                    pbar.set_postfix_str(f"ERROR {futures[fut].stem}: {exc}")
                pbar.update(1)

    print("Generating combined titration plot...")
    _plot_combined_titration(titration_dir, plt)
    print(f"Saved {titration_dir}/titration_combined_{{linear,log2,log10}}.{{png,svg}}")

    if minibinder_subset:
        mb_base = titration_dir / "minibinder"
        mb_library_csvs = sorted(mb_base.glob("**/*_titration_library.csv"))
        mb_scores_csvs = sorted(mb_base.glob("**/*_titration_scores.csv"))
        if mb_library_csvs:
            print("Generating minibinder library combined plot...")
            _plot_combined_titration(
                mb_base,
                plt,
                csv_glob="**/*_titration_library.csv",
                title_suffix="Minibinder Library",
            )
            print(
                f"Saved {mb_base}/titration_combined_{{linear,log2,log10}}.{{png,svg}}"
            )
        if mb_scores_csvs:
            print("Generating minibinder scores combined plot...")
            _plot_combined_titration(
                mb_base,
                plt,
                csv_glob="**/*_titration_scores.csv",
                title_suffix="Minibinder Scores",
                filename_prefix="titration_scores_combined",
            )
            print(
                f"Saved {mb_base}/titration_scores_combined_{{linear,log2,log10}}.{{png,svg}}"
            )


def main():
    args = _build_parser().parse_args()
    _logger = _init_logger()

    variant_dir = _resolve_output_dir(args)
    titration_subdir = "titration_min_exp" if args.min_exp_titration else "titration"
    titration_dir = variant_dir / titration_subdir

    if args.replot:
        titration_dir.mkdir(parents=True, exist_ok=True)
        _replot(titration_dir, minibinder_subset=args.minibinder_subset)
        return

    per_signal_dir = variant_dir / "per_signal"
    titration_dir.mkdir(parents=True, exist_ok=True)

    cells_files = sorted(per_signal_dir.glob("*_cells.h5ad"))
    if not cells_files:
        print(f"No cell-level h5ads (*_cells.h5ad) found in {per_signal_dir}")
        print("  Re-run Phase 1 (pca_optimization --slurm --clean) to generate them.")
        return

    print(f"Found {len(cells_files)} reporters in {per_signal_dir}")
    print(f"Titration output: {titration_dir}")

    if args.slurm:
        from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

        jobs = []
        for cf in cells_files:
            sig_safe = cf.stem.replace("_cells", "")[:40]
            jobs.append(
                {
                    "name": f"titr_{sig_safe}",
                    "func": titrate_single_reporter,
                    "kwargs": {
                        "cells_h5ad_path": str(cf),
                        "output_dir": str(titration_dir),
                        "norm_method": args.norm_method,
                        "minibinder_subset": args.minibinder_subset,
                        "min_exp": args.min_exp_titration,
                    },
                }
            )

        print(f"\nSubmitting {len(jobs)} SLURM titration jobs...")
        slurm_params = {
            "timeout_min": args.slurm_time,
            "mem": args.slurm_memory,
            "cpus_per_task": args.slurm_cpus,
            "slurm_partition": args.slurm_partition,
        }
        result = submit_parallel_jobs(
            jobs_to_submit=jobs,
            experiment="pca_titration",
            slurm_params=slurm_params,
            log_dir="pca_optimization",
            manifest_prefix="pca_titration",
            wait_for_completion=True,
        )

        if result.get("failed"):
            print(f"\n{len(result['failed'])} jobs failed")
        else:
            print(f"\nAll {len(jobs)} titration jobs complete")

        # Generate combined plot from all CSVs
        print("Generating combined titration plot...")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _plot_combined_titration(titration_dir, plt)
        print(
            f"Saved {titration_dir}/titration_combined_{{linear,log2,log10}}.{{png,svg}}"
        )

        if args.minibinder_subset:
            mb_base = titration_dir / "minibinder"
            if mb_base.exists():
                print("Generating minibinder combined plots...")
                _plot_combined_titration(
                    mb_base,
                    plt,
                    csv_glob="**/*_titration_library.csv",
                    title_suffix="Minibinder Library",
                )
                _plot_combined_titration(
                    mb_base,
                    plt,
                    csv_glob="**/*_titration_scores.csv",
                    title_suffix="Minibinder Scores",
                    filename_prefix="titration_scores_combined",
                )
                print(
                    f"Saved {mb_base}/titration_*_combined_{{linear,log2,log10}}.{{png,svg}}"
                )

    else:
        print("\nRunning locally (sequential)...")
        for cf in cells_files:
            result = titrate_single_reporter(
                cells_h5ad_path=str(cf),
                output_dir=str(titration_dir),
                norm_method=args.norm_method,
                minibinder_subset=args.minibinder_subset,
                min_exp=args.min_exp_titration,
            )
            print(f"  {result}")

        print("\nGenerating combined titration plot...")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _plot_combined_titration(titration_dir, plt)
        print(
            f"Saved {titration_dir}/titration_combined_{{linear,log2,log10}}.{{png,svg}}"
        )

        if args.minibinder_subset:
            mb_base = titration_dir / "minibinder"
            if mb_base.exists():
                print("Generating minibinder combined plots...")
                _plot_combined_titration(
                    mb_base,
                    plt,
                    csv_glob="**/*_titration_library.csv",
                    title_suffix="Minibinder Library",
                )
                _plot_combined_titration(
                    mb_base,
                    plt,
                    csv_glob="**/*_titration_scores.csv",
                    title_suffix="Minibinder Scores",
                    filename_prefix="titration_scores_combined",
                )
                print(
                    f"Saved {mb_base}/titration_*_combined_{{linear,log2,log10}}.{{png,svg}}"
                )


if __name__ == "__main__":
    main()
