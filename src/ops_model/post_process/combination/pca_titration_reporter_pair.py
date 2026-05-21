"""Phase + Fe (or any two reporters) combination titration comparison.

For each titration point (N cells per reporter):
  1. Downsample + aggregate reporter A to guide level
  2. Downsample + aggregate reporter B to guide level
  3. Normalize each independently (NTC or global)
  4. Reporter A / B alone scores: reused from the per-marker titration CSVs
     (interpolated in log(N)) when ``--per-marker-titration-dir`` resolves to
     existing CSVs, otherwise re-scored from cells in this run.
  5. Combine features (inner join on guide, horizontal concat) — score the raw
     concat directly.
  6. (``--dual-pass``) Fit PCA on the concatenated NTC-normalized guide matrix,
     sweep variance thresholds (default ``DEFAULT_SWEEP_THRESHOLDS`` from
     ``pca_optimization``), pick consensus winner (max of normalized
     act + dist + chad), and score that projection.

Output: 4 figures (one per mAP metric: activity, distinctiveness, CORUM, CHAD).
Each figure has 2 panels:
  - Left:  % ratio (significant) vs cell count
  - Right: mean mAP vs cell count
  - 3-4 lines per panel: A alone | B alone | A+B raw concat | A+B 2-pass PCA
With ``--dual-pass``, also writes ``titration_pair_dualpass_sweep.csv`` listing
every threshold scored at every N plus the chosen consensus.

Usage::

    python -m ops_model.post_process.combination.pca_titration_reporter_pair \\
        --reporter-a /path/to/pca_optimized_v2/dino/all/per_signal/Phase_cells.h5ad \\
        --reporter-b /path/to/pca_optimized_v2/dino/all/per_signal/FeRhoNox_cells.h5ad \\
        -o /hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v2/dino/all/titration/phase_fe_comparison

    # Phase vs Fe+ with dual-pass PCA + per-marker reuse:
    python -m ops_model.post_process.combination.pca_titration_reporter_pair \\
        --reporter-a /hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v2/dino/all/per_signal/Phase_cells.h5ad \\
        --reporter-b "/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v2/dino/all/per_signal/Fe2+_FeRhoNox_live-cell_dye_cells.h5ad" \\
        --name-a "Phase" --name-b "Fe+" \\
        --dual-pass \\
        -o /hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v2/dino/all/titration/phase_fe_comparison \\
        --slurm --yes
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

# Reuse utilities from the main titration script
from ops_model.post_process.combination.pca_titration import (
    DOWNSAMPLE_RATIO,
    MIN_CELLS,
    METRICS,
    NULL_SIZE,
    _init_logger,
    _subsample_and_aggregate,
    _score_all_metrics,
    _apply_x_scale,
    _format_cell_count,
    _prepare_for_copairs,
)
from ops_model.post_process.combination.pca_optimization import (
    DEFAULT_SWEEP_THRESHOLDS,
    DEFAULT_SWEEP_THRESHOLDS_CP,
    MIN_PCS,
)
from ops_utils.analysis.pca import fit_pca, n_pcs_for_threshold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-marker reuse helpers
# ---------------------------------------------------------------------------

# Columns in a per-marker titration CSV that we want to reuse for A-alone /
# B-alone curves. Anything ending in ``_ratio`` or ``_map_mean`` is a metric
# value; we copy them verbatim and skip everything else (signal, n_guides…).
_METRIC_COL_SUFFIXES = ("_ratio", "_map_mean")


def _is_metric_col(c: str) -> bool:
    return any(c.endswith(s) for s in _METRIC_COL_SUFFIXES)


def _per_marker_csv_path(per_marker_dir: Optional[Path], name: str) -> Optional[Path]:
    """Resolve <per_marker_dir>/<name>/<name>_titration.csv (handles names with
    spaces / special chars by trying the sanitized variant pca_titration uses)."""
    if per_marker_dir is None:
        return None
    per_marker_dir = Path(per_marker_dir)
    candidates = [per_marker_dir / name / f"{name}_titration.csv"]
    try:
        from ops_utils.data.feature_discovery import sanitize_signal_filename
        safe = sanitize_signal_filename(name)[:40]
        if safe != name:
            candidates.append(per_marker_dir / safe / f"{safe}_titration.csv")
    except Exception:
        pass
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_per_marker_titration(
    per_marker_dir: Optional[Path], name: str, target_n_cells: List[int]
) -> Optional[Dict[int, dict]]:
    """Load per-marker titration CSV and interpolate metrics at ``target_n_cells``.

    Returns ``{n_cells: {metric_col: value, ...}}`` with linear interpolation
    in log(N), or ``None`` if no CSV is available. Targets outside the
    per-marker schedule's [min, max] are clipped (np.interp default) — which
    matters for the smaller reporter only at extreme N where its schedule and
    the pair schedule should already match exactly.
    """
    csv_path = _per_marker_csv_path(per_marker_dir, name)
    if csv_path is None:
        return None
    df = pd.read_csv(csv_path)
    if "n_cells" not in df.columns or len(df) < 2:
        return None
    df = df.sort_values("n_cells")
    log_n = np.log(df["n_cells"].astype(float).values)
    metric_cols = [c for c in df.columns if _is_metric_col(c)]
    if not metric_cols:
        return None
    out: Dict[int, dict] = {}
    for target in target_n_cells:
        if target <= 0:
            continue
        log_target = float(np.log(target))
        scores = {
            col: float(np.interp(log_target, log_n, df[col].astype(float).values))
            for col in metric_cols
        }
        out[int(target)] = scores
    logger.info(
        f"  Reusing per-marker {name} from {csv_path.name} "
        f"({len(df)} N points → interpolated at {len(target_n_cells)} pair points)"
    )
    return out


# ---------------------------------------------------------------------------
# Dual-pass PCA helpers
# ---------------------------------------------------------------------------


def _dual_pass_sweep(
    g_concat: ad.AnnData,
    thresholds: List[float],
    _logger,
) -> Tuple[Optional[Dict[str, float]], Dict[str, object]]:
    """Fit PCA on the concatenated NTC-normalized guide matrix and sweep
    variance thresholds, scoring all 4 mAP metrics at each.

    Picks consensus = argmax of normalized (activity + distinctiveness + chad)
    ratios across thresholds with ``n_pcs >= MIN_PCS``. Returns
    ``(scores_at_consensus, sweep_info)`` or ``(None, sweep_info)`` if no
    threshold yields enough PCs to score.
    """
    X_in = np.asarray(g_concat.X, dtype=np.float32)
    X_in = np.nan_to_num(X_in, nan=0.0, posinf=0.0, neginf=0.0)
    n_in = X_in.shape[1]
    X_pcs, cumvar, _ = fit_pca(X_in)

    sweep_rows: List[Dict[str, object]] = []
    per_t_scores: Dict[float, dict] = {}
    for t in thresholds:
        n_pcs = int(n_pcs_for_threshold(cumvar, t))
        skipped = n_pcs < MIN_PCS
        row: Dict[str, object] = {
            "threshold": float(t),
            "n_pcs": n_pcs,
            "n_features_in": n_in,
            "skipped": bool(skipped),
        }
        if skipped:
            sweep_rows.append(row)
            _logger.info(
                f"    [dualpass] {t:.0%}: {n_pcs} PCs — skipped (< MIN_PCS={MIN_PCS})"
            )
            continue
        X_slice = X_pcs[:, :n_pcs].astype(np.float32)
        adata_t = ad.AnnData(
            X=X_slice,
            obs=g_concat.obs.copy(),
            var=pd.DataFrame(index=[f"sPC{j}" for j in range(n_pcs)]),
        )
        scores_t = _score_all_metrics(_prepare_for_copairs(adata_t), _logger)
        per_t_scores[float(t)] = scores_t
        for k, v in scores_t.items():
            if _is_metric_col(k):
                row[k] = v
        sweep_rows.append(row)
        _logger.info(
            f"    [dualpass] {t:.0%}: {n_pcs}/{n_in} PCs — "
            f"act={scores_t['activity_ratio']:.1%} "
            f"dist={scores_t['distinctiveness_ratio']:.1%} "
            f"chad={scores_t['chad_ratio']:.1%}"
        )

    valid = [r for r in sweep_rows if not r.get("skipped")]
    sweep_info: Dict[str, object] = {
        "sweep_rows": sweep_rows,
        "n_features_in": n_in,
        "consensus_t": None,
        "consensus_n": None,
    }
    if not valid:
        _logger.warning("    [dualpass] no threshold met MIN_PCS — skipping consensus pick")
        return None, sweep_info

    valid_df = pd.DataFrame(valid)

    def _norm(col: str) -> np.ndarray:
        v = valid_df[col].astype(float).values
        vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
        return (v - vmin) / (vmax - vmin) if vmax > vmin else np.ones_like(v)

    cscore = (
        _norm("activity_ratio")
        + _norm("distinctiveness_ratio")
        + _norm("chad_ratio")
    )
    best_idx = int(np.argmax(cscore))
    consensus_t = float(valid_df.iloc[best_idx]["threshold"])
    consensus_n = int(valid_df.iloc[best_idx]["n_pcs"])
    sweep_info["consensus_t"] = consensus_t
    sweep_info["consensus_n"] = consensus_n
    _logger.info(
        f"    [dualpass] consensus: {consensus_t:.0%} → {consensus_n}/{n_in} PCs"
    )
    return per_t_scores[consensus_t], sweep_info


def _both_inputs_are_cp(name_a: str, name_b: str) -> bool:
    """CP features are hand-crafted/independent and PCA is destructive at high
    thresholds — pca_optimization uses a 0.30-0.70 sweep grid for them. Fall
    back to that grid only if BOTH reporters are CP (mixed pairs use the full
    grid since the DINO side benefits from high-threshold PCs)."""
    return name_a.lower().startswith("cp") and name_b.lower().startswith("cp")


# ---------------------------------------------------------------------------
# Combination helper
# ---------------------------------------------------------------------------


def _combine_reporters(
    adata_a: ad.AnnData, adata_b: ad.AnnData
) -> Optional[ad.AnnData]:
    """Inner join two guide-level adatas on sgRNA key and concatenate features.

    Both adatas must already be normalized. Rows are aligned by sgRNA (the
    exact guide sequence), giving an unambiguous 1:1 mapping even when random
    subsampling causes slightly different guide sets to survive in each reporter.
    Falls back to ``perturbation`` (gene name) if sgRNA is not available.

    Returns None if fewer than 2 guides are shared.
    """
    import scipy.sparse as sp

    # Choose join key: sgRNA if available (exact guide), else gene-level perturbation
    key = (
        "sgRNA"
        if "sgRNA" in adata_a.obs.columns and "sgRNA" in adata_b.obs.columns
        else "perturbation"
    )

    keys_a = set(adata_a.obs[key])
    keys_b = set(adata_b.obs[key])
    common = sorted(keys_a & keys_b)

    if len(common) < 2:
        logger.warning(
            f"    Only {len(common)} shared {key}s — skipping combined scoring"
        )
        return None

    logger.info(f"    Combining on {key}: {len(common)} shared guides")

    def _filter_sort(adata: ad.AnnData) -> ad.AnnData:
        mask = adata.obs[key].isin(common)
        sub = adata[mask].copy()
        order = sub.obs[key].map({k: i for i, k in enumerate(common)})
        return sub[order.argsort()].copy()

    sub_a = _filter_sort(adata_a)
    sub_b = _filter_sort(adata_b)

    def _to_dense(x):
        return x.toarray() if sp.issparse(x) else np.asarray(x)

    X_combined = np.hstack([_to_dense(sub_a.X), _to_dense(sub_b.X)]).astype(np.float32)

    keep_cols = [
        c for c in ["perturbation", "sgRNA", "n_cells"] if c in sub_a.obs.columns
    ]
    combined = ad.AnnData(X=X_combined, obs=sub_a.obs[keep_cols].copy())
    return combined


# ---------------------------------------------------------------------------
# Core titration function
# ---------------------------------------------------------------------------


def titrate_pair(
    cells_a_path: Path,
    cells_b_path: Path,
    name_a: str,
    name_b: str,
    output_dir: Path,
    norm_method: str = "ntc",
    random_seed: int = 42,
    dual_pass: bool = False,
    sweep_thresholds: Optional[List[float]] = None,
    per_marker_titration_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Run titration for reporter A, reporter B, their raw concatenation, and
    optionally an A+B dual-pass PCA projection.

    At each titration point N, both reporters are independently downsampled to
    N cells, aggregated to guide level, normalized, then combined. A-alone /
    B-alone curves are loaded from existing per-marker titration CSVs (under
    ``per_marker_titration_dir``) when available — these are interpolated in
    log(N) onto the pair schedule — otherwise re-scored from cells.

    With ``dual_pass=True``, the concatenated NTC-normalized guide matrix is
    further reduced by a second PCA whose threshold is chosen by a per-N
    consensus sweep (max of normalized act + dist + chad ratios) over
    ``sweep_thresholds`` (default: ``DEFAULT_SWEEP_THRESHOLDS``).

    Returns ``(df_titration, df_dualpass_sweep)``. The sweep DataFrame is
    ``None`` when ``dual_pass=False``.
    """
    _logger = _init_logger()
    rng = np.random.RandomState(random_seed)

    _logger.info(f"Loading {name_a} from {cells_a_path}")
    adata_a = ad.read_h5ad(cells_a_path)
    _logger.info(f"  {name_a}: {adata_a.n_obs:,} cells, {adata_a.n_vars} PCs")

    _logger.info(f"Loading {name_b} from {cells_b_path}")
    adata_b = ad.read_h5ad(cells_b_path)
    _logger.info(f"  {name_b}: {adata_b.n_obs:,} cells, {adata_b.n_vars} PCs")

    # Drop signal column if present — can interfere with aggregation
    for adata in (adata_a, adata_b):
        if "signal" in adata.obs.columns:
            adata.obs = adata.obs.drop(columns=["signal"])

    # Titration schedule: based on the smaller reporter so N is matched
    min_total = min(adata_a.n_obs, adata_b.n_obs)
    cell_targets: List[int] = []
    n = min_total
    while n >= MIN_CELLS:
        cell_targets.append(int(n))
        n = int(n * DOWNSAMPLE_RATIO)
    if not cell_targets:
        cell_targets = [min_total]
    _logger.info(f"Titration points ({len(cell_targets)}): {cell_targets}")

    # Look up per-marker scores (interpolated to the pair schedule). When
    # available, skip re-scoring A-alone / B-alone at every N.
    pm_a = _load_per_marker_titration(per_marker_titration_dir, name_a, cell_targets)
    pm_b = _load_per_marker_titration(per_marker_titration_dir, name_b, cell_targets)
    if pm_a is None:
        _logger.info(f"  Per-marker {name_a} CSV not found — will rescore from cells")
    if pm_b is None:
        _logger.info(f"  Per-marker {name_b} CSV not found — will rescore from cells")

    if dual_pass:
        if sweep_thresholds is None:
            sweep_thresholds = (
                DEFAULT_SWEEP_THRESHOLDS_CP
                if _both_inputs_are_cp(name_a, name_b)
                else DEFAULT_SWEEP_THRESHOLDS
            )
        _logger.info(
            f"Dual-pass PCA enabled — sweep grid ({len(sweep_thresholds)}): "
            f"{[f'{t:.0%}' for t in sweep_thresholds]}"
        )

    rows: List[dict] = []
    sweep_records: List[dict] = []
    raw_label = f"{name_a}+{name_b}"
    dp_label = f"{name_a}+{name_b} (dualPCA)"

    for target in cell_targets:
        _logger.info(f"\n--- {target:,} cells ---")
        t = time.time()

        # A-alone: reuse per-marker CSV when available
        if pm_a is not None and target in pm_a:
            scores_a = pm_a[target]
            g_a = None
            _logger.info(
                f"  {name_a} (reused): act={scores_a.get('activity_ratio', float('nan')):.1%} "
                f"dist={scores_a.get('distinctiveness_ratio', float('nan')):.1%}"
            )
        else:
            g_a_raw = _subsample_and_aggregate(adata_a, target, rng)
            g_a = normalize_guide_adata(g_a_raw, norm_method)
            scores_a = _score_all_metrics(_prepare_for_copairs(g_a.copy()), _logger)
            _logger.info(
                f"  {name_a}: act={scores_a['activity_ratio']:.1%} "
                f"dist={scores_a['distinctiveness_ratio']:.1%} "
                f"corum={scores_a['corum_ratio']:.1%} "
                f"chad={scores_a['chad_ratio']:.1%}"
            )

        # B-alone: reuse per-marker CSV when available
        if pm_b is not None and target in pm_b:
            scores_b = pm_b[target]
            g_b = None
            _logger.info(
                f"  {name_b} (reused): act={scores_b.get('activity_ratio', float('nan')):.1%} "
                f"dist={scores_b.get('distinctiveness_ratio', float('nan')):.1%}"
            )
        else:
            g_b_raw = _subsample_and_aggregate(adata_b, target, rng)
            g_b = normalize_guide_adata(g_b_raw, norm_method)
            scores_b = _score_all_metrics(_prepare_for_copairs(g_b.copy()), _logger)
            _logger.info(
                f"  {name_b}: act={scores_b['activity_ratio']:.1%} "
                f"dist={scores_b['distinctiveness_ratio']:.1%} "
                f"corum={scores_b['corum_ratio']:.1%} "
                f"chad={scores_b['chad_ratio']:.1%}"
            )

        # Concat needs the actual per-N normalized guide adatas. When per-marker
        # was reused above, materialize them here for the combine step.
        if g_a is None:
            g_a_raw = _subsample_and_aggregate(adata_a, target, rng)
            g_a = normalize_guide_adata(g_a_raw, norm_method)
        if g_b is None:
            g_b_raw = _subsample_and_aggregate(adata_b, target, rng)
            g_b = normalize_guide_adata(g_b_raw, norm_method)

        # Raw concat
        g_combined = _combine_reporters(g_a, g_b)
        if g_combined is not None:
            scores_combined = _score_all_metrics(
                _prepare_for_copairs(g_combined.copy()), _logger
            )
            _logger.info(
                f"  Combined ({len(set(g_combined.obs['perturbation']))} guides, "
                f"{g_combined.n_vars} concat features): "
                f"act={scores_combined['activity_ratio']:.1%} "
                f"dist={scores_combined['distinctiveness_ratio']:.1%} "
                f"corum={scores_combined['corum_ratio']:.1%} "
                f"chad={scores_combined['chad_ratio']:.1%}"
            )
        else:
            scores_combined = {k: math.nan for k in scores_a}

        # Dual-pass PCA on the concat
        scores_dualpass: Optional[Dict[str, float]] = None
        dualpass_meta: Dict[str, object] = {}
        if dual_pass:
            if g_combined is None:
                _logger.warning("    [dualpass] skipping — no shared guides")
            else:
                scores_dp, sweep_info = _dual_pass_sweep(
                    g_combined, sweep_thresholds, _logger
                )
                scores_dualpass = scores_dp
                dualpass_meta = {
                    "consensus_t": sweep_info.get("consensus_t"),
                    "consensus_n": sweep_info.get("consensus_n"),
                    "n_features_in": sweep_info.get("n_features_in"),
                }
                for srow in sweep_info.get("sweep_rows", []):
                    rec = {"n_cells": int(target)}
                    rec.update(srow)
                    rec["is_consensus"] = (
                        srow.get("threshold") == dualpass_meta.get("consensus_t")
                    )
                    sweep_records.append(rec)

        _logger.info(f"  Step time: {time.time()-t:.0f}s")

        rows.append({"n_cells": target, "reporter": name_a, **scores_a})
        rows.append({"n_cells": target, "reporter": name_b, **scores_b})
        rows.append({"n_cells": target, "reporter": raw_label, **scores_combined})
        if dual_pass:
            dp_row = {
                "n_cells": target,
                "reporter": dp_label,
                "dualpass_threshold": dualpass_meta.get("consensus_t"),
                "dualpass_n_pcs": dualpass_meta.get("consensus_n"),
                "dualpass_n_features_in": dualpass_meta.get("n_features_in"),
            }
            if scores_dualpass is not None:
                dp_row.update(scores_dualpass)
            else:
                dp_row.update({k: math.nan for k in scores_a if _is_metric_col(k)})
            rows.append(dp_row)

    df_titration = pd.DataFrame(rows)
    df_sweep = pd.DataFrame(sweep_records) if sweep_records else None
    return df_titration, df_sweep


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Visual style for the three reporters
_REPORTER_STYLES = [
    # (suffix to match in reporter label, color, linestyle, marker, lw)
    (0, "#2166ac", "-", "o", 2.5),  # reporter A alone
    (1, "#d6604d", "--", "s", 2.5),  # reporter B alone
    (2, "#4dac26", "-", "^", 3.5),  # combined
]

_METRIC_TITLES = {
    "activity": "Activity",
    "distinctiveness": "Distinctiveness",
    "corum": "CORUM Consistency",
    "chad": "CHAD Consistency",
}

_RATIO_YLABELS = {
    "activity": "% Active",
    "distinctiveness": "% Distinctive",
    "corum": "% CORUM Consistent",
    "chad": "% CHAD Consistent",
}


def _plot_pair_comparison(
    df: pd.DataFrame,
    name_a: str,
    name_b: str,
    output_dir: Path,
) -> None:
    """Generate 4 figures (one per mAP metric) × 3 scale variants.

    Each figure: 2 panels (ratio | mean_map). Lines drawn for whichever of
    {A alone, B alone, A+B raw concat, A+B dualPCA} are present in ``df``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw_label = f"{name_a}+{name_b}"
    dp_label = f"{name_a}+{name_b} (dualPCA)"
    series_styles = [
        # (label,        color,     linestyle, marker, linewidth)
        (name_a,         "#2166ac", "-",       "o",    2.5),
        (name_b,         "#d6604d", "--",      "s",    2.5),
        (raw_label,      "#4dac26", "-",       "^",    3.0),
        (dp_label,       "#7b3294", "-",       "D",    3.5),
    ]
    # Drop styles for series that have no rows in this df
    present = set(df["reporter"].unique())
    series_styles = [s for s in series_styles if s[0] in present]

    x_all = df["n_cells"].unique()

    for metric in METRICS:
        ratio_col = f"{metric}_ratio"
        map_col = f"{metric}_map_mean"
        metric_title = _METRIC_TITLES[metric]
        ratio_ylabel = _RATIO_YLABELS[metric]

        for scale in ("log10", "linear"):
            fig, (ax_ratio, ax_map) = plt.subplots(1, 2, figsize=(20, 8))

            for rep_label, color, ls, marker, lw in series_styles:
                sub = df[df["reporter"] == rep_label].sort_values("n_cells")
                x = sub["n_cells"].values

                if ratio_col in sub.columns:
                    ax_ratio.plot(
                        x,
                        sub[ratio_col].values * 100,
                        color=color,
                        linestyle=ls,
                        marker=marker,
                        linewidth=lw,
                        markersize=8,
                        label=rep_label,
                    )

                if map_col in sub.columns:
                    ax_map.plot(
                        x,
                        sub[map_col].values,
                        color=color,
                        linestyle=ls,
                        marker=marker,
                        linewidth=lw,
                        markersize=8,
                        label=rep_label,
                    )

            for ax, ylabel, title_suffix in [
                (ax_ratio, f"{ratio_ylabel} (%)", "% Significant"),
                (ax_map, "Mean mAP", "Mean mAP"),
            ]:
                ax.set_xlabel(f"Cells per Reporter ({scale})", fontsize=18)
                ax.set_ylabel(ylabel, fontsize=18)
                ax.set_title(f"{metric_title} — {title_suffix}", fontsize=20)
                ax.tick_params(labelsize=14)
                ax.legend(fontsize=15)
                _apply_x_scale(ax, x_all, scale, tick_fontsize=14)

            has_dp = dp_label in present
            head = "vs ".join(s[0] for s in series_styles)
            fig.suptitle(
                f"{metric_title}: {head}  [{scale}]",
                fontsize=22,
                fontweight="bold",
            )
            fig.tight_layout()

            suffix = "_dualpass" if has_dp else ""
            stem = output_dir / f"{metric}_{scale}{suffix}"
            fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
            fig.savefig(f"{stem}.svg", bbox_inches="tight")
            plt.close(fig)
            logger.info(f"  Saved {stem}.{{png,svg}}")

    # Companion plot for dual-pass: chosen consensus threshold + n_pcs vs N
    if dp_label in present:
        sub = df[df["reporter"] == dp_label].sort_values("n_cells")
        if "dualpass_threshold" in sub.columns and sub["dualpass_threshold"].notna().any():
            fig, (ax_t, ax_n) = plt.subplots(1, 2, figsize=(18, 6))
            x = sub["n_cells"].values
            ax_t.plot(x, sub["dualpass_threshold"].values * 100,
                      color="#7b3294", marker="D", linewidth=3, markersize=8)
            ax_t.set_xlabel("Cells per Reporter (log10)", fontsize=18)
            ax_t.set_ylabel("Consensus threshold (%)", fontsize=18)
            ax_t.set_title("Chosen variance threshold vs N", fontsize=18)
            ax_t.tick_params(labelsize=13)
            _apply_x_scale(ax_t, x_all, "log10", tick_fontsize=13)

            ax_n.plot(x, sub["dualpass_n_pcs"].values,
                      color="#7b3294", marker="D", linewidth=3, markersize=8)
            if "dualpass_n_features_in" in sub.columns:
                ax_n.plot(x, sub["dualpass_n_features_in"].values,
                          color="#888888", linestyle=":", linewidth=2,
                          label="concat features (input dim)")
                ax_n.legend(fontsize=12)
            ax_n.set_xlabel("Cells per Reporter (log10)", fontsize=18)
            ax_n.set_ylabel("# 2nd-pass PCs at consensus", fontsize=18)
            ax_n.set_title("PC count vs N", fontsize=18)
            ax_n.tick_params(labelsize=13)
            _apply_x_scale(ax_n, x_all, "log10", tick_fontsize=13)

            fig.suptitle(
                f"Dual-pass PCA — chosen threshold trajectory ({raw_label})",
                fontsize=20, fontweight="bold",
            )
            fig.tight_layout()
            stem = output_dir / "dualpass_threshold_trajectory"
            fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
            fig.savefig(f"{stem}.svg", bbox_inches="tight")
            plt.close(fig)
            logger.info(f"  Saved {stem}.{{png,svg}}")


# ---------------------------------------------------------------------------
# SLURM job function (top-level so submitit can pickle it)
# ---------------------------------------------------------------------------


def run_pair_titration_job(
    reporter_a: str,
    reporter_b: str,
    name_a: str,
    name_b: str,
    output_dir: str,
    norm_method: str = "ntc",
    seed: int = 42,
    dual_pass: bool = False,
    sweep_thresholds: Optional[List[float]] = None,
    per_marker_titration_dir: Optional[str] = None,
) -> str:
    """Run the full pair titration + plotting as a single SLURM job."""
    import traceback

    _init_logger()
    try:
        path_a = Path(reporter_a)
        path_b = Path(reporter_b)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        csv_path = out / "titration_pair.csv"
        sweep_csv_path = out / "titration_pair_dualpass_sweep.csv"

        pm_dir = Path(per_marker_titration_dir) if per_marker_titration_dir else None
        df, df_sweep = titrate_pair(
            path_a,
            path_b,
            name_a,
            name_b,
            out,
            norm_method,
            seed,
            dual_pass=dual_pass,
            sweep_thresholds=sweep_thresholds,
            per_marker_titration_dir=pm_dir,
        )
        df.to_csv(csv_path, index=False)
        if df_sweep is not None and len(df_sweep):
            df_sweep.to_csv(sweep_csv_path, index=False)
        _plot_pair_comparison(df, name_a, name_b, out)
        sweep_note = f", {len(df_sweep)} sweep rows" if df_sweep is not None else ""
        return f"OK: {len(df)} rows{sweep_note}, plots saved to {out}"
    except Exception as e:
        traceback.print_exc()
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Titration comparison: reporter A alone vs B alone vs A+B combined.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--reporter-a", default=None, help="Path to reporter A *_cells.h5ad"
    )
    parser.add_argument(
        "--reporter-b", default=None, help="Path to reporter B *_cells.h5ad"
    )
    parser.add_argument(
        "--name-a",
        default=None,
        help="Display name for reporter A (default: stem of filename)",
    )
    parser.add_argument(
        "--name-b",
        default=None,
        help="Display name for reporter B (default: stem of filename)",
    )
    parser.add_argument(
        "--all-pairs-with",
        default=None,
        metavar="REFERENCE_H5AD",
        help="Submit one SLURM job per reporter paired with this reference "
        "(e.g. Phase_cells.h5ad). Discovers all *_cells.h5ad in --per-signal-dir.",
    )
    parser.add_argument(
        "--per-signal-dir",
        default=None,
        help="Directory containing *_cells.h5ad files for --all-pairs-with mode.",
    )
    parser.add_argument(
        "-o", "--output-dir", required=True, help="Output directory for CSVs and plots"
    )
    parser.add_argument(
        "--norm-method",
        default="ntc",
        choices=["ntc", "global"],
        help="Normalization method (default: ntc)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for downsampling (default: 42)",
    )
    parser.add_argument(
        "--dual-pass",
        action="store_true",
        help="Also fit a 2nd-pass PCA on the concatenated A+B guide matrix at "
        "each titration point and add a 4th curve for that projection. "
        "Threshold chosen per-N by consensus sweep (max of normalized "
        "act+dist+chad).",
    )
    parser.add_argument(
        "--sweep-thresholds",
        type=str,
        default=None,
        metavar="0.20,0.30,...",
        help="Comma-separated variance thresholds for the dual-pass sweep "
        "(default: DEFAULT_SWEEP_THRESHOLDS, or DEFAULT_SWEEP_THRESHOLDS_CP "
        "when both reporters are CP).",
    )
    parser.add_argument(
        "--per-marker-titration-dir",
        type=str,
        default=None,
        help="Directory containing existing per-reporter titration CSVs "
        "(<dir>/<name>/<name>_titration.csv from pca_titration.py). When set, "
        "A-alone / B-alone curves are interpolated from these instead of "
        "rescored. Defaults to inferring from --reporter-a path "
        "(<variant>/per_signal/X_cells.h5ad → <variant>/titration).",
    )
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Regenerate plots from existing titration_pair.csv without recomputing",
    )
    parser.add_argument(
        "--slurm", action="store_true", help="Submit as a single SLURM job"
    )
    parser.add_argument("--slurm-memory", type=str, default="200GB")
    parser.add_argument(
        "--slurm-time",
        type=int,
        default=60,
        help="SLURM time limit in minutes (default: 60)",
    )
    parser.add_argument("--slurm-cpus", type=int, default=8)
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation prompt"
    )
    args = parser.parse_args()

    _init_logger()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse sweep thresholds (None → defaults inside titrate_pair)
    sweep_thresholds: Optional[List[float]] = None
    if args.sweep_thresholds:
        sweep_thresholds = [
            float(t) for t in args.sweep_thresholds.split(",") if t.strip()
        ]

    def _default_pm_dir(cells_path: Path) -> Optional[str]:
        """``<variant>/per_signal/X_cells.h5ad`` → ``<variant>/titration``."""
        if cells_path.parent.name != "per_signal":
            return None
        candidate = cells_path.parent.parent / "titration"
        return str(candidate) if candidate.exists() else None

    pm_dir_arg: Optional[str] = args.per_marker_titration_dir

    # ------------------------------------------------------------------
    # --all-pairs-with: submit one job per reporter paired with reference
    # ------------------------------------------------------------------
    if args.all_pairs_with:
        from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
        from ops_utils.data.feature_discovery import sanitize_signal_filename

        ref_path = Path(args.all_pairs_with)
        ref_name = args.name_a or ref_path.stem.replace("_cells", "")

        per_signal_dir = (
            Path(args.per_signal_dir) if args.per_signal_dir else ref_path.parent
        )
        other_files = sorted(
            f
            for f in per_signal_dir.glob("*_cells.h5ad")
            if "_sub" not in f.name and f.resolve() != ref_path.resolve()
        )

        slurm_params = {
            "timeout_min": args.slurm_time,
            "mem": args.slurm_memory,
            "cpus_per_task": args.slurm_cpus,
            "slurm_partition": "cpu,gpu",
        }

        pm_dir_resolved = pm_dir_arg or _default_pm_dir(ref_path)

        jobs = []
        for other in other_files:
            other_name = other.stem.replace("_cells", "")
            safe = sanitize_signal_filename(other_name)[:35]
            pair_out = output_dir / safe
            jobs.append(
                {
                    "name": f"titr_{ref_name[:10]}_{safe[:20]}",
                    "func": run_pair_titration_job,
                    "kwargs": {
                        "reporter_a": str(ref_path),
                        "reporter_b": str(other),
                        "name_a": ref_name,
                        "name_b": other_name,
                        "output_dir": str(pair_out),
                        "norm_method": args.norm_method,
                        "seed": args.seed,
                        "dual_pass": args.dual_pass,
                        "sweep_thresholds": sweep_thresholds,
                        "per_marker_titration_dir": pm_dir_resolved,
                    },
                }
            )

        if not args.yes:
            print(f"\nPhase-pairs SLURM submission:")
            print(f"  Reference:   {ref_name}  ({ref_path.name})")
            print(f"  Partners:    {len(jobs)} reporters from {per_signal_dir}")
            print(f"  Output root: {output_dir}")
            print(f"  Dual-pass:   {args.dual_pass}")
            print(f"  Per-marker:  {pm_dir_resolved or '(rescore from cells)'}")
            print(
                f"  Memory: {args.slurm_memory}  |  Time: {args.slurm_time} min  |  CPUs: {args.slurm_cpus}"
            )
            if input("\nSubmit? [y/N] ").strip().lower() != "y":
                print("Cancelled.")
                return

        result = submit_parallel_jobs(
            jobs_to_submit=jobs,
            experiment="pca_titration_phase_pairs",
            slurm_params=slurm_params,
            log_dir="pca_optimization",
            manifest_prefix="pca_titration_phase_pairs",
            wait_for_completion=True,
        )
        if result.get("failed"):
            print(f"\n{len(result['failed'])} jobs failed")
        else:
            print(f"\nAll {len(jobs)} jobs complete → {output_dir}")
        return

    # ------------------------------------------------------------------
    # Single-pair mode
    # ------------------------------------------------------------------
    if not args.reporter_a or not args.reporter_b:
        print("ERROR: provide --reporter-a and --reporter-b, or use --all-pairs-with")
        return

    path_a = Path(args.reporter_a)
    path_b = Path(args.reporter_b)
    name_a = args.name_a or path_a.stem.replace("_cells", "")
    name_b = args.name_b or path_b.stem.replace("_cells", "")

    pm_dir_resolved = pm_dir_arg or _default_pm_dir(path_a)

    csv_path = output_dir / "titration_pair.csv"
    sweep_csv_path = output_dir / "titration_pair_dualpass_sweep.csv"

    if args.replot:
        if not csv_path.exists():
            print(f"ERROR: {csv_path} not found — run without --replot first")
            return
        print(f"--replot: loading {csv_path}")
        df = pd.read_csv(csv_path)
        _plot_pair_comparison(df, name_a, name_b, output_dir)
        print("Done.")
        return

    if args.slurm:
        from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

        slurm_params = {
            "timeout_min": args.slurm_time,
            "mem": args.slurm_memory,
            "cpus_per_task": args.slurm_cpus,
            "slurm_partition": "cpu,gpu",
        }
        jobs = [
            {
                "name": f"titr_pair_{name_a}_{name_b}",
                "func": run_pair_titration_job,
                "kwargs": {
                    "reporter_a": str(path_a),
                    "reporter_b": str(path_b),
                    "name_a": name_a,
                    "name_b": name_b,
                    "output_dir": str(output_dir),
                    "norm_method": args.norm_method,
                    "seed": args.seed,
                    "dual_pass": args.dual_pass,
                    "sweep_thresholds": sweep_thresholds,
                    "per_marker_titration_dir": pm_dir_resolved,
                },
            }
        ]

        if not args.yes:
            print(f"\nPair Titration SLURM Job:")
            print(f"  Reporter A: {name_a}  ({path_a.name})")
            print(f"  Reporter B: {name_b}  ({path_b.name})")
            print(f"  Output:     {output_dir}")
            print(f"  Dual-pass:  {args.dual_pass}")
            print(f"  Per-marker: {pm_dir_resolved or '(rescore from cells)'}")
            print(
                f"  Memory:     {args.slurm_memory}  |  Time: {args.slurm_time} min  |  CPUs: {args.slurm_cpus}"
            )
            if input("\nSubmit? [y/N] ").strip().lower() != "y":
                print("Cancelled.")
                return

        result = submit_parallel_jobs(
            jobs_to_submit=jobs,
            experiment="pca_titration_pair",
            slurm_params=slurm_params,
            log_dir="pca_optimization",
            manifest_prefix="pca_titration_pair",
            wait_for_completion=True,
        )
        if result.get("success"):
            print(f"\nJob submitted: {result.get('base_job_id')}")
        else:
            print("\nJob submission failed!")
        return

    # Local mode
    df, df_sweep = titrate_pair(
        cells_a_path=path_a,
        cells_b_path=path_b,
        name_a=name_a,
        name_b=name_b,
        output_dir=output_dir,
        norm_method=args.norm_method,
        random_seed=args.seed,
        dual_pass=args.dual_pass,
        sweep_thresholds=sweep_thresholds,
        per_marker_titration_dir=Path(pm_dir_resolved) if pm_dir_resolved else None,
    )
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    if df_sweep is not None and len(df_sweep):
        df_sweep.to_csv(sweep_csv_path, index=False)
        print(f"Saved: {sweep_csv_path}")
    print(f"Generating plots → {output_dir}")
    _plot_pair_comparison(df, name_a, name_b, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
