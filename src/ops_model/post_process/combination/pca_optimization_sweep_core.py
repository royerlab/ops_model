"""Sweep + per-signal output helpers for pca_optimization.

These functions form the inner loop of Phase 1 (one signal group at a
time): score a single PC threshold, sweep a list of thresholds picking
the consensus peak, then save the per-signal h5ads + sweep CSV +
diagnostic plot.

Functions exported:

* ``_prepare_for_copairs`` / ``_score_activity_per_threshold`` — scoring
  primitives that wrap copairs/map_scores helpers.
* ``_init_sweep_logger`` — shared logger setup for SLURM workers.
* ``_run_threshold_sweep`` / ``_run_guide_threshold_sweep`` — sweep
  variants for cell-level and guide-level (post-aggregation) inputs.
* ``_save_sweep_outputs`` / ``_save_raw_outputs`` — write the
  per_signal/*.h5ad outputs the rest of the pipeline expects.

In-module references to ``pca_optimization`` globals (``MIN_PCS``,
``CHAD_ANNOTATION_PATH``) are imported lazily inside the worker
functions so this module and its parent can re-import each other at
load time without a circular dependency.
"""

from __future__ import annotations

import logging
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
    phenotypic_consistency_manual_annotation,
    phenotypic_distinctivness,
)
from ops_utils.analysis.pca import n_pcs_for_threshold
from ops_utils.analysis.pca_sweep_plots import plot_pca_sweep


# =============================================================================
# Scoring helper (specific to this sweep — wraps copairs for per-threshold use)
# =============================================================================


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


def _score_activity_per_threshold(
    adata_guide: ad.AnnData, null_size: int = 100_000, distance: str = "cosine"
) -> Tuple[float, float]:
    """Score guide-level AnnData. Returns (active_ratio, auc)."""
    adata_guide = _prepare_for_copairs(adata_guide)
    activity_map, active_ratio = phenotypic_activity_assesment(
        adata_guide,
        plot_results=False,
        null_size=null_size,
        distance=distance,
    )
    auc = compute_auc_score(activity_map)
    return active_ratio, auc


# =============================================================================
# Shared sweep + output helpers (used by both Phase 1a and 1b)
# =============================================================================


def _init_sweep_logger():
    """Common logger setup for SLURM sweep functions."""
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger("copairs").setLevel(logging.WARNING)
    return logging.getLogger("ops_model.post_process.combination.pca_optimization")


def _run_threshold_sweep(
    X_pcs: np.ndarray,
    cumvar: np.ndarray,
    obs_df: pd.DataFrame,
    thresholds: List[float],
    norm_method: str,
    extra_sweep_cols: Dict[str, object],
    _logger,
    distance: str = "cosine",
) -> Optional[Dict]:
    """Sweep variance thresholds, score activity + distinctiveness_all + chad_all at each.

    Returns dict with keys:
        sweep_rows, consensus_t, consensus_n, consensus_r, consensus_a,
        peak_act_t, peak_dist_t, peak_chad_t
    or None if no valid threshold found.

    consensus_t is the threshold maximizing the normalized sum of all 3 mAP ratios.
    """
    from ops_model.post_process.combination.pca_optimization import (
        CHAD_ANNOTATION_PATH,
        MIN_PCS,
    )

    sweep_rows = []

    for threshold in thresholds:
        n_pcs = n_pcs_for_threshold(cumvar, threshold)
        X_slice = X_pcs[:, :n_pcs].astype(np.float32)
        pc_names = [f"PC{j}" for j in range(n_pcs)]

        adata_tmp = ad.AnnData(
            X=X_slice,
            obs=obs_df.copy(),
            var=pd.DataFrame(index=pc_names),
        )
        guide_tmp = aggregate_to_level(
            adata_tmp, level="guide", method="mean", preserve_batch_info=False
        )
        gene_tmp = aggregate_to_level(
            adata_tmp, level="gene", method="mean", preserve_batch_info=False
        )
        del adata_tmp
        guide_tmp.X = np.asarray(guide_tmp.X, dtype=np.float32)
        gene_tmp.X = np.asarray(gene_tmp.X, dtype=np.float32)

        guide_norm = normalize_guide_adata(guide_tmp.copy(), norm_method)
        guide_norm.X = np.asarray(guide_norm.X, dtype=np.float32)
        gene_norm = aggregate_to_level(
            guide_norm, "gene", preserve_batch_info=False, subsample_controls=False
        )
        del guide_tmp, gene_tmp

        r, a = 0.0, 0.0
        dist_all, chad_all = 0.0, 0.0
        try:
            r, a = _score_activity_per_threshold(guide_norm, distance=distance)
            g_cp = _prepare_for_copairs(guide_norm.copy())
            e_cp = _prepare_for_copairs(gene_norm.copy())
            _, dist_all = phenotypic_distinctivness(
                g_cp,
                plot_results=False,
                null_size=100_000,
                distance=distance,
            )
            _, chad_all = phenotypic_consistency_manual_annotation(
                e_cp,
                plot_results=False,
                null_size=100_000,
                cache_similarity=True,
                distance=distance,
                annotation_path=CHAD_ANNOTATION_PATH,
            )
        except Exception as e:
            _logger.warning(f"  Scoring failed at {threshold:.0%}: {e}")
        del guide_norm, gene_norm

        row = {
            "threshold": threshold,
            "n_pcs": n_pcs,
            "activity": r,
            "auc": a,
            "distinctiveness_all": dist_all,
            "chad_all": chad_all,
        }
        row.update(extra_sweep_cols)
        sweep_rows.append(row)

        skip = " [skipped for peak]" if n_pcs < MIN_PCS else ""
        _logger.info(
            f"  {threshold:.0%}: {n_pcs} PCs — act={r:.1%} dist_all={dist_all:.1%} chad_all={chad_all:.1%}{skip}"
        )

    # Only consider thresholds with enough PCs for peak selection
    valid = [row for row in sweep_rows if row["n_pcs"] >= MIN_PCS]
    if not valid:
        return None

    valid_df = pd.DataFrame(valid)

    def _peak_threshold(col):
        idx = valid_df[col].idxmax()
        return valid_df.loc[idx, "threshold"], valid_df.loc[idx, "n_pcs"]

    peak_act_t, _ = _peak_threshold("activity")
    peak_dist_t, _ = _peak_threshold("distinctiveness_all")
    peak_chad_t, _ = _peak_threshold("chad_all")

    # Consensus: normalize each metric to [0,1] across valid thresholds, sum, pick argmax
    def _norm(col):
        vals = valid_df[col].values.astype(float)
        vmin, vmax = vals.min(), vals.max()
        return (vals - vmin) / (vmax - vmin) if vmax > vmin else np.ones_like(vals)

    consensus_scores = (
        _norm("activity") + _norm("distinctiveness_all") + _norm("chad_all")
    )
    best_idx = int(np.argmax(consensus_scores))
    consensus_t = valid_df.iloc[best_idx]["threshold"]
    consensus_n = int(valid_df.iloc[best_idx]["n_pcs"])
    consensus_r = float(valid_df.iloc[best_idx]["activity"])
    consensus_a = float(valid_df.iloc[best_idx]["auc"])

    _logger.info(f"  Peak activity:       {peak_act_t:.0%}")
    _logger.info(f"  Peak distinctiveness:{peak_dist_t:.0%}")
    _logger.info(f"  Peak CHAD:           {peak_chad_t:.0%}")
    _logger.info(
        f"  Consensus peak:      {consensus_t:.0%} ({consensus_n} PCs) → act={consensus_r:.1%}"
    )

    return {
        "sweep_rows": sweep_rows,
        "consensus_t": consensus_t,
        "consensus_n": consensus_n,
        "consensus_r": consensus_r,
        "consensus_a": consensus_a,
        "peak_act_t": peak_act_t,
        "peak_dist_t": peak_dist_t,
        "peak_chad_t": peak_chad_t,
    }


def _run_guide_threshold_sweep(
    X_pcs: np.ndarray,
    cumvar: np.ndarray,
    obs_df: pd.DataFrame,
    thresholds: List[float],
    _logger,
    distance: str = "cosine",
) -> Optional[Dict]:
    """Sweep variance thresholds at *guide* level — input is assumed already
    NTC-normalized (e.g. the output of ``aggregate_channels``). For each
    threshold, slice the second-pass PCs, aggregate to gene, and score
    activity / distinctiveness / chad. No further normalization is applied.

    Returns the same dict shape as :func:`_run_threshold_sweep` so
    :func:`plot_pca_sweep` can be reused.
    """
    from ops_model.post_process.combination.pca_optimization import (
        CHAD_ANNOTATION_PATH,
        MIN_PCS,
    )

    sweep_rows: List[Dict] = []

    for threshold in thresholds:
        n_pcs = n_pcs_for_threshold(cumvar, threshold)
        X_slice = X_pcs[:, :n_pcs].astype(np.float32)
        pc_names = [f"sPC{j}" for j in range(n_pcs)]

        guide_tmp = ad.AnnData(
            X=X_slice,
            obs=obs_df.copy(),
            var=pd.DataFrame(index=pc_names),
        )
        guide_tmp = _prepare_for_copairs(guide_tmp)
        gene_tmp = aggregate_to_level(
            guide_tmp,
            "gene",
            preserve_batch_info=False,
            subsample_controls=False,
        )
        gene_tmp = _prepare_for_copairs(gene_tmp)

        r, a = 0.0, 0.0
        dist_all, chad_all = 0.0, 0.0
        try:
            activity_map, r = phenotypic_activity_assesment(
                guide_tmp,
                plot_results=False,
                null_size=100_000,
                distance=distance,
            )
            a = compute_auc_score(activity_map)
            _, dist_all = phenotypic_distinctivness(
                guide_tmp,
                plot_results=False,
                null_size=100_000,
                distance=distance,
            )
            _, chad_all = phenotypic_consistency_manual_annotation(
                gene_tmp,
                plot_results=False,
                null_size=100_000,
                cache_similarity=True,
                distance=distance,
                annotation_path=CHAD_ANNOTATION_PATH,
            )
        except Exception as exc:
            _logger.warning(f"  Scoring failed at {threshold:.0%}: {exc}")
        del guide_tmp, gene_tmp

        sweep_rows.append(
            {
                "threshold": threshold,
                "n_pcs": n_pcs,
                "activity": float(r),
                "auc": float(a),
                "distinctiveness_all": float(dist_all),
                "chad_all": float(chad_all),
            }
        )
        skip = " [skipped for peak]" if n_pcs < MIN_PCS else ""
        _logger.info(
            f"  {threshold:.0%}: {n_pcs} PCs — "
            f"act={r:.1%} dist_all={dist_all:.1%} chad_all={chad_all:.1%}{skip}"
        )

    valid = [row for row in sweep_rows if row["n_pcs"] >= MIN_PCS]
    if not valid:
        return None
    valid_df = pd.DataFrame(valid)

    def _peak_threshold(col):
        idx = valid_df[col].idxmax()
        return valid_df.loc[idx, "threshold"], int(valid_df.loc[idx, "n_pcs"])

    peak_act_t, _ = _peak_threshold("activity")
    peak_dist_t, _ = _peak_threshold("distinctiveness_all")
    peak_chad_t, _ = _peak_threshold("chad_all")

    def _norm(col):
        vals = valid_df[col].values.astype(float)
        vmin, vmax = vals.min(), vals.max()
        return (vals - vmin) / (vmax - vmin) if vmax > vmin else np.ones_like(vals)

    consensus_scores = (
        _norm("activity") + _norm("distinctiveness_all") + _norm("chad_all")
    )
    best_idx = int(np.argmax(consensus_scores))
    consensus_t = float(valid_df.iloc[best_idx]["threshold"])
    consensus_n = int(valid_df.iloc[best_idx]["n_pcs"])

    _logger.info(f"  Peak activity:        {peak_act_t:.0%}")
    _logger.info(f"  Peak distinctiveness: {peak_dist_t:.0%}")
    _logger.info(f"  Peak CHAD:            {peak_chad_t:.0%}")
    _logger.info(f"  Consensus peak:       {consensus_t:.0%} ({consensus_n} PCs)")

    return {
        "sweep_rows": sweep_rows,
        "consensus_t": consensus_t,
        "consensus_n": consensus_n,
        "peak_act_t": peak_act_t,
        "peak_dist_t": peak_dist_t,
        "peak_chad_t": peak_chad_t,
    }


def _save_sweep_outputs(
    X_pcs: np.ndarray,
    obs_df: pd.DataFrame,
    cumvar: np.ndarray,
    peak_n: int,
    peak_t: float,
    peak_activity_r: float,
    peak_activity_auc: float,
    best_act_t: float,  # kept for back-compat, now unused in plot (use metric_peaks)
    signal: str,
    sweep_rows: List[Dict],
    uns_metadata: Dict[str, object],
    output_dir: Path,
    subdir: str,
    file_prefix: str,
    suptitle: str,
    rng: np.random.RandomState,
    _logger,
    drop_obs_cols: Optional[List[str]] = None,
    fixed_threshold: Optional[float] = None,
    sweep_peak_t: Optional[float] = None,
    metric_peaks: Optional[Dict] = None,
    preserve_batch: bool = False,
    output_suffix: str = "",
    agg_method: str = "mean",
):
    """Build AnnData at peak PCs, save cell subsample, aggregate to guide/gene, write outputs.

    Args:
        drop_obs_cols: Columns to drop from obs before aggregation (e.g. ['experiment'] for copairs).
        preserve_batch: If True, aggregate preserving experiment identity in obs.
        output_suffix: Appended to file_prefix in all output filenames (e.g. '_batch', '_nopca').
        agg_method: aggregation method for cells→guide/gene (default ``mean``).
    """
    X_reduced = X_pcs[:, :peak_n].astype(np.float32)
    pc_names = [f"{signal}_PC{j}" for j in range(peak_n)]

    adata_cells = ad.AnnData(
        X=X_reduced,
        obs=obs_df.copy(),
        var=pd.DataFrame(index=pc_names),
    )

    out_subdir = output_dir / subdir
    out_subdir.mkdir(parents=True, exist_ok=True)

    # Stamp the cell-level adata's uns with the PCA artifacts BEFORE writing,
    # so downstream "reuse this transform" workflows can pull components +
    # mean + feature_names + variance_ratio off the cells.h5ad directly
    # (matches scanpy's `uns["pca"]` convention plus the existing top-level
    # ``pca_components`` / ``pca_feature_names`` fields the rest of this
    # pipeline reads off the guide/gene h5ads).
    cell_pca_uns = {
        "params": {
            "n_components": int(peak_n),
            "threshold": float(peak_t),
            "zero_center": True,
        },
        "variance_ratio": np.diff(np.concatenate([[0.0], cumvar])).astype(np.float32)[:peak_n],
    }
    if "pca_components" in uns_metadata:
        cell_pca_uns["components"] = np.asarray(uns_metadata["pca_components"], dtype=np.float32)
    if "pca_feature_names" in uns_metadata:
        cell_pca_uns["feature_names"] = list(uns_metadata["pca_feature_names"])
    if uns_metadata.get("pca_mean") is not None:
        cell_pca_uns["mean"] = np.asarray(uns_metadata["pca_mean"], dtype=np.float32)
    adata_cells.uns["pca"] = cell_pca_uns
    adata_cells.uns["pca_components"] = cell_pca_uns.get("components")
    if "feature_names" in cell_pca_uns:
        adata_cells.uns["pca_feature_names"] = cell_pca_uns["feature_names"]
    if "mean" in cell_pca_uns:
        adata_cells.uns["pca_mean"] = cell_pca_uns["mean"]
    adata_cells.uns["pca_threshold"] = float(peak_t)
    adata_cells.uns["n_pcs"] = int(peak_n)
    adata_cells.uns["signal"] = signal

    # Save full cell-level h5ad (PCA-reduced) for downstream titration analysis
    adata_cells.obs["signal"] = signal
    adata_cells.write_h5ad(out_subdir / f"{file_prefix}{output_suffix}_cells.h5ad")

    # Save a subsampled cell-level h5ad for cross-signal UMAP/PHATE
    n_sub = min(25000, adata_cells.n_obs)
    sub_idx = rng.choice(adata_cells.n_obs, n_sub, replace=False)
    sub_idx.sort()
    cells_sub = adata_cells[sub_idx].copy()
    # `cells_sub` inherits adata_cells.uns by reference — re-stamp explicitly
    # to be safe (some anndata slicing operations drop/copy uns inconsistently).
    cells_sub.uns.update(adata_cells.uns)
    cells_sub.write_h5ad(out_subdir / f"{file_prefix}{output_suffix}_cells_sub.h5ad")
    del cells_sub
    # Remove signal col before aggregation (not needed and may interfere)
    adata_cells.obs = adata_cells.obs.drop(columns=["signal"])

    # Drop specified columns before aggregation (e.g. 'experiment' for copairs compatibility)
    if drop_obs_cols:
        adata_cells.obs = adata_cells.obs[
            [c for c in adata_cells.obs.columns if c not in drop_obs_cols]
        ]

    g = aggregate_to_level(
        adata_cells, level="guide", method=agg_method, preserve_batch_info=preserve_batch
    )
    e = aggregate_to_level(
        adata_cells, level="gene", method=agg_method, preserve_batch_info=preserve_batch
    )
    del adata_cells
    g.X = np.asarray(g.X, dtype=np.float32)
    e.X = np.asarray(e.X, dtype=np.float32)

    # Store PCA embeddings in obsm (X IS the PCA-reduced space)
    variance_ratio_per_pc = np.diff(np.concatenate([[0.0], cumvar])).astype(np.float32)
    pca_uns = {
        "variance_ratio": variance_ratio_per_pc[:peak_n],
        "params": {
            "n_components": peak_n,
            "threshold": float(peak_t),
            "zero_center": True,
        },
    }
    for adata in [g, e]:
        adata.obsm["X_pca"] = np.asarray(adata.X, dtype=np.float32)
        adata.uns["pca"] = pca_uns

    # Store metadata
    base_uns = {
        "pca_threshold": float(peak_t),
        "n_pcs": int(peak_n),
        "signal": signal,
        "explained_variance": (
            float(cumvar[peak_n - 1]) if peak_n <= len(cumvar) else 1.0
        ),
        "peak_activity": float(peak_activity_r),
        "peak_auc": float(peak_activity_auc),
    }
    base_uns.update(uns_metadata)
    for adata in [g, e]:
        adata.uns.update(base_uns)

    g.write_h5ad(out_subdir / f"{file_prefix}{output_suffix}_guide.h5ad")
    e.write_h5ad(out_subdir / f"{file_prefix}{output_suffix}_gene.h5ad")

    # Save sweep CSV and plot (skipped when sweep was bypassed, e.g. preserve_batch mode)
    if sweep_rows:
        sweep_df = pd.DataFrame(sweep_rows)
        sweep_df.to_csv(
            out_subdir / f"{file_prefix}{output_suffix}_sweep.csv", index=False
        )
        try:
            plot_pca_sweep(
                sweep_df,
                signal,
                peak_t,
                peak_n,
                suptitle=suptitle,
                plots_dir=out_subdir / "plots",
                file_prefix=f"{file_prefix}{output_suffix}",
                fixed_threshold=fixed_threshold,
                sweep_peak_t=sweep_peak_t,
                metric_peaks=metric_peaks,
            )
            _logger.info(
                f"  Saved plot: {subdir}/plots/{file_prefix}{output_suffix}_sweep.png"
            )
        except Exception as plot_err:
            _logger.warning(f"  Plot failed: {plot_err}")


def _save_raw_outputs(
    X_raw: np.ndarray,
    obs_df: pd.DataFrame,
    feature_names: List[str],
    signal: str,
    uns_metadata: Dict[str, object],
    output_dir: Path,
    subdir: str,
    file_prefix: str,
    rng: np.random.RandomState,
    _logger,
    drop_obs_cols: Optional[List[str]] = None,
    preserve_batch: bool = False,
    output_suffix: str = "",
    agg_method: str = "mean",
) -> None:
    """Save cell, guide, and gene h5ads without PCA reduction.

    Used by --no-pca mode. The full feature matrix (post z-score if configured)
    is aggregated directly to guide/gene level without dimensionality reduction.
    No sweep CSV or sweep plot is produced.
    """
    n_feats = X_raw.shape[1]
    X_float = np.asarray(X_raw, dtype=np.float32)

    adata_cells = ad.AnnData(
        X=X_float,
        obs=obs_df.copy(),
        var=pd.DataFrame(index=feature_names),
    )

    out_subdir = output_dir / subdir
    out_subdir.mkdir(parents=True, exist_ok=True)

    adata_cells.obs["signal"] = signal
    adata_cells.write_h5ad(out_subdir / f"{file_prefix}{output_suffix}_cells.h5ad")

    n_sub = min(25000, adata_cells.n_obs)
    sub_idx = rng.choice(adata_cells.n_obs, n_sub, replace=False)
    sub_idx.sort()
    cells_sub = adata_cells[sub_idx].copy()
    cells_sub.write_h5ad(out_subdir / f"{file_prefix}{output_suffix}_cells_sub.h5ad")
    del cells_sub

    adata_cells.obs = adata_cells.obs.drop(columns=["signal"])
    if drop_obs_cols:
        adata_cells.obs = adata_cells.obs[
            [c for c in adata_cells.obs.columns if c not in drop_obs_cols]
        ]

    g = aggregate_to_level(
        adata_cells, level="guide", method=agg_method, preserve_batch_info=preserve_batch
    )
    e = aggregate_to_level(
        adata_cells, level="gene", method=agg_method, preserve_batch_info=preserve_batch
    )
    del adata_cells
    g.X = np.asarray(g.X, dtype=np.float32)
    e.X = np.asarray(e.X, dtype=np.float32)

    base_uns = {
        "pca_applied": False,
        "n_features": int(n_feats),
        "signal": signal,
        "agg_method": agg_method,
    }
    base_uns.update(uns_metadata)
    for adata in [g, e]:
        adata.uns.update(base_uns)

    g.write_h5ad(out_subdir / f"{file_prefix}{output_suffix}_guide.h5ad")
    e.write_h5ad(out_subdir / f"{file_prefix}{output_suffix}_gene.h5ad")
    _logger.info(
        f"  Saved {file_prefix}{output_suffix}_guide/gene.h5ad (no PCA, {n_feats} raw features)"
    )
