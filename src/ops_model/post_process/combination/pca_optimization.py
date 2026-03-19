"""Per-signal pooled PCA optimization & pre-reduction.

Pools cells across experiments sharing the same biological signal, fits PCA, sweeps
variance thresholds to find the optimal number of PCs, then aggregates all signals into
combined guide/gene h5ads and scores 4 phenotypic metrics (activity, distinctiveness,
CORUM consistency, CHAD consistency).

Two-phase SLURM architecture
-----------------------------
Phase 1  One SLURM job per biological signal group -- pool & downsample cells, PCA sweep,
         save per-signal h5ad.  Output → <root>/per_signal/
Phase 2  One aggregation job -- load per-signal h5ads, hconcat, NTC-normalize, score all
         4 metrics (also per-reporter), compute embeddings, save plots.

ROOT = /hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_all

8 variants — feature type × channel subset
------------------------------------------
Each variant produces an independent output subtree and can be compared via
compare_map_scores.py.  Replace --slurm with --aggregate-only --slurm to re-run
Phase 2 only (e.g. after code changes) without redoing the PCA sweeps.

  Variant                  Flags                                    Output subdir
  ──────────────────────── ──────────────────────────────────────── ─────────────────────────
  DINO        all          --slurm                                  dino/all/
  DINO        phase-only   --phase-only --slurm                     dino/phase_only/
  DINO        no-phase     --no-phase --slurm                       dino/no_phase/
  DINO        downsampled  --downsampled --slurm                    dino/downsampled/
  CellProfiler all         --cell-profiler --slurm                  cellprofiler/all/
  CellProfiler phase-only  --phase-only --cell-profiler --slurm     cellprofiler/phase_only/
  CellProfiler no-phase    --no-phase --cell-profiler --slurm       cellprofiler/no_phase/
  CellProfiler downsampled --downsampled --cell-profiler --slurm    cellprofiler/downsampled/

  Channel subsets:
    (default)    all fluorescent + phase channels, all cells pooled per signal group
    --phase-only label-free brightfield (Phase) only
    --no-phase   fluorescent channels only (excludes Phase)
    --downsampled all channels but cells equalised across signal groups (floor 750k/group)

  Append --aggregate-only to re-run Phase 2 only (e.g. after code changes).
  Use run_aggregate_all.sh to submit all 8 --aggregate-only jobs in parallel.

Output structure
----------------
  <root>/
    dino/
      all/          (default)
      phase_only/   (--phase-only)
      no_phase/     (--no-phase)
      downsampled/  (--downsampled)
    cellprofiler/
      all/
      phase_only/
      no_phase/
      downsampled/
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd

from ops_model.features.anndata_utils import (
    aggregate_to_level,
    hconcat_by_perturbation,
    normalize_guide_adata,
    split_ntc_for_embedding,
)
from ops_utils.analysis.embedding_plots import (
    build_metric_lookup,
    clean_X_for_embedding,
    get_perts_col,
    plot_embedding_overlay,
)
from ops_utils.analysis.map_scores import (
    compute_auc_score,
    phenotypic_activity_assesment,
    plot_map_scatter,
)
from ops_utils.analysis.pca import fit_pca, n_pcs_for_threshold
from ops_utils.analysis.pca_sweep_plots import (
    plot_channel_peaks_bar,
    plot_metric_map_bar,
    plot_pca_sweep,
    plot_sweep_curves_summary,
)
from ops_utils.data.feature_discovery import (
    build_signal_groups,
    count_cells_per_signal_group,
    discover_cellprofiler_experiments,
    discover_dino_experiments,
    find_cell_h5ad_path,
    get_channel_maps_path,
    get_storage_roots,
    load_attribution_config,
    load_cell_h5ad,
    resolve_channel_label,
    sanitize_signal_filename,
)
from ops_utils.data.positive_controls import (
    plot_positive_controls_grid,
)

logger = logging.getLogger(__name__)

DEFAULT_SWEEP_THRESHOLDS = [0.60, 0.70, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.88, 0.90, 0.95]
# CellProfiler features are hand-crafted and independent (not redundant like DINO embeddings),
# so PCA is destructive at high thresholds. Optimal region is ~50% variance explained.
DEFAULT_SWEEP_THRESHOLDS_CP = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
MIN_PCS = 10  # Minimum PCs for peak selection (avoids degenerate 1-PC artifact)
PCA_FIT_CAP = 5_000_000  # Cells used to fit PCA axes; larger datasets use passthrough (fit subsample, transform all)


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


def _score_activity_per_threshold(adata_guide: ad.AnnData, null_size: int = 100_000) -> Tuple[float, float]:
    """Score guide-level AnnData. Returns (active_ratio, auc)."""
    adata_guide = _prepare_for_copairs(adata_guide)
    activity_map, active_ratio = phenotypic_activity_assesment(
        adata_guide, plot_results=False, null_size=null_size,
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("copairs").setLevel(logging.WARNING)
    return logging.getLogger(__name__)


def _run_threshold_sweep(
    X_pcs: np.ndarray,
    cumvar: np.ndarray,
    obs_df: pd.DataFrame,
    thresholds: List[float],
    norm_method: str,
    extra_sweep_cols: Dict[str, object],
    _logger,
):
    """Sweep variance thresholds, score activity at each, return results + best peaks.

    Returns:
        (sweep_rows, best_auc_t, best_auc_r, best_auc_a, best_auc_n, best_act_t)
        or None if no valid threshold found.
    """
    best_act_t, best_act_r, best_act_a, best_act_n = None, -1.0, -1.0, 0
    best_auc_t, best_auc_r, best_auc_a, best_auc_n = None, -1.0, -1.0, 0
    sweep_rows = []

    for threshold in thresholds:
        n_pcs = n_pcs_for_threshold(cumvar, threshold)
        X_slice = X_pcs[:, :n_pcs].astype(np.float32)
        pc_names = [f"PC{j}" for j in range(n_pcs)]

        adata_tmp = ad.AnnData(
            X=X_slice, obs=obs_df.copy(), var=pd.DataFrame(index=pc_names),
        )
        guide_tmp = aggregate_to_level(adata_tmp, level="guide", method="mean", preserve_batch_info=False)
        del adata_tmp
        guide_tmp.X = np.asarray(guide_tmp.X, dtype=np.float32)

        guide_norm = normalize_guide_adata(guide_tmp.copy(), norm_method)
        guide_norm.X = np.asarray(guide_norm.X, dtype=np.float32)
        try:
            r, a = _score_activity_per_threshold(guide_norm)
        except Exception as e:
            _logger.warning(f"  Scoring failed at {threshold:.0%}: {e}")
            r, a = 0.0, 0.0
        del guide_tmp, guide_norm

        row = {"threshold": threshold, "n_pcs": n_pcs, "activity": r, "auc": a}
        row.update(extra_sweep_cols)
        sweep_rows.append(row)

        if n_pcs < MIN_PCS:
            _logger.info(f"  {threshold:.0%}: {n_pcs} PCs (< {MIN_PCS}) — {r:.1%}, AUC={a:.4f} [skipped for peak]")
            continue
        _logger.info(f"  {threshold:.0%}: {n_pcs} PCs — {r:.1%}, AUC={a:.4f}")
        if r > best_act_r or (r == best_act_r and a > best_act_a):
            best_act_t, best_act_r, best_act_a, best_act_n = threshold, r, a, n_pcs
        if a > best_auc_a or (a == best_auc_a and r > best_auc_r):
            best_auc_t, best_auc_r, best_auc_a, best_auc_n = threshold, r, a, n_pcs

    if best_act_t is None:
        return None

    _logger.info(f"  Peak (activity): {best_act_t:.0%} ({best_act_n} PCs) → {best_act_r:.1%}, AUC={best_act_a:.4f}")
    _logger.info(f"  Peak (AUC):      {best_auc_t:.0%} ({best_auc_n} PCs) → {best_auc_r:.1%}, AUC={best_auc_a:.4f}")

    return sweep_rows, best_auc_t, best_auc_r, best_auc_a, best_auc_n, best_act_t


def _save_sweep_outputs(
    X_pcs: np.ndarray,
    obs_df: pd.DataFrame,
    cumvar: np.ndarray,
    peak_n: int,
    peak_t: float,
    best_auc_r: float,
    best_auc_a: float,
    best_act_t: float,
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
):
    """Build AnnData at peak PCs, save cell subsample, aggregate to guide/gene, write outputs.

    Args:
        drop_obs_cols: Columns to drop from obs before aggregation (e.g. ['experiment'] for copairs).
    """
    X_reduced = X_pcs[:, :peak_n].astype(np.float32)
    pc_names = [f"{signal}_PC{j}" for j in range(peak_n)]

    adata_cells = ad.AnnData(
        X=X_reduced, obs=obs_df.copy(), var=pd.DataFrame(index=pc_names),
    )

    # Save a subsampled cell-level h5ad for cross-signal UMAP/PHATE
    n_sub = min(25000, adata_cells.n_obs)
    sub_idx = rng.choice(adata_cells.n_obs, n_sub, replace=False)
    sub_idx.sort()
    cells_sub = adata_cells[sub_idx].copy()
    cells_sub.obs["signal"] = signal
    out_subdir = output_dir / subdir
    out_subdir.mkdir(parents=True, exist_ok=True)
    cells_sub.write_h5ad(out_subdir / f"{file_prefix}_cells_sub.h5ad")
    del cells_sub

    # Drop specified columns before aggregation (e.g. 'experiment' for copairs compatibility)
    if drop_obs_cols:
        adata_cells.obs = adata_cells.obs[[c for c in adata_cells.obs.columns if c not in drop_obs_cols]]

    g = aggregate_to_level(adata_cells, level="guide", method="mean", preserve_batch_info=False)
    e = aggregate_to_level(adata_cells, level="gene", method="mean", preserve_batch_info=False)
    del adata_cells
    g.X = np.asarray(g.X, dtype=np.float32)
    e.X = np.asarray(e.X, dtype=np.float32)

    # Store PCA embeddings in obsm (X IS the PCA-reduced space)
    variance_ratio_per_pc = np.diff(np.concatenate([[0.0], cumvar])).astype(np.float32)
    pca_uns = {
        "variance_ratio": variance_ratio_per_pc[:peak_n],
        "params": {"n_components": peak_n, "threshold": float(peak_t), "zero_center": True},
    }
    for adata in [g, e]:
        adata.obsm["X_pca"] = np.asarray(adata.X, dtype=np.float32)
        adata.uns["pca"] = pca_uns

    # Store metadata
    base_uns = {
        "pca_threshold": float(peak_t),
        "n_pcs": int(peak_n),
        "signal": signal,
        "explained_variance": float(cumvar[peak_n - 1]) if peak_n <= len(cumvar) else 1.0,
        "peak_activity": float(best_auc_r),
        "peak_auc": float(best_auc_a),
    }
    base_uns.update(uns_metadata)
    for adata in [g, e]:
        adata.uns.update(base_uns)

    g.write_h5ad(out_subdir / f"{file_prefix}_guide.h5ad")
    e.write_h5ad(out_subdir / f"{file_prefix}_gene.h5ad")

    # Save sweep CSV
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(out_subdir / f"{file_prefix}_sweep.csv", index=False)

    # Save sweep plot
    try:
        plot_pca_sweep(
            sweep_df, signal, peak_t, peak_n, best_act_t,
            suptitle=suptitle,
            plots_dir=out_subdir / "plots",
            file_prefix=file_prefix,
        )
        _logger.info(f"  Saved plot: {subdir}/plots/{file_prefix}_sweep.png")
    except Exception as plot_err:
        _logger.warning(f"  Plot failed: {plot_err}")


# =============================================================================
# Phase 1: Pooled PCA sweep (one SLURM job per biological signal group)
# =============================================================================

def pca_sweep_pooled_signal(
    signal: str,
    exp_channel_pairs: List[Tuple[str, str]],
    output_dir: str,
    target_n_cells: int,
    norm_method: str = "ntc",
    sweep_thresholds: Optional[List[float]] = None,
    random_seed: int = 42,
    feature_dir_override: Optional[str] = None,
) -> str:
    """PCA variance sweep on pooled & downsampled cells for a biological signal.

    Pools cells from multiple experiments that share the same biological signal
    (e.g. "early endosome, EEA1"), downsamples to ``target_n_cells`` using
    proportional sampling, then runs the same PCA sweep as
    :func:`pca_sweep_single_experiment`.

    Used in ``--downsampled`` mode where each signal group gets equal
    representation regardless of how many experiments contribute to it.

    Top-level function (not a method) so submitit can pickle it.

    Saves to output_dir/per_signal/:
      - {signal}_guide.h5ad
      - {signal}_gene.h5ad
      - {signal}_sweep.csv
    """
    _logger = _init_sweep_logger()
    t_start = time.time()
    output_dir = Path(output_dir)
    thresholds = sweep_thresholds or DEFAULT_SWEEP_THRESHOLDS
    rng = np.random.RandomState(random_seed)

    # Resolve storage roots from config
    attr_config = load_attribution_config()
    storage_roots = get_storage_roots(attr_config)
    feature_dir = feature_dir_override or attr_config.get("feature_dir", "dino_features")
    maps_path = get_channel_maps_path()

    n_exps = len(exp_channel_pairs)
    _logger.info(f"Processing signal group: {signal} ({n_exps} experiments, features: {feature_dir})")

    # --- Pass 1: Lightweight pre-scan for cell counts (no full matrix load) ---
    import h5py

    exp_cell_counts = {}  # (exp, ch) -> n_cells
    for exp, ch in exp_channel_pairs:
        cell_file = find_cell_h5ad_path(exp, ch, storage_roots, feature_dir, maps_path)
        if cell_file is not None:
            try:
                with h5py.File(cell_file, "r") as f:
                    exp_cell_counts[(exp, ch)] = f["X"].shape[0]
            except Exception:
                exp_cell_counts[(exp, ch)] = 0

    n_cells_pooled = sum(exp_cell_counts.values())
    if n_cells_pooled == 0:
        return f"FAILED: {signal} — no cell data found for any experiment"

    actual_target = min(target_n_cells, n_cells_pooled)
    _logger.info(f"  Total pooled: {n_cells_pooled:,} cells, target: {actual_target:,}")

    # --- Pass 2: Load one experiment at a time, subsample, collect raw arrays ---
    X_blocks = []       # list of np arrays
    obs_blocks = []     # list of DataFrames
    n_vars_expected = None
    loaded_exps = []
    var_names = None

    t_load = time.time()
    for exp, ch in exp_channel_pairs:
        if (exp, ch) not in exp_cell_counts or exp_cell_counts[(exp, ch)] == 0:
            continue

        adata = load_cell_h5ad(exp, ch, storage_roots, feature_dir, maps_path)
        if adata is None:
            continue

        # Skip experiments with mismatched feature counts
        if n_vars_expected is None:
            n_vars_expected = adata.n_vars
            var_names = list(adata.var_names)
        elif adata.n_vars != n_vars_expected:
            _logger.warning(f"  SKIPPING {exp}/{ch}: {adata.n_vars} features (expected {n_vars_expected}) — feature count mismatch")
            del adata
            continue

        # Proportional subsample: each experiment contributes proportionally to its cell count
        fraction = adata.n_obs / n_cells_pooled
        n_take = max(1, int(round(fraction * actual_target)))
        n_take = min(n_take, adata.n_obs)
        if n_take < adata.n_obs:
            idx = rng.choice(adata.n_obs, n_take, replace=False)
            idx.sort()
            adata = adata[idx].copy()

        # Ensure label_str exists (CellProfiler uses 'perturbation' instead)
        if "label_str" not in adata.obs.columns and "perturbation" in adata.obs.columns:
            adata.obs["label_str"] = adata.obs["perturbation"]
        # Keep obs cols needed for aggregation + track provenance
        keep_cols = [c for c in ["sgRNA", "perturbation", "label_str"] if c in adata.obs.columns]
        obs = adata.obs[keep_cols].copy()
        obs["experiment"] = exp.split("_")[0]

        X_blocks.append(np.asarray(adata.X, dtype=np.float32))
        obs_blocks.append(obs)
        loaded_exps.append(exp)
        _logger.info(f"  {exp.split('_')[0]}/{ch}: {exp_cell_counts[(exp, ch)]:,} → {n_take:,} cells")
        del adata

    _logger.info(f"  Loading complete: {len(loaded_exps)} experiments in {time.time() - t_load:.0f}s")

    if not X_blocks:
        return f"FAILED: {signal} — no cell data found for any experiment"

    # Concatenate using numpy vstack (much faster than ad.concat for uniform features)
    t_concat = time.time()
    _logger.info(f"  Concatenating {len(X_blocks)} blocks ({sum(x.shape[0] for x in X_blocks):,} cells)...")
    X_raw = np.vstack(X_blocks)
    del X_blocks
    obs_df_full = pd.concat(obs_blocks, ignore_index=True)
    del obs_blocks
    _logger.info(f"  Concatenation done in {time.time() - t_concat:.0f}s — {X_raw.shape[0]:,} x {X_raw.shape[1]} matrix")

    n_cells = X_raw.shape[0]
    n_feats = X_raw.shape[1]
    feature_names = var_names
    _logger.info(f"  Pooled: {n_cells_pooled} total cells → {n_cells} pooled ({n_feats} shared features from {len(loaded_exps)} experiments)")

    # Obs DataFrames for scoring (without experiment col which breaks copairs)
    score_cols = [c for c in ["sgRNA", "perturbation", "label_str"] if c in obs_df_full.columns]
    obs_df = obs_df_full[score_cols].copy()

    # Per-experiment z-score before PCA for CellProfiler features (different scales need standardization)
    # Must be per-experiment to avoid batch effects dominating variance across experiments
    if feature_dir_override and "cell-profiler" in feature_dir_override:
        from sklearn.preprocessing import StandardScaler
        experiments = obs_df_full["experiment"].values if "experiment" in obs_df_full.columns else None
        if experiments is not None:
            for exp_id in np.unique(experiments):
                mask = experiments == exp_id
                X_raw[mask] = StandardScaler().fit_transform(X_raw[mask])
            _logger.info(f"  Applied per-experiment z-score scaling (CellProfiler mode, {len(np.unique(experiments))} experiments)")
        else:
            X_raw = StandardScaler().fit_transform(X_raw)
            _logger.info(f"  Applied global z-score scaling (CellProfiler mode, no experiment info)")

    # --- Fit PCA on subsample, transform all cells in chunks ---
    t_pca = time.time()
    n_total = X_raw.shape[0]

    if n_total > PCA_FIT_CAP:
        # Subsample for fit
        fit_idx = rng.choice(n_total, PCA_FIT_CAP, replace=False)
        fit_idx.sort()
        _logger.info(f"  Fitting PCA on {PCA_FIT_CAP:,}/{n_total:,} subsampled cells...")
        _, cumvar, pca_model = fit_pca(X_raw[fit_idx])
        del fit_idx

        # Transform all cells in chunks (avoids 40M x 500 float64 all at once)
        _logger.info(f"  Transforming all {n_total:,} cells in chunks...")
        chunk_size = 2_000_000
        X_pcs_chunks = []
        for i in range(0, n_total, chunk_size):
            chunk = np.asarray(X_raw[i:i + chunk_size], dtype=np.float64)
            chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
            X_pcs_chunks.append(pca_model.transform(chunk).astype(np.float32))
            _logger.info(f"    Transformed chunk {i:,}-{min(i + chunk_size, n_total):,}")
        X_pcs = np.vstack(X_pcs_chunks)
        del X_pcs_chunks
    else:
        _logger.info(f"  Fitting PCA on {n_total:,} x {X_raw.shape[1]} matrix...")
        X_pcs, cumvar, pca_model = fit_pca(X_raw)

    _logger.info(f"  PCA done in {time.time() - t_pca:.0f}s — {X_pcs.shape[1]} components")
    pca_components = pca_model.components_.copy()
    pca_var_ratio  = pca_model.explained_variance_ratio_.copy()
    del X_raw, pca_model

    # Sweep thresholds
    t_sweep = time.time()
    _logger.info(f"  Starting threshold sweep ({len(thresholds)} thresholds)...")
    result = _run_threshold_sweep(
        X_pcs, cumvar, obs_df, thresholds, norm_method,
        extra_sweep_cols={"signal": signal, "n_experiments": n_exps},
        _logger=_logger,
    )
    _logger.info(f"  Sweep done in {time.time() - t_sweep:.0f}s")
    if result is None:
        return f"FAILED: {signal} — no valid threshold found (all < {MIN_PCS} PCs)"
    sweep_rows, best_auc_t, best_auc_r, best_auc_a, best_auc_n, best_act_t = result

    # Save outputs at AUC-optimized peak
    file_prefix = sanitize_signal_filename(signal)
    exps_str = ", ".join(exp.split("_")[0] for exp in loaded_exps[:5])
    if len(loaded_exps) > 5:
        exps_str += f" +{len(loaded_exps)-5} more"

    _save_sweep_outputs(
        X_pcs, obs_df_full, cumvar,
        peak_n=best_auc_n, peak_t=best_auc_t,
        best_auc_r=best_auc_r, best_auc_a=best_auc_a, best_act_t=best_act_t,
        signal=signal, sweep_rows=sweep_rows,
        uns_metadata={
            "experiment": ",".join(exp.split("_")[0] for exp in loaded_exps),
            "channel": ",".join(ch for _, ch in exp_channel_pairs),
            "n_cells": int(n_cells), "n_cells_pooled": int(n_cells_pooled),
            "n_experiments": int(n_exps), "n_features_raw": int(n_feats),
            "pca_components": pca_components[:best_auc_n].tolist(),
            "pca_feature_names": feature_names,
        },
        output_dir=output_dir, subdir="per_signal", file_prefix=file_prefix,
        suptitle=f"{signal} ({n_exps} exps: {exps_str}) — {n_cells:,}/{n_cells_pooled:,} cells, {n_feats} raw features",
        rng=rng, _logger=_logger,
        drop_obs_cols=["experiment"],
    )

    elapsed = time.time() - t_start
    _logger.info(f"  Done: {signal} in {elapsed:.0f}s — {best_auc_n} PCs @ {best_auc_t:.0%}")

    return f"SUCCESS: {signal} — {best_auc_n} PCs @ {best_auc_t:.0%}, {best_auc_r:.1%} active, AUC={best_auc_a:.4f} ({n_exps} exps, {n_cells}/{n_cells_pooled} cells)"


# =============================================================================
# Phase 2: Aggregation sub-steps (used by aggregate_channels)
# =============================================================================

def _score_single_reporter_metrics(g_raw, norm_method, _logger, null_size=10_000):
    """Score all 4 phenotypic metrics for one reporter's guide h5ad.

    Uses a smaller null_size than the aggregate run for speed.
    Returns dict with activity, auc, distinctiveness, corum, chad (NaN on failure).
    """
    import math
    result = {k: math.nan for k in ("activity", "auc", "distinctiveness", "corum", "chad")}
    try:
        from ops_utils.analysis.map_scores import (
            phenotypic_activity_assesment,
            phenotypic_distinctivness,
            phenotypic_consistency_corum,
            phenotypic_consistency_manual_annotation,
        )
        g_norm = normalize_guide_adata(g_raw.copy(), norm_method)
        g_norm = _prepare_for_copairs(g_norm)

        activity_map, active_ratio = phenotypic_activity_assesment(
            g_norm, plot_results=False, null_size=null_size,
        )
        result["activity"] = float(active_ratio)
        result["auc"] = float(compute_auc_score(activity_map))

        _, dist_ratio = phenotypic_distinctivness(
            g_norm, activity_map, plot_results=False, null_size=null_size,
        )
        result["distinctiveness"] = float(dist_ratio)

        e_norm = aggregate_to_level(g_norm, "gene", preserve_batch_info=False, subsample_controls=False)
        e_norm = _prepare_for_copairs(e_norm)

        _, corum_ratio = phenotypic_consistency_corum(
            e_norm, activity_map, plot_results=False, null_size=null_size, cache_similarity=True,
        )
        result["corum"] = float(corum_ratio)

        _, chad_ratio = phenotypic_consistency_manual_annotation(
            e_norm, activity_map, plot_results=False, null_size=null_size, cache_similarity=True,
        )
        result["chad"] = float(chad_ratio)

    except Exception as exc:
        _logger.warning(f"  Per-reporter metrics scoring failed: {exc}")
    return result


def _load_per_unit_blocks(per_unit_dir, norm_method, _logger):
    """Load per-channel/per-signal guide+gene h5ads, return blocks + report rows."""
    guide_files = sorted(per_unit_dir.glob("*_guide.h5ad"))
    if not guide_files:
        return None, None, [], 0

    _logger.info(f"Found {len(guide_files)} per-unit guide files")
    guide_blocks, gene_blocks, report_rows = [], [], []
    total_cells = 0

    for gf in guide_files:
        file_prefix = gf.stem.replace("_guide", "")
        gene_file = per_unit_dir / f"{file_prefix}_gene.h5ad"
        if not gene_file.exists():
            _logger.warning(f"  Skipping {file_prefix}: no gene file")
            continue

        g = ad.read_h5ad(gf)
        sig = g.uns.get("signal", file_prefix)
        if sig == "unknown" or sig.startswith("(unmapped:"):
            _logger.warning(f"  Skipping {file_prefix}: unmapped channel (signal={sig!r})")
            continue

        e = ad.read_h5ad(gene_file)
        guide_blocks.append(g)
        gene_blocks.append(e)
        n_cells = int(g.uns.get("n_cells", 0))
        total_cells += n_cells

        _logger.info(f"  {sig}: scoring all 4 metrics per-reporter...")
        reporter_metrics = _score_single_reporter_metrics(g, norm_method, _logger)

        report_rows.append({
            "experiment": g.uns.get("experiment", ""),
            "channel": g.uns.get("channel", ""),
            "signal": sig,
            "n_cells": n_cells,
            "n_features_raw": int(g.uns.get("n_features_raw", 0)),
            "peak_threshold": float(g.uns.get("pca_threshold", 0)),
            "n_pcs": int(g.uns.get("n_pcs", 0)),
            "explained_variance": float(g.uns.get("explained_variance", 0)),
            "activity":        reporter_metrics["activity"],
            "auc":             reporter_metrics["auc"],
            "distinctiveness": reporter_metrics["distinctiveness"],
            "corum":           reporter_metrics["corum"],
            "chad":            reporter_metrics["chad"],
        })
        _logger.info(
            f"  {sig}: {g.n_obs} guides x {g.n_vars} PCs @ {g.uns.get('pca_threshold', '?')} | "
            f"act={reporter_metrics['activity']:.1%} dist={reporter_metrics['distinctiveness']:.1%} "
            f"corum={reporter_metrics['corum']:.1%} chad={reporter_metrics['chad']:.1%}"
        )

    return guide_blocks or None, gene_blocks, report_rows, total_cells


def _concat_and_normalize(guide_blocks, gene_blocks, norm_method, _logger):
    """Horizontal concat, NTC normalize, re-aggregate to gene, strip obs for copairs."""
    adata_guide = hconcat_by_perturbation(guide_blocks, "guide")
    adata_gene = hconcat_by_perturbation(gene_blocks, "gene")
    del guide_blocks, gene_blocks

    _logger.info(f"Concatenated: {adata_guide.n_obs} guides, {adata_guide.n_vars} features")
    _logger.info(f"NTC normalizing at guide level...")
    adata_guide = normalize_guide_adata(adata_guide, norm_method)
    adata_guide.X = np.asarray(adata_guide.X, dtype=np.float32)

    adata_gene = aggregate_to_level(
        adata_guide, "gene", preserve_batch_info=False, subsample_controls=False,
    )
    _logger.info(f"  Guide: {adata_guide.n_obs} obs, {adata_guide.n_vars} features")
    _logger.info(f"  Gene: {adata_gene.n_obs} obs, {adata_gene.n_vars} features")

    # Strip obs to copairs-required columns (extra string cols cause isnan error)
    adata_guide = _prepare_for_copairs(adata_guide)
    adata_gene = _prepare_for_copairs(adata_gene)

    return adata_guide, adata_gene


def _score_activity_aggregated(adata_guide, metrics_dir, _logger):
    """Run phenotypic activity scoring on aggregated data. Returns (activity_map, ratio, auc)."""
    _logger.info(f"Running activity scoring...")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    try:
        activity_map, active_ratio = phenotypic_activity_assesment(
            adata_guide, plot_results=False, null_size=100_000,
        )
        activity_map.to_csv(metrics_dir / "phenotypic_activity.csv", index=False)
        auc = compute_auc_score(activity_map)
        _logger.info(f"  Activity: {active_ratio:.1%} active, AUC={auc:.4f}")

        inactive = activity_map[~activity_map["below_corrected_p"]]
        inactive = inactive[inactive["perturbation"] != "NTC"]
        if len(inactive) > 0:
            names = sorted(inactive["perturbation"].tolist())
            _logger.info(f"  Inactive genes ({len(names)}): {', '.join(names)}")
        else:
            _logger.info(f"  All non-NTC perturbations are active")
        return activity_map, active_ratio, auc
    except Exception as exc:
        _logger.error(f"  Activity scoring failed: {exc}")
        return None, 0.0, 0.0


def _save_aggregated_h5ads(adata_guide, adata_gene, report_rows, output_dir,
                           r, a, norm_method, total_cells, _logger):
    """Write guide/gene h5ads and report CSV with metadata."""
    adata_guide.uns["pca_optimized"] = True
    adata_guide.uns["pca_report"] = pd.DataFrame(report_rows).to_dict(orient="list")
    adata_guide.uns["baseline_activity"] = float(r)
    adata_guide.uns["baseline_auc"] = float(a)
    adata_guide.uns["norm_method"] = norm_method
    adata_guide.uns["total_cells"] = total_cells

    adata_gene.uns["pca_optimized"] = True
    adata_gene.uns["norm_method"] = norm_method

    adata_guide.write_h5ad(output_dir / "guide_pca_optimized.h5ad")
    adata_gene.write_h5ad(output_dir / "gene_pca_optimized.h5ad")
    pd.DataFrame(report_rows).to_csv(output_dir / "pca_report.csv", index=False)
    _logger.info(f"  Saved guide_pca_optimized.h5ad, gene_pca_optimized.h5ad, pca_report.csv")


def _compute_and_plot_embeddings(adata_guide, metric_lookup, plots_dir, plt, _logger):
    """Compute UMAP + PHATE embeddings for guide/gene levels, plot overlays + positive controls.

    Returns adata_gene_embed with embeddings stored in obsm — caller should save it.
    """
    adata_gene_embed = split_ntc_for_embedding(adata_guide, random_seed=42)
    _logger.info(f"  Gene (NTC-split for embedding): {adata_gene_embed.n_obs} obs")
    # Propagate X_pca and pca uns from guide (same feature space, gene-level aggregation)
    adata_gene_embed.obsm["X_pca"] = np.asarray(adata_gene_embed.X, dtype=np.float32)
    if "pca" in adata_guide.uns:
        adata_gene_embed.uns["pca"] = adata_guide.uns["pca"]

    embed_pairs = [("guide", adata_guide), ("gene", adata_gene_embed)]
    level_embeddings = {}
    level_perts = {}

    def _make_embedder(name):
        """Return fit_fn(X, n_obs) -> (coords, params_dict) or None if library missing."""
        if name == "UMAP":
            from umap import UMAP
            def _fit(X, n_obs):
                nn = min(15, n_obs - 1)
                if nn < 2:
                    return None, {}
                model = UMAP(n_components=2, n_neighbors=nn, random_state=42)
                coords = model.fit_transform(X)
                params = {"n_neighbors": nn, "random_state": 42, "metric": "euclidean",
                          "a": float(getattr(model, "a_", None) or getattr(model, "_a", None) or 0),
                          "b": float(getattr(model, "b_", None) or getattr(model, "_b", None) or 0)}
                return coords, params
            return _fit
        elif name == "PHATE":
            import phate
            def _fit(X, n_obs):
                knn = min(15 if n_obs > 2000 else 10, n_obs - 1)
                if knn < 2:
                    return None, {}
                coords = phate.PHATE(
                    n_components=2, knn=knn, decay=15, t="auto",
                    n_jobs=-1, random_state=42, verbose=0,
                ).fit_transform(X)
                params = {"knn": knn, "decay": 15, "t": "auto", "random_state": 42}
                return coords, params
            return _fit

    for embed_name, pkg_hint in [("UMAP", "umap-learn"), ("PHATE", "phate")]:
        try:
            fit_fn = _make_embedder(embed_name)
        except ImportError:
            _logger.warning(f"  {embed_name} plots skipped: install {pkg_hint}")
            continue
        try:
            for level_name, adata_level in embed_pairs:
                _logger.info(f"  Computing {level_name} {embed_name} ({adata_level.n_obs} obs, {adata_level.n_vars} features)...")
                X_clean = clean_X_for_embedding(adata_level)
                coords, embed_params = fit_fn(X_clean, adata_level.n_obs)
                if coords is None:
                    _logger.warning(f"  {embed_name} skipped for {level_name}: too few observations")
                    continue
                perts = get_perts_col(adata_level)
                level_embeddings.setdefault(level_name, {})[embed_name] = coords
                level_perts[level_name] = perts
                # Store embedding in obsm/uns (guide level is adata_guide, passed by ref)
                obsm_key = f"X_{embed_name.lower()}"
                adata_level.obsm[obsm_key] = coords.astype(np.float32)
                adata_level.uns[embed_name.lower()] = {"params": embed_params}
                fname = plot_embedding_overlay(
                    coords, perts, metric_lookup, level_name, embed_name,
                    plots_dir, adata_level.n_obs, adata_level.n_vars, plt,
                )
                _logger.info(f"  Saved plots/{fname}")
        except Exception as err:
            _logger.warning(f"  {embed_name} plots failed: {err}")

    # Positive controls overlay grid (CHAD v4)
    for level_name in ["gene"]:
        if level_name in level_embeddings and level_name in level_perts:
            try:
                plot_positive_controls_grid(
                    level_embeddings[level_name], level_perts[level_name],
                    level_name, plots_dir, plt,
                )
            except Exception as pc_err:
                _logger.warning(f"  Positive controls grid failed for {level_name}: {pc_err}")

    return adata_gene_embed


def _score_distinctiveness(adata_guide, activity_map, r, total_feats, plots_dir, metrics_dir, plt, _logger):
    """Run phenotypic distinctiveness scoring, save CSV and plots. Returns (distinctiveness_map, ratio) or (None, 0)."""
    if activity_map is None:
        return None, 0.0
    try:
        from ops_utils.analysis.map_scores import phenotypic_distinctivness
        _logger.info(f"Running distinctiveness...")
        distinctiveness_map, distinctive_ratio = phenotypic_distinctivness(
            adata_guide, activity_map, plot_results=False, null_size=100_000,
        )
        distinctiveness_map.to_csv(metrics_dir / "phenotypic_distinctiveness.csv", index=False)
        _logger.info(f"  Distinctiveness: {distinctive_ratio:.1%}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        plot_map_scatter(ax1, activity_map, "Activity", r, show_ntc=False)
        plot_map_scatter(ax2, distinctiveness_map, "Distinctiveness", distinctive_ratio, show_ntc=False)
        fig.suptitle(f"Activity & Distinctiveness — {total_feats} features", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(plots_dir / "map_activity_distinctiveness.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        _logger.info(f"  Saved plots/map_activity_distinctiveness.png")
        return distinctiveness_map, distinctive_ratio
    except Exception as exc:
        _logger.error(f"  Distinctiveness failed: {exc}")
        return None, 0.0


def _score_consistency(adata_gene, activity_map, total_feats, plots_dir, metrics_dir, plt, _logger):
    """Run CORUM + CHAD consistency scoring, save CSVs and plots.

    Returns (corum_map, corum_ratio, chad_map, chad_ratio) or (None, 0, None, 0) on failure.
    """
    if activity_map is None:
        return None, 0.0, None, 0.0
    try:
        from ops_utils.analysis.map_scores import (
            phenotypic_consistency_corum,
            phenotypic_consistency_manual_annotation,
        )

        _logger.info(f"Running CORUM consistency...")
        consistency_corum_map, consistency_corum_ratio = phenotypic_consistency_corum(
            adata_gene, activity_map, plot_results=False, null_size=100_000,
            cache_similarity=True,
        )
        consistency_corum_map.to_csv(metrics_dir / "phenotypic_consistency_corum.csv", index=False)
        _logger.info(f"  CORUM: {consistency_corum_ratio:.1%}")

        _logger.info(f"Running CHAD consistency...")
        consistency_manual_map, consistency_manual_ratio = phenotypic_consistency_manual_annotation(
            adata_gene, activity_map, plot_results=False, null_size=100_000,
            cache_similarity=True,
        )
        consistency_manual_map.to_csv(metrics_dir / "phenotypic_consistency_manual.csv", index=False)
        _logger.info(f"  Manual (CHAD): {consistency_manual_ratio:.1%}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        plot_map_scatter(ax1, consistency_corum_map, "Consistency (CORUM)", consistency_corum_ratio, show_ntc=False)
        plot_map_scatter(ax2, consistency_manual_map, "Consistency (CHAD)", consistency_manual_ratio, show_ntc=False)
        fig.suptitle(f"Consistency Metrics — {total_feats} features", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(plots_dir / "map_consistency.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        _logger.info(f"  Saved plots/map_consistency.png")
        return consistency_corum_map, consistency_corum_ratio, consistency_manual_map, consistency_manual_ratio
    except Exception as exc:
        _logger.error(f"  Consistency metrics failed: {exc}")
        return None, 0.0, None, 0.0


# =============================================================================
# Phase 2: Aggregation (top-level entry point)
# =============================================================================

def aggregate_channels(
    output_dir: str,
    norm_method: str = "ntc",
    per_unit_subdir: str = "per_channel",
) -> str:
    """Load per-channel (or per-signal) h5ads, concatenate, normalize, score, save.

    Top-level function (not a method) so submitit can pickle it.

    Args:
        per_unit_subdir: subdirectory containing guide/gene h5ads.
            "per_channel" for standard mode, "per_signal" for downsampled mode.
    """
    _logger = _init_sweep_logger()
    t_start = time.time()
    output_dir = Path(output_dir)
    per_unit_dir = output_dir / per_unit_subdir

    # Step 1: Load per-channel/per-signal blocks
    guide_blocks, gene_blocks, report_rows, total_cells = _load_per_unit_blocks(per_unit_dir, norm_method, _logger)
    if guide_blocks is None:
        return "FAILED: no valid per-channel data loaded"

    # Step 2: Concat + normalize
    adata_guide, adata_gene = _concat_and_normalize(guide_blocks, gene_blocks, norm_method, _logger)
    total_feats = adata_guide.n_vars

    # Step 3: Activity scoring
    metrics_dir = output_dir / "metrics"
    activity_map, r, a = _score_activity_aggregated(adata_guide, metrics_dir, _logger)

    # Step 4: Save h5ads (before slow metrics) — store X_pca now; UMAP/PHATE added after step 6
    adata_guide.obsm["X_pca"] = np.asarray(adata_guide.X, dtype=np.float32)
    variance_ratio_per_pc = np.array([
        float(row.get("explained_variance", 0)) for row in report_rows
    ], dtype=np.float32)
    adata_guide.uns["pca"] = {"params": {"n_components": total_feats, "zero_center": True}}
    _save_aggregated_h5ads(adata_guide, adata_gene, report_rows, output_dir,
                           r, a, norm_method, total_cells, _logger)

    # Step 5: Plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if activity_map is not None:
        try:
            fig, ax = plt.subplots(figsize=(8, 7))
            plot_map_scatter(ax, activity_map, "Activity", r, show_ntc=False)
            fig.suptitle(f"Phenotypic Activity — {total_feats} features", fontsize=13, fontweight="bold")
            fig.tight_layout()
            fig.savefig(plots_dir / "map_activity.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            _logger.info(f"  Saved plots/map_activity.png")
        except Exception as e:
            _logger.warning(f"  Activity plot failed: {e}")

    plot_sweep_curves_summary(per_unit_dir, output_dir, plots_dir, r, a, plt, _logger, min_pcs=MIN_PCS)
    plot_channel_peaks_bar(report_rows, r, plots_dir, plt, _logger)

    # Step 6: Embeddings (UMAP + PHATE) — stored directly into adata_guide.obsm/uns
    # Returns adata_gene_embed (NTC-split gene-level object) with embeddings in obsm
    metric_lookup = build_metric_lookup(activity_map)
    adata_gene_embed = _compute_and_plot_embeddings(adata_guide, metric_lookup, plots_dir, plt, _logger)
    # Re-save guide + gene-embed with embeddings now populated
    adata_guide.write_h5ad(output_dir / "guide_pca_optimized.h5ad")
    if adata_gene_embed is not None:
        adata_gene_embed.write_h5ad(output_dir / "gene_embedding_pca_optimized.h5ad")
    _logger.info("  Re-saved guide_pca_optimized.h5ad + gene_embedding_pca_optimized.h5ad with embeddings")

    # Step 7: Distinctiveness + consistency (slow)
    dist_map, dist_ratio = _score_distinctiveness(adata_guide, activity_map, r, total_feats, plots_dir, metrics_dir, plt, _logger)
    corum_map, corum_ratio, chad_map, chad_ratio = _score_consistency(adata_gene, activity_map, total_feats, plots_dir, metrics_dir, plt, _logger)

    # Bar charts for all 4 metrics (per-perturbation mAP)
    plot_metric_map_bar(activity_map, "Activity", "perturbation", r, plots_dir, plt, _logger)
    plot_metric_map_bar(dist_map, "Distinctiveness", "perturbation", dist_ratio, plots_dir, plt, _logger)
    if corum_map is not None:
        corum_entity_col = "complex_id" if "complex_id" in corum_map.columns else corum_map.columns[0]
        plot_metric_map_bar(corum_map, "Consistency (CORUM)", corum_entity_col, corum_ratio, plots_dir, plt, _logger)
    if chad_map is not None:
        chad_entity_col = "complex_num" if "complex_num" in chad_map.columns else chad_map.columns[0]
        plot_metric_map_bar(chad_map, "Consistency (CHAD)", chad_entity_col, chad_ratio, plots_dir, plt, _logger)

    elapsed = time.time() - t_start
    _logger.info(f"\nDone in {elapsed/60:.1f} minutes")
    _logger.info(f"  {len(report_rows)} channels, {total_feats} total PCA features")
    _logger.info(f"  Baseline: {r:.1%} active, AUC={a:.4f}")

    return f"SUCCESS: {total_feats} features, {r:.1%} active, AUC={a:.4f}"


# =============================================================================
# CLI: mode handlers
# =============================================================================

def _discover_experiment_pairs(cp_override):
    """Common experiment discovery for SLURM modes. Returns (all_pairs, attr_config, storage_roots, feature_dir, maps_path)."""
    attr_config = load_attribution_config()
    storage_roots = get_storage_roots(attr_config)
    feature_dir = cp_override or attr_config.get("feature_dir", "dino_features")
    maps_path = get_channel_maps_path()
    if cp_override:
        all_pairs = discover_cellprofiler_experiments(storage_roots)
    else:
        all_pairs = discover_dino_experiments(storage_roots, feature_dir)
    return all_pairs, attr_config, storage_roots, feature_dir, maps_path


def _make_slurm_params(args):
    """Build standard SLURM params dict from parsed args."""
    return {
        "timeout_min": args.slurm_time,
        "mem": args.slurm_memory,
        "cpus_per_task": args.slurm_cpus,
        "slurm_partition": args.slurm_partition,
    }


def _make_agg_slurm_params(args):
    """Build aggregation SLURM params dict from parsed args."""
    return {
        "timeout_min": args.slurm_agg_time,
        "mem": args.slurm_agg_memory,
        "cpus_per_task": args.slurm_cpus,
        "slurm_partition": args.slurm_partition,
    }


def _submit_aggregation_slurm(agg_output, norm_method, per_unit_subdir, agg_slurm_params,
                               experiment_name, manifest_prefix):
    """Submit a single aggregation SLURM job."""
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    agg_jobs = [{
        "name": f"{manifest_prefix}_aggregate",
        "func": aggregate_channels,
        "kwargs": {
            "output_dir": agg_output,
            "norm_method": norm_method,
            "per_unit_subdir": per_unit_subdir,
        },
    }]
    agg_result = submit_parallel_jobs(
        jobs_to_submit=agg_jobs,
        experiment=experiment_name,
        slurm_params=agg_slurm_params,
        log_dir="pca_optimization",
        manifest_prefix=manifest_prefix,
        wait_for_completion=True,
    )
    if agg_result.get("failed"):
        print("Aggregation FAILED")
    else:
        print("Aggregation complete")


def _submit_phase1_slurm(jobs, args, agg_output, per_unit_subdir,
                          experiment_name, manifest_prefix, unit_label):
    """Submit Phase 1 SLURM jobs + auto-chain Phase 2 aggregation on completion."""
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

    slurm_params = _make_slurm_params(args)
    agg_slurm_params = _make_agg_slurm_params(args)

    def _on_phase1_complete(submitted_jobs, experiment):
        print(f"\nAll {unit_label} jobs complete. Submitting aggregation SLURM job...")
        _submit_aggregation_slurm(
            agg_output, args.norm_method, per_unit_subdir,
            agg_slurm_params, f"{manifest_prefix}_aggregation", f"{manifest_prefix}_agg",
        )

    print(f"\nSubmitting {len(jobs)} {unit_label} SLURM jobs...")
    result = submit_parallel_jobs(
        jobs_to_submit=jobs, experiment=experiment_name,
        slurm_params=slurm_params, log_dir="pca_optimization",
        manifest_prefix=f"{manifest_prefix}_opt", wait_for_completion=True,
        post_completion_callback=_on_phase1_complete,
    )
    if result.get("failed"):
        print(f"\nWarning: {len(result['failed'])} {unit_label} failed")
        for name in result["failed"]:
            print(f"  - {name}")


def _handle_umap_only(args, output_dir):
    """Generate UMAP + PHATE embedding plots from existing optimized h5ads."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _logger = logging.getLogger(__name__)

    umap_dir = output_dir
    guide_path = umap_dir / "guide_pca_optimized.h5ad"
    if not guide_path.exists():
        print(f"ERROR: {guide_path} not found. Run --aggregate-only first.")
        return

    _logger.info(f"Loading {guide_path}...")
    adata_guide = ad.read_h5ad(guide_path)

    # Load activity metrics for coloring
    activity_csv = umap_dir / "metrics" / "phenotypic_activity.csv"
    if activity_csv.exists():
        activity_map = pd.read_csv(activity_csv)
        metric_lookup = build_metric_lookup(activity_map)
        _logger.info(f"  Loaded activity metrics for {len(metric_lookup)} perturbations")
    else:
        metric_lookup = {}
        _logger.warning(f"  No activity CSV found at {activity_csv}, UMAPs will be uncolored")

    plots_dir = umap_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _compute_and_plot_embeddings(adata_guide, metric_lookup, plots_dir, plt, _logger)
    print("SUCCESS: Embedding plots saved")


def _handle_aggregate_only(args, output_dir):
    """Run only the aggregation step (Phase 2)."""
    agg_output = str(output_dir)
    agg_subdir = "per_signal"

    if args.slurm:
        print(f"Submitting aggregation as SLURM job ({args.slurm_agg_memory}, {args.slurm_agg_time}min)...")
        if args.downsampled or getattr(args, "all_cells", False):
            print(f"  Mode: signal-group (reading from {agg_output}/per_signal/)")
        _submit_aggregation_slurm(
            agg_output, args.norm_method, agg_subdir,
            _make_agg_slurm_params(args), "pca_aggregation", "pca_agg",
        )
    else:
        result = aggregate_channels(output_dir=agg_output, norm_method=args.norm_method, per_unit_subdir=agg_subdir)
        print(result)


def _handle_downsampled(args, output_dir, cp_override):
    """Pool cells by signal group, downsample, PCA sweep (local or SLURM)."""
    ds_output_dir = output_dir
    ds_output_dir.mkdir(parents=True, exist_ok=True)

    all_pairs, attr_config, storage_roots, feature_dir, maps_path = _discover_experiment_pairs(cp_override)
    if not all_pairs:
        print("No experiment-channel pairs found!")
        return

    from ops_utils.data.feature_metadata import FeatureMetadata
    fm = FeatureMetadata(metadata_path=maps_path)
    signal_groups = build_signal_groups(all_pairs, fm)

    # Apply phase filter (--phase-only / --no-phase) when running in pooled signal mode
    phase_filter = getattr(args, "phase_filter", None)
    if phase_filter == "phase_only":
        signal_groups = {s: p for s, p in signal_groups.items() if s == "Phase"}
        if not signal_groups:
            print("ERROR: --phase-only found no Phase signal group in the discovered channels.")
            return
    elif phase_filter == "no_phase":
        signal_groups = {s: p for s, p in signal_groups.items() if s != "Phase"}

    n_signals = len(signal_groups)

    mode_label = "CellProfiler" if cp_override else ("Downsampled" if args.downsampled else "All-Cells")
    print(f"\n{mode_label} PCA Optimization: {len(all_pairs)} channels -> {n_signals} signal groups")
    print(f"Output: {ds_output_dir}")

    # Pre-scan cell counts and compute per-signal target
    print("\nPre-scanning cell counts per signal group...")
    cell_counts = count_cells_per_signal_group(signal_groups, storage_roots, feature_dir, maps_path)

    if args.downsampled:
        # Equalise: all groups target the smallest group (floor 750k)
        MIN_CELLS_FLOOR = 750_000
        global_target = max(min(cell_counts.values()), MIN_CELLS_FLOOR)
        per_signal_target = {s: global_target for s in cell_counts}
        small_groups = {s: n for s, n in cell_counts.items() if n < global_target}
        if small_groups:
            print(f"\n  {len(small_groups)} signal group(s) have fewer than {global_target:,} cells (will use all available):")
            for s, n in sorted(small_groups.items(), key=lambda x: x[1]):
                print(f"    {s}: {n:,} cells")
        print(f"\nSignal group manifest (downsampling all to {global_target:,} cells):")
        print(f"  {'Signal':<45} {'Exps':>5} {'Cells':>10} {'-> Downsampled':>15}")
        print(f"  {'-'*45} {'-'*5} {'-'*10} {'-'*15}")
    else:
        # All-cells: load every cell; pca_sweep_pooled_signal uses passthrough PCA for >5M
        per_signal_target = dict(cell_counts)
        print(f"\nSignal group manifest (all cells — PCA fit subsampled at >{PCA_FIT_CAP:,}):")
        print(f"  {'Signal':<45} {'Exps':>5} {'Cells':>10}")
        print(f"  {'-'*45} {'-'*5} {'-'*10}")

    manifest_rows = []
    for signal in sorted(signal_groups.keys()):
        pairs = signal_groups[signal]
        n_cells = cell_counts[signal]
        t = per_signal_target[signal]
        if args.downsampled:
            print(f"  {signal:<45} {len(pairs):>5} {n_cells:>10,} -> {t:>15,}")
        else:
            pca_note = f" (passthrough PCA)" if n_cells > PCA_FIT_CAP else ""
            print(f"  {signal:<45} {len(pairs):>5} {n_cells:>10,}{pca_note}")
        manifest_rows.append({
            "signal": signal, "n_experiments": len(pairs),
            "n_cells_pooled": n_cells, "n_cells_used": t,
            "experiments": ",".join(e.split("_")[0] for e, c in pairs),
        })
    print(f"\n  Total: {n_signals} signal groups, {sum(cell_counts.values()):,} total cells")
    pd.DataFrame(manifest_rows).to_csv(ds_output_dir / "downsampled_manifest.csv", index=False)

    # Build common kwargs for signal-group jobs
    def _signal_job_kwargs(signal, pairs):
        kwargs = dict(
            signal=signal, exp_channel_pairs=pairs,
            output_dir=str(ds_output_dir), target_n_cells=per_signal_target[signal],
            norm_method=args.norm_method,
        )
        if cp_override:
            kwargs["feature_dir_override"] = cp_override
            kwargs["sweep_thresholds"] = DEFAULT_SWEEP_THRESHOLDS_CP
        return kwargs

    if not args.slurm:
        print("\nRunning locally (sequential)...")
        for signal, pairs in signal_groups.items():
            result = pca_sweep_pooled_signal(**_signal_job_kwargs(signal, pairs))
            print(f"  {result}")
        result = aggregate_channels(
            output_dir=str(ds_output_dir), norm_method=args.norm_method, per_unit_subdir="per_signal",
        )
        print(result)
        return

    # SLURM mode: one job per signal group — split Phase out for higher memory
    phase_jobs = []
    other_jobs = []
    for signal, pairs in signal_groups.items():
        sig_safe = sanitize_signal_filename(signal)[:40]
        job = {
            "name": f"pca_ds_{sig_safe}",
            "func": pca_sweep_pooled_signal,
            "kwargs": _signal_job_kwargs(signal, pairs),
            "metadata": {"signal": signal, "n_experiments": len(pairs)},
        }
        if signal == "Phase":
            phase_jobs.append(job)
        else:
            other_jobs.append(job)

    # Downsampled jobs need more time than per-channel
    args.slurm_time = max(args.slurm_time, 30)

    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs, wait_for_multiple_job_arrays
    slurm_params = _make_slurm_params(args)
    agg_slurm_params = _make_agg_slurm_params(args)

    # Submit both batches without waiting — they run in parallel on SLURM
    job_arrays = []

    if other_jobs:
        print(f"\nSubmitting {len(other_jobs)} non-Phase signal-group SLURM jobs ({slurm_params.get('mem', '?')} each)...")
        result_other = submit_parallel_jobs(
            jobs_to_submit=other_jobs, experiment="pca_ds_optimization",
            slurm_params=slurm_params, log_dir="pca_optimization",
            manifest_prefix="pca_ds_opt", wait_for_completion=False,
        )
        if result_other.get("submitted_jobs"):
            job_arrays.append({
                "submitted_jobs": result_other["submitted_jobs"],
                "base_job_id": result_other["base_job_id"],
                "label": "reporters",
                "slurm_params": slurm_params,
            })

    if phase_jobs:
        phase_memory = getattr(args, "phase_memory", "600GB")
        phase_slurm_params = {**slurm_params, "mem": phase_memory}
        print(f"\nSubmitting {len(phase_jobs)} Phase SLURM job(s) ({phase_memory} memory)...")
        result_phase = submit_parallel_jobs(
            jobs_to_submit=phase_jobs, experiment="pca_ds_optimization_phase",
            slurm_params=phase_slurm_params, log_dir="pca_optimization",
            manifest_prefix="pca_ds_phase_opt", wait_for_completion=False,
        )
        if result_phase.get("submitted_jobs"):
            job_arrays.append({
                "submitted_jobs": result_phase["submitted_jobs"],
                "base_job_id": result_phase["base_job_id"],
                "label": "Phase",
                "slurm_params": phase_slurm_params,
            })

    # Wait for ALL arrays with unified progress monitoring
    if job_arrays:
        wait_result = wait_for_multiple_job_arrays(
            job_arrays, experiment="pca_ds_optimization",
        )
        if wait_result.get("failed"):
            print(f"\nWarning: {len(wait_result['failed'])} jobs failed")
            for name in wait_result["failed"]:
                print(f"  - {name}")

    # Chain aggregation after all Phase 1 jobs complete
    print(f"\nAll signal-group jobs complete. Submitting aggregation SLURM job...")
    _submit_aggregation_slurm(
        str(ds_output_dir), args.norm_method, "per_signal",
        agg_slurm_params, "pca_ds_aggregation", "pca_ds_agg",
    )



# =============================================================================
# CLI
# =============================================================================

def _build_parser():
    """Build argparse parser for the PCA optimization CLI."""
    parser = argparse.ArgumentParser(
        description="Per-signal pooled PCA optimization for organelle attribution"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str,
        default="/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized",
        help="Root output directory (feature-type and channel-subset subdirs are added automatically)",
    )
    parser.add_argument("--norm-method", type=str, default="ntc", choices=["ntc", "global"],
                        help="Normalization method (default: ntc)")
    parser.add_argument("--slurm", action="store_true",
                        help="Submit Phase 1 signal-group SLURM jobs + Phase 2 aggregation job")
    parser.add_argument("--slurm-memory", type=str, default="100GB",
                        help="SLURM memory per signal-group job (default: 100GB)")
    parser.add_argument("--slurm-time", type=int, default=10,
                        help="SLURM time limit per signal-group job in minutes (default: 10)")
    parser.add_argument("--slurm-cpus", type=int, default=16,
                        help="SLURM CPUs per signal-group job (default: 16)")
    parser.add_argument("--slurm-partition", type=str, default="cpu,gpu",
                        help="SLURM partition (default: cpu,gpu)")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--slurm-agg-memory", type=str, default="500GB",
                        help="SLURM memory for aggregation job (default: 500GB)")
    parser.add_argument("--slurm-agg-time", type=int, default=60,
                        help="SLURM time limit for aggregation job in minutes (default: 60)")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Only run Phase 2 aggregation (skips PCA sweeps, reads existing per_signal/ h5ads).")
    parser.add_argument("--umap-only", action="store_true",
                        help="Only generate embedding plots from existing optimized h5ads.")
    parser.add_argument("--downsampled", action="store_true",
                        help="Equalise cells across signal groups by downsampling to the smallest group "
                             "(floor 750k). Default mode uses all cells per group. Output → downsampled/.")
    parser.add_argument("--phase-memory", type=str, default="600GB",
                        help="SLURM memory for Phase signal job (default: 600GB). Phase ~50M cells needs more.")
    parser.add_argument("--cell-profiler", action="store_true",
                        help="Use CellProfiler morphological features instead of DINO embeddings.")
    phase_group = parser.add_mutually_exclusive_group()
    phase_group.add_argument("--phase-only", action="store_true",
                             help="Include only Phase (label-free brightfield) channels. Output → phase_only/.")
    phase_group.add_argument("--no-phase", action="store_true",
                             help="Exclude Phase channels, fluorescent only. Output → no_phase/.")
    return parser


def main():
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)

    # Nest output under feature-type subdir: dino/ or cellprofiler/
    cp_override = None
    if args.cell_profiler:
        cp_override = "cell-profiler"
        output_dir = output_dir / "cellprofiler"
        print(f"CellProfiler mode: features from 3-assembly/cell-profiler/anndata_objects/")
        print(f"PCA sweep thresholds: {DEFAULT_SWEEP_THRESHOLDS_CP} (lower range — CP features are independent)")
        print(f"Output: {output_dir}")
    else:
        output_dir = output_dir / "dino"

    # Nest under channel-subset subdir; default (no filter) goes to all/
    if args.phase_only:
        if args.downsampled:
            print("ERROR: --phase-only is not compatible with --downsampled.")
            return
        output_dir = output_dir / "phase_only"
        args.phase_filter = "phase_only"
        print(f"Phase-only mode: output → {output_dir}")
    elif args.no_phase:
        if args.downsampled:
            print("ERROR: --no-phase is not compatible with --downsampled.")
            return
        output_dir = output_dir / "no_phase"
        args.phase_filter = "no_phase"
        print(f"No-phase mode: output → {output_dir}")
    elif args.downsampled:
        output_dir = output_dir / "downsampled"
        args.phase_filter = None
        print(f"Downsampled mode: output → {output_dir}")
    else:
        output_dir = output_dir / "all"
        args.phase_filter = None
        print(f"All-cells mode (default): output → {output_dir}")

    # all_cells=True is now always the default (non-downsampled path)
    args.all_cells = not args.downsampled

    output_dir.mkdir(parents=True, exist_ok=True)

    # Dispatch to mode handler
    if args.umap_only:
        _handle_umap_only(args, output_dir)
    elif args.aggregate_only:
        _handle_aggregate_only(args, output_dir)
    else:
        _handle_downsampled(args, output_dir, cp_override)


if __name__ == "__main__":
    main()
