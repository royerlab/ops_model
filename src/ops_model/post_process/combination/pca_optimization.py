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

12 variants — feature type × channel subset
-------------------------------------------
Each variant produces an independent output subtree and can be compared via
compare_map_scores.py.  Replace --slurm with --aggregate-only --slurm to re-run
Phase 2 only (e.g. after code changes) without redoing the PCA sweeps.

  Variant                           Flags                                              Output subdir
  ───────────────────────────────── ────────────────────────────────────────────────── ─────────────────────────────────
  DINO        all                   --slurm                                            dino/all/
  DINO        phase-only            --phase-only --slurm                               dino/phase_only/
  DINO        no-phase              --no-phase --slurm                                 dino/no_phase/
  DINO        downsampled           --downsampled --slurm                              dino/downsampled/
  DINO        phase-only-ds         --phase-only --downsampled --slurm                 dino/phase_only_downsampled/
  DINO        no-phase-ds           --no-phase --downsampled --slurm                   dino/no_phase_downsampled/
  CellProfiler all                  --cell-profiler --slurm                            cellprofiler/all/
  CellProfiler phase-only           --phase-only --cell-profiler --slurm               cellprofiler/phase_only/
  CellProfiler no-phase             --no-phase --cell-profiler --slurm                 cellprofiler/no_phase/
  CellProfiler downsampled          --downsampled --cell-profiler --slurm              cellprofiler/downsampled/
  CellProfiler phase-only-ds        --phase-only --downsampled --cell-profiler --slurm cellprofiler/phase_only_downsampled/
  CellProfiler no-phase-ds          --no-phase --downsampled --cell-profiler --slurm   cellprofiler/no_phase_downsampled/

  Channel subsets:
    (default)    all fluorescent + phase channels, all cells pooled per signal group
    --phase-only label-free brightfield (Phase) only
    --no-phase   fluorescent channels only (excludes Phase)
    --downsampled cells equalised across signal groups (floor 750k/group, top 3 exps per signal)
    Combine --phase-only/--no-phase with --downsampled for filtered + downsampled variants.

  Append --aggregate-only to re-run Phase 2 only (e.g. after code changes).
  Use run_aggregate_all.sh to submit all 12 --aggregate-only jobs in parallel.

Validation cohort run (4 experiments, Phase only, validation500 library)
------------------------------------------------------------------------
The validation experiments (ops0146/0147/0150/0151) use the validation500
library, which has its own CHAD cluster file. ``--run-tag`` tucks the cohort
under a ``paper_v1/<leaf>`` parent for organization (no actual --paper-v1 flag,
since these experiments aren't in the paper_v1 YAML)::

    python -m ops_model.post_process.combination.pca_optimization \\
        --output-dir /hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3 \\
        --cell-dino \\
        --zscore-per-experiment \\
        --run-tag paper_v1/validation_4exp_phase_only \\
        --experiments ops0146,ops0147,ops0150,ops0151 \\
        --phase-only \\
        --chad-annotation /hpc/projects/icd.fast.ops/configs/gene_clusters/val_library_chad_positive_controls_v1.yml \\
        --slurm

  → cell_dino/zscore_per_exp/paper_v1/validation_4exp_phase_only/phase_only/consensus_sweep/cosine/

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
    phenotypic_consistency_manual_annotation,
    phenotypic_distinctivness,
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

DEFAULT_SWEEP_THRESHOLDS = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    0.99,
]
# CellProfiler features are hand-crafted and independent (not redundant like DINO embeddings),
# so PCA is destructive at high thresholds. Optimal region is ~50% variance explained.
DEFAULT_SWEEP_THRESHOLDS_CP = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
MIN_PCS = 10  # Minimum PCs for peak selection (avoids degenerate 1-PC artifact)
PCA_FIT_CAP = 5_000_000  # Cells used to fit PCA axes; larger datasets use passthrough (fit subsample, transform all)

# Dud sgRNAs known to produce off-target/toxic phenotypes — filtered out by default.
# Source: cell_dino_final.yml cell_filters.
DUD_GUIDES = frozenset({
    "TCCCATGACTTGTTGTCATG",
    "GCAGGCAAATTCTGAACTTG",
    "GGGTGGTATCATAGCCACCC",
    "CACATCCCCAATGGGGAGTT",
    "TATTCAAAGTTGATGTTGGA",
})


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
    return logging.getLogger(__name__)


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
):
    """Build AnnData at peak PCs, save cell subsample, aggregate to guide/gene, write outputs.

    Args:
        drop_obs_cols: Columns to drop from obs before aggregation (e.g. ['experiment'] for copairs).
        preserve_batch: If True, aggregate preserving experiment identity in obs.
        output_suffix: Appended to file_prefix in all output filenames (e.g. '_batch', '_nopca').
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
        adata_cells, level="guide", method="mean", preserve_batch_info=preserve_batch
    )
    e = aggregate_to_level(
        adata_cells, level="gene", method="mean", preserve_batch_info=preserve_batch
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
        adata_cells, level="guide", method="mean", preserve_batch_info=preserve_batch
    )
    e = aggregate_to_level(
        adata_cells, level="gene", method="mean", preserve_batch_info=preserve_batch
    )
    del adata_cells
    g.X = np.asarray(g.X, dtype=np.float32)
    e.X = np.asarray(e.X, dtype=np.float32)

    base_uns = {
        "pca_applied": False,
        "n_features": int(n_feats),
        "signal": signal,
    }
    base_uns.update(uns_metadata)
    for adata in [g, e]:
        adata.uns.update(base_uns)

    g.write_h5ad(out_subdir / f"{file_prefix}{output_suffix}_guide.h5ad")
    e.write_h5ad(out_subdir / f"{file_prefix}{output_suffix}_gene.h5ad")
    _logger.info(
        f"  Saved {file_prefix}{output_suffix}_guide/gene.h5ad (no PCA, {n_feats} raw features)"
    )


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
    distance: str = "cosine",
    fixed_threshold: Optional[float] = None,
    preserve_batch: bool = False,
    no_pca: bool = False,
    zscore_per_experiment: bool = False,
    exclude_dud_guides: bool = True,
    downsample_per_guide: bool = False,
    cells_per_guide: int = 250,
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
    # When fixed_threshold is set, always run the full sweep for plotting but
    # override peak selection afterwards.
    thresholds = (
        DEFAULT_SWEEP_THRESHOLDS
        if fixed_threshold is not None
        else (sweep_thresholds or DEFAULT_SWEEP_THRESHOLDS)
    )
    rng = np.random.RandomState(random_seed)

    # Resolve storage roots from config
    attr_config = load_attribution_config()
    storage_roots = get_storage_roots(attr_config)
    feature_dir = feature_dir_override or attr_config.get(
        "feature_dir", "dino_features"
    )
    maps_path = get_channel_maps_path()
    preserve_batch = preserve_batch or attr_config.get("preserve_batch", False)

    n_exps = len(exp_channel_pairs)
    _logger.info(
        f"Processing signal group: {signal} ({n_exps} experiments, features: {feature_dir})"
    )

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

    # --- Guide-max pre-scan (optional) ---
    # When downsample_per_guide=True, cap each sgRNA at a fixed cells_per_guide.
    # Requires a pre-scan of obs (no X loaded) so we can compute per-experiment
    # fractions (cap / pooled_count) to achieve the global cap.
    pooled_sg_counts: Dict[str, int] = {}
    cells_per_guide_cap: Optional[int] = None
    if downsample_per_guide:
        _logger.info(
            f"  [per-guide] Pre-scanning sgRNA counts (cap = {cells_per_guide} cells/guide)..."
        )
        for exp, ch in exp_channel_pairs:
            if (exp, ch) not in exp_cell_counts or exp_cell_counts[(exp, ch)] == 0:
                continue
            path = find_cell_h5ad_path(exp, ch, storage_roots, feature_dir, maps_path)
            if path is None:
                continue
            try:
                adata_backed = ad.read_h5ad(path, backed="r")
                obs = adata_backed.obs
                if "sgRNA" in obs.columns:
                    if exclude_dud_guides:
                        obs = obs[~obs["sgRNA"].isin(DUD_GUIDES)]
                    vc = obs["sgRNA"].value_counts()
                    for s, c in vc.items():
                        pooled_sg_counts[s] = pooled_sg_counts.get(s, 0) + int(c)
                adata_backed.file.close()
            except Exception as exc:
                _logger.warning(f"  [per-guide] Pre-scan failed for {exp}/{ch}: {exc}")

        if pooled_sg_counts:
            cells_per_guide_cap = int(cells_per_guide)
            _logger.info(
                f"  [per-guide] {len(pooled_sg_counts):,} sgRNAs in pool — "
                f"capping each at {cells_per_guide_cap:,} cells"
            )
        else:
            _logger.warning("  [per-guide] No sgRNA info found — falling back to proportional sampling")
            downsample_per_guide = False

    # When downsampling and >2 experiments contribute to a signal group, use
    # the fewest experiments needed to reach target_n_cells (starting from the
    # largest).  Spreading a fixed cell budget across many batches lets batch
    # effects overwhelm the signal.
    #
    # ⚠ HARDCODE: For the "Phase" signal group, force ops0094 first then ops0100.
    # These were manually selected as the highest-quality Phase experiments.
    _PHASE_PREFERRED_ORDER = ("ops0094", "ops0100")

    if not downsample_per_guide and len(exp_cell_counts) > 2 and n_cells_pooled > target_n_cells:
        if signal == "Phase":
            _logger.warning(
                "  ⚠ HARDCODED EXPERIMENT ORDER FOR PHASE: %s — update pca_optimization.py to change",
                _PHASE_PREFERRED_ORDER,
            )
            # Put preferred experiments first (in order), then the rest ranked by cell count
            preferred = []
            remainder = []
            for pair in exp_cell_counts:
                exp_short = pair[0].split("_")[0]
                if exp_short in _PHASE_PREFERRED_ORDER:
                    preferred.append(pair)
                else:
                    remainder.append(pair)
            preferred.sort(
                key=lambda p: _PHASE_PREFERRED_ORDER.index(p[0].split("_")[0])
            )
            remainder.sort(key=lambda p: exp_cell_counts[p], reverse=True)
            ranked = preferred + remainder
        else:
            ranked = sorted(exp_cell_counts, key=exp_cell_counts.get, reverse=True)
        kept = []
        running_total = 0
        for pair in ranked:
            kept.append(pair)
            running_total += exp_cell_counts[pair]
            if running_total >= target_n_cells:
                break
        dropped = set(exp_cell_counts) - set(kept)
        for d in sorted(dropped, key=lambda k: exp_cell_counts[k], reverse=True):
            _logger.info(
                f"  Dropping {d[0].split('_')[0]}/{d[1]} ({exp_cell_counts[d]:,} cells) — enough cells from top {len(kept)} experiments"
            )
        exp_cell_counts = {k: v for k, v in exp_cell_counts.items() if k in set(kept)}
        n_cells_pooled = sum(exp_cell_counts.values())
        _logger.info(
            f"  Kept {len(kept)} of {len(ranked)} experiments ({n_cells_pooled:,} cells >= {target_n_cells:,} target)"
        )

    actual_target = min(target_n_cells, n_cells_pooled)
    _logger.info(
        f"  Total pooled: {n_cells_pooled:,} cells ({len(exp_cell_counts)} experiments), target: {actual_target:,}"
    )

    # --- Pass 2b: Load one experiment at a time, subsample, collect raw arrays ---
    X_blocks = []  # list of np arrays
    obs_blocks = []  # list of DataFrames
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

        # Filter out dud sgRNAs (off-target / toxic) — enabled by default
        if exclude_dud_guides and "sgRNA" in adata.obs.columns:
            n_before = adata.n_obs
            keep = ~adata.obs["sgRNA"].isin(DUD_GUIDES)
            n_dropped = int((~keep).sum())
            if n_dropped > 0:
                adata = adata[keep].copy()
                _logger.info(
                    f"  {exp.split('_')[0]}/{ch}: dropped {n_dropped:,} dud-guide cells "
                    f"({n_before:,} → {adata.n_obs:,})"
                )

        if n_vars_expected is None:
            n_vars_expected = adata.n_vars
            var_names = list(adata.var_names)
        elif adata.n_vars != n_vars_expected:
            _logger.warning(
                f"  SKIPPING {exp}/{ch}: {adata.n_vars} features (expected {n_vars_expected}) — feature count mismatch"
            )
            del adata
            continue

        if downsample_per_guide and cells_per_guide_cap is not None and "sgRNA" in adata.obs.columns:
            # Per-sgRNA cap: each sgRNA contributes (cap / pooled_count) of its cells
            # from each experiment, so global total per sgRNA ≈ cells_per_guide_cap.
            sgrnas = adata.obs["sgRNA"].values
            keep_mask = np.zeros(adata.n_obs, dtype=bool)
            for s in np.unique(sgrnas):
                s_idx = np.where(sgrnas == s)[0]
                pooled = pooled_sg_counts.get(s, len(s_idx))
                if pooled <= cells_per_guide_cap:
                    keep_mask[s_idx] = True
                else:
                    frac = cells_per_guide_cap / pooled
                    n_take_s = max(1, int(round(frac * len(s_idx))))
                    picked = rng.choice(s_idx, n_take_s, replace=False)
                    keep_mask[picked] = True
            n_take = int(keep_mask.sum())
            if n_take < adata.n_obs:
                adata = adata[keep_mask].copy()
        else:
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
        keep_cols = [
            c for c in ["sgRNA", "perturbation", "label_str"] if c in adata.obs.columns
        ]
        obs = adata.obs[keep_cols].copy()
        obs["experiment"] = exp.split("_")[0]

        X_blocks.append(np.asarray(adata.X, dtype=np.float32))
        obs_blocks.append(obs)
        loaded_exps.append(exp)
        _logger.info(
            f"  {exp.split('_')[0]}/{ch}: {exp_cell_counts[(exp, ch)]:,} → {n_take:,} cells"
        )
        del adata

    _logger.info(
        f"  Loading complete: {len(loaded_exps)} experiments in {time.time() - t_load:.0f}s"
    )

    if not X_blocks:
        return f"FAILED: {signal} — no cell data found for any experiment"

    # Concatenate using numpy vstack (much faster than ad.concat for uniform features)
    t_concat = time.time()
    _logger.info(
        f"  Concatenating {len(X_blocks)} blocks ({sum(x.shape[0] for x in X_blocks):,} cells)..."
    )
    X_raw = np.vstack(X_blocks)
    del X_blocks
    obs_df_full = pd.concat(obs_blocks, ignore_index=True)
    del obs_blocks
    _logger.info(
        f"  Concatenation done in {time.time() - t_concat:.0f}s — {X_raw.shape[0]:,} x {X_raw.shape[1]} matrix"
    )

    n_cells = X_raw.shape[0]
    n_feats = X_raw.shape[1]
    feature_names = var_names
    _logger.info(
        f"  Pooled: {n_cells_pooled} total cells → {n_cells} pooled ({n_feats} shared features from {len(loaded_exps)} experiments)"
    )

    # Obs DataFrames for scoring (without experiment col which breaks copairs)
    score_cols = [
        c for c in ["sgRNA", "perturbation", "label_str"] if c in obs_df_full.columns
    ]
    obs_df = obs_df_full[score_cols].copy()

    # Per-experiment z-score before PCA — via pca.normalize_before_pca config or --zscore-per-experiment
    if attr_config.get("pca", {}).get("normalize_before_pca", False) or zscore_per_experiment:
        from sklearn.preprocessing import StandardScaler

        experiments = (
            obs_df_full["experiment"].values
            if "experiment" in obs_df_full.columns
            else None
        )
        if experiments is not None:
            for exp_id in np.unique(experiments):
                mask = experiments == exp_id
                X_raw[mask] = StandardScaler().fit_transform(X_raw[mask])
            _logger.info(
                f"  Applied per-experiment z-score scaling ({len(np.unique(experiments))} experiments)"
            )
        else:
            X_raw = StandardScaler().fit_transform(X_raw)
            _logger.info(f"  Applied global z-score scaling (no experiment info)")

    # --- No-PCA early exit: aggregate raw features and return ---
    if no_pca:
        output_suffix = "_nopca" + ("_batch" if preserve_batch else "")
        file_prefix = sanitize_signal_filename(signal)
        exps_str = ", ".join(exp.split("_")[0] for exp in loaded_exps[:5])
        if len(loaded_exps) > 5:
            exps_str += f" +{len(loaded_exps)-5} more"
        _save_raw_outputs(
            X_raw=X_raw,
            obs_df=obs_df_full,
            feature_names=feature_names or [],
            signal=signal,
            uns_metadata={
                "experiment": ",".join(exp.split("_")[0] for exp in loaded_exps),
                "channel": ",".join(ch for _, ch in exp_channel_pairs),
                "n_cells": int(n_cells),
                "n_cells_pooled": int(n_cells_pooled),
                "n_experiments": int(n_exps),
                "n_features_raw": int(n_feats),
            },
            output_dir=output_dir,
            subdir="per_signal",
            file_prefix=file_prefix,
            rng=rng,
            _logger=_logger,
            drop_obs_cols=None if preserve_batch else ["experiment"],
            preserve_batch=preserve_batch,
            output_suffix=output_suffix,
        )
        elapsed = time.time() - t_start
        _logger.info(
            f"  Done: {signal} in {elapsed:.0f}s — no PCA ({n_feats} raw features)"
        )
        return f"SUCCESS: {signal} — no PCA, {n_feats} raw features ({n_exps} exps, {n_cells}/{n_cells_pooled} cells)"

    # --- Fit PCA on subsample, transform all cells in chunks ---
    t_pca = time.time()
    n_total = X_raw.shape[0]

    if n_total > PCA_FIT_CAP:
        # Subsample for fit
        fit_idx = rng.choice(n_total, PCA_FIT_CAP, replace=False)
        fit_idx.sort()
        _logger.info(
            f"  Fitting PCA on {PCA_FIT_CAP:,}/{n_total:,} subsampled cells..."
        )
        _, cumvar, pca_model = fit_pca(X_raw[fit_idx])
        del fit_idx

        # Transform all cells in chunks (avoids 40M x 500 float64 all at once)
        _logger.info(f"  Transforming all {n_total:,} cells in chunks...")
        chunk_size = 2_000_000
        X_pcs_chunks = []
        for i in range(0, n_total, chunk_size):
            chunk = np.asarray(X_raw[i : i + chunk_size], dtype=np.float64)
            chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
            X_pcs_chunks.append(pca_model.transform(chunk).astype(np.float32))
            _logger.info(
                f"    Transformed chunk {i:,}-{min(i + chunk_size, n_total):,}"
            )
        X_pcs = np.vstack(X_pcs_chunks)
        del X_pcs_chunks
    else:
        _logger.info(f"  Fitting PCA on {n_total:,} x {X_raw.shape[1]} matrix...")
        X_pcs, cumvar, pca_model = fit_pca(X_raw)

    _logger.info(
        f"  PCA done in {time.time() - t_pca:.0f}s — {X_pcs.shape[1]} components"
    )
    pca_components = pca_model.components_.copy()
    pca_var_ratio = pca_model.explained_variance_ratio_.copy()
    # Capture the fit mean so we can re-project new cells (e.g. validation
    # cohort) through this exact PCA: ``X_new_pcs = (X_new − pca_mean) @ pca_components.T``.
    # Without this, downstream "reuse PCA" runs would have to assume zero-mean
    # input, which only roughly holds after z-score.
    pca_mean = (
        pca_model.mean_.copy()
        if getattr(pca_model, "mean_", None) is not None
        else None
    )
    del X_raw, pca_model

    # preserve_batch: skip sweep and use pca.variance_cutoff from config directly
    if preserve_batch:
        variance_cutoff = attr_config.get("pca", {}).get("variance_cutoff", 0.80)
        _logger.info(
            f"  preserve_batch: skipping sweep, using variance_cutoff={variance_cutoff:.0%} from config"
        )
        selected_t = variance_cutoff
        selected_n = n_pcs_for_threshold(cumvar, variance_cutoff)
        selected_r, selected_a = 0.0, 0.0
        sweep_rows: List[Dict] = []
        metric_peaks: Dict = {}
        sweep_peak_t = variance_cutoff
    else:
        # Sweep thresholds
        t_sweep = time.time()
        _logger.info(f"  Starting threshold sweep ({len(thresholds)} thresholds)...")
        result = _run_threshold_sweep(
            X_pcs,
            cumvar,
            obs_df,
            thresholds,
            norm_method,
            extra_sweep_cols={"signal": signal, "n_experiments": n_exps},
            _logger=_logger,
            distance=distance,
        )
        _logger.info(f"  Sweep done in {time.time() - t_sweep:.0f}s")
        if result is None:
            return f"FAILED: {signal} — no valid threshold found (all < {MIN_PCS} PCs)"
        sweep_rows = result["sweep_rows"]
        consensus_t = result["consensus_t"]
        consensus_n = result["consensus_n"]
        consensus_r = result["consensus_r"]
        consensus_a = result["consensus_a"]
        metric_peaks = {
            k: result[k] for k in ("peak_act_t", "peak_dist_t", "peak_chad_t")
        }

        # Override peak with fixed_threshold if specified; otherwise use consensus
        sweep_peak_t = consensus_t  # preserve consensus for plot reference
        selected_t, selected_n, selected_r, selected_a = (
            consensus_t,
            consensus_n,
            consensus_r,
            consensus_a,
        )
        if fixed_threshold is not None:
            fixed_n = n_pcs_for_threshold(cumvar, fixed_threshold)
            fixed_row = next(
                (r for r in sweep_rows if r["threshold"] == fixed_threshold), None
            )
            selected_r = fixed_row["activity"] if fixed_row else consensus_r
            selected_a = fixed_row["auc"] if fixed_row else consensus_a
            _logger.info(
                f"  Fixed threshold override: {fixed_threshold:.0%} → {fixed_n} PCs (consensus was {consensus_t:.0%})"
            )
            selected_t, selected_n = fixed_threshold, fixed_n

    # Save outputs at selected peak
    file_prefix = sanitize_signal_filename(signal)
    exps_str = ", ".join(exp.split("_")[0] for exp in loaded_exps[:5])
    if len(loaded_exps) > 5:
        exps_str += f" +{len(loaded_exps)-5} more"

    output_suffix = "_batch" if preserve_batch else ""
    _save_sweep_outputs(
        X_pcs,
        obs_df_full,
        cumvar,
        peak_n=selected_n,
        peak_t=selected_t,
        peak_activity_r=selected_r,
        peak_activity_auc=selected_a,
        best_act_t=metric_peaks.get("peak_act_t", selected_t),
        metric_peaks=metric_peaks or None,
        signal=signal,
        sweep_rows=sweep_rows,
        uns_metadata={
            "experiment": ",".join(exp.split("_")[0] for exp in loaded_exps),
            "channel": ",".join(ch for _, ch in exp_channel_pairs),
            "n_cells": int(n_cells),
            "n_cells_pooled": int(n_cells_pooled),
            "n_experiments": int(n_exps),
            "n_features_raw": int(n_feats),
            "pca_components": pca_components[:selected_n].tolist(),
            "pca_feature_names": feature_names,
            # Source-fit mean: enables ``X_new_pcs = (X_new − mean) @ comps.T``
            # so downstream runs can re-project new cells through this exact PCA.
            "pca_mean": (
                pca_mean.tolist() if pca_mean is not None else None
            ),
        },
        output_dir=output_dir,
        subdir="per_signal",
        file_prefix=file_prefix,
        suptitle=f"{signal} ({n_exps} exps: {exps_str}) — {n_cells:,}/{n_cells_pooled:,} cells, {n_feats} raw features",
        rng=rng,
        _logger=_logger,
        drop_obs_cols=None if preserve_batch else ["experiment"],
        fixed_threshold=fixed_threshold,
        sweep_peak_t=sweep_peak_t,
        preserve_batch=preserve_batch,
        output_suffix=output_suffix,
    )

    elapsed = time.time() - t_start
    _logger.info(
        f"  Done: {signal} in {elapsed:.0f}s — {selected_n} PCs @ {selected_t:.0%}"
    )

    return f"SUCCESS: {signal} — {selected_n} PCs @ {selected_t:.0%}, {selected_r:.1%} active ({n_exps} exps, {n_cells}/{n_cells_pooled} cells)"


# =============================================================================
# Phase 2: Aggregation sub-steps (used by aggregate_channels)
# =============================================================================

def _plot_chad_umap(umap_coords, genes, gene_to_cluster, out_path, plt, _logger):
    """Plot UMAP colored by CHAD cluster with gene labels."""
    import seaborn as sns
    from itertools import product

    cats = [gene_to_cluster.get(g, "Uncategorized") for g in genes]
    is_ntc = np.array([str(g).startswith("NTC") for g in genes])
    unique_cats = sorted(set(c for c in cats if c != "Uncategorized"))

    # 10 dark colors x 6 markers = 60 unique combos for 50+ clusters
    # 20 colors x 10 markers = 200 unique combos
    colors_20 = sns.color_palette("dark", 10) + sns.color_palette("Set1", 9) + sns.color_palette("Set2", 1)
    markers_10 = ["o", "s", "D", "^", "v", "P", "p", "h", "*", "X"]
    combos = list(product(colors_20, markers_10))
    cat_to_color = {cat: combos[i % len(combos)][0] for i, cat in enumerate(unique_cats)}
    cat_to_marker = {cat: combos[i % len(combos)][1] for i, cat in enumerate(unique_cats)}

    # Square UMAP on left, legend fills right
    fig = plt.figure(figsize=(30, 12))
    ax = fig.add_axes([0.04, 0.04, 0.50, 0.94])  # [left, bottom, width, height]
    ax.set_box_aspect(0.7)

    # Pin axis limits to full UMAP extent so every plot has identical framing
    x_min, x_max = umap_coords[:, 0].min(), umap_coords[:, 0].max()
    y_min, y_max = umap_coords[:, 1].min(), umap_coords[:, 1].max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # Uncategorized background
    uncat_mask = np.array([c == "Uncategorized" for c in cats]) & ~is_ntc
    if uncat_mask.any():
        ax.scatter(umap_coords[uncat_mask, 0], umap_coords[uncat_mask, 1],
                   c=[(0.85, 0.85, 0.85)], s=40, alpha=0.2, edgecolors="none", label="Uncategorized")

    # Categorized genes — unique color+marker per cluster
    for cat in unique_cats:
        if cat == "OR controls":
            continue  # Plotted separately with special marker
        mask = np.array([c == cat for c in cats]) & ~is_ntc
        if mask.any():
            # Truncate long labels for legend readability
            label = cat if len(cat) <= 60 else cat[:57] + "..."
            ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       c=[cat_to_color[cat]], marker=cat_to_marker[cat],
                       s=150, alpha=0.85, edgecolors="white", linewidths=0.3, label=label)

    # OR controls — bright red X, larger than NTCs
    is_or = np.array([gene_to_cluster.get(g, "") == "OR controls" for g in genes])
    if is_or.any():
        ax.scatter(umap_coords[is_or, 0], umap_coords[is_or, 1],
                   c="#FF0000", marker="X", s=285, alpha=0.7, edgecolors="#CC0000",
                   linewidths=0.6, label="OR controls", zorder=11)

    # NTCs
    if is_ntc.any():
        ax.scatter(umap_coords[is_ntc, 0], umap_coords[is_ntc, 1],
                   c="#e08080", marker="X", s=195, alpha=0.4, edgecolors="#b05050",
                   linewidths=0.3, label="NTC", zorder=10)

    # Gene labels — only annotated genes, radial offset to avoid overlap
    rng = np.random.RandomState(42)
    for i, gene in enumerate(genes):
        if str(gene).startswith("NTC"):
            continue
        if gene_to_cluster.get(gene, "Uncategorized") == "Uncategorized":
            continue
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(40, 80)
        dx = radius * np.cos(angle)
        dy = radius * np.sin(angle)
        color = "#FF0000" if gene_to_cluster.get(gene) == "OR controls" else cat_to_color.get(gene_to_cluster.get(gene, ""), "black")
        ax.annotate(gene, xy=(umap_coords[i, 0], umap_coords[i, 1]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=14, alpha=0.75, ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", color=color, alpha=0.5, lw=0.5))

    # Use 1 column by default; only split to 2 if too many items to fit vertically
    # At fontsize 13 with labelspacing 0.4, ~40 rows fit in 18in figure
    ncol = 2 if len(unique_cats) > 40 else 1
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=16,
              framealpha=0.9, ncol=ncol, columnspacing=0.6, handletextpad=0.3,
              labelspacing=0.4)
    ax.set_title("Gene UMAP -- colored by CHAD cluster", fontsize=32, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=24)
    ax.set_ylabel("UMAP 2", fontsize=24)
    ax.tick_params(labelsize=18)
    # Fixed axes position so legend size doesn't distort UMAP aspect across plots
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    _logger.info(f"  Saved CHAD UMAP: {out_path}")


def _score_single_reporter_metrics(
    g_raw, norm_method, _logger, null_size=100_000, distance="cosine"
):
    """Score all 4 phenotypic metrics for one reporter's guide h5ad.

    Uses a smaller null_size than the aggregate run for speed.
    Returns dict with activity, auc, distinctiveness, corum, chad,
    and unfiltered variants: distinctiveness_all, corum_all, chad_all (NaN on failure).
    """
    import math

    result = {
        k: math.nan
        for k in (
            "activity",
            "auc",
            "distinctiveness",
            "corum",
            "chad",
            "distinctiveness_all",
            "corum_all",
            "chad_all",
        )
    }
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
            g_norm,
            plot_results=False,
            null_size=null_size,
            distance=distance,
        )
        result["activity"] = float(active_ratio)
        result["auc"] = float(compute_auc_score(activity_map))

        _, dist_ratio = phenotypic_distinctivness(
            g_norm,
            plot_results=False,
            null_size=null_size,
            distance=distance,
        )
        result["distinctiveness"] = float(dist_ratio)
        result["distinctiveness_all"] = result["distinctiveness"]

        e_norm = aggregate_to_level(
            g_norm, "gene", preserve_batch_info=False, subsample_controls=False
        )
        e_norm = _prepare_for_copairs(e_norm)

        _, corum_ratio = phenotypic_consistency_corum(
            e_norm,
            plot_results=False,
            null_size=null_size,
            cache_similarity=True,
            distance=distance,
        )
        result["corum"] = float(corum_ratio)
        result["corum_all"] = result["corum"]

        _, chad_ratio = phenotypic_consistency_manual_annotation(
            e_norm,
            plot_results=False,
            null_size=null_size,
            cache_similarity=True,
            distance=distance,
            annotation_path=CHAD_ANNOTATION_PATH,
        )
        result["chad"] = float(chad_ratio)
        result["chad_all"] = result["chad"]

    except Exception as exc:
        _logger.warning(f"  Per-reporter metrics scoring failed: {exc}")
    return result


def _load_per_unit_blocks(per_unit_dir, norm_method, _logger, distance="cosine"):
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
            _logger.warning(
                f"  Skipping {file_prefix}: unmapped channel (signal={sig!r})"
            )
            continue

        e = ad.read_h5ad(gene_file)
        guide_blocks.append(g)
        gene_blocks.append(e)
        n_cells = int(g.uns.get("n_cells", 0))
        total_cells += n_cells

        _logger.info(f"  {sig}: scoring all 4 metrics per-reporter...")
        reporter_metrics = _score_single_reporter_metrics(
            g, norm_method, _logger, distance=distance
        )

        report_rows.append(
            {
                "experiment": g.uns.get("experiment", ""),
                "channel": g.uns.get("channel", ""),
                "signal": sig,
                "n_cells": n_cells,
                "n_features_raw": int(g.uns.get("n_features_raw", 0)),
                "peak_threshold": float(g.uns.get("pca_threshold", 0)),
                "n_pcs": int(g.uns.get("n_pcs", 0)),
                "explained_variance": float(g.uns.get("explained_variance", 0)),
                "activity": reporter_metrics["activity"],
                "auc": reporter_metrics["auc"],
                "distinctiveness": reporter_metrics["distinctiveness"],
                "corum": reporter_metrics["corum"],
                "chad": reporter_metrics["chad"],
                "distinctiveness_all": reporter_metrics["distinctiveness_all"],
                "corum_all": reporter_metrics["corum_all"],
                "chad_all": reporter_metrics["chad_all"],
            }
        )
        _logger.info(
            f"  {sig}: {g.n_obs} guides x {g.n_vars} PCs @ {g.uns.get('pca_threshold', '?')} | "
            f"act={reporter_metrics['activity']:.1%} dist={reporter_metrics['distinctiveness']:.1%} "
            f"corum={reporter_metrics['corum']:.1%} chad={reporter_metrics['chad']:.1%} | "
            f"all: dist={reporter_metrics['distinctiveness_all']:.1%} "
            f"corum={reporter_metrics['corum_all']:.1%} chad={reporter_metrics['chad_all']:.1%}"
        )

    return guide_blocks or None, gene_blocks, report_rows, total_cells


def _concat_and_normalize(guide_blocks, gene_blocks, norm_method, _logger):
    """Horizontal concat, NTC normalize, re-aggregate to gene, strip obs for copairs."""
    adata_guide = hconcat_by_perturbation(guide_blocks, "guide")
    adata_gene = hconcat_by_perturbation(gene_blocks, "gene")
    del guide_blocks, gene_blocks

    _logger.info(
        f"Concatenated: {adata_guide.n_obs} guides, {adata_guide.n_vars} features"
    )
    _logger.info(f"NTC normalizing at guide level...")
    adata_guide = normalize_guide_adata(adata_guide, norm_method)
    adata_guide.X = np.asarray(adata_guide.X, dtype=np.float32)

    adata_gene = aggregate_to_level(
        adata_guide,
        "gene",
        preserve_batch_info=False,
        subsample_controls=False,
    )
    _logger.info(f"  Guide: {adata_guide.n_obs} obs, {adata_guide.n_vars} features")
    _logger.info(f"  Gene: {adata_gene.n_obs} obs, {adata_gene.n_vars} features")

    # Strip obs to copairs-required columns (extra string cols cause isnan error)
    adata_guide = _prepare_for_copairs(adata_guide)
    adata_gene = _prepare_for_copairs(adata_gene)

    return adata_guide, adata_gene


def _score_activity_aggregated(adata_guide, metrics_dir, _logger, distance="cosine"):
    """Run phenotypic activity scoring on aggregated data. Returns (activity_map, ratio, auc)."""
    _logger.info(f"Running activity scoring...")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    try:
        activity_map, active_ratio = phenotypic_activity_assesment(
            adata_guide,
            plot_results=False,
            null_size=100_000,
            distance=distance,
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


def _save_aggregated_h5ads(
    adata_guide,
    adata_gene,
    report_rows,
    output_dir,
    r,
    a,
    norm_method,
    total_cells,
    _logger,
):
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
    _logger.info(
        f"  Saved guide_pca_optimized.h5ad, gene_pca_optimized.h5ad, pca_report.csv"
    )


ANNOTATED_GENE_PANEL_PATH = Path(
    "/hpc/projects/icd.fast.ops/configs/annotated_gene_panel_July2025.csv"
)


def _atomic_write_h5ad(adata: ad.AnnData, path: Path, _logger) -> None:
    """Write h5ad to a sibling .tmp path, then os.replace into target.

    AnnData's write_h5ad is NOT atomic — a mid-write failure (e.g. unsupported
    obs dtype) leaves the target file partially overwritten and unreadable.
    This helper writes to a temp path first and only moves on success.

    Also duplicates the in-pipeline ``perturbation`` column as ``geneKO_name``
    for the saved file (matches reference gene-level h5ad layout; viewers pick
    it up as the default ID variable) while keeping ``perturbation`` so
    downstream readers (e.g. 2nd-pass PCA) still find it.
    """
    import os

    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    # Coerce object obs columns with mixed/NaN values to strings so h5py's
    # vlen-string writer doesn't choke (matches what existing pipelines do).
    for col in list(adata.obs.columns):
        if adata.obs[col].dtype == object:
            adata.obs[col] = adata.obs[col].astype(str).replace({"nan": ""})

    added_geneko = False
    if "perturbation" in adata.obs.columns and "geneKO_name" not in adata.obs.columns:
        adata.obs["geneKO_name"] = adata.obs["perturbation"]
        added_geneko = True
    try:
        adata.write_h5ad(tmp)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    finally:
        if added_geneko:
            adata.obs = adata.obs.drop(columns=["geneKO_name"])


def _annotate_genes_from_panel(adata_gene: ad.AnnData, _logger) -> None:
    """Merge annotated_gene_panel_July2025.csv metadata into adata_gene.obs by gene symbol.

    Adds columns like LongName, NCBI_ID, Funk/Ramezani/Replogle map coords,
    In_corum_complexes, funk_cluster, Gene_Category, ... — matching the reference
    gene-level h5ad layout (cropseq_gene_level.h5ad).
    """
    if not ANNOTATED_GENE_PANEL_PATH.exists():
        _logger.warning(f"  Gene panel not found at {ANNOTATED_GENE_PANEL_PATH}, skipping annotation")
        return
    panel = pd.read_csv(ANNOTATED_GENE_PANEL_PATH, index_col=0)
    panel = panel.set_index("Gene.name")
    # Preserve existing obs columns; only add panel columns that aren't already present
    cols_to_add = [c for c in panel.columns if c not in adata_gene.obs.columns]
    joined = adata_gene.obs.join(panel[cols_to_add], how="left")
    adata_gene.obs = joined
    n_matched = panel.index.intersection(adata_gene.obs_names).size
    _logger.info(
        f"  Annotated {n_matched}/{adata_gene.n_obs} genes from panel "
        f"({len(cols_to_add)} columns added)"
    )


def _compute_and_plot_embeddings(adata_guide, metric_lookup, plots_dir, plt, _logger, random_seed: int = 42):
    """Compute UMAP + PHATE embeddings for guide/gene levels, plot overlays + positive controls.

    Returns adata_gene_embed with embeddings stored in obsm — caller should save it.
    The same ``random_seed`` is threaded into ``split_ntc_for_embedding``, UMAP,
    and PHATE so a given seed deterministically reproduces the same embedding.
    """
    adata_gene_embed = split_ntc_for_embedding(adata_guide, random_seed=random_seed)
    _logger.info(
        f"  Gene (NTC-split for embedding, seed={random_seed}): {adata_gene_embed.n_obs} obs"
    )
    # Use gene symbol (perturbation) as obs index, matching reference gene-level h5ads
    if "perturbation" in adata_gene_embed.obs.columns:
        adata_gene_embed.obs_names = adata_gene_embed.obs["perturbation"].astype(str).values
        adata_gene_embed.obs_names_make_unique()
    # Join in annotated gene panel metadata (LongName, NCBI_ID, Funk/Ramezani/Replogle map
    # coords, complex/pathway membership, funk_cluster, Gene_Category, ...) keyed by gene symbol
    _annotate_genes_from_panel(adata_gene_embed, _logger)
    # Propagate X_pca and pca uns from guide (same feature space, gene-level aggregation)
    adata_gene_embed.obsm["X_pca"] = np.asarray(adata_gene_embed.X, dtype=np.float32)
    if "pca" in adata_guide.uns:
        adata_gene_embed.uns["pca"] = adata_guide.uns["pca"]

    embed_pairs = [("guide", adata_guide), ("gene", adata_gene_embed)]
    level_embeddings = {}
    level_perts = {}

    def _make_embedder(name):
        """Return fit_fn(X, n_obs, level) -> (coords, params_dict) or None if library missing."""
        if name == "UMAP":
            from umap import UMAP

            def _fit(X, n_obs, level):
                nn = min(10, n_obs - 1)
                if nn < 2:
                    return None, {}
                model = UMAP(
                    n_components=2,
                    n_neighbors=nn,
                    min_dist=0.25,
                    random_state=random_seed,
                )
                coords = model.fit_transform(X)
                params = {
                    "n_neighbors": nn,
                    "min_dist": 0.25,
                    "random_state": random_seed,
                    "metric": "euclidean",
                    "a": float(
                        getattr(model, "a_", None) or getattr(model, "_a", None) or 0
                    ),
                    "b": float(
                        getattr(model, "b_", None) or getattr(model, "_b", None) or 0
                    ),
                }
                return coords, params

            return _fit
        elif name == "PHATE":
            import phate

            def _fit(X, n_obs, level):
                knn = min(15 if n_obs > 2000 else 10, n_obs - 1)
                if knn < 2:
                    return None, {}
                coords = phate.PHATE(
                    n_components=2,
                    knn=knn,
                    decay=15,
                    t="auto",
                    n_jobs=-1,
                    random_state=random_seed,
                    verbose=0,
                ).fit_transform(X)
                params = {"knn": knn, "decay": 15, "t": "auto", "random_state": random_seed}
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
                _logger.info(
                    f"  Computing {level_name} {embed_name} ({adata_level.n_obs} obs, {adata_level.n_vars} features)..."
                )
                X_clean = clean_X_for_embedding(adata_level)
                coords, embed_params = fit_fn(X_clean, adata_level.n_obs, level_name)
                if coords is None:
                    _logger.warning(
                        f"  {embed_name} skipped for {level_name}: too few observations"
                    )
                    continue
                perts = get_perts_col(adata_level)
                level_embeddings.setdefault(level_name, {})[embed_name] = coords
                level_perts[level_name] = perts
                # Store embedding in obsm/uns (guide level is adata_guide, passed by ref)
                obsm_key = f"X_{embed_name.lower()}"
                adata_level.obsm[obsm_key] = coords.astype(np.float32)
                adata_level.uns[embed_name.lower()] = {"params": embed_params}
                fname = plot_embedding_overlay(
                    coords,
                    perts,
                    metric_lookup,
                    level_name,
                    embed_name,
                    plots_dir,
                    adata_level.n_obs,
                    adata_level.n_vars,
                    plt,
                )
                _logger.info(f"  Saved plots/{fname}")
                # Save embedding coordinates as CSV
                import pandas as pd
                embed_df = pd.DataFrame(coords, columns=[f"{embed_name}1", f"{embed_name}2"])
                embed_df.insert(0, "perturbation", perts.values if hasattr(perts, "values") else perts)
                embed_csv_name = f"{level_name}_{embed_name.lower()}_coords.csv"
                embed_df.to_csv(plots_dir / embed_csv_name, index=False)
                _logger.info(f"  Saved plots/{embed_csv_name}")
        except Exception as err:
            _logger.warning(f"  {embed_name} plots failed: {err}")

    # Positive controls overlay grid (CHAD v4)
    for level_name in ["gene"]:
        if level_name in level_embeddings and level_name in level_perts:
            try:
                plot_positive_controls_grid(
                    level_embeddings[level_name],
                    level_perts[level_name],
                    level_name,
                    plots_dir,
                    plt,
                )
            except Exception as pc_err:
                _logger.warning(
                    f"  Positive controls grid failed for {level_name}: {pc_err}"
                )

    return adata_gene_embed


def _score_distinctiveness(
    adata_guide,
    activity_map,
    r,
    total_feats,
    plots_dir,
    metrics_dir,
    plt,
    _logger,
    distance="cosine",
    suffix="",
):
    """Run phenotypic distinctiveness scoring, save CSV and plots. Returns (distinctiveness_map, ratio) or (None, 0).

    NOTE: The scoring call below does NOT pass ``active_only=True`` to
    ``phenotypic_distinctivness``, so it is computed across all geneKOs
    regardless of the ``suffix`` argument. The ``suffix`` argument only
    affects output filenames.
    """
    label = "all geneKOs"
    if activity_map is None:
        return None, 0.0
    try:
        from ops_utils.analysis.map_scores import phenotypic_distinctivness

        _logger.info(f"Running distinctiveness ({label})...")
        distinctiveness_map, distinctive_ratio = phenotypic_distinctivness(
            adata_guide,
            plot_results=False,
            null_size=100_000,
            distance=distance,
        )
        distinctiveness_map.to_csv(
            metrics_dir / f"phenotypic_distinctiveness{suffix}.csv", index=False
        )
        _logger.info(f"  Distinctiveness ({label}): {distinctive_ratio:.1%}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        plot_map_scatter(ax1, activity_map, "Activity", r, show_ntc=False)
        plot_map_scatter(
            ax2,
            distinctiveness_map,
            f"Distinctiveness ({label})",
            distinctive_ratio,
            show_ntc=False,
        )
        fig.suptitle(
            f"Activity & Distinctiveness ({label}) — {total_feats} features",
            fontsize=13,
            fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(
            plots_dir / f"map_activity_distinctiveness{suffix}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
        _logger.info(f"  Saved plots/map_activity_distinctiveness{suffix}.png")
        return distinctiveness_map, distinctive_ratio
    except Exception as exc:
        _logger.error(f"  Distinctiveness ({label}) failed: {exc}")
        return None, 0.0


def _score_consistency(
    adata_gene,
    activity_map,
    total_feats,
    plots_dir,
    metrics_dir,
    plt,
    _logger,
    distance="cosine",
    suffix="",
):
    """Run CORUM + CHAD consistency scoring, save CSVs and plots.

    Returns (corum_map, corum_ratio, chad_map, chad_ratio) or (None, 0, None, 0) on failure.

    NOTE: ``phenotypic_consistency_*`` is called WITHOUT ``activity_map``, so
    consistency is computed over all genes regardless of the ``suffix``
    argument (which only affects output filenames).
    """
    label = "all geneKOs"
    if activity_map is None:
        return None, 0.0, None, 0.0
    try:
        from ops_utils.analysis.map_scores import (
            phenotypic_consistency_corum,
            phenotypic_consistency_manual_annotation,
        )

        _logger.info(f"Running CORUM consistency ({label})...")
        consistency_corum_map, consistency_corum_ratio = phenotypic_consistency_corum(
            adata_gene,
            plot_results=False,
            null_size=100_000,
            cache_similarity=True,
            distance=distance,
        )
        consistency_corum_map.to_csv(
            metrics_dir / f"phenotypic_consistency_corum{suffix}.csv", index=False
        )
        _logger.info(f"  CORUM ({label}): {consistency_corum_ratio:.1%}")

        _logger.info(f"Running CHAD consistency ({label})...")
        consistency_manual_map, consistency_manual_ratio = (
            phenotypic_consistency_manual_annotation(
                adata_gene,
                plot_results=False,
                null_size=100_000,
                cache_similarity=True,
                distance=distance,
                annotation_path=CHAD_ANNOTATION_PATH,
            )
        )
        consistency_manual_map.to_csv(
            metrics_dir / f"phenotypic_consistency_manual{suffix}.csv", index=False
        )
        _logger.info(f"  Manual CHAD ({label}): {consistency_manual_ratio:.1%}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        plot_map_scatter(
            ax1,
            consistency_corum_map,
            f"Consistency CORUM ({label})",
            consistency_corum_ratio,
            show_ntc=False,
        )
        plot_map_scatter(
            ax2,
            consistency_manual_map,
            f"Consistency CHAD ({label})",
            consistency_manual_ratio,
            show_ntc=False,
        )
        fig.suptitle(
            f"Consistency Metrics ({label}) — {total_feats} features",
            fontsize=13,
            fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(
            plots_dir / f"map_consistency{suffix}.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)
        _logger.info(f"  Saved plots/map_consistency{suffix}.png")
        return (
            consistency_corum_map,
            consistency_corum_ratio,
            consistency_manual_map,
            consistency_manual_ratio,
        )
    except Exception as exc:
        _logger.error(f"  Consistency metrics ({label}) failed: {exc}")
        return None, 0.0, None, 0.0


# =============================================================================
# Phase 2: Aggregation (top-level entry point)
# =============================================================================


def aggregate_channels(
    output_dir: str,
    norm_method: str = "ntc",
    per_unit_subdir: str = "per_channel",
    distance: str = "cosine",
    random_seed: int = 42,
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
    guide_blocks, gene_blocks, report_rows, total_cells = _load_per_unit_blocks(
        per_unit_dir, norm_method, _logger, distance=distance
    )
    if guide_blocks is None:
        return "FAILED: no valid per-channel data loaded"

    # Step 2: Concat + normalize
    adata_guide, adata_gene = _concat_and_normalize(
        guide_blocks, gene_blocks, norm_method, _logger
    )
    total_feats = adata_guide.n_vars

    # Step 3: Activity scoring
    metrics_dir = output_dir / "metrics"
    activity_map, r, a = _score_activity_aggregated(
        adata_guide, metrics_dir, _logger, distance=distance
    )

    # Step 4: Save h5ads (before slow metrics) — store X_pca now; UMAP/PHATE added after step 6
    adata_guide.obsm["X_pca"] = np.asarray(adata_guide.X, dtype=np.float32)
    variance_ratio_per_pc = np.array(
        [float(row.get("explained_variance", 0)) for row in report_rows],
        dtype=np.float32,
    )
    adata_guide.uns["pca"] = {
        "params": {"n_components": total_feats, "zero_center": True}
    }
    _save_aggregated_h5ads(
        adata_guide,
        adata_gene,
        report_rows,
        output_dir,
        r,
        a,
        norm_method,
        total_cells,
        _logger,
    )

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
            fig.suptitle(
                f"Phenotypic Activity — {total_feats} features",
                fontsize=13,
                fontweight="bold",
            )
            fig.tight_layout()
            fig.savefig(plots_dir / "map_activity.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            _logger.info(f"  Saved plots/map_activity.png")
        except Exception as e:
            _logger.warning(f"  Activity plot failed: {e}")

    plot_sweep_curves_summary(
        per_unit_dir, output_dir, plots_dir, r, a, plt, _logger, min_pcs=MIN_PCS
    )

    # Step 6: Embeddings (UMAP + PHATE) — stored directly into adata_guide.obsm/uns
    # Returns adata_gene_embed (NTC-split gene-level object) with embeddings in obsm
    metric_lookup = build_metric_lookup(activity_map)
    adata_gene_embed = _compute_and_plot_embeddings(
        adata_guide, metric_lookup, plots_dir, plt, _logger,
        random_seed=random_seed,
    )
    # Re-save guide + gene-embed with embeddings now populated
    _atomic_write_h5ad(adata_guide, output_dir / "guide_pca_optimized.h5ad", _logger)
    if adata_gene_embed is not None:
        _atomic_write_h5ad(
            adata_gene_embed, output_dir / "gene_embedding_pca_optimized.h5ad", _logger
        )
    _logger.info(
        "  Re-saved guide_pca_optimized.h5ad + gene_embedding_pca_optimized.h5ad with embeddings"
    )

    # Step 7: Distinctiveness + consistency (slow) — active-only
    dist_map, dist_ratio = _score_distinctiveness(
        adata_guide,
        activity_map,
        r,
        total_feats,
        plots_dir,
        metrics_dir,
        plt,
        _logger,
        distance=distance,
    )
    corum_map, corum_ratio, chad_map, chad_ratio = _score_consistency(
        adata_gene,
        activity_map,
        total_feats,
        plots_dir,
        metrics_dir,
        plt,
        _logger,
        distance=distance,
    )

    # Per-reporter bar chart with all 4 aggregate baselines (active-only)
    plot_channel_peaks_bar(
        report_rows,
        r,
        plots_dir,
        plt,
        _logger,
        dist_ratio=dist_ratio,
        corum_ratio=corum_ratio,
        chad_ratio=chad_ratio,
    )

    # Per-reporter bar chart — unfiltered (all genes, not just active)
    dist_ratio_all, corum_ratio_all, chad_ratio_all = None, None, None
    try:
        from ops_utils.analysis.map_scores import (
            phenotypic_distinctivness,
            phenotypic_consistency_corum,
            phenotypic_consistency_manual_annotation,
        )

        _, dist_ratio_all = phenotypic_distinctivness(
            adata_guide,
            plot_results=False,
            null_size=100_000,
            distance=distance,
        )
        _, corum_ratio_all = phenotypic_consistency_corum(
            adata_gene,
            plot_results=False,
            cache_similarity=True,
            distance=distance,
        )
        _, chad_ratio_all = phenotypic_consistency_manual_annotation(
            adata_gene,
            plot_results=False,
            cache_similarity=True,
            distance=distance,
            annotation_path=CHAD_ANNOTATION_PATH,
        )
        _logger.info(
            f"  Unfiltered aggregate baselines: dist={dist_ratio_all:.1%} corum={corum_ratio_all:.1%} chad={chad_ratio_all:.1%}"
        )
    except Exception as e:
        _logger.warning(f"  Unfiltered aggregate baselines failed: {e}")

    # Build report rows remapped to the _all columns for the unfiltered plot
    unfiltered_rows = []
    for row in report_rows:
        r2 = dict(row)
        r2["distinctiveness"] = r2.get("distinctiveness_all", float("nan"))
        r2["corum"] = r2.get("corum_all", float("nan"))
        r2["chad"] = r2.get("chad_all", float("nan"))
        unfiltered_rows.append(r2)
    plot_channel_peaks_bar(
        unfiltered_rows,
        r,
        plots_dir,
        plt,
        _logger,
        dist_ratio=dist_ratio_all,
        corum_ratio=corum_ratio_all,
        chad_ratio=chad_ratio_all,
        filename="per_channel_peaks_all_genes.png",
    )

    # Bar charts for all 4 metrics (per-perturbation mAP)
    plot_metric_map_bar(
        activity_map, "Activity", "perturbation", r, plots_dir, plt, _logger
    )
    plot_metric_map_bar(
        dist_map, "Distinctiveness", "perturbation", dist_ratio, plots_dir, plt, _logger
    )
    if corum_map is not None:
        corum_entity_col = (
            "complex_id" if "complex_id" in corum_map.columns else corum_map.columns[0]
        )
        plot_metric_map_bar(
            corum_map,
            "Consistency (CORUM)",
            corum_entity_col,
            corum_ratio,
            plots_dir,
            plt,
            _logger,
        )
    if chad_map is not None:
        chad_entity_col = (
            "complex_num" if "complex_num" in chad_map.columns else chad_map.columns[0]
        )
        plot_metric_map_bar(
            chad_map,
            "Consistency (CHAD)",
            chad_entity_col,
            chad_ratio,
            plots_dir,
            plt,
            _logger,
        )

    _chad_path = CHAD_ANNOTATION_PATH or "/hpc/projects/icd.ops/configs/gene_clusters/chad_positive_controls_v4.yml"
    if adata_gene_embed is not None and "X_umap" in adata_gene_embed.obsm:
        try:
            import yaml as _yaml
            with open(_chad_path) as f:
                chad_clusters = _yaml.safe_load(f)
            gene_to_cluster = {}
            for cid, cdata in chad_clusters.items():
                name = cdata.get("name", f"cluster_{cid}")
                for gene in cdata.get("genes", []):
                    gene_to_cluster[gene.strip()] = name

            _plot_chad_umap(
                adata_gene_embed.obsm["X_umap"],
                adata_gene_embed.obs["perturbation"].values,
                gene_to_cluster,
                plots_dir / "umap_chad_clusters.png",
                plt, _logger,
            )
        except Exception as e:
            _logger.warning(f"  CHAD UMAP failed: {e}")

    # Extra overlays (super-category, Leiden, interactive HTML)
    try:
        from ops_model.post_process.combination.embedding_overlays import (
            save_extra_overlays,
        )

        save_extra_overlays(
            adata_guide=adata_guide,
            adata_gene_embed=adata_gene_embed,
            plots_dir=plots_dir,
            plt=plt,
            activity_map=activity_map,
            dist_map=dist_map,
            corum_map=corum_map,
            chad_map=chad_map,
            chad_path_override=CHAD_ANNOTATION_PATH,
            _logger=_logger,
        )
        # Re-save h5ads now that leiden_r* columns + neighbors graph have been
        # added to the in-memory adata objects by save_extra_overlays
        _atomic_write_h5ad(adata_guide, output_dir / "guide_pca_optimized.h5ad", _logger)
        if adata_gene_embed is not None:
            _atomic_write_h5ad(
                adata_gene_embed,
                output_dir / "gene_embedding_pca_optimized.h5ad",
                _logger,
            )
    except Exception as e:
        _logger.warning(f"  Extra overlays failed: {e}")

    elapsed = time.time() - t_start
    _logger.info(f"\nDone in {elapsed/60:.1f} minutes")
    _logger.info(f"  {len(report_rows)} channels, {total_feats} total PCA features")
    _logger.info(f"  Baseline: {r:.1%} active, AUC={a:.4f}")

    return f"SUCCESS: {total_feats} features, {r:.1%} active, AUC={a:.4f}"


# =============================================================================
# Phase 3 (optional): Second-pass PCA on concatenated guide features
# =============================================================================


def _save_pc_marker_contributions(
    components: np.ndarray,
    input_feature_names: List[str],
    n_pcs: int,
    cumvar: np.ndarray,
    out_dir: Path,
    plots_dir: Path,
    plt,
    _logger,
    top_n_pcs_for_bars: int = 50,
    top_k_markers_per_pc: int = 3,
) -> None:
    """For each second-pass PC, attribute variance back to source markers.

    Input feature names follow the ``<signal>_PC<n>`` convention from Phase 1
    (``_save_sweep_outputs``). For each second-pass PC ``sPCi`` we sum the
    squared loadings (``components[i, j] ** 2``) of all input features that
    came from a given signal — that's the share of ``sPCi``'s unit variance
    explained by that marker.

    Outputs:
      ``second_pca_marker_contributions.csv`` — wide table, rows=sPCi,
        columns=marker. Values sum to 1 across markers within each row.
      ``plots/second_pca_marker_contributions.png`` — heatmap (n_pcs x markers).
      ``plots/second_pca_marker_contributions_stacked.png`` — stacked bar of
        the top ``top_n_pcs_for_bars`` PCs.
      ``second_pca_top_markers_per_pc.csv`` — the top ``top_k_markers_per_pc``
        markers for each PC with their fractional contribution.
    """
    import re

    pat = re.compile(r"^(?P<signal>.+)_PC\d+$")
    feat_to_signal: List[str] = []
    for f in input_feature_names:
        m = pat.match(f)
        feat_to_signal.append(m.group("signal") if m else f)

    feat_to_signal_arr = np.asarray(feat_to_signal)
    unique_signals = sorted(set(feat_to_signal))

    # Squared loadings, normalized so each PC's row sums to 1
    sq = np.square(components).astype(np.float64)  # (n_pcs, n_feats_in)
    row_sums = sq.sum(axis=1, keepdims=True)
    sq_norm = np.divide(sq, row_sums, out=np.zeros_like(sq), where=row_sums > 0)

    contrib = np.zeros((n_pcs, len(unique_signals)), dtype=np.float64)
    for s_idx, signal in enumerate(unique_signals):
        cols = np.where(feat_to_signal_arr == signal)[0]
        if cols.size == 0:
            continue
        contrib[:, s_idx] = sq_norm[:, cols].sum(axis=1)

    pc_names = [f"sPC{i}" for i in range(n_pcs)]
    explained_var_per_pc = np.diff(np.concatenate([[0.0], cumvar])).astype(np.float64)
    df = pd.DataFrame(contrib, index=pc_names, columns=unique_signals)
    df.insert(0, "explained_variance", explained_var_per_pc[:n_pcs])
    df.to_csv(out_dir / "second_pca_marker_contributions.csv", index_label="pc")
    _logger.info("  Saved second_pca_marker_contributions.csv")

    # Top-K markers per PC (long-format CSV)
    top_rows = []
    for i in range(n_pcs):
        order = np.argsort(-contrib[i])
        for rank in range(min(top_k_markers_per_pc, len(unique_signals))):
            j = order[rank]
            top_rows.append({
                "pc": pc_names[i],
                "rank": rank + 1,
                "signal": unique_signals[j],
                "contribution": float(contrib[i, j]),
                "explained_variance": float(explained_var_per_pc[i]),
            })
    pd.DataFrame(top_rows).to_csv(out_dir / "second_pca_top_markers_per_pc.csv", index=False)
    _logger.info("  Saved second_pca_top_markers_per_pc.csv")

    # Heatmap (n_pcs x markers)
    try:
        fig_w = max(10, 0.25 * len(unique_signals) + 4)
        fig_h = max(6, 0.18 * n_pcs + 2)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(contrib, aspect="auto", cmap="viridis", vmin=0, vmax=contrib.max() or 1)
        ax.set_xticks(range(len(unique_signals)))
        ax.set_xticklabels(unique_signals, rotation=80, ha="right", fontsize=8)
        step = max(1, n_pcs // 60)
        ax.set_yticks(range(0, n_pcs, step))
        ax.set_yticklabels([pc_names[i] for i in range(0, n_pcs, step)], fontsize=8)
        ax.set_xlabel("Source marker (Phase 1 signal)")
        ax.set_ylabel("Second-pass PC")
        ax.set_title(
            f"Second-pass PC × marker contribution (squared loadings, row-normalized) — "
            f"{n_pcs} PCs × {len(unique_signals)} markers"
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cbar.set_label("share of PC variance")
        fig.tight_layout()
        fig.savefig(plots_dir / "second_pca_marker_contributions.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        _logger.info("  Saved plots/second_pca_marker_contributions.png")
    except Exception as exc:
        _logger.warning(f"  Heatmap failed: {exc}")

    # Stacked bar of top PCs — adjacent signals must look very different in
    # BOTH color and hatch, so any segment in the stack can be matched to its
    # legend entry by either dimension. We walk color and hatch on independent
    # coprime strides near n/phi (~138°), so each step i→i+1 jumps far in hue
    # AND far in the hatch list, and neither dimension repeats until its full
    # cycle is exhausted.
    try:
        n_show = min(top_n_pcs_for_bars, n_pcs)
        import math

        import seaborn as sns

        hatch_patterns = [
            "", "///", "\\\\\\", "xxx", "...", "+++",
            "ooo", "|||", "---", "**", "OO", "//\\\\",
        ]
        n_colors = 40
        n_hatches = len(hatch_patterns)
        husl = sns.color_palette("husl", n_colors)
        phi = (1 + math.sqrt(5)) / 2

        def _coprime_stride(n: int, target: float) -> int:
            s = max(1, int(round(target)))
            for d in range(n):
                for sign in (1, -1):
                    cand = s + sign * d
                    if 1 <= cand < n and math.gcd(cand, n) == 1:
                        return cand
            return 1

        color_stride = _coprime_stride(n_colors, n_colors / phi)
        hatch_stride = _coprime_stride(n_hatches, n_hatches / phi)
        signal_style = {
            sig: (
                husl[(i * color_stride) % n_colors],
                hatch_patterns[(i * hatch_stride) % n_hatches],
            )
            for i, sig in enumerate(unique_signals)
        }

        fig, ax = plt.subplots(figsize=(max(14, 0.50 * n_show + 4), 9))
        bottom = np.zeros(n_show)
        for s_idx, signal in enumerate(unique_signals):
            vals = contrib[:n_show, s_idx]
            if vals.sum() < 1e-6:
                continue
            color, hatch = signal_style[signal]
            ax.bar(
                range(n_show),
                vals,
                bottom=bottom,
                color=color,
                hatch=hatch,
                label=signal if len(signal) <= 35 else signal[:32] + "...",
                width=0.85,
                edgecolor="black",
                linewidth=0.4,
            )
            bottom += vals
        ax.set_xticks(range(n_show))
        ax.set_xticklabels(pc_names[:n_show], rotation=70, fontsize=8)
        ax.set_ylabel("share of PC variance")
        ax.set_xlabel("Second-pass PC (in order)")
        ax.set_title(
            f"Marker contribution to top {n_show} second-pass PCs"
        )
        ax.set_ylim(0, 1.0)
        ncol = 2 if len(unique_signals) > 20 else 1
        ax.legend(
            bbox_to_anchor=(1.02, 1.0), loc="upper left",
            fontsize=8, ncol=ncol, columnspacing=0.6, handletextpad=0.3,
        )
        fig.tight_layout()
        fig.savefig(
            plots_dir / "second_pca_marker_contributions_stacked.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)
        _logger.info("  Saved plots/second_pca_marker_contributions_stacked.png")
    except Exception as exc:
        _logger.warning(f"  Stacked bar failed: {exc}")

    # Top-3 summary log
    summary_lines = []
    for i in range(min(50, n_pcs)):
        order = np.argsort(-contrib[i])[:3]
        bits = ", ".join(
            f"{unique_signals[j]}({contrib[i, j]*100:.0f}%)" for j in order
        )
        summary_lines.append(f"  {pc_names[i]} (var={explained_var_per_pc[i]:.3f}): {bits}")
    if summary_lines:
        _logger.info("  Top contributors per second-pass PC:")
        for line in summary_lines:
            _logger.info(line)


def apply_second_pass_pca(
    output_dir: str,
    threshold: float = 0.0,
    distance: str = "cosine",
    norm_method: str = "ntc",
    subdir: Optional[str] = None,
    run_sweep: bool = True,
    sweep_thresholds: Optional[List[float]] = None,
    random_seed: int = 42,
) -> str:
    """Second-pass PCA on the already-concatenated NTC-normalized guide features.

    Reads ``<output_dir>/guide_pca_optimized.h5ad`` (output of
    :func:`aggregate_channels`), fits a PCA on its horizontally-concatenated
    guide features, retains the top ``threshold`` of cumulative variance,
    re-aggregates to gene level, re-scores all phenotypic metrics, and writes
    the results to ``<output_dir>/<subdir>/`` (default
    ``second_pca_{threshold:.0%}``).
    """
    _logger = _init_sweep_logger()
    t_start = time.time()
    output_dir = Path(output_dir)

    # threshold <= 0 means "consensus sweep" — pick the threshold that maximizes the
    # normalized sum of activity + distinctiveness + chad. Forces the sweep on.
    use_consensus = threshold is None or threshold <= 0
    if use_consensus:
        run_sweep = True
        if subdir is None:
            subdir = "second_pca_consensus"
    else:
        if subdir is None:
            subdir = f"second_pca_{int(round(threshold * 100)):02d}"
    out_dir = output_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    guide_path = output_dir / "guide_pca_optimized.h5ad"
    if not guide_path.exists():
        return f"FAILED: {guide_path} not found — run aggregation first"

    _logger.info(f"Loading first-pass concatenated guide h5ad: {guide_path}")
    adata_in = ad.read_h5ad(guide_path)
    n_guides = adata_in.n_obs
    n_feats_in = adata_in.n_vars
    _logger.info(
        f"  Input: {n_guides:,} guides x {n_feats_in:,} concatenated NTC-normalized features"
    )

    X_in = np.asarray(adata_in.X, dtype=np.float32)
    X_in = np.nan_to_num(X_in, nan=0.0, posinf=0.0, neginf=0.0)
    _logger.info(f"  Fitting second-pass PCA on {n_guides:,} x {n_feats_in:,} matrix...")
    t_pca = time.time()
    X_pcs, cumvar, pca_model = fit_pca(X_in)
    _logger.info(
        f"  PCA done in {time.time() - t_pca:.0f}s — {X_pcs.shape[1]} components computed"
    )

    # Sweep variance thresholds at guide level (input is already NTC-normalized)
    sweep_result = None
    if run_sweep:
        thresholds = sweep_thresholds or DEFAULT_SWEEP_THRESHOLDS
        _logger.info(
            f"  Sweeping {len(thresholds)} variance thresholds at guide level..."
        )
        t_sweep = time.time()
        sweep_result = _run_guide_threshold_sweep(
            X_pcs,
            cumvar,
            adata_in.obs,
            thresholds,
            _logger=_logger,
            distance=distance,
        )
        _logger.info(f"  Sweep done in {time.time() - t_sweep:.0f}s")

    if use_consensus:
        if sweep_result is None:
            return f"FAILED: consensus mode requested but sweep produced no valid thresholds"
        threshold = float(sweep_result["consensus_t"])
        n_pcs = int(sweep_result["consensus_n"])
        _logger.info(
            f"  Consensus mode: chose {threshold:.0%} ({n_pcs} PCs) — "
            f"max of normalized act+dist+chad"
        )
    else:
        n_pcs = n_pcs_for_threshold(cumvar, threshold)
        _logger.info(
            f"  Retaining {n_pcs} PCs at {threshold:.0%} "
            f"(explained={float(cumvar[n_pcs - 1]):.3f})"
        )
        if sweep_result is not None:
            _logger.info(
                f"  (sweep consensus peak: {sweep_result['consensus_t']:.0%} = "
                f"{sweep_result['consensus_n']} PCs)"
            )

    X_reduced = X_pcs[:, :n_pcs].astype(np.float32)
    pc_names = [f"sPC{i}" for i in range(n_pcs)]

    obs = adata_in.obs.copy()
    adata_guide = ad.AnnData(
        X=X_reduced,
        obs=obs,
        var=pd.DataFrame(index=pc_names),
    )
    variance_ratio_per_pc = np.diff(np.concatenate([[0.0], cumvar])).astype(np.float32)
    input_feature_names = list(adata_in.var_names)
    second_pca_uns = {
        "input_features": int(n_feats_in),
        "input_path": str(guide_path),
        "threshold": float(threshold),
        "n_pcs": int(n_pcs),
        "explained_variance": float(cumvar[n_pcs - 1]),
        "components": pca_model.components_[:n_pcs].astype(np.float32),
        "input_feature_names": input_feature_names,
    }
    adata_guide.obsm["X_pca"] = X_reduced.copy()
    adata_guide.uns["pca"] = {
        "variance_ratio": variance_ratio_per_pc[:n_pcs],
        "params": {
            "n_components": int(n_pcs),
            "threshold": float(threshold),
            "zero_center": True,
        },
    }
    adata_guide.uns["second_pca"] = second_pca_uns
    adata_guide.uns["norm_method"] = norm_method

    adata_guide = _prepare_for_copairs(adata_guide)
    adata_gene = aggregate_to_level(
        adata_guide, "gene", preserve_batch_info=False, subsample_controls=False
    )
    adata_gene = _prepare_for_copairs(adata_gene)
    adata_gene.obsm["X_pca"] = np.asarray(adata_gene.X, dtype=np.float32)
    adata_gene.uns["pca"] = adata_guide.uns["pca"]
    adata_gene.uns["second_pca"] = second_pca_uns
    adata_gene.uns["norm_method"] = norm_method

    total_feats = adata_guide.n_vars
    _logger.info(
        f"  After 2nd PCA: {adata_guide.n_obs:,} guides x {total_feats} PCs, "
        f"{adata_gene.n_obs:,} genes"
    )

    metrics_dir = out_dir / "metrics"
    activity_map, r, a = _score_activity_aggregated(
        adata_guide, metrics_dir, _logger, distance=distance
    )

    adata_guide.uns["pca_optimized"] = True
    adata_guide.uns["second_pca_applied"] = True
    adata_guide.uns["baseline_activity"] = float(r)
    adata_guide.uns["baseline_auc"] = float(a)
    adata_gene.uns["pca_optimized"] = True
    adata_gene.uns["second_pca_applied"] = True
    adata_guide.write_h5ad(out_dir / "guide_pca_optimized.h5ad")
    adata_gene.write_h5ad(out_dir / "gene_pca_optimized.h5ad")
    _logger.info(f"  Saved guide/gene_pca_optimized.h5ad in {subdir}/")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(np.arange(1, len(cumvar) + 1), cumvar, lw=2, color="steelblue")
        ax.axhline(threshold, ls="--", color="red", alpha=0.6, label=f"{threshold:.0%}")
        ax.axvline(n_pcs, ls=":", color="black", alpha=0.5, label=f"{n_pcs} PCs")
        ax.set_xlabel("PC index")
        ax.set_ylabel("Cumulative explained variance")
        ax.set_title(
            f"Second-pass PCA — {n_feats_in} features → {n_pcs} PCs at {threshold:.0%}"
        )
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "second_pca_cumvar.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        _logger.info("  Saved plots/second_pca_cumvar.png")
    except Exception as exc:
        _logger.warning(f"  Cumvar plot failed: {exc}")

    if sweep_result is not None:
        sweep_df = pd.DataFrame(sweep_result["sweep_rows"])
        sweep_csv = out_dir / "second_pca_sweep.csv"
        sweep_df.to_csv(sweep_csv, index=False)
        _logger.info(f"  Saved {sweep_csv.name}")
        try:
            plot_pca_sweep(
                sweep_df,
                signal="2nd PCA",
                peak_t=threshold,
                peak_n=n_pcs,
                suptitle=(
                    f"2nd-pass PCA sweep — {n_feats_in} concat features → "
                    f"chosen={n_pcs} PCs @ {threshold:.0%} "
                    f"(consensus={sweep_result['consensus_t']:.0%} = "
                    f"{sweep_result['consensus_n']} PCs)"
                ),
                plots_dir=plots_dir,
                file_prefix="second_pca",
                fixed_threshold=threshold,
                sweep_peak_t=sweep_result["consensus_t"],
                metric_peaks={
                    "peak_act_t": sweep_result["peak_act_t"],
                    "peak_dist_t": sweep_result["peak_dist_t"],
                    "peak_chad_t": sweep_result["peak_chad_t"],
                },
            )
            _logger.info("  Saved plots/second_pca_sweep.png")
        except Exception as exc:
            _logger.warning(f"  Sweep plot failed: {exc}")

    # Per-PC marker contribution (which signals drive each second-pass PC)
    try:
        _save_pc_marker_contributions(
            components=pca_model.components_[:n_pcs],
            input_feature_names=input_feature_names,
            n_pcs=n_pcs,
            cumvar=cumvar,
            out_dir=out_dir,
            plots_dir=plots_dir,
            plt=plt,
            _logger=_logger,
        )
    except Exception as exc:
        _logger.warning(f"  Per-PC marker contribution plot failed: {exc}")

    if activity_map is not None:
        try:
            fig, ax = plt.subplots(figsize=(8, 7))
            plot_map_scatter(ax, activity_map, "Activity", r, show_ntc=False)
            fig.suptitle(
                f"Phenotypic Activity (2nd PCA) — {total_feats} features",
                fontsize=13,
                fontweight="bold",
            )
            fig.tight_layout()
            fig.savefig(plots_dir / "map_activity.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            _logger.warning(f"  Activity plot failed: {exc}")

    metric_lookup = build_metric_lookup(activity_map) if activity_map is not None else {}
    adata_gene_embed = _compute_and_plot_embeddings(
        adata_guide, metric_lookup, plots_dir, plt, _logger,
        random_seed=random_seed,
    )
    _atomic_write_h5ad(adata_guide, out_dir / "guide_pca_optimized.h5ad", _logger)
    if adata_gene_embed is not None:
        _atomic_write_h5ad(
            adata_gene_embed, out_dir / "gene_embedding_pca_optimized.h5ad", _logger
        )

    dist_map, dist_ratio = _score_distinctiveness(
        adata_guide,
        activity_map,
        r,
        total_feats,
        plots_dir,
        metrics_dir,
        plt,
        _logger,
        distance=distance,
    )
    corum_map, corum_ratio, chad_map, chad_ratio = _score_consistency(
        adata_gene,
        activity_map,
        total_feats,
        plots_dir,
        metrics_dir,
        plt,
        _logger,
        distance=distance,
    )

    if activity_map is not None:
        plot_metric_map_bar(
            activity_map, "Activity", "perturbation", r, plots_dir, plt, _logger
        )
    if dist_map is not None:
        plot_metric_map_bar(
            dist_map,
            "Distinctiveness",
            "perturbation",
            dist_ratio,
            plots_dir,
            plt,
            _logger,
        )
    if corum_map is not None:
        col = (
            "complex_id" if "complex_id" in corum_map.columns else corum_map.columns[0]
        )
        plot_metric_map_bar(
            corum_map,
            "Consistency (CORUM)",
            col,
            corum_ratio,
            plots_dir,
            plt,
            _logger,
        )
    if chad_map is not None:
        col = (
            "complex_num" if "complex_num" in chad_map.columns else chad_map.columns[0]
        )
        plot_metric_map_bar(
            chad_map,
            "Consistency (CHAD)",
            col,
            chad_ratio,
            plots_dir,
            plt,
            _logger,
        )

    _chad_path = (
        CHAD_ANNOTATION_PATH
        or "/hpc/projects/icd.ops/configs/gene_clusters/chad_positive_controls_v4.yml"
    )
    if adata_gene_embed is not None and "X_umap" in adata_gene_embed.obsm:
        try:
            import yaml as _yaml

            with open(_chad_path) as f:
                chad_clusters = _yaml.safe_load(f)
            gene_to_cluster = {}
            for cid, cdata in chad_clusters.items():
                name = cdata.get("name", f"cluster_{cid}")
                for gene in cdata.get("genes", []):
                    gene_to_cluster[gene.strip()] = name
            _plot_chad_umap(
                adata_gene_embed.obsm["X_umap"],
                adata_gene_embed.obs["perturbation"].values,
                gene_to_cluster,
                plots_dir / "umap_chad_clusters.png",
                plt,
                _logger,
            )
        except Exception as exc:
            _logger.warning(f"  CHAD UMAP failed: {exc}")

    # Extra overlays (super-category, Leiden, interactive HTML)
    try:
        from ops_model.post_process.combination.embedding_overlays import (
            save_extra_overlays,
        )

        save_extra_overlays(
            adata_guide=adata_guide,
            adata_gene_embed=adata_gene_embed,
            plots_dir=plots_dir,
            plt=plt,
            activity_map=activity_map,
            dist_map=dist_map,
            corum_map=corum_map,
            chad_map=chad_map,
            chad_path_override=CHAD_ANNOTATION_PATH,
            _logger=_logger,
        )
        # Re-save h5ads with leiden_r* columns + neighbors graph just added
        _atomic_write_h5ad(adata_guide, out_dir / "guide_pca_optimized.h5ad", _logger)
        if adata_gene_embed is not None:
            _atomic_write_h5ad(
                adata_gene_embed,
                out_dir / "gene_embedding_pca_optimized.h5ad",
                _logger,
            )
    except Exception as exc:
        _logger.warning(f"  Extra overlays failed: {exc}")

    summary = {
        "n_guides": int(n_guides),
        "n_features_input": int(n_feats_in),
        "threshold": float(threshold),
        "n_pcs": int(n_pcs),
        "explained_variance": float(cumvar[n_pcs - 1]),
        "activity": float(r),
        "auc": float(a),
        "distinctiveness": float(dist_ratio),
        "corum": float(corum_ratio),
        "chad": float(chad_ratio),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "second_pca_summary.csv", index=False)

    elapsed = time.time() - t_start
    _logger.info(
        f"Done in {elapsed/60:.1f} min — {n_pcs}/{n_feats_in} PCs @ {threshold:.0%}, "
        f"act={r:.1%} AUC={a:.4f} dist={dist_ratio:.1%} corum={corum_ratio:.1%} chad={chad_ratio:.1%}"
    )
    return (
        f"SUCCESS: {n_pcs}/{n_feats_in} PCs @ {threshold:.0%}, "
        f"act={r:.1%} AUC={a:.4f}"
    )


# =============================================================================
# CLI: mode handlers
# =============================================================================


def _discover_experiment_pairs(
    cp_override,
    include_cellpainting: bool = False,
    include_4i: bool = False,
    include_cp: bool = False,
    include_standard: bool = True,
    force_include: Optional[set] = None,
):
    """Common experiment discovery for SLURM modes. Returns (all_pairs, attr_config, storage_roots, feature_dir, maps_path).

    ``force_include`` (set of exp short names like ``{"ops0146"}``) bypasses
    discovery's bad-experiment / non-default-library filters for the listed
    experiments — used when the caller explicitly asked for them via
    ``--experiments``.
    """
    attr_config = load_attribution_config()
    storage_roots = get_storage_roots(attr_config)
    feature_dir = cp_override or attr_config.get("feature_dir", "dino_features")
    maps_path = get_channel_maps_path()
    if cp_override == "cell-profiler":
        all_pairs = discover_cellprofiler_experiments(
            storage_roots,
            include_cellpainting=include_cellpainting,
            include_4i=include_4i,
            include_cp=include_cp,
            include_standard=include_standard,
            force_include=force_include,
        )
    else:
        all_pairs = discover_dino_experiments(
            storage_roots,
            feature_dir,
            include_cellpainting=include_cellpainting,
            include_4i=include_4i,
            include_cp=include_cp,
            include_standard=include_standard,
            force_include=force_include,
        )
    # 4i nucleus (DAPI) channels are nucleus segmentation refs, not biological
    # signals — drop them from the pool. CP DAPI stays since CP uses it as a
    # bona-fide channel in the multiplex panel.
    if include_4i:
        before = len(all_pairs)
        dropped_4i_dapi = [e for e, c in all_pairs if c == "4i:DAPI"]
        all_pairs = [(e, c) for e, c in all_pairs if c != "4i:DAPI"]
        if dropped_4i_dapi:
            print(
                f"  WARNING: dropping 4i:DAPI nucleus channel — it's a "
                f"segmentation ref, not a biological signal "
                f"({len(dropped_4i_dapi)} pairs across "
                f"{len(set(dropped_4i_dapi))} experiments). "
                f"CP DAPI is kept."
            )
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


def _aggregate_then_second_pca(
    output_dir: str,
    norm_method: str,
    per_unit_subdir: str,
    distance: str,
    second_pca_threshold: float,
    second_pca_subdir: Optional[str],
    second_pca_run_sweep: bool,
    second_pca_sweep_thresholds: Optional[List[float]],
    random_seed: int = 42,
) -> str:
    """Run Phase 2 aggregation and the 2nd-pass PCA back-to-back.

    Top-level (picklable) helper for SLURM jobs that want both steps in one slot.
    """
    agg_result = aggregate_channels(
        output_dir=output_dir,
        norm_method=norm_method,
        per_unit_subdir=per_unit_subdir,
        distance=distance,
        random_seed=random_seed,
    )
    if str(agg_result).startswith("FAILED"):
        return agg_result
    second_result = apply_second_pass_pca(
        output_dir=output_dir,
        threshold=second_pca_threshold,
        distance=distance,
        norm_method=norm_method,
        subdir=second_pca_subdir,
        run_sweep=second_pca_run_sweep,
        sweep_thresholds=second_pca_sweep_thresholds,
        random_seed=random_seed,
    )
    return f"{agg_result} | 2nd-pca: {second_result}"


def _build_second_pca_kwargs(args) -> Optional[Dict]:
    """Return kwargs to pass to apply_second_pass_pca, or None if --second-pca was not set."""
    if not getattr(args, "second_pca", False):
        return None
    sweep_thresholds = None
    raw = getattr(args, "second_pca_sweep_thresholds", None)
    if raw:
        sweep_thresholds = [float(t) for t in raw.split(",") if t.strip()]
    return {
        "second_pca_threshold": args.second_pca_threshold,
        "second_pca_subdir": args.second_pca_subdir,
        "second_pca_run_sweep": not args.second_pca_no_sweep,
        "second_pca_sweep_thresholds": sweep_thresholds,
    }


def _submit_aggregation_slurm(
    agg_output,
    norm_method,
    per_unit_subdir,
    agg_slurm_params,
    experiment_name,
    manifest_prefix,
    distance="cosine",
    second_pca_kwargs: Optional[Dict] = None,
    random_seed: int = 42,
):
    """Submit a single aggregation SLURM job (optionally chained with 2nd-pass PCA)."""
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

    if second_pca_kwargs is not None:
        job_func = _aggregate_then_second_pca
        job_kwargs = {
            "output_dir": agg_output,
            "norm_method": norm_method,
            "per_unit_subdir": per_unit_subdir,
            "distance": distance,
            "random_seed": random_seed,
            **second_pca_kwargs,
        }
        job_name = f"{manifest_prefix}_aggregate_2pca"
    else:
        job_func = aggregate_channels
        job_kwargs = {
            "output_dir": agg_output,
            "norm_method": norm_method,
            "per_unit_subdir": per_unit_subdir,
            "distance": distance,
            "random_seed": random_seed,
        }
        job_name = f"{manifest_prefix}_aggregate"

    agg_jobs = [
        {
            "name": job_name,
            "func": job_func,
            "kwargs": job_kwargs,
        }
    ]
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


def _submit_phase1_slurm(
    jobs,
    args,
    agg_output,
    per_unit_subdir,
    experiment_name,
    manifest_prefix,
    unit_label,
):
    """Submit Phase 1 SLURM jobs + auto-chain Phase 2 aggregation on completion."""
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

    slurm_params = _make_slurm_params(args)
    agg_slurm_params = _make_agg_slurm_params(args)

    def _on_phase1_complete(submitted_jobs, experiment):
        print(f"\nAll {unit_label} jobs complete. Submitting aggregation SLURM job...")
        _submit_aggregation_slurm(
            agg_output,
            args.norm_method,
            per_unit_subdir,
            agg_slurm_params,
            f"{manifest_prefix}_aggregation",
            f"{manifest_prefix}_agg",
            distance=args.distance,
            second_pca_kwargs=_build_second_pca_kwargs(args),
            random_seed=getattr(args, "seed", 42),
        )

    print(f"\nSubmitting {len(jobs)} {unit_label} SLURM jobs...")
    result = submit_parallel_jobs(
        jobs_to_submit=jobs,
        experiment=experiment_name,
        slurm_params=slurm_params,
        log_dir="pca_optimization",
        manifest_prefix=f"{manifest_prefix}_opt",
        wait_for_completion=True,
        post_completion_callback=_on_phase1_complete,
    )
    if result.get("failed"):
        print(f"\nWarning: {len(result['failed'])} {unit_label} failed")
        for name in result["failed"]:
            print(f"  - {name}")


def _handle_chad_umap_only(args, output_dir):
    """Only regenerate the CHAD-colored UMAP from existing gene embeddings."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import yaml as _yaml

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _logger = logging.getLogger(__name__)

    embed_path = output_dir / "gene_embedding_pca_optimized.h5ad"
    if not embed_path.exists():
        print(f"ERROR: {embed_path} not found. Run --aggregate-only first.")
        return

    _logger.info(f"Loading {embed_path}...")
    adata = ad.read_h5ad(embed_path)
    if "X_umap" not in adata.obsm:
        print("ERROR: No X_umap in gene embedding.")
        return

    chad_path = args.chad_annotation or "/hpc/projects/icd.ops/configs/gene_clusters/chad_positive_controls_v4.yml"
    with open(chad_path) as f:
        chad_clusters = _yaml.safe_load(f)

    # Optional cluster range filter
    if args.chad_cluster_range:
        lo, hi = map(int, args.chad_cluster_range.split("-"))
        chad_clusters = {k: v for k, v in chad_clusters.items() if isinstance(k, int) and lo <= k <= hi}
        _logger.info(f"Filtered to clusters {lo}-{hi} ({len(chad_clusters)} clusters)")

    gene_to_cluster = {}
    for cid, cdata in chad_clusters.items():
        name = cdata.get("name", f"cluster_{cid}")
        for gene in cdata.get("genes", []):
            gene_to_cluster[gene.strip()] = name

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.chad_umap_output or "umap_chad_clusters.png"
    out_path = plots_dir / out_name

    _plot_chad_umap(
        adata.obsm["X_umap"],
        adata.obs["perturbation"].values,
        gene_to_cluster,
        out_path,
        plt, _logger,
    )


def _handle_umap_only(args, output_dir):
    """Generate UMAP + PHATE embedding plots from existing optimized h5ads."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
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
        _logger.info(
            f"  Loaded activity metrics for {len(metric_lookup)} perturbations"
        )
    else:
        metric_lookup = {}
        _logger.warning(
            f"  No activity CSV found at {activity_csv}, UMAPs will be uncolored"
        )

    plots_dir = umap_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _compute_and_plot_embeddings(adata_guide, metric_lookup, plots_dir, plt, _logger)
    print("SUCCESS: Embedding plots saved")


def _stored_embedding_seed(adata, embed_name: str):
    """Pull the random_state used when an embedding (umap/phate) was last fit.
    Returns None if absent."""
    rec = (adata.uns.get(embed_name) or {}).get("params") or {}
    val = rec.get("random_state")
    try:
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _try_load_swept_umap(
    out_dir: Path, seed: int, _logger, nn: int = 15, md: float = 0.1,
):
    """Look for ``gene_umap_seed_sweep_coords_nn{nn}_md{md}.npz`` in
    ``out_dir`` and return the coords for ``seed`` if present. Otherwise
    return None. The cache is written by ``_run_seed_sweep`` and is the
    authoritative source for layouts users selected from the sweep canvas."""
    cache_path = Path(out_dir) / f"gene_umap_seed_sweep_coords_nn{nn}_md{md:g}.npz"
    if not cache_path.exists():
        return None
    try:
        z = np.load(cache_path)
        seeds = z["seeds"]
        coords_all = z["coords"]
        idx = np.where(seeds == int(seed))[0]
        if idx.size == 0:
            _logger.info(
                "  Sweep cache %s lacks seed=%d — falling back to refit",
                cache_path.name, seed,
            )
            return None
        _logger.info(
            "  Loaded gene UMAP coords for seed=%d from %s "
            "(matches --sweep-seed exactly)", seed, cache_path.name,
        )
        return np.asarray(coords_all[idx[0]], dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        _logger.warning("  Sweep cache unreadable (%s) — falling back to refit", exc)
        return None


def _recompute_embeddings_for_seed(
    adata,
    level_name: str,
    seed: int,
    _logger,
    out_dir=None,
    umap_n_neighbors: Optional[int] = None,
    umap_min_dist: Optional[float] = None,
):
    """Refit UMAP + PHATE in-place using ``seed``. Skips embeddings whose
    stored params already match. Returns the set of embedding names that
    were actually refit (so the caller knows whether to rewrite the h5ad).

    ``umap_n_neighbors`` / ``umap_min_dist`` are optional overrides — when
    set they bypass the sweep cache (the cache is keyed on default
    n_neighbors=15, min_dist=0.1) and force a refit at the requested params.
    """
    refit: set = set()
    n_obs = adata.n_obs
    nn = int(umap_n_neighbors) if umap_n_neighbors is not None else min(15, n_obs - 1)
    nn = min(nn, n_obs - 1)
    if nn < 2:
        _logger.warning(
            "  %s: too few obs (%d) — skipping embedding recompute", level_name, n_obs,
        )
        return refit
    md = float(umap_min_dist) if umap_min_dist is not None else 0.1

    if "X_pca" in adata.obsm:
        X = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
    else:
        X = np.asarray(adata.X, dtype=np.float32)

    stored_params = ((adata.uns.get("umap") or {}).get("params") or {})
    stored_seed = _stored_embedding_seed(adata, "umap")
    stored_nn = stored_params.get("n_neighbors")
    stored_md = stored_params.get("min_dist")

    # Look up the cache for the (nn, md) the user requested — sweep writes
    # one cache per param combo, so there's a hit if and only if a sweep was
    # run at these same params.
    cache_coords = None
    if level_name == "gene" and out_dir is not None:
        cache_coords = _try_load_swept_umap(out_dir, int(seed), _logger, nn=nn, md=md)

    needs_umap = (
        cache_coords is not None
        or stored_seed != int(seed)
        or (umap_n_neighbors is not None and stored_nn != nn)
        or (umap_min_dist is not None and stored_md != md)
    )
    if needs_umap:
        coords = cache_coords
        if coords is None:
            _logger.info(
                "  Refitting %s UMAP at seed=%d, n_neighbors=%d, min_dist=%g "
                "(stored: seed=%s, nn=%s, md=%s)",
                level_name, seed, nn, md,
                stored_seed, stored_nn, stored_md,
            )
            from umap import UMAP

            try:
                model = UMAP(
                    n_components=2, n_neighbors=nn, min_dist=md,
                    random_state=int(seed),
                )
                coords = model.fit_transform(X)
            except Exception as exc:  # noqa: BLE001
                _logger.warning("  %s UMAP refit failed: %s", level_name, exc)
                coords = None

        if coords is not None:
            adata.obsm["X_umap"] = np.asarray(coords, dtype=np.float32)
            adata.uns["umap"] = {"params": {
                "n_neighbors": nn,
                "min_dist": md,
                "random_state": int(seed),
                "metric": "euclidean",
            }}
            refit.add("umap")

    if _stored_embedding_seed(adata, "phate") != int(seed):
        try:
            import phate

            knn = min(15 if n_obs > 2000 else 10, n_obs - 1)
            if knn >= 2:
                _logger.info(
                    "  Recomputing %s PHATE at seed=%d (was %s)",
                    level_name, seed, _stored_embedding_seed(adata, "phate"),
                )
                coords = phate.PHATE(
                    n_components=2, knn=knn, decay=15, t="auto",
                    n_jobs=-1, random_state=int(seed), verbose=0,
                ).fit_transform(X)
                adata.obsm["X_phate"] = coords.astype(np.float32)
                adata.uns["phate"] = {"params": {
                    "knn": knn, "decay": 15, "t": "auto", "random_state": int(seed),
                }}
                refit.add("phate")
        except ImportError:
            _logger.warning("  PHATE recompute skipped: install phate")
        except Exception as exc:
            _logger.warning("  PHATE recompute failed: %s", exc)

    return refit


def _run_overlays_only(
    output_dir: str,
    seed: int = 42,
    umap_n_neighbors: Optional[int] = None,
    umap_min_dist: Optional[float] = None,
) -> str:
    """Picklable SLURM worker: regenerate HTML overlays from existing h5ads.

    If ``seed`` differs from the random_state stored in ``adata.uns["umap"]``
    (or ``["phate"]``), the corresponding embedding is refit in-place and the
    h5ad is rewritten so subsequent overlay regenerations stay consistent.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    _logger = logging.getLogger(__name__)

    out = Path(output_dir)
    guide_path = out / "guide_pca_optimized.h5ad"
    gene_path = out / "gene_embedding_pca_optimized.h5ad"

    if not guide_path.exists():
        return f"ERROR: {guide_path} not found. Run pipeline first."
    if not gene_path.exists():
        return f"ERROR: {gene_path} not found. Run pipeline first."

    _logger.info(f"Loading {guide_path}...")
    adata_guide = ad.read_h5ad(guide_path)
    _logger.info(f"Loading {gene_path}...")
    adata_gene_embed = ad.read_h5ad(gene_path)

    # Guide level: don't pass UMAP param overrides (gene-level only).
    if _recompute_embeddings_for_seed(adata_guide, "guide", seed, _logger, out_dir=out):
        _logger.info(f"  Rewriting {guide_path} with refit embeddings")
        _atomic_write_h5ad(adata_guide, guide_path, _logger)
    if _recompute_embeddings_for_seed(
        adata_gene_embed, "gene", seed, _logger, out_dir=out,
        umap_n_neighbors=umap_n_neighbors, umap_min_dist=umap_min_dist,
    ):
        _logger.info(f"  Rewriting {gene_path} with refit embeddings")
        _atomic_write_h5ad(adata_gene_embed, gene_path, _logger)

    metrics_dir = out / "metrics"

    def _load_csv(name):
        path = metrics_dir / name
        if path.exists():
            _logger.info(f"  Loaded {name}")
            return pd.read_csv(path)
        _logger.warning(f"  {name} not found — skipping")
        return None

    activity_map = _load_csv("phenotypic_activity.csv")
    dist_map = _load_csv("phenotypic_distinctiveness.csv")
    corum_map = _load_csv("phenotypic_consistency_corum.csv")
    chad_map = _load_csv("phenotypic_consistency_manual.csv")

    plots_dir = out / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    from ops_model.post_process.combination.embedding_overlays import save_extra_overlays

    save_extra_overlays(
        adata_guide=adata_guide,
        adata_gene_embed=adata_gene_embed,
        plots_dir=plots_dir,
        plt=plt,
        activity_map=activity_map,
        dist_map=dist_map,
        corum_map=corum_map,
        chad_map=chad_map,
        chad_path_override=CHAD_ANNOTATION_PATH,
        _logger=_logger,
    )
    # Persist obs additions from save_extra_overlays (CHAD / CORUM / supercategory
    # / leiden_r* / GO BP/CC / Reactome / KEGG / neighbors graph) onto disk.
    _atomic_write_h5ad(adata_guide, guide_path, _logger)
    _atomic_write_h5ad(adata_gene_embed, gene_path, _logger)
    return "SUCCESS: Overlays regenerated"


def _fit_umap_one_seed(seed: int, X, nn: int, min_dist: float = 0.1):
    """Picklable worker: fit UMAP at one seed and return (seed, coords, err)."""
    try:
        from umap import UMAP

        model = UMAP(
            n_components=2,
            n_neighbors=nn,
            min_dist=float(min_dist),
            random_state=int(seed),
        )
        coords = model.fit_transform(X)
        return int(seed), np.asarray(coords, dtype=np.float32), None
    except Exception as exc:  # noqa: BLE001
        return int(seed), None, str(exc)


def _run_seed_sweep(
    output_dir: str,
    n_seeds: int = 36,
    base_seed: int = 0,
    n_workers: int = 1,
    umap_n_neighbors: Optional[int] = None,
    umap_min_dist: Optional[float] = None,
) -> str:
    """Fit UMAP at ``n_seeds`` consecutive seeds on the gene-level h5ad and
    write a single PNG canvas (sqrt(n)×sqrt(n) panels) so different layouts
    can be compared at a glance. NTCs render as faded red x's. Fits run in
    parallel with ``n_workers`` joblib workers (each pinned to 1 thread)."""
    import math
    import time
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    _logger = logging.getLogger(__name__)

    out = Path(output_dir)
    gene_path = out / "gene_embedding_pca_optimized.h5ad"
    if not gene_path.exists():
        return f"ERROR: {gene_path} not found. Run pipeline first."

    _logger.info(f"Loading {gene_path}...")
    adata = ad.read_h5ad(gene_path)
    if "X_pca" in adata.obsm:
        X = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
    else:
        X = np.asarray(adata.X, dtype=np.float32)

    n_obs = adata.n_obs
    nn = int(umap_n_neighbors) if umap_n_neighbors is not None else min(15, n_obs - 1)
    nn = min(nn, n_obs - 1)
    if nn < 2:
        return f"ERROR: too few obs ({n_obs}) for UMAP"
    md = float(umap_min_dist) if umap_min_dist is not None else 0.1

    perts = (
        adata.obs["perturbation"].astype(str).values
        if "perturbation" in adata.obs.columns
        else np.asarray(adata.obs_names.values, dtype=str)
    )
    is_ntc = np.array([p.startswith("NTC") for p in perts])

    seeds = list(range(int(base_seed), int(base_seed) + int(n_seeds)))
    n_workers = max(1, min(int(n_workers), len(seeds)))
    _logger.info(
        f"Fitting {len(seeds)} UMAPs (n_obs={n_obs}, n_features={X.shape[1]}, "
        f"n_neighbors={nn}, min_dist={md:g}) on {n_workers} worker(s)..."
    )

    t0 = time.time()
    if n_workers > 1:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_workers, backend="loky", verbose=5)(
            delayed(_fit_umap_one_seed)(s, X, nn, md) for s in seeds
        )
    else:
        results = [_fit_umap_one_seed(s, X, nn, md) for s in seeds]
    _logger.info(f"All UMAP fits done in {time.time() - t0:.1f}s")

    coords_by_seed = {s: c for s, c, _ in results if c is not None}
    for s, _, err in results:
        if err is not None:
            _logger.warning(f"  UMAP failed for seed={s}: {err}")

    n_cols = int(math.ceil(math.sqrt(len(seeds))))
    n_rows = int(math.ceil(len(seeds) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.5, n_rows * 2.5),
        squeeze=False,
    )
    for i, seed in enumerate(seeds):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]
        coords = coords_by_seed.get(seed)
        if coords is None:
            ax.set_title(f"seed={seed} (failed)", fontsize=8)
            ax.axis("off")
            continue
        ax.scatter(
            coords[~is_ntc, 0], coords[~is_ntc, 1],
            s=2, c="#3a7", alpha=0.5, linewidths=0,
        )
        if is_ntc.any():
            ax.scatter(
                coords[is_ntc, 0], coords[is_ntc, 1],
                s=6, c="#cc6060", marker="x", alpha=0.7, linewidths=0.4,
            )
        ax.set_title(f"seed={seed}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
    for i in range(len(seeds), n_rows * n_cols):
        r, c = divmod(i, n_cols)
        axes[r][c].axis("off")

    param_tag = f"nn{nn}_md{md:g}"
    plots_dir = out / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / (
        f"gene_umap_seed_sweep_{n_seeds}_base{base_seed}_{param_tag}.png"
    )
    fig.suptitle(
        f"Gene UMAP seed sweep — {n_seeds} seeds (base={base_seed}, "
        f"n_neighbors={nn}, min_dist={md:g})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Persist the exact coords each seed produced so --overlays-only --seed N
    # can load them directly instead of re-fitting (UMAP's parallel reductions
    # aren't bit-deterministic across runs even with random_state pinned).
    cache_path = out / f"gene_umap_seed_sweep_coords_{param_tag}.npz"
    np.savez(
        cache_path,
        seeds=np.asarray(list(coords_by_seed.keys()), dtype=np.int64),
        coords=np.stack(list(coords_by_seed.values()), axis=0).astype(np.float32),
        n_neighbors=np.int32(nn),
        min_dist=np.float32(md),
    )
    _logger.info(f"Saved per-seed coords cache → {cache_path}")
    return f"SUCCESS: {out_path}"


def _handle_sweep_seed(args, output_dir):
    """Submit / run the seed-sweep canvas for the gene-level UMAP."""
    if getattr(args, "second_pca_only", False):
        subdir = getattr(args, "second_pca_subdir", None)
        if subdir is None:
            threshold = getattr(args, "second_pca_threshold", 0.0)
            subdir = (
                "second_pca_consensus"
                if threshold == 0.0
                else f"second_pca_{int(round(threshold * 100))}"
            )
        output_dir = output_dir / subdir
        print(f"Second-pass seed-sweep mode: output → {output_dir}")

    n_seeds = int(getattr(args, "sweep_seed_n", 36))
    base_seed = int(getattr(args, "sweep_seed_base", 0))
    n_workers = int(getattr(args, "slurm_cpus", 36))
    umap_nn = getattr(args, "umap_n_neighbors", None)
    umap_md = getattr(args, "umap_min_dist", None)
    sweep_kwargs = {
        "output_dir": str(output_dir),
        "n_seeds": n_seeds,
        "base_seed": base_seed,
        "n_workers": n_workers,
        "umap_n_neighbors": int(umap_nn) if umap_nn is not None else None,
        "umap_min_dist": float(umap_md) if umap_md is not None else None,
    }

    if args.slurm:
        from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

        slurm_params = {
            "timeout_min": 10,
            "mem": "64GB",
            "cpus_per_task": n_workers,
            "slurm_partition": args.slurm_partition,
        }
        print(
            f"Submitting --sweep-seed as SLURM job "
            f"({slurm_params['mem']}, {slurm_params['timeout_min']}min, "
            f"{n_workers} CPUs, n={n_seeds}, base={base_seed}, "
            f"umap nn={sweep_kwargs['umap_n_neighbors']}, "
            f"md={sweep_kwargs['umap_min_dist']})..."
        )
        result = submit_parallel_jobs(
            jobs_to_submit=[
                {
                    "name": "pca_seed_sweep",
                    "func": _run_seed_sweep,
                    "kwargs": sweep_kwargs,
                }
            ],
            experiment="pca_seed_sweep",
            slurm_params=slurm_params,
            log_dir="pca_optimization",
            manifest_prefix="pca_seed_sweep",
            wait_for_completion=True,
        )
        if result.get("failed"):
            print("Seed sweep FAILED")
        else:
            print("Seed sweep complete")
    else:
        print(_run_seed_sweep(**sweep_kwargs))


def _handle_overlays_only(args, output_dir):
    """Re-generate interactive HTML overlays from existing optimised h5ads."""
    # When combined with --second-pca-only, descend into the second-pass subdir
    if getattr(args, "second_pca_only", False):
        subdir = getattr(args, "second_pca_subdir", None)
        if subdir is None:
            threshold = getattr(args, "second_pca_threshold", 0.0)
            if threshold == 0.0:
                subdir = "second_pca_consensus"
            else:
                subdir = f"second_pca_{int(round(threshold * 100))}"
        output_dir = output_dir / subdir
        print(f"Second-pass overlays mode: output → {output_dir}")

    seed = int(getattr(args, "seed", 42))
    umap_nn = getattr(args, "umap_n_neighbors", None)
    umap_md = getattr(args, "umap_min_dist", None)
    umap_kwargs = {
        "umap_n_neighbors": int(umap_nn) if umap_nn is not None else None,
        "umap_min_dist": float(umap_md) if umap_md is not None else None,
    }
    if args.slurm:
        from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

        slurm_params = {
            "timeout_min": 30,
            "mem": "64GB",
            "cpus_per_task": args.slurm_cpus,
            "slurm_partition": args.slurm_partition,
        }
        print(
            f"Submitting --overlays-only as SLURM job "
            f"({slurm_params['mem']}, {slurm_params['timeout_min']}min, "
            f"seed={seed}, umap nn={umap_kwargs['umap_n_neighbors']}, "
            f"md={umap_kwargs['umap_min_dist']})..."
        )
        result = submit_parallel_jobs(
            jobs_to_submit=[
                {
                    "name": "pca_overlays_only",
                    "func": _run_overlays_only,
                    "kwargs": {
                        "output_dir": str(output_dir), "seed": seed, **umap_kwargs,
                    },
                }
            ],
            experiment="pca_overlays_only",
            slurm_params=slurm_params,
            log_dir="pca_optimization",
            manifest_prefix="pca_overlays_only",
            wait_for_completion=True,
        )
        if result.get("failed"):
            print("Overlays regeneration FAILED")
        else:
            print("Overlays regeneration complete")
    else:
        print(_run_overlays_only(str(output_dir), seed=seed, **umap_kwargs))


def _handle_aggregate_only(args, output_dir):
    """Run only the aggregation step (Phase 2).

    When a non-default distance metric is used, the per_signal/ h5ads live in
    the *parent* directory (the original cosine sweep output).  We read from
    there but write results into the distance-specific subdirectory.
    """
    agg_output = str(output_dir)
    agg_subdir = "per_signal"

    # If per_signal/ doesn't exist here but does in the parent, read from parent
    per_signal_dir = Path(agg_output) / agg_subdir
    if not per_signal_dir.exists() and (Path(agg_output).parent / agg_subdir).exists():
        # e.g. output_dir = .../all/euclidean, per_signal lives in .../all/per_signal
        source_subdir = str(Path(agg_output).parent / agg_subdir)
        print(f"Reading swept h5ads from {source_subdir}")
        print(f"Writing aggregated results to {agg_output}")
        # Symlink per_signal into the output dir so aggregate_channels can find it
        per_signal_dir.parent.mkdir(parents=True, exist_ok=True)
        per_signal_dir.symlink_to(Path(agg_output).parent / agg_subdir)

    second_pca_kwargs = _build_second_pca_kwargs(args)
    if args.slurm:
        print(
            f"Submitting aggregation as SLURM job ({args.slurm_agg_memory}, {args.slurm_agg_time}min)..."
        )
        if args.downsampled or getattr(args, "all_cells", False):
            print(f"  Mode: signal-group (reading from {agg_output}/per_signal/)")
        if second_pca_kwargs is not None:
            print("  Chained: 2nd-pass PCA will run after aggregation in the same SLURM job.")
        _submit_aggregation_slurm(
            agg_output,
            args.norm_method,
            agg_subdir,
            _make_agg_slurm_params(args),
            "pca_aggregation",
            "pca_agg",
            distance=args.distance,
            second_pca_kwargs=second_pca_kwargs,
            random_seed=getattr(args, "seed", 42),
        )
    else:
        result = aggregate_channels(
            output_dir=agg_output,
            norm_method=args.norm_method,
            per_unit_subdir=agg_subdir,
            distance=args.distance,
            random_seed=getattr(args, "seed", 42),
        )
        print(result)
        if second_pca_kwargs is not None:
            print("\nChaining 2nd-pass PCA on aggregate output...")
            _handle_second_pca(args, output_dir)


def _handle_second_pca(args, output_dir):
    """Run a second-pass PCA on the existing aggregated guide h5ad."""
    sweep_thresholds = None
    if args.second_pca_sweep_thresholds:
        sweep_thresholds = [
            float(t) for t in args.second_pca_sweep_thresholds.split(",") if t.strip()
        ]

    kwargs = dict(
        output_dir=str(output_dir),
        threshold=args.second_pca_threshold,
        distance=args.distance,
        norm_method=args.norm_method,
        subdir=args.second_pca_subdir,
        run_sweep=not args.second_pca_no_sweep,
        sweep_thresholds=sweep_thresholds,
        random_seed=getattr(args, "seed", 42),
    )

    if args.slurm:
        from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

        slurm_params = _make_agg_slurm_params(args)
        print(
            f"Submitting --second-pca as SLURM job "
            f"({slurm_params.get('mem')}, {slurm_params.get('timeout_min')}min)..."
        )
        agg_result = submit_parallel_jobs(
            jobs_to_submit=[
                {
                    "name": "pca_second_pass",
                    "func": apply_second_pass_pca,
                    "kwargs": kwargs,
                }
            ],
            experiment="pca_second_pass",
            slurm_params=slurm_params,
            log_dir="pca_optimization",
            manifest_prefix="pca_second_pass",
            wait_for_completion=True,
        )
        if agg_result.get("failed"):
            print("Second-pass PCA FAILED")
        else:
            print("Second-pass PCA complete")
    else:
        print(apply_second_pass_pca(**kwargs))


def _handle_downsampled(args, output_dir, cp_override):
    """Pool cells by signal group, downsample, PCA sweep (local or SLURM)."""
    ds_output_dir = output_dir
    ds_output_dir.mkdir(parents=True, exist_ok=True)

    # --clean: wipe stale per-signal h5ads so Phase 1 runs from scratch
    if getattr(args, "clean", False):
        import shutil

        per_signal_dir = ds_output_dir / "per_signal"
        if per_signal_dir.exists():
            print(f"--clean: removing {per_signal_dir}")
            shutil.rmtree(per_signal_dir)

    # When the user explicitly named experiments via --experiments, let them
    # through discovery's bad-experiment / non-default-library filters.
    exp_whitelist = getattr(args, "experiments", None)
    force_set: Optional[set] = None
    if exp_whitelist:
        force_set = {e.strip() for e in exp_whitelist.split(",") if e.strip()}

    all_pairs, attr_config, storage_roots, feature_dir, maps_path = (
        _discover_experiment_pairs(
            cp_override,
            include_cellpainting=getattr(args, "include_cellpainting", False),
            include_4i=getattr(args, "include_4i", False),
            include_cp=getattr(args, "include_cp", False),
            include_standard=getattr(args, "include_standard", True),
            force_include=force_set,
        )
    )
    if not all_pairs:
        print("No experiment-channel pairs found!")
        return

    paper_v1_path = getattr(args, "paper_v1", None)
    if paper_v1_path:
        import yaml as _yaml

        v1_path = Path(paper_v1_path)
        if not v1_path.exists():
            print(f"ERROR: --paper-v1 manifest not found: {v1_path}")
            return
        with open(v1_path) as f:
            v1_doc = _yaml.safe_load(f) or {}
        v1_map = v1_doc.get("experiments_channels", {}) or {}
        if not v1_map:
            print(f"ERROR: --paper-v1 manifest has empty 'experiments_channels': {v1_path}")
            return
        # YAML keys are full experiment names (e.g. ops0094_20251217). Match by full name.
        expected_full = set(v1_map.keys())
        expected_short = {e.split("_")[0] for e in expected_full}
        before = len(all_pairs)
        all_pairs = [(exp, ch) for exp, ch in all_pairs if exp in expected_full]
        print(
            f"--paper-v1: {before} → {len(all_pairs)} pairs "
            f"(restricted to {len(expected_full)} experiments from {v1_path.name})"
        )
        discovered_full = {exp for exp, _ in all_pairs}
        missing = sorted(expected_full - discovered_full)
        # --only-4i / --only-cp legitimately scope to a sub-cohort that lacks
        # most paper_v1 experiments; relax the strict-equality check there.
        only_subset = getattr(args, "only_4i", False) or getattr(args, "only_cp", False)
        if missing and not only_subset:
            print(
                f"\nERROR: --paper-v1 expects {len(expected_full)} experiments but "
                f"{len(missing)} are MISSING from discovery (must be present, no more, no less):"
            )
            for m in missing:
                print(f"  - {m}")
            return
        if missing and only_subset:
            print(
                f"  --only-4i/--only-cp scope: {len(missing)}/{len(expected_full)} paper_v1 "
                f"experiments have no data in this sub-cohort and are skipped."
            )
        extra_short = {exp.split("_")[0] for exp, _ in all_pairs} - expected_short
        if extra_short:
            print(
                f"\nERROR: --paper-v1 saw {len(extra_short)} extra experiments after filtering "
                f"(this should not happen): {sorted(extra_short)}"
            )
            return

        # Per-experiment channel allow-list from the YAML. Discovery names
        # channels by the sanitized label (e.g. "SEC61B", "no_label", "Phase");
        # the YAML lists fluorophores (Phase, GFP, mCherry, Cy5). Walk the
        # channel_maps once to invert label→channel_name so we can compare.
        from ops_utils.data.feature_metadata import FeatureMetadata
        from ops_utils.data.filesystem import extract_ops_key

        meta = FeatureMetadata()
        marker_to_fluor: Dict[str, Dict[str, str]] = {
            exp_short: {
                FeatureMetadata.sanitize_label(e["label"]): e["channel_name"]
                for e in (entries or [])
                if isinstance(e, dict) and e.get("channel_name") and e.get("label")
            }
            for exp_short, entries in meta.metadata.items()
        }
        exp_to_allowed_fluors: Dict[str, set] = {
            exp: {meta.normalize_channel_name(str(c).strip())
                  for c in (chs or []) if str(c).strip()}
            for exp, chs in v1_map.items()
        }

        before_ch = len(all_pairs)
        kept, excluded, unresolved = [], [], []
        for exp, ch in all_pairs:
            allowed = exp_to_allowed_fluors.get(exp) or set()
            if not allowed:
                kept.append((exp, ch)); continue
            exp_short = extract_ops_key(exp) or exp.split("_")[0]
            fluor = (marker_to_fluor.get(exp_short) or {}).get(ch)
            if fluor is None:
                kept.append((exp, ch)); unresolved.append((exp, ch))
            elif fluor in allowed:
                kept.append((exp, ch))
            else:
                excluded.append((exp, ch, fluor))
        all_pairs = kept
        print(
            f"--paper-v1 channel filter: {before_ch} → {len(all_pairs)} pairs "
            f"({len(excluded)} excluded by per-experiment channel allow-list)"
        )

        def _print_groups(title: str, groups: Dict[Tuple[str, ...], List[str]]) -> None:
            print(title)
            for key, exps in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
                exps_sorted = sorted(set(exps))
                sample = ", ".join(exps_sorted[:3])
                more = f" ... (+{len(exps_sorted) - 3} more)" if len(exps_sorted) > 3 else ""
                label = key[0] if len(key) == 1 else f"{key[0]} → {key[1]}"
                print(f"    {label}: {len(exps_sorted)} experiments — {sample}{more}")

        if excluded:
            grouped = {}
            for exp, ch, fluor in excluded:
                grouped.setdefault((ch, fluor or "?"), []).append(exp)
            _print_groups("  Excluded (marker → fluorophore, not in allow-list):", grouped)
        if unresolved:
            grouped = {}
            for exp, ch in unresolved:
                grouped.setdefault((ch,), []).append(exp)
            _print_groups("  WARNING: kept (marker not resolvable via channel_maps):", grouped)
        if not all_pairs:
            print("ERROR: no experiment-channel pairs remain after --paper-v1 channel filter.")
            return

    exp_whitelist = getattr(args, "experiments", None)
    if getattr(args, "match_v02", False):
        v02_manifest = Path("/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.2/dino/all/consensus_sweep/cosine/downsampled_manifest.csv")
        if not v02_manifest.exists():
            print(f"ERROR: --match-v02 requires {v02_manifest}")
            return
        v02_df = pd.read_csv(v02_manifest)
        allowed = set()
        for exps_str in v02_df["experiments"].dropna():
            allowed.update(e.strip() for e in exps_str.split(",") if e.strip())
        before = len(all_pairs)
        all_pairs = [(exp, ch) for exp, ch in all_pairs if exp.split("_")[0] in allowed]
        print(f"--match-v02: {before} → {len(all_pairs)} pairs (matched {len(allowed)} experiments from v0.2)")
        if not all_pairs:
            print("ERROR: no experiment-channel pairs remain after --match-v02 filter.")
            return
    elif exp_whitelist:
        allowed = {e.strip() for e in exp_whitelist.split(",") if e.strip()}
        before = len(all_pairs)
        all_pairs = [(exp, ch) for exp, ch in all_pairs if exp.split("_")[0] in allowed]
        print(f"--experiments filter: {before} → {len(all_pairs)} pairs (allowed {len(allowed)} experiments)")
        if not all_pairs:
            print("ERROR: no experiment-channel pairs remain after --experiments filter.")
            return

    from ops_utils.data.feature_metadata import FeatureMetadata

    fm = FeatureMetadata(metadata_path=maps_path)
    signal_groups = build_signal_groups(all_pairs, fm)

    # Apply phase filter (--phase-only / --no-phase) when running in pooled signal mode
    phase_filter = getattr(args, "phase_filter", None)
    if phase_filter == "phase_only":
        signal_groups = {s: p for s, p in signal_groups.items() if s == "Phase"}
        if not signal_groups:
            print(
                "ERROR: --phase-only found no Phase signal group in the discovered channels."
            )
            return
    elif phase_filter == "no_phase":
        signal_groups = {s: p for s, p in signal_groups.items() if s != "Phase"}

    n_signals = len(signal_groups)

    mode_label = (
        "CellProfiler"
        if cp_override == "cell-profiler"
        else ("Cell-DINO" if cp_override == "cell_dino_features"
              else ("Downsampled" if args.downsampled else "All-Cells"))
    )
    print(
        f"\n{mode_label} PCA Optimization: {len(all_pairs)} channels -> {n_signals} signal groups"
    )
    print(f"Output: {ds_output_dir}")

    # Pre-scan cell counts and compute per-signal target
    print("\nPre-scanning cell counts per signal group...")
    cell_counts = count_cells_per_signal_group(
        signal_groups, storage_roots, feature_dir, maps_path
    )

    if getattr(args, "downsample_per_guide", False):
        # Per-guide cap happens inside pca_sweep_pooled_signal — outer target is
        # just the total cell count (so target_n_cells doesn't artificially trim).
        per_signal_target = dict(cell_counts)
        cap = int(getattr(args, "cells_per_guide", 250))
        # Quick pre-scan: count unique sgRNAs per signal (backed-mode obs reads only)
        print(f"\nPre-scanning unique sgRNAs per signal (for post-cap estimate)...")
        sgrna_counts: Dict[str, int] = {}
        for signal, pairs in signal_groups.items():
            seen = set()
            for exp, ch in pairs:
                path = find_cell_h5ad_path(exp, ch, storage_roots, feature_dir, maps_path)
                if path is None:
                    continue
                try:
                    a = ad.read_h5ad(path, backed="r")
                    if "sgRNA" in a.obs.columns:
                        seen.update(a.obs["sgRNA"].astype(str).unique())
                    a.file.close()
                except Exception:
                    pass
            sgrna_counts[signal] = len(seen)
        print(f"\nSignal group manifest (per-sgRNA cap = {cap:,} cells/guide applied inside each job):")
        print(f"  {'Signal':<45} {'Exps':>5} {'Cells':>10} {'sgRNAs':>7} {'-> Expected':>15}")
        print(f"  {'-'*45} {'-'*5} {'-'*10} {'-'*7} {'-'*15}")
    elif args.downsampled:
        # Equalise: all groups target the smallest group (floor 750k)
        MIN_CELLS_FLOOR = 750_000
        global_target = max(min(cell_counts.values()), MIN_CELLS_FLOOR)
        per_signal_target = {s: global_target for s in cell_counts}
        small_groups = {s: n for s, n in cell_counts.items() if n < global_target}
        if small_groups:
            print(
                f"\n  {len(small_groups)} signal group(s) have fewer than {global_target:,} cells (will use all available):"
            )
            for s, n in sorted(small_groups.items(), key=lambda x: x[1]):
                print(f"    {s}: {n:,} cells")
        print(f"\nSignal group manifest (downsampling all to {global_target:,} cells):")
        print(f"  {'Signal':<45} {'Exps':>5} {'Cells':>10} {'-> Downsampled':>15}")
        print(f"  {'-'*45} {'-'*5} {'-'*10} {'-'*15}")
    else:
        # All-cells: load every cell; pca_sweep_pooled_signal uses passthrough PCA for >5M
        per_signal_target = dict(cell_counts)
        print(
            f"\nSignal group manifest (all cells — PCA fit subsampled at >{PCA_FIT_CAP:,}):"
        )
        print(f"  {'Signal':<45} {'Exps':>5} {'Cells':>10}")
        print(f"  {'-'*45} {'-'*5} {'-'*10}")

    manifest_rows = []
    for signal in sorted(signal_groups.keys()):
        pairs = signal_groups[signal]
        n_cells = cell_counts[signal]
        t = per_signal_target[signal]
        if getattr(args, "downsample_per_guide", False):
            n_sg = sgrna_counts.get(signal, 0)
            expected = min(n_cells, n_sg * int(getattr(args, "cells_per_guide", 250)))
            print(f"  {signal:<45} {len(pairs):>5} {n_cells:>10,} {n_sg:>7,} -> {expected:>12,}")
        elif args.downsampled:
            print(f"  {signal:<45} {len(pairs):>5} {n_cells:>10,} -> {t:>15,}")
        else:
            pca_note = f" (passthrough PCA)" if n_cells > PCA_FIT_CAP else ""
            print(f"  {signal:<45} {len(pairs):>5} {n_cells:>10,}{pca_note}")
        manifest_rows.append(
            {
                "signal": signal,
                "n_experiments": len(pairs),
                "n_cells_pooled": n_cells,
                "n_cells_used": t,
                "experiments": ",".join(e.split("_")[0] for e, c in pairs),
            }
        )
    n_unique_exps = len({exp.split("_")[0] for pairs in signal_groups.values() for exp, _ in pairs})
    if getattr(args, "downsample_per_guide", False):
        total_expected = sum(
            min(cell_counts[s], sgrna_counts.get(s, 0) * int(getattr(args, "cells_per_guide", 250)))
            for s in cell_counts
        )
        print(
            f"\n  Total: {n_signals} signal groups, {n_unique_exps} experiments, "
            f"{sum(cell_counts.values()):,} cells → {total_expected:,} expected after cap"
        )
    else:
        print(
            f"\n  Total: {n_signals} signal groups, {n_unique_exps} experiments, "
            f"{sum(cell_counts.values()):,} total cells"
        )

    if getattr(args, "dry_run", False):
        print("\n--dry-run: exiting before processing.")
        return

    pd.DataFrame(manifest_rows).to_csv(
        ds_output_dir / "downsampled_manifest.csv", index=False
    )

    skip_phase2 = (
        getattr(args, "preserve_batch", False)
        or getattr(args, "no_pca", False)
        or attr_config.get("preserve_batch", False)
    )

    # Build common kwargs for signal-group jobs
    def _signal_job_kwargs(signal, pairs):
        kwargs = dict(
            signal=signal,
            exp_channel_pairs=pairs,
            output_dir=str(ds_output_dir),
            target_n_cells=per_signal_target[signal],
            norm_method=args.norm_method,
            distance=args.distance,
        )
        if cp_override:
            kwargs["feature_dir_override"] = cp_override
            if cp_override == "cell-profiler":
                kwargs["sweep_thresholds"] = DEFAULT_SWEEP_THRESHOLDS_CP
        if args.fixed_threshold is not None and args.fixed_threshold > 0:
            kwargs["fixed_threshold"] = args.fixed_threshold
        if getattr(args, "preserve_batch", False):
            kwargs["preserve_batch"] = True
        if getattr(args, "no_pca", False):
            kwargs["no_pca"] = True
        if getattr(args, "zscore_per_experiment", False):
            kwargs["zscore_per_experiment"] = True
        if not getattr(args, "exclude_dud_guides", True):
            kwargs["exclude_dud_guides"] = False
        if getattr(args, "downsample_per_guide", False):
            kwargs["downsample_per_guide"] = True
            kwargs["cells_per_guide"] = int(getattr(args, "cells_per_guide", 250))
        return kwargs

    if not args.slurm:
        print("\nRunning locally (sequential)...")
        for signal, pairs in signal_groups.items():
            result = pca_sweep_pooled_signal(**_signal_job_kwargs(signal, pairs))
            print(f"  {result}")
        if not skip_phase2:
            result = aggregate_channels(
                output_dir=str(ds_output_dir),
                norm_method=args.norm_method,
                per_unit_subdir="per_signal",
                distance=args.distance,
                random_seed=getattr(args, "seed", 42),
            )
            print(result)
            if getattr(args, "second_pca", False):
                print("\nChaining 2nd-pass PCA on aggregate output...")
                _handle_second_pca(args, ds_output_dir)
        else:
            print("  Phase 2 aggregation skipped (--preserve-batch or --no-pca mode)")
        return

    # SLURM mode: one job per signal group — split into high-memory (>4M cells)
    # and standard-memory batches
    HIGH_MEMORY_CELL_THRESHOLD = 4_000_000
    high_mem_jobs = []
    other_jobs = []
    for signal, pairs in signal_groups.items():
        sig_safe = sanitize_signal_filename(signal)[:40]
        job = {
            "name": f"pca_ds_{sig_safe}",
            "func": pca_sweep_pooled_signal,
            "kwargs": _signal_job_kwargs(signal, pairs),
            "metadata": {"signal": signal, "n_experiments": len(pairs)},
        }
        if cell_counts.get(signal, 0) > HIGH_MEMORY_CELL_THRESHOLD:
            high_mem_jobs.append(job)
        else:
            other_jobs.append(job)

    # Signal-group jobs need more time than per-channel (copairs scoring is slow)
    args.slurm_time = max(args.slurm_time, 60)

    from ops_utils.hpc.slurm_batch_utils import (
        submit_parallel_jobs,
        wait_for_multiple_job_arrays,
    )

    slurm_params = _make_slurm_params(args)
    agg_slurm_params = _make_agg_slurm_params(args)

    # Submit both batches without waiting — they run in parallel on SLURM
    job_arrays = []

    if other_jobs:
        print(
            f"\nSubmitting {len(other_jobs)} non-Phase signal-group SLURM jobs ({slurm_params.get('mem', '?')} each)..."
        )
        result_other = submit_parallel_jobs(
            jobs_to_submit=other_jobs,
            experiment="pca_ds_optimization",
            slurm_params=slurm_params,
            log_dir="pca_optimization",
            manifest_prefix="pca_ds_opt",
            wait_for_completion=False,
        )
        if result_other.get("submitted_jobs"):
            job_arrays.append(
                {
                    "submitted_jobs": result_other["submitted_jobs"],
                    "base_job_id": result_other["base_job_id"],
                    "label": "reporters",
                    "slurm_params": slurm_params,
                }
            )

    if high_mem_jobs:
        phase_memory = getattr(args, "phase_memory", "600GB")
        high_mem_slurm_params = {
            **slurm_params,
            "mem": phase_memory,
            "timeout_min": max(slurm_params.get("timeout_min", 60), 360),
        }
        high_mem_names = [j["metadata"]["signal"] for j in high_mem_jobs]
        print(
            f"\nSubmitting {len(high_mem_jobs)} high-memory SLURM job(s) "
            f"({phase_memory}, >4M cells): {', '.join(high_mem_names)}"
        )
        result_high = submit_parallel_jobs(
            jobs_to_submit=high_mem_jobs,
            experiment="pca_ds_optimization_high_mem",
            slurm_params=high_mem_slurm_params,
            log_dir="pca_optimization",
            manifest_prefix="pca_ds_high_opt",
            wait_for_completion=False,
        )
        if result_high.get("submitted_jobs"):
            job_arrays.append(
                {
                    "submitted_jobs": result_high["submitted_jobs"],
                    "base_job_id": result_high["base_job_id"],
                    "label": "high_mem",
                    "slurm_params": high_mem_slurm_params,
                }
            )

    # Wait for ALL arrays with unified progress monitoring
    if job_arrays:
        wait_result = wait_for_multiple_job_arrays(
            job_arrays,
            experiment="pca_ds_optimization",
        )
        if wait_result.get("failed"):
            print(f"\nWarning: {len(wait_result['failed'])} jobs failed")
            for name in wait_result["failed"]:
                print(f"  - {name}")

    # Chain aggregation after all Phase 1 jobs complete (skipped for --preserve-batch and --no-pca)
    if skip_phase2:
        print(
            f"\nAll signal-group jobs complete. Phase 2 aggregation skipped (--preserve-batch or --no-pca mode)."
        )
    else:
        sp_kwargs = _build_second_pca_kwargs(args)
        msg = "aggregation SLURM job"
        if sp_kwargs is not None:
            msg += " (chained with 2nd-pass PCA)"
        print(f"\nAll signal-group jobs complete. Submitting {msg}...")
        _submit_aggregation_slurm(
            str(ds_output_dir),
            args.norm_method,
            "per_signal",
            agg_slurm_params,
            "pca_ds_aggregation",
            "pca_ds_agg",
            distance=args.distance,
            second_pca_kwargs=sp_kwargs,
            random_seed=getattr(args, "seed", 42),
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
        "-o",
        "--output-dir",
        type=str,
        default="/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3",
        help="Root output directory (feature-type and channel-subset subdirs are added automatically)",
    )
    parser.add_argument(
        "--norm-method",
        type=str,
        default="ntc",
        choices=["ntc", "global"],
        help="Normalization method (default: ntc)",
    )
    parser.add_argument(
        "--distance",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean"],
        help="Distance metric for mAP scoring (default: cosine)",
    )
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=0.80,
        help="Skip the variance sweep and use a single fixed PCA threshold (default: 0.80). "
        "Pass --fixed-threshold 0 to disable and run the full consensus sweep instead.",
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Submit Phase 1 signal-group SLURM jobs + Phase 2 aggregation job",
    )
    parser.add_argument(
        "--slurm-memory",
        type=str,
        default="100GB",
        help="SLURM memory per signal-group job (default: 100GB)",
    )
    parser.add_argument(
        "--slurm-time",
        type=int,
        default=10,
        help="SLURM time limit per signal-group job in minutes (default: 10)",
    )
    parser.add_argument(
        "--slurm-cpus",
        type=int,
        default=16,
        help="SLURM CPUs per signal-group job (default: 16)",
    )
    parser.add_argument(
        "--slurm-partition",
        type=str,
        default="cpu,gpu",
        help="SLURM partition (default: cpu,gpu)",
    )
    parser.add_argument(
        "-y", "--yes", action="store_true", help="Skip confirmation prompt"
    )
    parser.add_argument(
        "--slurm-agg-memory",
        type=str,
        default="500GB",
        help="SLURM memory for aggregation job (default: 500GB)",
    )
    parser.add_argument(
        "--slurm-agg-time",
        type=int,
        default=60,
        help="SLURM time limit for aggregation job in minutes (default: 60)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing per_signal/ directory before Phase 1 to ensure a fresh run.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only run Phase 2 aggregation (skips PCA sweeps, reads existing per_signal/ h5ads).",
    )
    parser.add_argument(
        "--umap-only",
        action="store_true",
        help="Only generate embedding plots from existing optimized h5ads.",
    )
    parser.add_argument(
        "--overlays-only",
        action="store_true",
        help="Re-generate the interactive HTML overlay pages from existing "
        "guide_pca_optimized.h5ad + gene_embedding_pca_optimized.h5ad. "
        "Skips the full pipeline, PCA sweep, and scoring. UMAP/PHATE are "
        "reused from the h5ads unless --seed differs from the stored "
        "random_state, in which case those embeddings are refit and the "
        "h5ads are rewritten before overlays are regenerated.",
    )
    parser.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=None,
        help="Override gene-level UMAP n_neighbors when used with "
        "--overlays-only (forces a refit at this value; bypasses the "
        "--sweep-seed cache, which is keyed on default n_neighbors=15).",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=None,
        help="Override gene-level UMAP min_dist when used with --overlays-only "
        "(forces a refit; default UMAP value is 0.1).",
    )
    parser.add_argument(
        "--sweep-seed",
        action="store_true",
        help="Fit gene-level UMAP at --sweep-seed-n consecutive seeds and "
        "save a single PNG canvas (sqrt(n)×sqrt(n) panels) so different "
        "seed-driven layouts can be compared at a glance. Reads the existing "
        "gene_embedding_pca_optimized.h5ad — no other outputs are touched.",
    )
    parser.add_argument(
        "--sweep-seed-n",
        type=int,
        default=36,
        help="Number of consecutive seeds to fit in --sweep-seed mode "
        "(default: 36 → 6×6 grid).",
    )
    parser.add_argument(
        "--sweep-seed-base",
        type=int,
        default=0,
        help="Starting seed value for --sweep-seed mode. Seeds tried are "
        "[base, base+1, ..., base+n-1] (default: 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for UMAP / PHATE / NTC-split, threaded through "
             "aggregate_channels, apply_second_pass_pca, and the SLURM helpers. "
             "Same seed → bit-identical embeddings (default: 42).",
    )
    parser.add_argument(
        "--second-pca",
        dest="second_pca",
        action="store_true",
        default=True,
        help="Also run a second-pass PCA after the main pipeline (full run or --aggregate-only) "
        "finishes. The 2nd-pass reads <output_dir>/guide_pca_optimized.h5ad, fits PCA on the "
        "horizontally concatenated NTC-normalized guide features, retains top --second-pca-threshold "
        "of variance, re-aggregates to gene level, re-scores all metrics, and writes results to "
        "<output_dir>/<--second-pca-subdir>/. In SLURM mode this is bundled into the same SLURM job "
        "as the aggregation step. Default: True.",
    )
    parser.add_argument(
        "--no-second-pca",
        dest="second_pca",
        action="store_false",
        help="Disable the chained 2nd-pass PCA.",
    )
    parser.add_argument(
        "--second-pca-only",
        action="store_true",
        help="Only run the 2nd-pass PCA on an existing aggregate output (skips Phase 1 + Phase 2). "
        "Use this when you've already run the main pipeline and just want to (re-)compute the "
        "2nd-pass.",
    )
    parser.add_argument(
        "--second-pca-threshold",
        type=float,
        default=0.0,
        help="Cumulative variance threshold for the second-pass PCA. "
        "Default: 0 → run the sweep and pick the consensus peak (max of "
        "normalized activity+distinctiveness+CHAD across thresholds), "
        "writing to second_pca_consensus/. "
        "Pass a positive value (e.g. 0.80) to use a fixed threshold, "
        "writing to second_pca_<pct>/.",
    )
    parser.add_argument(
        "--second-pca-subdir",
        type=str,
        default=None,
        help="Subdir name under <output_dir> for second-pass PCA outputs "
        "(default: second_pca_<threshold>).",
    )
    parser.add_argument(
        "--second-pca-no-sweep",
        action="store_true",
        help="Skip the variance-threshold sweep in --second-pca mode (faster, but no "
        "sweep CSV/plot to compare against the chosen threshold).",
    )
    parser.add_argument(
        "--second-pca-sweep-thresholds",
        type=str,
        default=None,
        help="Comma-separated variance thresholds for the second-pass sweep "
        "(default: same as DEFAULT_SWEEP_THRESHOLDS).",
    )
    parser.add_argument(
        "--downsampled",
        action="store_true",
        help="Equalise cells across signal groups by downsampling to the smallest group "
        "(floor 750k). Default mode uses all cells per group. Output → downsampled/.",
    )
    parser.add_argument(
        "--downsample-per-guide",
        dest="downsample_per_guide",
        action="store_true",
        help="Cap each sgRNA at --cells-per-guide cells (pooled across experiments). "
             "Replaces proportional-per-experiment downsampling with a global per-sgRNA cap. "
             "Implies --downsampled. Output → downsampled_per_guide/.",
    )
    parser.add_argument(
        "--cells-per-guide",
        type=int,
        default=250,
        help="Per-sgRNA cell cap used with --downsample-per-guide (default: 250).",
    )
    parser.add_argument(
        "--phase-memory",
        type=str,
        default="600GB",
        help="SLURM memory for Phase signal job (default: 600GB). Phase ~50M cells needs more.",
    )
    parser.add_argument(
        "--cell-profiler",
        action="store_true",
        help="Use CellProfiler morphological features instead of DINO embeddings.",
    )
    parser.add_argument(
        "--cell-dino",
        action="store_true",
        help="Use cell-level DINO features (feature_dir=cell_dino_features) instead of default DINO. "
             "Output → cell_dino/ subdir.",
    )
    parser.add_argument(
        "--exclude-dud-guides", dest="exclude_dud_guides",
        action="store_true", default=True,
        help="Filter out known dud sgRNAs (default: True). See DUD_GUIDES constant.",
    )
    parser.add_argument(
        "--no-exclude-dud-guides", dest="exclude_dud_guides",
        action="store_false",
        help="Keep dud sgRNAs in the cell pool.",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Use -o/--output-dir as the exact output path (skip automatic dino/all/… nesting).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover experiments and print the signal-group manifest, then exit without processing.",
    )
    parser.add_argument(
        "--include-cellpainting",
        action="store_true",
        help="Include legacy CP1_*/CP2_* channels stored inside the standard feature_dir "
             "(older Cell Painting layout). For the newer per-experiment _cp/ sibling layout, "
             "use --with-cp instead. Output → with_cellpainting/ subdir.",
    )
    parser.add_argument(
        "--with-4i",
        dest="include_4i",
        action="store_true",
        help="Also scan parallel <feature_dir>_4i/ sibling directories (e.g. dino_features_4i/) "
             "and pool those channels alongside standard ones. 4i channels are tagged so they "
             "form their own signal groups (label suffix ' (4i)'). Composes with --with-cp and "
             "--include-cellpainting. Output → with_4i/ subdir.",
    )
    parser.add_argument(
        "--with-cp",
        dest="include_cp",
        action="store_true",
        help="Also scan parallel <feature_dir>_cp/ sibling directories (e.g. cell_dino_features_cp/) "
             "for the new Cell Painting layout (e.g. ops0094 ConA/Hoechst/NPM1/Phalloidin/TOMM20/Tubulin/WGA). "
             "Channels are tagged so they form their own signal groups (label suffix ' (cp)'). "
             "Composes with --with-4i. Output → with_cp/ subdir.",
    )
    parser.add_argument(
        "--only-4i",
        action="store_true",
        help="Run only the 4i sibling-dir channels (skip the standard <feature_dir>/). "
             "Implies --with-4i and turns off the standard scan. Output → only_4i/ subdir.",
    )
    parser.add_argument(
        "--only-cp",
        action="store_true",
        help="Run only the cp sibling-dir channels (skip the standard <feature_dir>/). "
             "Implies --with-cp and turns off the standard scan. Output → only_cp/ subdir. "
             "Composes with --only-4i (then both 4i and cp siblings, no standard).",
    )
    parser.add_argument(
        "--paper-v1",
        type=str,
        nargs="?",
        const="/hpc/projects/icd.fast.ops/configs/good_experiment_list_v1.yml",
        default=None,
        help="Restrict discovery to the exact experiment list in the paper-v1 YAML "
             "(default path: /hpc/projects/icd.fast.ops/configs/good_experiment_list_v1.yml). "
             "Errors out if any expected experiment is missing from discovery. "
             "Output → paper_v1/ subdir.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional cohort/run subfolder inserted into the output path after "
             "the feature/zscore/paper_v1 subdirs and before the channel-set / "
             "threshold subdirs. Accepts multi-segment paths "
             "(e.g. 'paper_v1/validation_4exp_phase_only'). Pure organization — "
             "does not filter experiments.",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help="Comma-separated experiment short names (e.g. ops0031,ops0035) to restrict to. "
             "Only these experiments will be included in signal groups. Useful for A/B comparisons.",
    )
    parser.add_argument(
        "--match-v02",
        action="store_true",
        help="Restrict to the same experiments used in pca_optimized_v0.2 (reads v0.2 manifest). "
             "Useful for controlled A/B comparison of features with identical experiment sets.",
    )
    parser.add_argument(
        "--chad-annotation",
        type=str,
        default=None,
        help="Path to custom CHAD annotation YAML for consistency scoring. "
             "Defaults to chad_positive_controls_v4.yml.",
    )
    parser.add_argument(
        "--chad-umap-only", action="store_true",
        help="Only regenerate the CHAD-colored UMAP from existing gene_embedding_pca_optimized.h5ad.",
    )
    parser.add_argument(
        "--chad-umap-output", type=str, default=None,
        help="Output filename for CHAD UMAP (saved under plots/).",
    )
    parser.add_argument(
        "--chad-cluster-range", type=str, default=None,
        help="Filter CHAD clusters to a range by integer key (e.g. '1-76' or '100-162').",
    )
    parser.add_argument(
        "--zscore-per-experiment", dest="zscore_per_experiment",
        action="store_true", default=True,
        help="Apply per-experiment z-score scaling to features before PCA. "
             "Output → zscore_per_exp/ subdir. Default: True.",
    )
    parser.add_argument(
        "--no-zscore-per-experiment", dest="zscore_per_experiment",
        action="store_false",
        help="Disable per-experiment z-score scaling.",
    )
    phase_group = parser.add_mutually_exclusive_group()
    phase_group.add_argument(
        "--phase-only",
        action="store_true",
        help="Include only Phase (label-free brightfield) channels. Output → phase_only/.",
    )
    phase_group.add_argument(
        "--no-phase",
        action="store_true",
        help="Exclude Phase channels, fluorescent only. Output → no_phase/.",
    )
    parser.add_argument(
        "--preserve-batch",
        action="store_true",
        help="Preserve experiment identity in guide/gene aggregation (for batch effect inspection). "
        "Skips the variance sweep; uses pca.variance_cutoff from the attribution config. "
        "Phase 2 aggregation is skipped. Output → batch/ subdir.",
    )
    parser.add_argument(
        "--no-pca",
        action="store_true",
        help="Skip PCA reduction entirely; export the full feature matrix. "
        "Phase 2 aggregation is skipped. Output → no_pca/ subdir.",
    )
    return parser


def main():
    global CHAD_ANNOTATION_PATH
    args = _build_parser().parse_args()
    CHAD_ANNOTATION_PATH = args.chad_annotation
    output_dir = Path(args.output_dir)

    # --only-4i / --only-cp imply the corresponding --with-* and turn off the
    # standard scan. Apply once here so both --direct and standard paths see it.
    only_4i = getattr(args, "only_4i", False)
    only_cp = getattr(args, "only_cp", False)
    if only_4i:
        args.include_4i = True
    if only_cp:
        args.include_cp = True
    args.include_standard = not (only_4i or only_cp)

    # --direct: use the given path as-is, skip all automatic nesting
    if args.direct:
        args.phase_filter = None
        args.all_cells = True
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Direct mode: output → {output_dir}")
        if args.sweep_seed:
            _handle_sweep_seed(args, output_dir)
        elif args.chad_umap_only:
            _handle_chad_umap_only(args, output_dir)
        elif args.umap_only:
            _handle_umap_only(args, output_dir)
        elif args.overlays_only:
            _handle_overlays_only(args, output_dir)
        elif args.second_pca_only:
            _handle_second_pca(args, output_dir)
        elif args.aggregate_only:
            _handle_aggregate_only(args, output_dir)
        else:
            _handle_downsampled(args, output_dir, None)
        return

    # Nest output under feature-type subdir: dino/, cellprofiler/, or cell_dino/
    cp_override = None
    if args.cell_profiler and args.cell_dino:
        raise ValueError("--cell-profiler and --cell-dino are mutually exclusive")
    if args.cell_profiler:
        cp_override = "cell-profiler"
        output_dir = output_dir / "cellprofiler"
        print(
            f"CellProfiler mode: features from 3-assembly/cell-profiler/anndata_objects/"
        )
        print(
            f"PCA sweep thresholds: {DEFAULT_SWEEP_THRESHOLDS_CP} (lower range — CP features are independent)"
        )
        print(f"Output: {output_dir}")
    elif args.cell_dino:
        cp_override = "cell_dino_features"
        output_dir = output_dir / "cell_dino"
        print(f"Cell-DINO mode: features from 3-assembly/cell_dino_features/")
        print(f"Output: {output_dir}")
    else:
        output_dir = output_dir / "dino"

    # Nest under zscore subdir if requested
    if args.zscore_per_experiment:
        output_dir = output_dir / "zscore_per_exp"
        print(f"Per-experiment z-score scaling enabled: output → {output_dir}")

    # paper_v1 sits at the top of the channel-set hierarchy so the v1 cohort
    # is the primary partition; with_cp / with_4i / cellpainting nest under it.
    if getattr(args, "paper_v1", None):
        output_dir = output_dir / "paper_v1"
        print(f"Paper-v1 experiment list enforced: output → {output_dir}")

    # --run-tag accepts a multi-segment relative path (e.g.
    # "paper_v1/validation_4exp_phase_only") so callers can recreate cohort
    # folders that don't have a dedicated flag.
    if getattr(args, "run_tag", None):
        tag = args.run_tag.strip().strip("/")
        if tag:
            output_dir = output_dir / tag
            print(f"Run tag: output → {output_dir}")

    # Nest under cell-painting subdir if requested
    if args.include_cellpainting:
        output_dir = output_dir / "with_cellpainting"
        print(f"Cell Painting channels included: output → {output_dir}")

    # Nest under cp-sibling subdir if requested (new layout). Goes ABOVE
    # with_4i so combined --with-cp --with-4i nests as with_cp/with_4i/.
    if getattr(args, "include_cp", False):
        sub = "only_cp" if getattr(args, "only_cp", False) and not args.include_standard else "with_cp"
        output_dir = output_dir / sub
        print(f"CP sibling-dir channels included: output → {output_dir}")

    # Nest under 4i subdir if requested (composes with --include-cellpainting / --with-cp)
    if getattr(args, "include_4i", False):
        sub = "only_4i" if getattr(args, "only_4i", False) and not args.include_standard else "with_4i"
        output_dir = output_dir / sub
        print(f"4i sibling-dir channels included: output → {output_dir}")

    # --downsample-per-guide implies --downsampled
    if args.downsample_per_guide:
        args.downsampled = True
    _ds_suffix = "_per_guide" if args.downsample_per_guide else ""

    # Nest under channel-subset subdir
    if args.phase_only and args.downsampled:
        output_dir = output_dir / f"phase_only_downsampled{_ds_suffix}"
        args.phase_filter = "phase_only"
        print(f"Phase-only downsampled mode: output → {output_dir}")
    elif args.no_phase and args.downsampled:
        output_dir = output_dir / f"no_phase_downsampled{_ds_suffix}"
        args.phase_filter = "no_phase"
        print(f"No-phase downsampled mode: output → {output_dir}")
    elif args.phase_only:
        output_dir = output_dir / "phase_only"
        args.phase_filter = "phase_only"
        print(f"Phase-only mode: output → {output_dir}")
    elif args.no_phase:
        output_dir = output_dir / "no_phase"
        args.phase_filter = "no_phase"
        print(f"No-phase mode: output → {output_dir}")
    elif args.downsampled:
        output_dir = output_dir / f"downsampled{_ds_suffix}"
        args.phase_filter = None
        print(f"Downsampled mode: output → {output_dir}")
    else:
        output_dir = output_dir / "all_livecell"
        args.phase_filter = None
        print(f"All live-cell mode (default): output → {output_dir}")

    # all_cells=True is now always the default (non-downsampled path)
    args.all_cells = not args.downsampled

    # Nest under mode-specific subdir
    if args.no_pca:
        mode_tag = "no_pca_batch" if args.preserve_batch else "no_pca"
        output_dir = output_dir / mode_tag
        print(f"No-PCA mode — output → {output_dir}")
    elif args.preserve_batch:
        output_dir = output_dir / "batch"
        print(f"Preserve-batch mode — output → {output_dir}")
    elif args.fixed_threshold is not None and args.fixed_threshold > 0:
        thresh_tag = f"fixed_{args.fixed_threshold:.0%}"
        output_dir = output_dir / thresh_tag
        print(f"Fixed threshold: {args.fixed_threshold:.0%} — output → {output_dir}")
    else:
        output_dir = output_dir / "consensus_sweep"
        print(f"Consensus sweep mode — output → {output_dir}")

    # Nest under distance metric subdir
    output_dir = output_dir / args.distance
    print(f"Distance metric: {args.distance} — output → {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Dispatch to mode handler. ``--sweep-seed`` is checked first so that
    # combining it with ``--second-pca-only`` (which is also a subdir
    # navigator) doesn't accidentally trigger the full second-pass pipeline.
    if args.sweep_seed:
        _handle_sweep_seed(args, output_dir)
    elif args.chad_umap_only:
        _handle_chad_umap_only(args, output_dir)
    elif args.umap_only:
        _handle_umap_only(args, output_dir)
    elif args.overlays_only:
        _handle_overlays_only(args, output_dir)
    elif args.second_pca_only:
        _handle_second_pca(args, output_dir)
    elif args.aggregate_only:
        _handle_aggregate_only(args, output_dir)
    else:
        _handle_downsampled(args, output_dir, cp_override)


if __name__ == "__main__":
    main()
