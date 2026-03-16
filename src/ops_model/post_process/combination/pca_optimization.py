"""Per-channel cell-level PCA optimization & pre-reduction.

Produces PCA-reduced guide/gene h5ad files for the attribution stage.

Two-phase SLURM architecture
-----------------------------
Phase 1  One SLURM job per channel -- load cells, PCA sweep, save per-channel h5ad.
Phase 2  One aggregation job -- load per-channel h5ads, hconcat, normalize, score, plot.

Usage
-----
  # Phase 1 (parallel SLURM):
  python -m ops_model.post_process.combination.pca_optimization \\
      --slurm -o /hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized -y

  # Phase 2 (aggregation):
  python -m ops_model.post_process.combination.pca_optimization \\
      --aggregate-only -o /hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import yaml

from ops_model.features.anndata_utils import aggregate_to_level
from ops_utils.analysis.map_scores import (
    compute_auc_score,
    phenotypic_activity_assesment,
    plot_map_scatter,
)
from ops_utils.analysis.normalization import zscore_normalize
from ops_utils.data.feature_discovery import (
    build_signal_groups,
    count_cells_per_signal_group,
    discover_cellprofiler_experiments,
    discover_dino_experiments,
    find_cell_h5ad_path,
    find_experiment_dir,
    get_channel_maps_path,
    get_storage_roots,
    load_attribution_config,
    load_cell_h5ad,
    resolve_channel_label,
    sanitize_signal_filename,
)

logger = logging.getLogger(__name__)

DEFAULT_SWEEP_THRESHOLDS = [0.60, 0.70, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.88, 0.90, 0.95]
# CellProfiler features are hand-crafted and independent (not redundant like DINO embeddings),
# so PCA is destructive at high thresholds. Optimal region is ~50% variance explained.
DEFAULT_SWEEP_THRESHOLDS_CP = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
MIN_PCS = 10  # Minimum PCs for peak selection (avoids degenerate 1-PC artifact)


# =============================================================================
# Data loading & scoring helpers
# =============================================================================



def _fit_pca(X: np.ndarray, max_pcs: int = 500):
    """Fit PCA on cell matrix. Returns (X_transformed, cumvar, pca_model)."""
    from sklearn.decomposition import PCA

    X = np.asarray(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    n_pcs = min(X.shape[0] - 1, X.shape[1] - 1, max_pcs)
    pca = PCA(n_components=n_pcs)
    X_pcs = pca.fit_transform(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    return X_pcs, cumvar, pca


def _n_pcs_for_threshold(cumvar: np.ndarray, threshold: float) -> int:
    n = int(np.searchsorted(cumvar, threshold) + 1)
    return min(n, len(cumvar))


def _score_activity(adata_guide: ad.AnnData, null_size: int = 100_000) -> Tuple[float, float]:
    """Score guide-level AnnData. Returns (active_ratio, auc)."""
    # Strip obs to only copairs-required columns — extra string/categorical columns
    # (e.g. label_str, experiment) cause 'ufunc isnan not supported' in copairs
    if "n_cells" not in adata_guide.obs.columns:
        adata_guide.obs["n_cells"] = 1
    keep = [c for c in ["sgRNA", "perturbation", "n_cells"] if c in adata_guide.obs.columns]
    adata_guide.obs = adata_guide.obs[keep].copy()
    for col in adata_guide.obs.columns:
        if adata_guide.obs[col].dtype.name == "category":
            adata_guide.obs[col] = adata_guide.obs[col].astype(str)
    adata_guide.X = np.asarray(adata_guide.X, dtype=np.float64)

    activity_map, active_ratio = phenotypic_activity_assesment(
        adata_guide, plot_results=False, null_size=null_size,
    )
    auc = compute_auc_score(activity_map)
    return active_ratio, auc


def _normalize_guide(
    adata_guide: ad.AnnData,
    norm_method: str = "ntc",
) -> ad.AnnData:
    """Z-score normalize guide-level data."""
    feature_cols = list(adata_guide.var_names)
    df = pd.DataFrame(adata_guide.X, columns=feature_cols)
    for col in adata_guide.obs.columns:
        df[col] = adata_guide.obs[col].values

    df = zscore_normalize(
        df,
        feature_cols,
        method=norm_method,
        perturbation_col="perturbation",
    )
    adata_guide.X = df[feature_cols].values.astype(np.float32)
    return adata_guide


def _hconcat(blocks: List[ad.AnnData], level: str = "guide") -> ad.AnnData:
    """Horizontally concat AnnData blocks by matching on perturbation key."""
    key = "sgRNA" if level == "guide" and "sgRNA" in blocks[0].obs.columns else "perturbation"
    common = set(blocks[0].obs[key].values)
    for b in blocks[1:]:
        common &= set(b.obs[key].values)
    common = sorted(common)

    matrices = []
    var_names = []
    ref_obs = None
    for b in blocks:
        mask = b.obs[key].isin(common)
        sub = b[mask].copy()
        order = sub.obs[key].map({k: i for i, k in enumerate(common)}).values
        sub = sub[np.argsort(order)]
        if ref_obs is None:
            ref_obs = sub.obs.copy()
        matrices.append(np.asarray(sub.X, dtype=np.float32))
        var_names.extend(sub.var_names.tolist())

    result = ad.AnnData(
        X=np.hstack(matrices),
        obs=ref_obs,
        var=pd.DataFrame(index=var_names),
    )
    result.var_names_make_unique()
    return result


def _plot_embedding_overlay(
    coords, perts, metric_lookup, level_name, embed_name,
    plots_dir, n_obs, n_vars, plt,
):
    """Plot 2-panel embedding (mAP viridis + p-value plasma) with NTC red diamonds."""
    mAP_vals = np.array([metric_lookup.get(p, {}).get("mean_average_precision", np.nan) for p in perts])
    log10p_vals = np.array([metric_lookup.get(p, {}).get("-log10(p-value)", np.nan) for p in perts])
    is_sig = np.array([metric_lookup.get(p, {}).get("below_corrected_p", False) for p in perts])
    is_ntc = np.array([str(p).upper().startswith("NTC") or "non-targeting" in str(p).lower() for p in perts])

    ntc_mask = is_ntc
    sig_mask = is_sig & ~is_ntc
    nonsig_mask = ~is_sig & ~is_ntc
    n_ntc = ntc_mask.sum()
    n_sig = sig_mask.sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    for ax, color_vals, cmap_name, cbar_label, panel_title in [
        (ax1, mAP_vals, "viridis", "Mean Average Precision", "mAP"),
        (ax2, log10p_vals, "plasma", "-log10(p-value)", "-log10(p)"),
    ]:
        if nonsig_mask.any():
            ax.scatter(coords[nonsig_mask, 0], coords[nonsig_mask, 1],
                       c="lightgrey", s=40, alpha=0.5, label="Not significant")
        if sig_mask.any():
            sc = ax.scatter(coords[sig_mask, 0], coords[sig_mask, 1],
                            c=color_vals[sig_mask], s=50, alpha=0.8,
                            cmap=cmap_name, edgecolors="black", linewidths=0.3,
                            label=f"Significant ({n_sig})")
            plt.colorbar(sc, ax=ax, label=cbar_label, shrink=0.8)
        if ntc_mask.any():
            ax.scatter(coords[ntc_mask, 0], coords[ntc_mask, 1],
                       c="#E03030", s=80, alpha=0.9, marker="D",
                       edgecolors="black", linewidths=0.5,
                       label=f"NTC ({n_ntc})", zorder=5)

        ax.set_xlabel(f"{embed_name} 1", fontsize=11)
        ax.set_ylabel(f"{embed_name} 2", fontsize=11)
        ax.set_title(f"{level_name.title()} — Activity: {panel_title}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="best")

    fig.suptitle(
        f"{level_name.title()}-Level {embed_name} — {n_obs} obs, {n_vars} features\n"
        f"{n_sig}/{n_obs - n_ntc} significant ({100 * n_sig / max(n_obs - n_ntc, 1):.1f}%)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fname = f"{embed_name.lower()}_{level_name}.png"
    fig.savefig(plots_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


# ---- Positive controls overlay on embeddings ----

def _split_ntc_for_embedding(
    adata_guide: ad.AnnData,
    group_size: int = 4,
    random_seed: int = 42,
) -> ad.AnnData:
    """
    Create a gene-level adata where NTC guides are split into random groups
    of `group_size` instead of being aggregated into one NTC gene.

    This makes gene-level UMAP/PHATE show NTCs as multiple dots matching
    KO gene group sizes, providing a visual null baseline.

    Returns a new gene-level AnnData (does not modify adata_guide).
    """
    rng = np.random.RandomState(random_seed)
    obs = adata_guide.obs.copy()
    pert_col = "perturbation" if "perturbation" in obs.columns else "label_str"

    # Identify NTC guides
    is_ntc = obs[pert_col].apply(
        lambda p: str(p).upper().startswith("NTC") or "non-targeting" in str(p).lower()
    )
    ntc_idx = np.where(is_ntc.values)[0]

    if len(ntc_idx) < group_size:
        # Not enough NTCs to split — just aggregate normally
        return aggregate_to_level(
            adata_guide, "gene", preserve_batch_info=False, subsample_controls=False,
        )

    # Assign NTC guides to random groups of `group_size`
    shuffled = ntc_idx.copy()
    rng.shuffle(shuffled)
    # Convert to string dtype to avoid Categorical issues with new categories
    new_pert = obs[pert_col].astype(str).copy()
    grp_num = 1
    for i in range(0, len(shuffled), group_size):
        chunk = shuffled[i:i + group_size]
        if len(chunk) < group_size:
            # Leftover NTCs (< group_size) — assign to last full group
            new_pert.iloc[chunk] = f"NTC_grp{grp_num - 1}" if grp_num > 1 else f"NTC_grp{grp_num}"
        else:
            new_pert.iloc[chunk] = f"NTC_grp{grp_num}"
            grp_num += 1

    # Create a copy with modified perturbation labels
    adata_mod = adata_guide.copy()
    adata_mod.obs[pert_col] = new_pert

    # Aggregate to gene level — NTC_grp1..N become separate gene entries
    return aggregate_to_level(
        adata_mod, "gene", preserve_batch_info=False, subsample_controls=False,
    )


CHAD_V4_PATH = Path("/hpc/projects/icd.ops/configs/gene_clusters/chad_positive_controls_v4.yml")
SKIP_CLUSTERS = {"NTCs"}
MIN_GENES_PER_CLUSTER = 2


def _load_positive_controls(path: Path = CHAD_V4_PATH) -> Dict[str, List[str]]:
    """Load CHAD positive control clusters from YAML. Returns {name: [genes]}."""
    if not path.exists():
        return {}
    with open(path) as f:
        raw = yaml.safe_load(f)
    clusters = {}
    for _id, data in raw.items():
        name = data.get("name", f"cluster_{_id}")
        genes = data.get("genes", [])
        if name in SKIP_CLUSTERS or len(genes) < MIN_GENES_PER_CLUSTER:
            continue
        clusters[name] = genes
    return clusters


def _plot_positive_controls_grid(
    embeddings: Dict[str, np.ndarray],
    perts: np.ndarray,
    level_name: str,
    plots_dir: Path,
    plt_mod,
    random_seed: int = 42,
):
    """
    Generate separate UMAP and PHATE canvases with CHAD positive control groups highlighted.

    Each canvas is a multi-column grid:
      - First panel: NTC groups highlighted (null baseline)
      - Remaining panels: one per CHAD cluster

    Parameters
    ----------
    embeddings : dict
        {"UMAP": coords, "PHATE": coords} — 2D arrays of shape (n_obs, 2)
    perts : np.ndarray
        Perturbation labels for each observation
    level_name : str
        "gene" or "guide"
    plots_dir : Path
        Where to save the figures
    plt_mod : module
        matplotlib.pyplot
    random_seed : int
        Seed for NTC subsampling
    """
    import seaborn as sns

    clusters = _load_positive_controls()
    if not clusters:
        logger.warning("  Positive controls grid skipped: could not load CHAD v4")
        return

    embed_names = [k for k in ["UMAP", "PHATE"] if k in embeddings]
    if not embed_names:
        return

    # Filter clusters to genes present in data
    all_perts = set(perts)
    filtered = {}
    for name, genes in clusters.items():
        present = [g for g in genes if g in all_perts]
        if len(present) >= MIN_GENES_PER_CLUSTER:
            filtered[name] = present
    if not filtered:
        logger.warning("  Positive controls grid: no clusters found in data")
        return

    # Find NTC indices for the first panel
    is_ntc = np.array([
        str(p).upper().startswith("NTC") or "non-targeting" in str(p).lower()
        for p in perts
    ])
    ntc_indices = np.where(is_ntc)[0]

    n_clusters = len(filtered)
    cluster_colors = sns.color_palette("husl", n_clusters)

    # Generate one canvas per embedding type
    for embed_name in embed_names:
        coords = embeddings[embed_name]

        # Total panels: 1 (NTC) + n_clusters
        n_panels = 1 + n_clusters
        n_cols = min(6, n_panels)
        n_rows = (n_panels + n_cols - 1) // n_cols

        fig_width = 4.5 * n_cols
        fig_height = 4 * n_rows
        fig, axes = plt_mod.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        flat_axes = axes.flatten()

        # --- Panel 0: NTC groups highlighted ---
        ax0 = flat_axes[0]
        ax0.scatter(
            coords[:, 0], coords[:, 1],
            c="lightgray", s=12, alpha=0.4, rasterized=True,
        )
        if len(ntc_indices) > 0:
            ax0.scatter(
                coords[ntc_indices, 0], coords[ntc_indices, 1],
                c="#E03030", s=60, alpha=0.9, marker="D",
                edgecolors="black", linewidths=0.6, zorder=4,
                label=f"NTC ({len(ntc_indices)})",
            )
            ax0.legend(fontsize=7, loc="best")
        ax0.set_title(f"NTCs ({len(ntc_indices)} groups of 4)", fontsize=9, fontweight="bold")
        ax0.set_xlabel(f"{embed_name} 1", fontsize=8)
        ax0.set_ylabel(f"{embed_name} 2", fontsize=8)
        ax0.tick_params(labelsize=6)

        # --- Panels 1..N: one per CHAD cluster ---
        for panel_idx, (cluster_name, genes) in enumerate(filtered.items(), start=1):
            if panel_idx >= len(flat_axes):
                break
            ax = flat_axes[panel_idx]
            gene_mask = np.isin(perts, genes)

            # Background
            ax.scatter(
                coords[:, 0], coords[:, 1],
                c="lightgray", s=12, alpha=0.4, rasterized=True,
            )

            # Highlight cluster
            if gene_mask.any():
                color_idx = panel_idx - 1
                ax.scatter(
                    coords[gene_mask, 0], coords[gene_mask, 1],
                    c=[cluster_colors[color_idx % len(cluster_colors)]],
                    s=70, alpha=0.95,
                    edgecolors="black", linewidths=0.7, rasterized=True, zorder=4,
                )

                # Label genes
                if level_name == "gene" and len(genes) <= 15:
                    for gene in genes:
                        g_mask = perts == gene
                        if g_mask.any():
                            idx = np.where(g_mask)[0][0]
                            ax.annotate(
                                gene, coords[idx],
                                fontsize=6, alpha=0.9,
                                xytext=(3, 3), textcoords="offset points",
                            )

            ax.set_title(f"{cluster_name} ({len(genes)}g)", fontsize=8, fontweight="bold")
            ax.set_xlabel(f"{embed_name} 1", fontsize=8)
            ax.set_ylabel(f"{embed_name} 2", fontsize=8)
            ax.tick_params(labelsize=6)

        # Hide empty panels
        for idx in range(n_panels, len(flat_axes)):
            flat_axes[idx].axis("off")

        fig.suptitle(
            f"CHAD Positive Controls — {level_name.title()} {embed_name}",
            fontsize=13, fontweight="bold", y=1.01,
        )
        fig.tight_layout()
        fname = f"positive_controls_{embed_name.lower()}_{level_name}.png"
        fig.savefig(plots_dir / fname, dpi=150, bbox_inches="tight")
        plt_mod.close(fig)
        logger.info(f"  Saved plots/{fname}")



def _plot_sweep(sweep_df, signal, peak_t, peak_n, best_act_t, suptitle, plots_dir, file_prefix):
    """Shared 3-panel sweep plot: Activity, AUC, and #PCs vs variance threshold."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    ts = sweep_df["threshold"].values
    acts = sweep_df["activity"].values * 100
    aucs = sweep_df["auc"].values
    pcs = sweep_df["n_pcs"].values

    # Activity vs threshold
    ax1.plot(ts, acts, "o-", color="steelblue", linewidth=2, markersize=6)
    ax1.axvline(peak_t, color="red", linestyle="--", alpha=0.6, label=f"AUC peak={peak_t:.0%}")
    if best_act_t is not None and best_act_t != peak_t:
        ax1.axvline(best_act_t, color="green", linestyle="--", alpha=0.4, label=f"Act peak={best_act_t:.0%}")
    ax1.set_xlabel("Explained Variance Threshold")
    ax1.set_ylabel("% Active Perturbations")
    ax1.set_title(f"{signal}: Activity vs Threshold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # AUC vs threshold
    ax2.plot(ts, aucs, "o-", color="darkorange", linewidth=2, markersize=6)
    ax2.axvline(peak_t, color="red", linestyle="--", alpha=0.6, label=f"AUC peak={peak_t:.0%}")
    ax2.set_xlabel("Explained Variance Threshold")
    ax2.set_ylabel("Activity AUC")
    ax2.set_title(f"{signal}: AUC vs Threshold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # N PCs vs threshold
    ax3.plot(ts, pcs, "o-", color="green", linewidth=2, markersize=6)
    ax3.axhline(MIN_PCS, color="red", linestyle="--", alpha=0.5, label=f"MIN_PCS={MIN_PCS}")
    ax3.axvline(peak_t, color="red", linestyle="--", alpha=0.6, label=f"Selected={peak_n} PCs")
    ax3.set_xlabel("Explained Variance Threshold")
    ax3.set_ylabel("Number of PCs")
    ax3.set_title(f"{signal}: PCs vs Threshold")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / f"{file_prefix}_sweep.png", dpi=150)
    plt.close(fig)


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
        n_pcs = _n_pcs_for_threshold(cumvar, threshold)
        X_slice = X_pcs[:, :n_pcs].astype(np.float32)
        pc_names = [f"PC{j}" for j in range(n_pcs)]

        adata_tmp = ad.AnnData(
            X=X_slice, obs=obs_df.copy(), var=pd.DataFrame(index=pc_names),
        )
        guide_tmp = aggregate_to_level(adata_tmp, level="guide", method="mean", preserve_batch_info=False)
        del adata_tmp
        guide_tmp.X = np.asarray(guide_tmp.X, dtype=np.float32)

        guide_norm = _normalize_guide(guide_tmp.copy(), norm_method)
        guide_norm.X = np.asarray(guide_norm.X, dtype=np.float32)
        try:
            r, a = _score_activity(guide_norm)
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
        _plot_sweep(
            sweep_df, signal, peak_t, peak_n, best_act_t,
            suptitle=suptitle,
            plots_dir=out_subdir / "plots",
            file_prefix=file_prefix,
        )
        _logger.info(f"  Saved plot: {subdir}/plots/{file_prefix}_sweep.png")
    except Exception as plot_err:
        _logger.warning(f"  Plot failed: {plot_err}")


# =============================================================================
# Phase 1a: Per-experiment PCA sweep (one SLURM job per experiment-channel)
# =============================================================================

def pca_sweep_single_experiment(
    exp: str,
    channel: str,
    output_dir: str,
    norm_method: str = "ntc",
    sweep_thresholds: Optional[List[float]] = None,
    feature_dir_override: Optional[str] = None,
) -> str:
    """PCA variance sweep for a single experiment-channel pair.

    Loads cell-level features from one experiment/channel, fits PCA, sweeps
    variance thresholds to find the optimal number of PCs, and saves
    guide/gene-level h5ads at the best threshold.

    Top-level function (not a method) so submitit can pickle it.

    Saves to output_dir/per_channel/:
      - {signal}_guide.h5ad  (aggregated at peak PCs, pre-NTC-normalization)
      - {signal}_gene.h5ad
      - {signal}_sweep.csv   (full sweep data for this channel)
    """
    _logger = _init_sweep_logger()
    t_start = time.time()
    output_dir = Path(output_dir)
    thresholds = sweep_thresholds or DEFAULT_SWEEP_THRESHOLDS

    # Resolve metadata using robust channel resolution (same as attribution stage)
    import io, contextlib
    maps_path = get_channel_maps_path()
    from ops_utils.data.feature_metadata import FeatureMetadata
    fm = FeatureMetadata(metadata_path=maps_path)
    exp_short = exp.split("_")[0]
    with contextlib.redirect_stdout(io.StringIO()):
        resolved = resolve_channel_label(fm, exp, channel)
    sig = resolved["label"]

    # Skip unmapped channels — notify user and return early
    if sig == "unknown" or sig.startswith("(unmapped:"):
        msg = f"SKIPPED: {exp} / {channel} — could not resolve channel to a biological signal (label={sig!r})"
        _logger.warning(msg)
        return msg

    # Resolve storage roots from config
    attr_config = load_attribution_config()
    storage_roots = get_storage_roots(attr_config)
    feature_dir = feature_dir_override or attr_config.get("feature_dir", "dino_features")

    label = f"{exp_short}/{channel} ({sig})"
    _logger.info(f"Processing {label} [features: {feature_dir}]...")

    # Load cell data
    adata_cells = load_cell_h5ad(exp, channel, storage_roots, feature_dir, maps_path)
    if adata_cells is None:
        return f"SKIPPED: {label} — cell data not found"

    n_cells = adata_cells.n_obs
    n_feats = adata_cells.n_vars
    _logger.info(f"  {n_cells} cells, {n_feats} features")

    # Ensure label_str exists (CellProfiler uses 'perturbation' instead)
    if "label_str" not in adata_cells.obs.columns and "perturbation" in adata_cells.obs.columns:
        adata_cells.obs["label_str"] = adata_cells.obs["perturbation"]

    # Keep obs cols needed for aggregation
    keep_cols = [c for c in ["sgRNA", "perturbation", "label_str"] if c in adata_cells.obs.columns]
    obs_df = adata_cells.obs[keep_cols].copy()
    X_raw = np.asarray(adata_cells.X, dtype=np.float32)
    del adata_cells

    # Global z-score before PCA for CellProfiler features (different scales need standardization)
    if feature_dir_override and "cell-profiler" in feature_dir_override:
        from sklearn.preprocessing import StandardScaler
        X_raw = StandardScaler().fit_transform(X_raw)
        _logger.info(f"  Applied global z-score scaling (CellProfiler mode)")

    # Fit PCA once
    X_pcs, cumvar, pca_model = _fit_pca(X_raw)
    del X_raw, pca_model

    # Sweep thresholds
    result = _run_threshold_sweep(
        X_pcs, cumvar, obs_df, thresholds, norm_method,
        extra_sweep_cols={"experiment": exp, "channel": channel, "signal": sig},
        _logger=_logger,
    )
    if result is None:
        return f"FAILED: {label} — no valid threshold found (all < {MIN_PCS} PCs)"
    sweep_rows, best_auc_t, best_auc_r, best_auc_a, best_auc_n, best_act_t = result

    # Save outputs at AUC-optimized peak
    obs_df["experiment"] = exp_short
    file_prefix = f"{exp_short}_{sig}"
    _save_sweep_outputs(
        X_pcs, obs_df, cumvar,
        peak_n=best_auc_n, peak_t=best_auc_t,
        best_auc_r=best_auc_r, best_auc_a=best_auc_a, best_act_t=best_act_t,
        signal=sig, sweep_rows=sweep_rows,
        uns_metadata={
            "experiment": exp, "channel": channel,
            "n_cells": int(n_cells), "n_features_raw": int(n_feats),
        },
        output_dir=output_dir, subdir="per_channel", file_prefix=file_prefix,
        suptitle=f"{exp_short}/{channel} → {sig} ({n_cells:,} cells, {n_feats} raw features)",
        rng=np.random.RandomState(42), _logger=_logger,
    )

    elapsed = time.time() - t_start
    _logger.info(f"  Done: {sig} in {elapsed:.0f}s — {best_auc_n} PCs @ {best_auc_t:.0%}")

    return f"SUCCESS: {sig} — {best_auc_n} PCs @ {best_auc_t:.0%}, {best_auc_r:.1%} active, AUC={best_auc_a:.4f}"


# =============================================================================
# Phase 1b: Pooled PCA sweep (one SLURM job per biological signal group)
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

    # --- Pass 2: Load one experiment at a time, subsample, free immediately ---
    all_blocks = []
    n_vars_expected = None
    loaded_exps = []

    for exp, ch in exp_channel_pairs:
        if (exp, ch) not in exp_cell_counts or exp_cell_counts[(exp, ch)] == 0:
            continue

        adata = load_cell_h5ad(exp, ch, storage_roots, feature_dir, maps_path)
        if adata is None:
            continue

        # Track feature counts (no longer skip — use inner join on concat)
        if n_vars_expected is None:
            n_vars_expected = adata.n_vars
        elif adata.n_vars != n_vars_expected:
            _logger.info(f"  {exp}/{ch}: {adata.n_vars} features (vs {n_vars_expected}), will use shared features")

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
        all_blocks.append(ad.AnnData(
            X=np.asarray(adata.X, dtype=np.float32),
            obs=obs,
            var=adata.var.copy(),
        ))
        loaded_exps.append(exp)
        _logger.info(f"  {exp.split('_')[0]}/{ch}: {exp_cell_counts[(exp, ch)]:,} → {n_take:,} cells")
        del adata

    if not all_blocks:
        return f"FAILED: {signal} — no cell data found for any experiment"

    # Concatenate subsampled blocks — inner join keeps only features shared across all experiments
    adata_cells = ad.concat(all_blocks, join="inner")
    del all_blocks
    # Fill any NaNs from partial feature overlap
    if np.isnan(adata_cells.X).any():
        adata_cells.X = np.nan_to_num(adata_cells.X, nan=0.0)
    n_cells = adata_cells.n_obs
    n_feats = adata_cells.n_vars
    _logger.info(f"  Pooled: {n_cells_pooled} total cells → {n_cells} downsampled ({n_feats} shared features from {len(loaded_exps)} experiments)")

    # Keep obs and raw X
    # Note: 'experiment' is kept for provenance but must be dropped before scoring
    # (copairs treats any non-meta obs column as a feature, breaking mAP)
    keep_cols = [c for c in ["sgRNA", "perturbation", "label_str", "experiment"] if c in adata_cells.obs.columns]
    obs_df_full = adata_cells.obs[keep_cols].copy()
    score_cols = [c for c in ["sgRNA", "perturbation", "label_str"] if c in obs_df_full.columns]
    obs_df = obs_df_full[score_cols].copy()
    X_raw = np.asarray(adata_cells.X, dtype=np.float32)
    del adata_cells

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

    # --- Fit PCA once ---
    X_pcs, cumvar, pca_model = _fit_pca(X_raw)
    del X_raw, pca_model

    # Sweep thresholds
    result = _run_threshold_sweep(
        X_pcs, cumvar, obs_df, thresholds, norm_method,
        extra_sweep_cols={"signal": signal, "n_experiments": n_exps},
        _logger=_logger,
    )
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
# Phase 2: Aggregation (load per-channel h5ads, hconcat, normalize, score, plot)
# =============================================================================

def aggregate_channels(
    output_dir: str,
    norm_method: str = "ntc",
    per_unit_subdir: str = "per_channel",
) -> str:
    """
    Load per-channel (or per-signal) h5ads, horizontally concatenate, normalize, score, save.

    Top-level function (not a method) so submitit can pickle it.

    Args:
        per_unit_subdir: subdirectory containing guide/gene h5ads.
            "per_channel" for standard mode, "per_signal" for downsampled mode.
    """
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("copairs").setLevel(logging.WARNING)
    _logger = logging.getLogger(__name__)

    t_start = time.time()
    output_dir = Path(output_dir)
    per_channel_dir = output_dir / per_unit_subdir

    # Discover per-channel outputs
    guide_files = sorted(per_channel_dir.glob("*_guide.h5ad"))
    if not guide_files:
        return "FAILED: no per-channel h5ad files found"

    _logger.info(f"Found {len(guide_files)} per-channel guide files")

    # Load all per-channel blocks
    guide_blocks = []
    gene_blocks = []
    report_rows = []
    total_cells = 0

    for gf in guide_files:
        file_prefix = gf.stem.replace("_guide", "")
        gene_file = per_channel_dir / f"{file_prefix}_gene.h5ad"
        if not gene_file.exists():
            _logger.warning(f"  Skipping {file_prefix}: no gene file")
            continue

        g = ad.read_h5ad(gf)
        sig = g.uns.get("signal", file_prefix)

        # Skip unmapped channels that slipped through
        if sig == "unknown" or sig.startswith("(unmapped:"):
            _logger.warning(f"  Skipping {file_prefix}: unmapped channel (signal={sig!r})")
            continue

        e = ad.read_h5ad(gene_file)
        guide_blocks.append(g)
        gene_blocks.append(e)
        n_cells = int(g.uns.get("n_cells", 0))
        total_cells += n_cells
        report_rows.append({
            "experiment": g.uns.get("experiment", ""),
            "channel": g.uns.get("channel", ""),
            "signal": sig,
            "n_cells": n_cells,
            "n_features_raw": int(g.uns.get("n_features_raw", 0)),
            "peak_threshold": float(g.uns.get("pca_threshold", 0)),
            "n_pcs": int(g.uns.get("n_pcs", 0)),
            "explained_variance": float(g.uns.get("explained_variance", 0)),
            "activity": float(g.uns.get("peak_activity", 0)),
            "auc": float(g.uns.get("peak_auc", 0)),
        })
        _logger.info(f"  {sig}: {g.n_obs} guides x {g.n_vars} PCs @ {g.uns.get('pca_threshold', '?')}")

    if not guide_blocks:
        return "FAILED: no valid per-channel data loaded"

    # Horizontal concat
    adata_guide = _hconcat(guide_blocks, "guide")
    adata_gene = _hconcat(gene_blocks, "gene")
    del guide_blocks, gene_blocks

    total_feats = adata_guide.n_vars
    _logger.info(f"Concatenated: {adata_guide.n_obs} guides, {total_feats} features")

    # NTC normalize
    _logger.info(f"NTC normalizing at guide level...")
    adata_guide = _normalize_guide(adata_guide, norm_method)
    adata_guide.X = np.asarray(adata_guide.X, dtype=np.float32)

    # Re-aggregate to gene level
    adata_gene = aggregate_to_level(
        adata_guide, "gene",
        preserve_batch_info=False,
        subsample_controls=False,
    )
    _logger.info(f"  Guide: {adata_guide.n_obs} obs, {adata_guide.n_vars} features")
    _logger.info(f"  Gene: {adata_gene.n_obs} obs, {adata_gene.n_vars} features")

    # Strip obs to copairs-required columns (extra string cols cause isnan error in copairs)
    for _ad in [adata_guide, adata_gene]:
        if _ad is not None:
            if "n_cells" not in _ad.obs.columns:
                _ad.obs["n_cells"] = 1
            keep = [c for c in ["sgRNA", "perturbation", "n_cells"] if c in _ad.obs.columns]
            _ad.obs = _ad.obs[keep].copy()
            for col in _ad.obs.columns:
                if _ad.obs[col].dtype.name == "category":
                    _ad.obs[col] = _ad.obs[col].astype(str)
            _ad.X = np.asarray(_ad.X, dtype=np.float64)

    # --- Phase 2a: Activity scoring (fast) ---
    _logger.info(f"Running activity scoring...")
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    map_results = None
    try:
        activity_map, active_ratio = phenotypic_activity_assesment(
            adata_guide, plot_results=False, null_size=100_000,
        )
        activity_map.to_csv(metrics_dir / "phenotypic_activity.csv", index=False)
        r = active_ratio
        a = compute_auc_score(activity_map)
        _logger.info(f"  Activity: {r:.1%} active, AUC={a:.4f}")

        # Print inactive genes
        inactive = activity_map[~activity_map["below_corrected_p"]]
        inactive = inactive[inactive["perturbation"] != "NTC"]
        if len(inactive) > 0:
            names = sorted(inactive["perturbation"].tolist())
            _logger.info(f"  Inactive genes ({len(names)}): {', '.join(names)}")
        else:
            _logger.info(f"  All non-NTC perturbations are active")
    except Exception as exc:
        _logger.error(f"  Activity scoring failed: {exc}")
        activity_map = None
        r, a = 0.0, 0.0

    # Save h5ads immediately (before slow metrics)
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

    # --- Setup plots ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Save activity mAP scatter immediately
    if activity_map is not None:
        try:
            fig, ax = plt.subplots(figsize=(8, 7))
            plot_map_scatter(ax, activity_map, "Activity", r)
            fig.suptitle(f"Phenotypic Activity — {total_feats} features", fontsize=13, fontweight="bold")
            fig.tight_layout()
            fig.savefig(plots_dir / "map_activity.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            _logger.info(f"  Saved plots/map_activity.png")
        except Exception as e:
            _logger.warning(f"  Activity plot failed: {e}")

    # Collect all per-channel sweep CSVs into one
    sweep_csvs = sorted(per_channel_dir.glob("*_sweep.csv"))
    if sweep_csvs:
        sweep_df = pd.concat([pd.read_csv(f) for f in sweep_csvs], ignore_index=True)
        sweep_df.to_csv(output_dir / "pca_sweep_all_channels.csv", index=False)
        _logger.info(f"  Saved pca_sweep_all_channels.csv ({len(sweep_df)} rows)")

    if sweep_csvs:
        sweep_df = pd.read_csv(output_dir / "pca_sweep_all_channels.csv")
        signals_in_sweep = sweep_df["signal"].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(signals_in_sweep)))

        # Plot 1: Per-channel sweep curves
        n_signals = len(signals_in_sweep)
        legend_cols = max(1, n_signals // 20)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        for i, sig in enumerate(sorted(signals_in_sweep)):
            sub = sweep_df[sweep_df["signal"] == sig].sort_values("threshold")
            ax1.plot(sub["threshold"], sub["activity"] * 100, "o-", color=colors[i],
                     linewidth=1.5, markersize=4, label=sig, alpha=0.8)
            ax2.plot(sub["threshold"], sub["auc"], "o-", color=colors[i],
                     linewidth=1.5, markersize=4, label=sig, alpha=0.8)
        ax1.axhline(r * 100, color="black", linestyle=":", alpha=0.5, label=f"Pooled baseline ({r:.1%})")
        ax2.axhline(a, color="black", linestyle=":", alpha=0.5, label=f"Pooled baseline ({a:.4f})")
        ax1.axvline(0.80, color="gray", linestyle="--", alpha=0.3, label="80% threshold")
        ax2.axvline(0.80, color="gray", linestyle="--", alpha=0.3, label="80% threshold")
        ax1.set_xlabel("Explained Variance Threshold")
        ax1.set_ylabel("% Active Perturbations")
        ax1.set_title("Per-Channel PCA Sweep: Activity")
        ax1.grid(True, alpha=0.3)
        ax2.set_xlabel("Explained Variance Threshold")
        ax2.set_ylabel("Activity AUC")
        ax2.set_title("Per-Channel PCA Sweep: AUC")
        ax2.grid(True, alpha=0.3)
        # Single legend outside the plots (from ax2, which has same labels)
        ax2.legend(fontsize=5, loc="center left", bbox_to_anchor=(1.02, 0.5),
                   ncol=legend_cols, borderaxespad=0, frameon=True)
        fig.tight_layout(rect=[0, 0, 0.82, 1])
        fig.savefig(plots_dir / "per_channel_sweep.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        _logger.info(f"  Saved plots/per_channel_sweep.png")

        # Plot 2: N PCs vs threshold
        fig, ax = plt.subplots(figsize=(16, 8))
        for i, sig in enumerate(sorted(signals_in_sweep)):
            sub = sweep_df[sweep_df["signal"] == sig].sort_values("threshold")
            ax.plot(sub["threshold"], sub["n_pcs"], "o-", color=colors[i],
                    linewidth=1.5, markersize=4, label=sig, alpha=0.8)
        ax.axhline(MIN_PCS, color="red", linestyle="--", alpha=0.5, label=f"MIN_PCS={MIN_PCS}")
        ax.set_xlabel("Explained Variance Threshold")
        ax.set_ylabel("Number of PCs")
        ax.set_title("PCs Selected vs Variance Threshold (per channel)")
        ax.legend(fontsize=5, loc="center left", bbox_to_anchor=(1.02, 0.5),
                  ncol=legend_cols, borderaxespad=0, frameon=True)
        ax.grid(True, alpha=0.3)
        fig.tight_layout(rect=[0, 0, 0.82, 1])
        fig.savefig(plots_dir / "n_pcs_vs_threshold.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        _logger.info(f"  Saved plots/n_pcs_vs_threshold.png")

        # Plot 3: Summary sweep with mean curve
        # All channels as light lines, bold mean overlay
        fig, axes = plt.subplots(1, 3, figsize=(26, 8))
        ax_act, ax_auc, ax_pcs = axes

        # Compute mean curves by interpolating all channels onto common thresholds
        all_thresholds = sorted(sweep_df["threshold"].unique())
        act_matrix = []  # (n_channels, n_thresholds)
        auc_matrix = []
        pcs_matrix = []

        for sig in sorted(signals_in_sweep):
            sub = sweep_df[sweep_df["signal"] == sig].sort_values("threshold")
            # Interpolate to common thresholds
            act_interp = np.interp(all_thresholds, sub["threshold"], sub["activity"])
            auc_interp = np.interp(all_thresholds, sub["threshold"], sub["auc"])
            pcs_interp = np.interp(all_thresholds, sub["threshold"], sub["n_pcs"])
            act_matrix.append(act_interp)
            auc_matrix.append(auc_interp)
            pcs_matrix.append(pcs_interp)

        act_matrix = np.array(act_matrix)
        auc_matrix = np.array(auc_matrix)
        pcs_matrix = np.array(pcs_matrix)
        mean_act = act_matrix.mean(axis=0)
        mean_auc = auc_matrix.mean(axis=0)
        mean_pcs = pcs_matrix.mean(axis=0)

        # Plot individual channels as light lines
        for i, sig in enumerate(sorted(signals_in_sweep)):
            sub = sweep_df[sweep_df["signal"] == sig].sort_values("threshold")
            ax_act.plot(sub["threshold"], sub["activity"] * 100, "-", color=colors[i],
                        linewidth=0.8, alpha=0.35, label=sig)
            ax_auc.plot(sub["threshold"], sub["auc"], "-", color=colors[i],
                        linewidth=0.8, alpha=0.35, label=sig)
            ax_pcs.plot(sub["threshold"], sub["n_pcs"], "-", color=colors[i],
                        linewidth=0.8, alpha=0.35, label=sig)

        # Overlay mean as bold black line
        ax_act.plot(all_thresholds, mean_act * 100, "o-", color="black",
                    linewidth=2.5, markersize=5, label="Mean", zorder=10)
        ax_auc.plot(all_thresholds, mean_auc, "o-", color="black",
                    linewidth=2.5, markersize=5, label="Mean", zorder=10)
        ax_pcs.plot(all_thresholds, mean_pcs, "o-", color="black",
                    linewidth=2.5, markersize=5, label="Mean", zorder=10)

        # Baselines and reference lines
        ax_act.axhline(r * 100, color="red", linestyle=":", alpha=0.6, label=f"Pooled baseline ({r:.1%})")
        ax_auc.axhline(a, color="red", linestyle=":", alpha=0.6, label=f"Pooled baseline ({a:.4f})")
        ax_pcs.axhline(MIN_PCS, color="red", linestyle="--", alpha=0.5, label=f"MIN_PCS={MIN_PCS}")
        for ax in axes:
            ax.axvline(0.80, color="gray", linestyle="--", alpha=0.3)
            ax.set_xlabel("Explained Variance Threshold")
            ax.grid(True, alpha=0.3)
        # Single legend outside plots (from ax_act which has all channel labels + mean)
        ax_pcs.legend(fontsize=5, loc="center left", bbox_to_anchor=(1.02, 0.5),
                      ncol=legend_cols, borderaxespad=0, frameon=True)

        ax_act.set_ylabel("% Active Perturbations")
        ax_act.set_title("Activity vs Threshold")
        ax_auc.set_ylabel("Activity AUC")
        ax_auc.set_title("AUC vs Threshold")
        ax_pcs.set_ylabel("Number of PCs")
        ax_pcs.set_title("PCs vs Threshold")

        fig.suptitle(f"Summary Sweep: {len(signals_in_sweep)} channels, mean shown in black", fontsize=12)
        fig.tight_layout(rect=[0, 0, 0.82, 1])
        fig.savefig(plots_dir / "summary_sweep.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        _logger.info(f"  Saved plots/summary_sweep.png")

    # Plot 4: Bar chart of per-channel peaks
    report_df = pd.DataFrame(report_rows)
    if len(report_df) > 0:
        # Create exp_signal labels (e.g. ops0046_Phase)
        exp_short = report_df["experiment"].apply(lambda e: str(e).split("_")[0] if pd.notna(e) else "")
        report_df["bar_label"] = exp_short + "_" + report_df["signal"].astype(str)

        # Sort by activity (highest to lowest) for top plot
        df_by_activity = report_df.sort_values("activity", ascending=False).reset_index(drop=True)
        # Sort by AUC (highest to lowest) for bottom plot
        df_by_auc = report_df.sort_values("auc", ascending=False).reset_index(drop=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(16, len(report_df) * 1.0), 14))

        x1 = np.arange(len(df_by_activity))
        bars1 = ax1.bar(x1, df_by_activity["activity"] * 100, color="steelblue", alpha=0.8)
        ax1.axhline(r * 100, color="red", linestyle="--", alpha=0.6, label=f"Pooled baseline ({r:.1%})")
        for i, (_, row) in enumerate(df_by_activity.iterrows()):
            ax1.text(x1[i], row["activity"] * 100 + 0.5,
                     f"thr={row['peak_threshold']:.0%}\n{row['n_pcs']}PCs",
                     ha="center", va="bottom", fontsize=7)
        ax1.set_xticks(x1)
        ax1.set_xticklabels(df_by_activity["bar_label"], rotation=45, ha="right", fontsize=7)
        ax1.set_ylabel("% Active Perturbations")
        ax1.set_title("Per-Channel Peak Activity (at optimal PCA threshold) — sorted by activity")
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")

        x2 = np.arange(len(df_by_auc))
        bars2 = ax2.bar(x2, df_by_auc["auc"], color="darkorange", alpha=0.8)
        ax2.axhline(a, color="red", linestyle="--", alpha=0.6, label=f"Pooled baseline ({a:.4f})")
        for i, (_, row) in enumerate(df_by_auc.iterrows()):
            ax2.text(x2[i], row["auc"] + 0.002,
                     f"thr={row['peak_threshold']:.0%}\n{row['n_pcs']}PCs",
                     ha="center", va="bottom", fontsize=7)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(df_by_auc["bar_label"], rotation=45, ha="right", fontsize=7)
        ax2.set_ylabel("Activity AUC")
        ax2.set_title("Per-Channel Peak AUC (at optimal PCA threshold) — sorted by AUC")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")
        fig.subplots_adjust(bottom=0.25, hspace=0.45)
        fig.savefig(plots_dir / "per_channel_peaks.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        _logger.info(f"  Saved plots/per_channel_peaks.png")

    # --- Phase 2b: Embedding plots (UMAP + PHATE, before slow metrics) ---

    # Build metric lookup from activity_map
    metric_lookup = {}
    if activity_map is not None:
        act_df = activity_map
        if "-log10(p-value)" not in act_df.columns and "corrected_p_value" in act_df.columns:
            act_df = act_df.copy()
            act_df["-log10(p-value)"] = -np.log10(act_df["corrected_p_value"].clip(lower=1e-300))
        metric_lookup = act_df.set_index("perturbation")[
            ["mean_average_precision", "-log10(p-value)", "below_corrected_p"]
        ].to_dict("index")

    # Helper to get perts array from adata
    def _get_perts(adata_level):
        obs = adata_level.obs
        pert_col = "perturbation" if "perturbation" in obs.columns else "label_str"
        return obs[pert_col].values

    def _clean_X(adata_level):
        X = np.asarray(adata_level.X, dtype=np.float32)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Build gene-level adata with NTCs split into random groups of 4
    # This way the gene UMAP/PHATE shows NTCs as multiple dots matching KO group sizes
    adata_gene_embed = _split_ntc_for_embedding(adata_guide, random_seed=42)
    _logger.info(f"  Gene (NTC-split for embedding): {adata_gene_embed.n_obs} obs")

    # Track embeddings for positive controls grid
    # {level_name: {"UMAP": coords, "PHATE": coords}}
    level_embeddings = {}
    level_perts = {}

    # UMAP
    try:
        from umap import UMAP

        for level_name, adata_level in [("guide", adata_guide), ("gene", adata_gene_embed)]:
            _logger.info(f"  Computing {level_name} UMAP ({adata_level.n_obs} obs, {adata_level.n_vars} features)...")
            X_clean = _clean_X(adata_level)
            n_neighbors = min(15, adata_level.n_obs - 1)
            if n_neighbors < 2:
                _logger.warning(f"  UMAP skipped for {level_name}: too few observations")
                continue

            umap_model = UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
            coords = umap_model.fit_transform(X_clean)
            perts = _get_perts(adata_level)

            level_embeddings.setdefault(level_name, {})["UMAP"] = coords
            level_perts[level_name] = perts

            fname = _plot_embedding_overlay(
                coords, perts, metric_lookup, level_name, "UMAP",
                plots_dir, adata_level.n_obs, adata_level.n_vars, plt,
            )
            _logger.info(f"  Saved plots/{fname}")

    except ImportError:
        _logger.warning("  UMAP plots skipped: install umap-learn (pip install umap-learn)")
    except Exception as umap_err:
        _logger.warning(f"  UMAP plots failed: {umap_err}")

    # PHATE
    try:
        import phate

        for level_name, adata_level in [("guide", adata_guide), ("gene", adata_gene_embed)]:
            _logger.info(f"  Computing {level_name} PHATE ({adata_level.n_obs} obs, {adata_level.n_vars} features)...")
            X_clean = _clean_X(adata_level)

            # Optimal PHATE params:
            #   knn=10 for gene (~1000 obs), knn=15 for guide (~4000 obs)
            #   t='auto' lets PHATE pick diffusion time via knee-point
            #   decay=15 sharpens the kernel for cleaner structure
            #   n_jobs=-1 for full parallelism
            knn = min(15 if adata_level.n_obs > 2000 else 10, adata_level.n_obs - 1)
            if knn < 2:
                _logger.warning(f"  PHATE skipped for {level_name}: too few observations")
                continue

            phate_op = phate.PHATE(
                n_components=2,
                knn=knn,
                decay=15,
                t="auto",
                n_jobs=-1,
                random_state=42,
                verbose=0,
            )
            coords = phate_op.fit_transform(X_clean)
            perts = _get_perts(adata_level)

            level_embeddings.setdefault(level_name, {})["PHATE"] = coords
            level_perts[level_name] = perts

            fname = _plot_embedding_overlay(
                coords, perts, metric_lookup, level_name, "PHATE",
                plots_dir, adata_level.n_obs, adata_level.n_vars, plt,
            )
            _logger.info(f"  Saved plots/{fname}")

    except ImportError:
        _logger.warning("  PHATE plots skipped: install phate (pip install phate)")
    except Exception as phate_err:
        _logger.warning(f"  PHATE plots failed: {phate_err}")

    # Positive controls overlay grid (CHAD v4)
    for level_name in ["gene"]:
        if level_name in level_embeddings and level_name in level_perts:
            try:
                _plot_positive_controls_grid(
                    level_embeddings[level_name],
                    level_perts[level_name],
                    level_name,
                    plots_dir,
                    plt,
                )
            except Exception as pc_err:
                _logger.warning(f"  Positive controls grid failed for {level_name}: {pc_err}")

    # --- Phase 2c: Distinctiveness (medium speed) ---
    distinctiveness_map = None
    if activity_map is not None:
        try:
            from ops_utils.analysis.map_scores import phenotypic_distinctivness
            _logger.info(f"Running distinctiveness...")
            distinctiveness_map, distinctive_ratio = phenotypic_distinctivness(
                adata_guide, activity_map, plot_results=False, null_size=100_000,
            )
            distinctiveness_map.to_csv(metrics_dir / "phenotypic_distinctiveness.csv", index=False)
            _logger.info(f"  Distinctiveness: {distinctive_ratio:.1%}")

            # Save activity + distinctiveness 2-panel plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            plot_map_scatter(ax1, activity_map, "Activity", r)
            plot_map_scatter(ax2, distinctiveness_map, "Distinctiveness", distinctive_ratio)
            fig.suptitle(f"Activity & Distinctiveness — {total_feats} features", fontsize=13, fontweight="bold")
            fig.tight_layout()
            fig.savefig(plots_dir / "map_activity_distinctiveness.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            _logger.info(f"  Saved plots/map_activity_distinctiveness.png")
        except Exception as exc:
            _logger.error(f"  Distinctiveness failed: {exc}")

    # --- Phase 2d: Consistency metrics (slow — CORUM + CHAD) ---
    if activity_map is not None:
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

            # Save CORUM + CHAD 2-panel plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            plot_map_scatter(ax1, consistency_corum_map, "Consistency (CORUM)", consistency_corum_ratio)
            plot_map_scatter(ax2, consistency_manual_map, "Consistency (CHAD)", consistency_manual_ratio)
            fig.suptitle(f"Consistency Metrics — {total_feats} features", fontsize=13, fontweight="bold")
            fig.tight_layout()
            fig.savefig(plots_dir / "map_consistency.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            _logger.info(f"  Saved plots/map_consistency.png")

        except Exception as exc:
            _logger.error(f"  Consistency metrics failed: {exc}")

    elapsed = time.time() - t_start
    _logger.info(f"\nDone in {elapsed/60:.1f} minutes")
    _logger.info(f"  {len(report_rows)} channels, {total_feats} total PCA features")
    _logger.info(f"  Baseline: {r:.1%} active, AUC={a:.4f}")

    return f"SUCCESS: {total_feats} features, {r:.1%} active, AUC={a:.4f}"


# =============================================================================
# Local mode: sequential (for testing or small runs)
# =============================================================================

def run_pca_optimization(
    output_dir: str,
    sweep_thresholds: Optional[List[float]] = None,
    norm_method: str = "ntc",
) -> str:
    """Run full pipeline locally (sequential): discover, per-channel sweep, aggregate."""
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("copairs").setLevel(logging.WARNING)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover experiments
    attr_config = load_attribution_config()
    storage_roots = get_storage_roots(attr_config)
    feature_dir = attr_config.get("feature_dir", "dino_features")
    all_pairs = discover_dino_experiments(storage_roots, feature_dir)
    logger.info(f"Found {len(all_pairs)} experiment-channel pairs")

    if not all_pairs:
        return "FAILED: no experiment-channel pairs found"

    # Phase 1: process each channel sequentially
    for i, (exp, ch) in enumerate(all_pairs):
        logger.info(f"\n[{i+1}/{len(all_pairs)}] Processing {exp}/{ch}...")
        result = pca_sweep_single_experiment(
            exp=exp,
            channel=ch,
            output_dir=str(output_dir),
            norm_method=norm_method,
            sweep_thresholds=sweep_thresholds,
        )
        logger.info(f"  {result}")

    # Phase 2: aggregate
    return aggregate_channels(output_dir=str(output_dir), norm_method=norm_method)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Per-channel cell-level PCA optimization for organelle attribution"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized",
        help="Output directory for PCA-optimized h5ad files",
    )
    parser.add_argument(
        "--norm-method",
        type=str,
        default="ntc",
        choices=["ntc", "global"],
        help="Normalization method (default: ntc)",
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Submit per-channel SLURM jobs + aggregation job",
    )
    parser.add_argument(
        "--slurm-memory",
        type=str,
        default="100GB",
        help="SLURM memory per channel job (default: 100GB)",
    )
    parser.add_argument(
        "--slurm-time",
        type=int,
        default=10,
        help="SLURM time limit per channel job in minutes (default: 10)",
    )
    parser.add_argument(
        "--slurm-cpus",
        type=int,
        default=16,
        help="SLURM CPUs per channel job (default: 16)",
    )
    parser.add_argument(
        "--slurm-partition",
        type=str,
        default="cpu,gpu",
        help="SLURM partition (default: cpu,gpu)",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt",
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
        "--aggregate-only",
        action="store_true",
        help="Only run the aggregation step (Phase 2). "
             "Use after per-channel jobs have completed.",
    )
    parser.add_argument(
        "--umap-only",
        action="store_true",
        help="Only generate embedding plots (UMAP + PHATE) from existing optimized h5ads. "
             "Loads guide/gene_pca_optimized.h5ad and metrics CSVs.",
    )
    parser.add_argument(
        "--downsampled",
        action="store_true",
        help="Pool cells by biological signal across experiments, downsample to "
             "equal cell counts per signal, then run PCA optimization. "
             "Outputs to {output_dir}/downsampled/.",
    )
    parser.add_argument(
        "--cell-profiler",
        action="store_true",
        help="Use CellProfiler morphological features instead of DINO embeddings. "
             "Loads from 3-assembly/cell-profiler/anndata_objects/. "
             "Outputs to {output_dir}/cellprofiler/.",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    # --cell-profiler: override feature_dir and nest output in cellprofiler/ subdir
    _cp_feature_dir_override = None
    if args.cell_profiler:
        _cp_feature_dir_override = "cell-profiler"
        output_dir = output_dir / "cellprofiler"
        print(f"CellProfiler mode: features from 3-assembly/cell-profiler/anndata_objects/")
        print(f"PCA sweep thresholds: {DEFAULT_SWEEP_THRESHOLDS_CP} (lower range — CP features are independent, not redundant like DINO)")
        print(f"Output: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # UMAP-only mode
    if args.umap_only:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        _logger = logging.getLogger(__name__)

        umap_dir = output_dir / "downsampled" if args.downsampled else output_dir
        guide_path = umap_dir / "guide_pca_optimized.h5ad"
        gene_path = umap_dir / "gene_pca_optimized.h5ad"
        if not guide_path.exists():
            print(f"ERROR: {guide_path} not found. Run --aggregate-only first.")
            return
        _logger.info(f"Loading {guide_path}...")
        adata_guide = ad.read_h5ad(guide_path)
        adata_gene = ad.read_h5ad(gene_path) if gene_path.exists() else None
        if adata_gene is None:
            _logger.warning(f"  {gene_path} not found, skipping gene-level UMAP")

        # Load activity metrics if available
        activity_csv = umap_dir / "metrics" / "phenotypic_activity.csv"
        metric_lookup = {}
        if activity_csv.exists():
            act_df = pd.read_csv(activity_csv)
            if "-log10(p-value)" not in act_df.columns and "corrected_p_value" in act_df.columns:
                act_df["-log10(p-value)"] = -np.log10(act_df["corrected_p_value"].clip(lower=1e-300))
            metric_lookup = act_df.set_index("perturbation")[
                ["mean_average_precision", "-log10(p-value)", "below_corrected_p"]
            ].to_dict("index")
            _logger.info(f"  Loaded activity metrics for {len(metric_lookup)} perturbations")
        else:
            _logger.warning(f"  No activity CSV found at {activity_csv}, UMAPs will be uncolored")

        plots_dir = umap_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Build gene-level with NTCs split into groups of 4 for embedding
        adata_gene_embed = _split_ntc_for_embedding(adata_guide, random_seed=42)
        _logger.info(f"  Gene (NTC-split for embedding): {adata_gene_embed.n_obs} obs")

        embed_pairs = [("guide", adata_guide), ("gene", adata_gene_embed)]

        def _clean_X(adata_level):
            X = np.asarray(adata_level.X, dtype=np.float32)
            return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        def _get_perts(adata_level):
            obs = adata_level.obs
            pert_col = "perturbation" if "perturbation" in obs.columns else "label_str"
            return obs[pert_col].values

        # Track embeddings for positive controls grid
        level_embeddings = {}
        level_perts_map = {}

        # UMAP
        try:
            from umap import UMAP
            for level_name, adata_level in embed_pairs:
                _logger.info(f"  Computing {level_name} UMAP ({adata_level.n_obs} obs, {adata_level.n_vars} features)...")
                X_clean = _clean_X(adata_level)
                n_neighbors = min(15, adata_level.n_obs - 1)
                if n_neighbors < 2:
                    continue
                coords = UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42).fit_transform(X_clean)
                perts = _get_perts(adata_level)
                level_embeddings.setdefault(level_name, {})["UMAP"] = coords
                level_perts_map[level_name] = perts
                fname = _plot_embedding_overlay(
                    coords, perts, metric_lookup, level_name, "UMAP",
                    plots_dir, adata_level.n_obs, adata_level.n_vars, plt,
                )
                _logger.info(f"  Saved plots/{fname}")
        except ImportError:
            _logger.warning("  UMAP skipped: install umap-learn")
        except Exception as e:
            _logger.warning(f"  UMAP failed: {e}")

        # PHATE
        try:
            import phate
            for level_name, adata_level in embed_pairs:
                _logger.info(f"  Computing {level_name} PHATE ({adata_level.n_obs} obs, {adata_level.n_vars} features)...")
                X_clean = _clean_X(adata_level)
                knn = min(15 if adata_level.n_obs > 2000 else 10, adata_level.n_obs - 1)
                if knn < 2:
                    continue
                phate_op = phate.PHATE(
                    n_components=2, knn=knn, decay=15, t="auto",
                    n_jobs=-1, random_state=42, verbose=0,
                )
                coords = phate_op.fit_transform(X_clean)
                perts = _get_perts(adata_level)
                level_embeddings.setdefault(level_name, {})["PHATE"] = coords
                level_perts_map[level_name] = perts
                fname = _plot_embedding_overlay(
                    coords, perts, metric_lookup, level_name, "PHATE",
                    plots_dir, adata_level.n_obs, adata_level.n_vars, plt,
                )
                _logger.info(f"  Saved plots/{fname}")
        except ImportError:
            _logger.warning("  PHATE skipped: install phate")
        except Exception as e:
            _logger.warning(f"  PHATE failed: {e}")

        # Positive controls overlay grid (CHAD v4)
        for level_name in ["gene"]:
            if level_name in level_embeddings and level_name in level_perts_map:
                try:
                    _plot_positive_controls_grid(
                        level_embeddings[level_name],
                        level_perts_map[level_name],
                        level_name,
                        plots_dir,
                        plt,
                    )
                except Exception as pc_err:
                    _logger.warning(f"  Positive controls grid failed for {level_name}: {pc_err}")

        print("SUCCESS: Embedding plots saved")
        return

    # Aggregate-only mode (local or SLURM)
    if args.aggregate_only:
        # Resolve paths and subdir for downsampled mode
        if args.downsampled:
            agg_output = str(output_dir / "downsampled")
            agg_subdir = "per_signal"
        else:
            agg_output = str(output_dir)
            agg_subdir = "per_channel"
        if args.slurm:
            from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
            print(f"Submitting aggregation as SLURM job ({args.slurm_agg_memory}, {args.slurm_agg_time}min, {args.slurm_cpus}cpus)...")
            if args.downsampled:
                print(f"  Mode: signal-group (reading from {agg_output}/per_signal/)")
            agg_jobs = [{
                "name": "pca_aggregate",
                "func": aggregate_channels,
                "kwargs": {
                    "output_dir": agg_output,
                    "norm_method": args.norm_method,
                    "per_unit_subdir": agg_subdir,
                },
            }]
            agg_slurm_params = {
                "timeout_min": args.slurm_agg_time,
                "mem": args.slurm_agg_memory,
                "cpus_per_task": args.slurm_cpus,
                "slurm_partition": args.slurm_partition,
            }
            agg_result = submit_parallel_jobs(
                jobs_to_submit=agg_jobs,
                experiment="pca_aggregation",
                slurm_params=agg_slurm_params,
                log_dir="pca_optimization",
                manifest_prefix="pca_agg",
                wait_for_completion=True,
            )
            if agg_result.get("failed"):
                print("Aggregation FAILED")
            else:
                print("Aggregation complete")
        else:
            result = aggregate_channels(output_dir=agg_output, norm_method=args.norm_method, per_unit_subdir=agg_subdir)
            print(result)
        return

    # --downsampled mode: pool cells by signal group, downsample, PCA
    if args.downsampled:
        from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

        if args.cell_profiler:
            ds_output_dir = output_dir / "downsampled"  # {output_dir}/cellprofiler/downsampled/
        else:
            ds_output_dir = output_dir / "downsampled"
        ds_output_dir.mkdir(parents=True, exist_ok=True)

        # Discover experiments
        attr_config = load_attribution_config()
        storage_roots = get_storage_roots(attr_config)
        feature_dir = _cp_feature_dir_override or attr_config.get("feature_dir", "dino_features")
        maps_path = get_channel_maps_path()

        if _cp_feature_dir_override:
            all_pairs = discover_cellprofiler_experiments(storage_roots)
        else:
            all_pairs = discover_dino_experiments(storage_roots, feature_dir)

        if not all_pairs:
            print("No experiment-channel pairs found!")
            return
        from ops_utils.data.feature_metadata import FeatureMetadata
        fm = FeatureMetadata(metadata_path=maps_path)

        # Build signal groups
        signal_groups = build_signal_groups(all_pairs, fm)
        n_signals = len(signal_groups)
        mode_label = "CellProfiler" if _cp_feature_dir_override else "Downsampled"
        print(f"\n{mode_label} PCA Optimization: {len(all_pairs)} channels → {n_signals} signal groups")
        print(f"Output: {ds_output_dir}")

        # Pre-scan cell counts
        print("\nPre-scanning cell counts per signal group...")
        cell_counts = count_cells_per_signal_group(signal_groups, storage_roots, feature_dir, maps_path)
        MIN_CELLS_FLOOR = 750_000
        raw_min = min(cell_counts.values())
        target_n_cells = max(raw_min, MIN_CELLS_FLOOR)
        max_n_cells = max(cell_counts.values())

        # Warn about signal groups that have fewer cells than the floor
        small_groups = {s: n for s, n in cell_counts.items() if n < target_n_cells}
        if small_groups:
            print(f"\n⚠️  {len(small_groups)} signal group(s) have fewer than {target_n_cells:,} cells (will use all available):")
            for s, n in sorted(small_groups.items(), key=lambda x: x[1]):
                print(f"    {s}: {n:,} cells")

        # Print manifest
        print(f"\nSignal group manifest (downsampling all to {target_n_cells:,} cells):")
        print(f"  {'Signal':<45} {'Exps':>5} {'Cells':>10} {'→ Downsampled':>15}")
        print(f"  {'-'*45} {'-'*5} {'-'*10} {'-'*15}")
        manifest_rows = []
        for signal in sorted(signal_groups.keys()):
            pairs = signal_groups[signal]
            n_cells = cell_counts[signal]
            print(f"  {signal:<45} {len(pairs):>5} {n_cells:>10,} → {target_n_cells:>12,}")
            manifest_rows.append({
                "signal": signal,
                "n_experiments": len(pairs),
                "n_cells_pooled": n_cells,
                "n_cells_downsampled": target_n_cells,
                "experiments": ",".join(e.split("_")[0] for e, c in pairs),
            })
        print(f"\n  Total: {n_signals} signal groups, {sum(cell_counts.values()):,} total cells → {target_n_cells:,} per group")

        # Save manifest
        manifest_df = pd.DataFrame(manifest_rows)
        manifest_df.to_csv(ds_output_dir / "downsampled_manifest.csv", index=False)

        if not args.slurm:
            # Local sequential mode
            print("\nRunning locally (sequential)...")
            for signal, pairs in signal_groups.items():
                sg_kwargs = dict(
                    signal=signal,
                    exp_channel_pairs=pairs,
                    output_dir=str(ds_output_dir),
                    target_n_cells=target_n_cells,
                    norm_method=args.norm_method,
                )
                if _cp_feature_dir_override:
                    sg_kwargs["feature_dir_override"] = _cp_feature_dir_override
                    sg_kwargs["sweep_thresholds"] = DEFAULT_SWEEP_THRESHOLDS_CP
                result = pca_sweep_pooled_signal(**sg_kwargs)
                print(f"  {result}")
            # Aggregate
            result = aggregate_channels(
                output_dir=str(ds_output_dir),
                norm_method=args.norm_method,
                per_unit_subdir="per_signal",
            )
            print(result)
            return

        # SLURM mode: one job per signal group
        jobs = []
        for signal, pairs in signal_groups.items():
            sig_safe = sanitize_signal_filename(signal)[:40]
            jobs.append({
                "name": f"pca_ds_{sig_safe}",
                "func": pca_sweep_pooled_signal,
                "kwargs": {
                    "signal": signal,
                    "exp_channel_pairs": pairs,
                    "output_dir": str(ds_output_dir),
                    "target_n_cells": target_n_cells,
                    "norm_method": args.norm_method,
                    **({"feature_dir_override": _cp_feature_dir_override} if _cp_feature_dir_override else {}),
                    **({"sweep_thresholds": DEFAULT_SWEEP_THRESHOLDS_CP} if _cp_feature_dir_override else {}),
                },
                "metadata": {"signal": signal, "n_experiments": len(pairs)},
            })

        # Downsampled jobs load many experiments per signal — need more time than per-channel
        ds_time = max(args.slurm_time, 30)
        slurm_params = {
            "timeout_min": ds_time,
            "mem": args.slurm_memory,
            "cpus_per_task": args.slurm_cpus,
            "slurm_partition": args.slurm_partition,
        }

        # Phase 1: Submit per-signal-group jobs
        def _on_ds_phase1_complete(submitted_jobs, experiment):
            """Callback after all signal-group jobs complete — submit aggregation."""
            print(f"\nAll signal-group jobs complete. Submitting aggregation SLURM job...")
            agg_jobs = [{
                "name": "pca_ds_aggregate",
                "func": aggregate_channels,
                "kwargs": {
                    "output_dir": str(ds_output_dir),
                    "norm_method": args.norm_method,
                    "per_unit_subdir": "per_signal",
                },
            }]
            agg_slurm_params = {
                "timeout_min": args.slurm_agg_time,
                "mem": args.slurm_agg_memory,
                "cpus_per_task": args.slurm_cpus,
                "slurm_partition": args.slurm_partition,
            }
            agg_result = submit_parallel_jobs(
                jobs_to_submit=agg_jobs,
                experiment="pca_ds_aggregation",
                slurm_params=agg_slurm_params,
                log_dir="pca_optimization",
                manifest_prefix="pca_ds_agg",
                wait_for_completion=True,
            )
            if agg_result.get("failed"):
                print(f"\nAggregation FAILED")
            else:
                print(f"\nAggregation complete")

        print(f"\nSubmitting {len(jobs)} signal-group SLURM jobs...")
        result = submit_parallel_jobs(
            jobs_to_submit=jobs,
            experiment="pca_ds_optimization",
            slurm_params=slurm_params,
            log_dir="pca_optimization",
            manifest_prefix="pca_ds_opt",
            wait_for_completion=True,
            post_completion_callback=_on_ds_phase1_complete,
        )

        if result.get("failed"):
            print(f"\nWarning: {len(result['failed'])} signal groups failed")
            for name in result["failed"]:
                print(f"  - {name}")

        return

    if args.slurm:
        from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

        # Discover experiments
        attr_config = load_attribution_config()
        slurm_storage_roots = get_storage_roots(attr_config)
        slurm_feature_dir = _cp_feature_dir_override or attr_config.get("feature_dir", "dino_features")

        if _cp_feature_dir_override:
            all_pairs = discover_cellprofiler_experiments(slurm_storage_roots)
        else:
            all_pairs = discover_dino_experiments(slurm_storage_roots, slurm_feature_dir)

        if not all_pairs:
            print("No experiment-channel pairs found!")
            return

        # Build signal map using robust channel resolution
        import contextlib
        import io

        maps_path = get_channel_maps_path()
        from ops_utils.data.feature_metadata import FeatureMetadata
        fm = FeatureMetadata(metadata_path=maps_path)

        print(f"\nPCA Optimization: {len(all_pairs)} channels to process")
        print(f"Output: {output_dir}")

        # Build per-channel jobs, skipping unmapped channels
        jobs = []
        skipped_unmapped = []
        for exp, ch in all_pairs:
            exp_short = exp.split("_")[0]
            with contextlib.redirect_stdout(io.StringIO()):
                resolved = resolve_channel_label(fm, exp, ch)
            sig = resolved["label"]
            if sig == "unknown" or sig.startswith("(unmapped:"):
                skipped_unmapped.append((exp, ch, sig))
                continue
            # Sanitize for SLURM job name (no spaces/special chars)
            sig_safe = sig.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")[:40]
            job_kwargs = {
                    "exp": exp,
                    "channel": ch,
                    "output_dir": str(output_dir),
                    "norm_method": args.norm_method,
            }
            if _cp_feature_dir_override:
                job_kwargs["feature_dir_override"] = _cp_feature_dir_override
                job_kwargs["sweep_thresholds"] = DEFAULT_SWEEP_THRESHOLDS_CP
            jobs.append({
                "name": f"pca_{sig_safe}_{exp_short}",
                "func": pca_sweep_single_experiment,
                "kwargs": job_kwargs,
                "metadata": {"signal": sig, "experiment": exp, "channel": ch},
            })

        if skipped_unmapped:
            print(f"\n⚠️  {len(skipped_unmapped)} experiment-channel pairs could not be mapped to a biological signal and will be SKIPPED:")
            for exp, ch, sig in skipped_unmapped:
                print(f"    {exp} / {ch}  →  {sig!r}")
            print()

        slurm_params = {
            "timeout_min": args.slurm_time,
            "mem": args.slurm_memory,
            "cpus_per_task": args.slurm_cpus,
            "slurm_partition": args.slurm_partition,
        }

        # Phase 1: Submit per-channel jobs
        def _on_phase1_complete(submitted_jobs, experiment):
            """Callback after all per-channel jobs complete — submit aggregation SLURM job."""
            print(f"\nAll per-channel jobs complete. Submitting aggregation SLURM job...")
            agg_jobs = [{
                "name": "pca_aggregate",
                "func": aggregate_channels,
                "kwargs": {
                    "output_dir": str(output_dir),
                    "norm_method": args.norm_method,
                },
            }]
            agg_slurm_params = {
                "timeout_min": args.slurm_agg_time,
                "mem": args.slurm_agg_memory,
                "cpus_per_task": args.slurm_cpus,
                "slurm_partition": args.slurm_partition,
            }
            agg_result = submit_parallel_jobs(
                jobs_to_submit=agg_jobs,
                experiment="pca_aggregation",
                slurm_params=agg_slurm_params,
                log_dir="pca_optimization",
                manifest_prefix="pca_agg",
                wait_for_completion=True,
            )
            if agg_result.get("failed"):
                print(f"\nAggregation FAILED")
            else:
                print(f"\nAggregation complete")

        result = submit_parallel_jobs(
            jobs_to_submit=jobs,
            experiment="pca_optimization",
            slurm_params=slurm_params,
            log_dir="pca_optimization",
            manifest_prefix="pca_opt",
            wait_for_completion=True,
            post_completion_callback=_on_phase1_complete,
        )

        if result.get("failed"):
            print(f"\nWarning: {len(result['failed'])} channels failed")
            for name in result["failed"]:
                print(f"  - {name}")

    else:
        # Local mode
        result = run_pca_optimization(
            output_dir=str(output_dir),
            sweep_thresholds=DEFAULT_SWEEP_THRESHOLDS_CP if _cp_feature_dir_override else None,
            norm_method=args.norm_method,
        )
        print(result)


if __name__ == "__main__":
    main()
