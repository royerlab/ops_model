import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import anndata as ad

from .config_handler import CombinationConfig
from ops_model.features.anndata_utils import concatenate_experiments_comprehensive

# Initialize logger
logger = logging.getLogger(__name__)


@contextmanager
def temp_log_level(level, module_names):
    """Temporarily set the logging level for a list of modules."""
    original_levels = {}
    for name in module_names:
        module_logger = logging.getLogger(name)
        original_levels[name] = module_logger.level
        module_logger.setLevel(level)
    try:
        yield
    finally:
        for name, original_level in original_levels.items():
            logging.getLogger(name).setLevel(original_level)


class ComprehensiveCombiner:
    """
    Orchestrates the comprehensive combination of experiments by wrapping
    the core logic in `anndata_utils.concatenate_experiments_comprehensive`.
    """

    def __init__(self, config: CombinationConfig):
        """
        Initializes the combiner with a validated configuration.
        """
        self.config = config

    def combine(self) -> Tuple[ad.AnnData, ad.AnnData]:
        """
        Execute the full combination pipeline by calling the utility function.
        """
        logger.info("Starting comprehensive combination process...")

        # Prepare arguments for the utility function from the config
        # Note: Currently, the same embedding config is used for both guide and gene levels
        # in concatenate_experiments_comprehensive. Using gene_level config since that's
        # typically more important for downstream analysis.
        embedding_config = self.config.embeddings.get("gene_level")
        if embedding_config is None:
            # Fallback to guide_level if gene_level missing
            embedding_config = self.config.embeddings.get("guide_level")
        if embedding_config is None:
            # Final fallback if both missing (shouldn't happen)
            from .config_handler import EmbeddingConfig

            embedding_config = EmbeddingConfig()

        # Convert experiments_channels from Dict[str, List[str]] to List[Tuple[str, str]]
        experiments_channels_list = [
            (exp, ch)
            for exp, channels in (self.config.experiments_channels or {}).items()
            for ch in channels
        ]

        if not experiments_channels_list:
            raise ValueError("No experiment/channel pairs were found to combine.")

        # Temporarily silence verbose logs from underlying libraries
        with temp_log_level(logging.WARNING, ["scanpy", "umap"]):
            adata_guide, adata_gene = concatenate_experiments_comprehensive(
                experiments_channels=experiments_channels_list,
                feature_type=self.config.feature_type,
                base_dir=self.config.base_dir,
                feature_dir=self.config.feature_dir,
                recompute_embeddings=embedding_config.compute_embeddings,
                n_pca_components=embedding_config.n_pca_components,
                n_umap_neighbors=embedding_config.n_neighbors,
                compute_pca=embedding_config.pca,
                compute_umap=embedding_config.umap,
                compute_phate=embedding_config.phate,
                normalize_on_pooling=self.config.normalization.get(
                    "normalize_on_pooling", True
                ),
                normalize_on_controls=self.config.normalization.get(
                    "normalize_on_controls", False
                ),
                subsample_controls=self.config.control_subsampling.get(
                    "enabled", False
                ),
                control_gene=self.config.control_subsampling.get("control_gene", "NTC"),
                control_group_size=self.config.control_subsampling.get("group_size", 4),
                random_seed=self.config.control_subsampling.get("random_seed"),
                fit_on_aggregated_controls=self.config.fitted_embeddings.get(
                    "enabled", False
                ),
                use_pca_for_umap=self.config.fitted_embeddings.get(
                    "use_pca_for_umap", True
                ),
                leiden_resolutions=(
                    self.config.leiden_clustering.get("resolutions")
                    if self.config.leiden_clustering.get("enabled", False)
                    else None
                ),
            )

        logger.info("Comprehensive combination process complete.")
        return adata_guide, adata_gene


# =============================================================================
# PCA-optimized combiner — constants
# =============================================================================

_SWEEP_THRESHOLDS_DINO = [
    0.60,
    0.70,
    0.74,
    0.76,
    0.78,
    0.80,
    0.82,
    0.84,
    0.88,
    0.90,
    0.95,
]
_SWEEP_THRESHOLDS_CP = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
_MIN_PCS = 10  # skip thresholds that yield fewer PCs than this
_MIN_CELLS_FLOOR = 750_000  # floor for auto target_n_cells
_PCA_FIT_CAP = 5_000_000  # cells used to fit PCA axes; larger datasets use passthrough (fit subsample, transform all)


# =============================================================================
# PCA-optimized combiner — module-level helpers (must be picklable for SLURM)
# =============================================================================


def _prepare_cells_for_scoring(adata: "ad.AnnData") -> "ad.AnnData":
    """Strip obs to copairs-required columns and cast X to float64."""
    import numpy as np

    if "n_cells" not in adata.obs.columns:
        adata.obs["n_cells"] = 1
    keep = [c for c in ["sgRNA", "perturbation", "n_cells"] if c in adata.obs.columns]
    adata.obs = adata.obs[keep].copy()
    for col in adata.obs.columns:
        if adata.obs[col].dtype.name == "category":
            adata.obs[col] = adata.obs[col].astype(str)
    adata.X = adata.X.astype("float64")
    return adata


def _sweep_pca_thresholds(
    X_pcs: "np.ndarray",
    cumvar: "np.ndarray",
    obs_df: "pd.DataFrame",
    thresholds: List[float],
    norm_method: str,
    _logger,
) -> Optional[Tuple[List[Dict], float, int]]:
    """Sweep variance thresholds, score AUC at each, return best result.

    Returns (sweep_rows, best_threshold, best_n_pcs) or None if no valid threshold found.
    The normalization here is temporary (for scoring only) and discarded after selection.
    """
    import numpy as np
    import pandas as pd
    import anndata as ad
    from ops_utils.analysis.pca import n_pcs_for_threshold
    from ops_utils.analysis.map_scores import (
        compute_auc_score,
        phenotypic_activity_assesment,
    )
    from ops_model.features.anndata_utils import (
        aggregate_to_level,
        normalize_guide_adata,
    )

    best_auc_t, best_auc_r, best_auc_a, best_auc_n = None, -1.0, -1.0, 0
    sweep_rows = []

    for threshold in thresholds:
        n_pcs = n_pcs_for_threshold(cumvar, threshold)
        X_slice = X_pcs[:, :n_pcs].astype(np.float32)
        pc_names = [f"PC{j}" for j in range(n_pcs)]

        adata_tmp = ad.AnnData(
            X=X_slice, obs=obs_df.copy(), var=pd.DataFrame(index=pc_names)
        )
        guide_tmp = aggregate_to_level(
            adata_tmp, level="guide", method="mean", preserve_batch_info=False
        )
        del adata_tmp
        guide_tmp.X = guide_tmp.X.astype(np.float32)

        guide_norm = normalize_guide_adata(guide_tmp.copy(), norm_method)
        guide_norm.X = guide_norm.X.astype(np.float32)
        guide_norm = _prepare_cells_for_scoring(guide_norm)

        try:
            activity_map, active_ratio = (
                phenotypic_activity_assesment(  # distance default="cosine"
                    guide_norm,
                    plot_results=False,
                    null_size=100_000,
                )
            )
            auc = compute_auc_score(activity_map)
        except Exception as e:
            _logger.warning(f"  Scoring failed at {threshold:.0%}: {e}")
            active_ratio, auc = 0.0, 0.0
        del guide_tmp, guide_norm

        row = {
            "threshold": threshold,
            "n_pcs": n_pcs,
            "activity": active_ratio,
            "auc": auc,
        }
        sweep_rows.append(row)

        if n_pcs < _MIN_PCS:
            _logger.info(
                f"  {threshold:.0%}: {n_pcs} PCs (< {_MIN_PCS}) — {active_ratio:.1%}, AUC={auc:.4f} [skipped]"
            )
            continue

        _logger.info(
            f"  {threshold:.0%}: {n_pcs} PCs — {active_ratio:.1%}, AUC={auc:.4f}"
        )
        if auc > best_auc_a or (auc == best_auc_a and active_ratio > best_auc_r):
            best_auc_t, best_auc_r, best_auc_a, best_auc_n = (
                threshold,
                active_ratio,
                auc,
                n_pcs,
            )

    if best_auc_t is None:
        return None

    _logger.info(
        f"  Best (AUC): {best_auc_t:.0%} ({best_auc_n} PCs) → {best_auc_r:.1%}, AUC={best_auc_a:.4f}"
    )
    return sweep_rows, best_auc_t, best_auc_n


def _process_signal_group(
    signal: str,
    exp_channel_pairs: List[Tuple[str, str]],
    output_dir: str,
    base_dir: str,
    feature_dir: str,
    pca_config: Dict[str, Any],
    downsampling_config: Dict[str, Any],
    norm_method: str,
    random_seed: int = 42,
    preserve_batch: bool = False,
    no_pca: bool = False,
    cell_filter: Optional[Any] = None,
) -> str:
    """Phase 1: pool cells for one biological signal, fit PCA, select n_pcs, save h5ads.

    Top-level function (not a method) so submitit can pickle it for SLURM dispatch.

    Saves to output_dir/per_channel/:
      - {signal_prefix}_guide.h5ad  (aggregated at selected n_pcs, un-normalized)
      - {signal_prefix}_gene.h5ad
      - {signal_prefix}_sweep.csv   (one row per threshold, or one row for fixed mode)
    """
    import logging
    import time
    import warnings
    import numpy as np
    import pandas as pd
    import anndata as ad
    from pathlib import Path

    warnings.filterwarnings("ignore", category=FutureWarning)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger("copairs").setLevel(logging.WARNING)
    _logger = logging.getLogger(__name__)
    t_start = time.time()

    from ops_utils.data.feature_discovery import (
        find_cell_h5ad_path,
        load_cell_h5ad,
        get_channel_maps_path,
        sanitize_signal_filename,
    )
    from ops_utils.analysis.pca import fit_pca, n_pcs_for_threshold
    from ops_utils.analysis.normalization import zscore_normalize
    from ops_model.features.anndata_utils import aggregate_to_level

    output_dir = Path(output_dir)
    per_channel_dir = output_dir / "per_channel"
    per_channel_dir.mkdir(parents=True, exist_ok=True)

    maps_path = get_channel_maps_path()
    storage_roots = [Path(base_dir)]

    _logger.info(f"Processing signal: {signal} ({len(exp_channel_pairs)} experiments)")

    # --- Pre-scan cell counts (lightweight h5py read) ---
    import h5py

    exp_cell_counts = {}
    for exp, ch in exp_channel_pairs:
        cell_file = find_cell_h5ad_path(exp, ch, storage_roots, feature_dir, maps_path)
        if cell_file is not None:
            try:
                with h5py.File(cell_file, "r") as f:
                    exp_cell_counts[(exp, ch)] = f["X"].shape[0]
            except Exception:
                pass

    n_cells_pooled = sum(exp_cell_counts.values())
    if n_cells_pooled == 0:
        return f"FAILED: {signal} — no cell data found for any experiment"

    # --- Resolve downsampling target ---
    downsampling_enabled = downsampling_config.get("enabled", False)
    if downsampling_enabled:
        raw_target = downsampling_config.get("target_n_cells", "auto")
        actual_target = (
            int(raw_target)
            if raw_target != "auto"
            else max(min(exp_cell_counts.values()), _MIN_CELLS_FLOOR)
        )
        actual_target = min(actual_target, n_cells_pooled)
    else:
        actual_target = n_cells_pooled

    _logger.info(f"  Total cells: {n_cells_pooled:,}, target: {actual_target:,}")

    # --- Load cells, optionally downsample, optionally z-score per experiment ---
    rng = np.random.RandomState(random_seed)
    all_blocks = []
    n_vars_expected = None
    normalize_before_pca = pca_config.get("normalize_before_pca", False)
    inferred_cell_type = None  # read from first successfully loaded h5ad

    for exp, ch in exp_channel_pairs:
        if exp_cell_counts.get((exp, ch), 0) == 0:
            continue

        adata = load_cell_h5ad(exp, ch, storage_roots, feature_dir, maps_path)
        if adata is None:
            continue

        if cell_filter is not None:
            n_before = adata.n_obs
            adata = cell_filter(adata)
            if adata.n_obs == 0:
                _logger.warning(f"  {exp}/{ch}: all cells removed by filter, skipping")
                continue
            _logger.info(f"  {exp}/{ch}: {n_before} → {adata.n_obs} cells after filtering")

        if inferred_cell_type is None:
            inferred_cell_type = adata.uns.get("cell_type", "cell")

        if n_vars_expected is None:
            n_vars_expected = adata.n_vars
        elif adata.n_vars != n_vars_expected:
            _logger.info(
                f"  {exp}/{ch}: {adata.n_vars} features (vs {n_vars_expected}), will use shared features on concat"
            )

        # Ensure label_str exists
        if "label_str" not in adata.obs.columns and "perturbation" in adata.obs.columns:
            adata.obs["label_str"] = adata.obs["perturbation"]

        # Proportional subsample
        if downsampling_enabled and actual_target < n_cells_pooled:
            fraction = exp_cell_counts[(exp, ch)] / n_cells_pooled
            n_take = max(1, int(round(fraction * actual_target)))
            n_take = min(n_take, adata.n_obs)
            if n_take < adata.n_obs:
                idx = rng.choice(adata.n_obs, n_take, replace=False)
                idx.sort()
                adata = adata[idx].copy()

        keep_cols = [
            c for c in ["sgRNA", "perturbation", "label_str"] if c in adata.obs.columns
        ]
        obs = adata.obs[keep_cols].copy()
        obs["experiment"] = exp.split("_")[0]

        X_block = np.asarray(adata.X, dtype=np.float32)
        feature_cols = list(adata.var_names)

        # Per-experiment z-score before pooling (uses ops_utils backend)
        if normalize_before_pca:
            df_block = pd.DataFrame(X_block, columns=feature_cols)
            df_norm = zscore_normalize(
                df_block, feature_cols=feature_cols, method="global"
            )
            X_block = df_norm[feature_cols].values.astype(np.float32)

        all_blocks.append(ad.AnnData(X=X_block, obs=obs, var=adata.var.copy()))
        _logger.info(
            f"  {exp.split('_')[0]}/{ch}: {exp_cell_counts[(exp, ch)]:,} → {len(obs):,} cells"
        )
        del adata, X_block

    if not all_blocks:
        return f"FAILED: {signal} — failed to load cell data for all experiments"

    # Concatenate blocks; inner join keeps only shared features across experiments.
    # index_unique ensures obs_names are unique across experiments (avoids AnnData warning).
    adata_cells = ad.concat(all_blocks, join="inner", index_unique="-")
    del all_blocks
    if np.isnan(adata_cells.X).any():
        adata_cells.X = np.nan_to_num(adata_cells.X, nan=0.0)

    n_cells = adata_cells.n_obs
    n_feats = adata_cells.n_vars
    feature_names = list(adata_cells.var_names)
    _logger.info(f"  Pooled: {n_cells:,} cells, {n_feats} features")

    # Separate obs for scoring (no experiment column — copairs doesn't handle extra string cols)
    score_cols = [
        c
        for c in ["sgRNA", "perturbation", "label_str"]
        if c in adata_cells.obs.columns
    ]
    obs_for_scoring = adata_cells.obs[score_cols].copy()
    obs_full = adata_cells.obs[[c for c in adata_cells.obs.columns]].copy()

    X_raw = np.asarray(adata_cells.X, dtype=np.float32)
    del adata_cells

    if no_pca:
        _logger.info(
            f"  no_pca=True: skipping PCA, using {n_feats} raw features directly"
        )
        X_reduced = X_raw
        del X_raw
        pc_names = feature_names
        n_pcs = n_feats
        pca_components = None
        peak_t = None
        sweep_rows = []
    else:
        # --- Fit PCA on subsample, transform all cells in chunks ---
        t_pca = time.time()
        n_total = X_raw.shape[0]

        if n_total > _PCA_FIT_CAP:
            fit_idx = rng.choice(n_total, _PCA_FIT_CAP, replace=False)
            fit_idx.sort()
            _logger.info(
                f"  Fitting PCA on {_PCA_FIT_CAP:,}/{n_total:,} subsampled cells..."
            )
            _, cumvar, pca_model = fit_pca(X_raw[fit_idx])
            del fit_idx
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
        del X_raw, pca_model

        # --- Select n_pcs: sweep or fixed ---
        selection = pca_config.get("selection", "sweep")
        if preserve_batch:
            selection = "fixed"  # skip sweep when preserving batch info

        if selection == "fixed":
            cutoff = float(pca_config.get("variance_cutoff", 0.80))
            n_pcs = n_pcs_for_threshold(cumvar, cutoff)
            peak_t = cutoff
            sweep_rows = [
                {"threshold": cutoff, "n_pcs": n_pcs, "activity": None, "auc": None}
            ]
            _logger.info(f"  Fixed cutoff {cutoff:.0%}: {n_pcs} PCs")
        else:
            thresholds = pca_config.get("_sweep_thresholds", _SWEEP_THRESHOLDS_DINO)
            result = _sweep_pca_thresholds(
                X_pcs, cumvar, obs_for_scoring, thresholds, norm_method, _logger
            )
            if result is None:
                return f"FAILED: {signal} — no valid threshold found (all thresholds yield < {_MIN_PCS} PCs)"
            sweep_rows, peak_t, n_pcs = result

        # --- Build AnnData at selected n_pcs and aggregate ---
        X_reduced = X_pcs[:, :n_pcs].astype(np.float32)
        pc_names = [f"{signal}_PC{j}" for j in range(n_pcs)]
        del X_pcs

    # Compute n_experiments per sgRNA from obs_full before dropping the experiment column.
    # Injected directly into g.obs after guide aggregation (aggregate_to_level at guide level
    # does not carry arbitrary cell-obs columns through, so cell-level injection is lost).
    sgRNA_to_n_exp = obs_full.groupby("sgRNA")["experiment"].nunique()

    # Drop experiment column before aggregation unless preserving batch info
    # (copairs is incompatible with extra string cols, but preserve_batch skips copairs scoring)
    if preserve_batch:
        obs_for_agg = obs_full
    else:
        obs_for_agg = obs_full[[c for c in obs_full.columns if c != "experiment"]]
    adata_reduced = ad.AnnData(
        X=X_reduced,
        obs=obs_for_agg,
        var=pd.DataFrame(index=pc_names),
    )
    del X_reduced

    g = aggregate_to_level(
        adata_reduced, level="guide", method="mean", preserve_batch_info=preserve_batch
    )
    e = aggregate_to_level(
        adata_reduced, level="gene", method="mean", preserve_batch_info=preserve_batch
    )
    del adata_reduced

    g.X = g.X.astype(np.float32)
    e.X = e.X.astype(np.float32)

    # Inject n_experiments into guide obs so Phase 2's guide→gene aggregation picks it up
    # via aggregate_to_level's max() path (lines 804-808 of anndata_utils.py).
    g.obs["n_experiments"] = g.obs["sgRNA"].map(sgRNA_to_n_exp).fillna(1).astype(int)
    g.uns["aggregation_method"] = "mean"
    e.uns["aggregation_method"] = "mean"

    uns = {
        "signal": signal,
        "pca_applied": not no_pca,
        "n_cells": int(n_cells),
        "n_cells_pooled": int(n_cells_pooled),
        "n_features_raw": int(n_feats),
        "pca_feature_names": feature_names,
        "experiments": ",".join(exp.split("_")[0] for exp, _ in exp_channel_pairs),
        "exp_cell_counts": {
            exp.split("_")[0]: int(cnt) for (exp, ch), cnt in exp_cell_counts.items()
        },
        "channels": list({ch for _, ch in exp_channel_pairs}),
        "cell_type": inferred_cell_type or "cell",
        "embedding_type": feature_dir,
    }
    if no_pca:
        uns["n_features"] = int(n_feats)
    else:
        uns.update(
            {
                "pca_threshold": float(peak_t),
                "n_pcs": int(n_pcs),
                "explained_variance": (
                    float(cumvar[n_pcs - 1]) if n_pcs <= len(cumvar) else 1.0
                ),
                "pca_components": pca_components[:n_pcs].tolist(),
            }
        )
    for adata in [g, e]:
        adata.uns.update(uns)

    from ops_model.post_process.anndata_processing.anndata_validator import (
        AnndataValidator,
    )

    _validator = AnndataValidator()
    for _adata, _level in [(g, "guide"), (e, "gene")]:
        _report = _validator.validate(_adata, level=_level, strict=False)
        if not _report.is_valid:
            _logger.warning(
                f"  {signal} {_level}-level AnnData failed validation:\n{_report.summary()}"
            )
        else:
            _logger.info(
                f"  {signal} {_level}-level AnnData passed validation ({_report.get_warning_count()} warnings)"
            )

    file_prefix = sanitize_signal_filename(signal)
    output_suffix = ("_nopca" if no_pca else "") + ("_batch" if preserve_batch else "")
    g.write_h5ad(per_channel_dir / f"{file_prefix}{output_suffix}_guide.h5ad")
    e.write_h5ad(per_channel_dir / f"{file_prefix}{output_suffix}_gene.h5ad")
    if sweep_rows:
        pd.DataFrame(sweep_rows).to_csv(
            per_channel_dir / f"{file_prefix}{output_suffix}_sweep.csv", index=False
        )

    elapsed = time.time() - t_start
    if no_pca:
        _logger.info(
            f"  Done: {signal} in {elapsed:.0f}s — {n_feats} raw features (no PCA)"
        )
        return f"SUCCESS: {signal} — {n_feats} raw features, no PCA ({n_cells:,}/{n_cells_pooled:,} cells)"
    else:
        _logger.info(f"  Done: {signal} in {elapsed:.0f}s — {n_pcs} PCs @ {peak_t:.0%}")
        return f"SUCCESS: {signal} — {n_pcs} PCs @ {peak_t:.0%} ({n_cells:,}/{n_cells_pooled:,} cells)"


# =============================================================================
# PcaOptimizationCombiner
# =============================================================================


class PcaOptimizationCombiner:
    """Config-driven combiner that follows the pca_optimization two-phase pipeline.

    Phase 1 — per biological signal group:
        Pool cells from all experiments → optionally downsample → optionally z-score →
        fit PCA → select n_pcs (sweep or fixed cutoff) → save per-signal guide/gene h5ads.

    Phase 2 — aggregation:
        Load per-signal guide h5ads → hconcat → NTC normalize → aggregate to gene →
        optionally compute UMAP/PHATE → return (adata_guide, adata_gene).
    """

    def __init__(self, config: CombinationConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Experiment resolution
    # ------------------------------------------------------------------

    def _resolve_experiments(self) -> List[Tuple[str, str]]:
        """Return final (experiment, channel) list from config.

        If auto_discover=True: scan base_dir via feature_discovery functions.
        If auto_discover=False: flatten experiments_channels from config.
        Applies reporters filter to the result in both cases.
        """
        from ops_utils.data.feature_discovery import (
            discover_dino_experiments,
            discover_cellprofiler_experiments,
        )

        if self.config.auto_discover:
            if self.config.experiments_channels:
                logger.warning(
                    "auto_discover=True and experiments_channels are both set; "
                    "ignoring experiments_channels and using auto_discover."
                )
            storage_roots = [Path(self.config.base_dir)]
            if self.config.feature_type == "cellprofiler":
                pairs = discover_cellprofiler_experiments(storage_roots)
            else:
                pairs = discover_dino_experiments(
                    storage_roots, self.config.feature_dir
                )
            logger.info(
                f"Auto-discovered {len(pairs)} experiment-channel pairs from {self.config.base_dir}"
            )
        else:
            if not self.config.experiments_channels:
                raise ValueError(
                    "experiments_channels must be set when auto_discover=False"
                )
            pairs = [
                (exp, ch)
                for exp, channels in self.config.experiments_channels.items()
                for ch in channels
            ]

        if self.config.reporters:
            pairs = [(exp, ch) for exp, ch in pairs if ch in self.config.reporters]

        if not pairs:
            raise ValueError("No experiment-channel pairs remain after filtering.")

        return pairs

    def _group_by_signal(
        self, pairs: List[Tuple[str, str]]
    ) -> Dict[str, List[Tuple[str, str]]]:
        """Group (experiment, channel) pairs by biological signal label."""
        from ops_utils.data.feature_discovery import (
            get_channel_maps_path,
            build_signal_groups,
        )
        from ops_utils.data.feature_metadata import FeatureMetadata

        maps_path = get_channel_maps_path()
        fm = FeatureMetadata(metadata_path=maps_path)
        return build_signal_groups(pairs, fm)

    # ------------------------------------------------------------------
    # Auto target_n_cells computation
    # ------------------------------------------------------------------

    def _compute_auto_target(
        self, signal_groups: Dict[str, List[Tuple[str, str]]]
    ) -> int:
        """Scan cell counts per signal group and return max(min_count, floor)."""
        import h5py
        from ops_utils.data.feature_discovery import (
            find_cell_h5ad_path,
            get_channel_maps_path,
        )

        maps_path = get_channel_maps_path()
        storage_roots = [Path(self.config.base_dir)]
        feature_dir = self.config.feature_dir

        logger.info("Pre-scanning cell counts to compute auto target_n_cells...")
        min_count = float("inf")
        for signal, pairs in signal_groups.items():
            group_total = 0
            for exp, ch in pairs:
                cell_file = find_cell_h5ad_path(
                    exp, ch, storage_roots, feature_dir, maps_path
                )
                if cell_file is not None:
                    try:
                        with h5py.File(cell_file, "r") as f:
                            group_total += f["X"].shape[0]
                    except Exception:
                        pass
            if group_total > 0:
                min_count = min(min_count, group_total)
                logger.info(f"  {signal}: {group_total:,} cells")

        if min_count == float("inf"):
            min_count = _MIN_CELLS_FLOOR

        target = max(int(min_count), _MIN_CELLS_FLOOR)
        logger.info(f"  Auto target_n_cells = {target:,}")
        return target

    # ------------------------------------------------------------------
    # PCA config builder
    # ------------------------------------------------------------------

    def _build_pca_config(self) -> Dict[str, Any]:
        """Return pca config dict with default sweep thresholds injected."""
        pca_cfg = dict(self.config.pca)
        if pca_cfg.get("selection", "sweep") == "sweep":
            if self.config.feature_type == "cellprofiler":
                pca_cfg["_sweep_thresholds"] = _SWEEP_THRESHOLDS_CP
            else:
                pca_cfg["_sweep_thresholds"] = _SWEEP_THRESHOLDS_DINO
        return pca_cfg

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------

    def _run_phase1_local(
        self,
        signal_groups: Dict[str, List[Tuple[str, str]]],
        output_dir: Path,
        downsampling_config: Dict[str, Any],
        cell_filter: Optional[Any] = None,
    ) -> None:
        """Run Phase 1 sequentially in the calling process."""
        pca_cfg = self._build_pca_config()
        norm_method = self.config.normalization.get("method", "ntc")

        for signal, pairs in signal_groups.items():
            result = _process_signal_group(
                signal=signal,
                exp_channel_pairs=pairs,
                output_dir=str(output_dir),
                base_dir=self.config.base_dir,
                feature_dir=self.config.feature_dir,
                pca_config=pca_cfg,
                downsampling_config=downsampling_config,
                norm_method=norm_method,
                preserve_batch=self.config.preserve_batch,
                no_pca=self.config.no_pca,
                cell_filter=cell_filter,
            )
            logger.info(f"  {result}")

    def _run_phase1_slurm(
        self,
        signal_groups: Dict[str, List[Tuple[str, str]]],
        output_dir: Path,
        downsampling_config: Dict[str, Any],
        cell_filter: Optional[Any] = None,
    ) -> None:
        """Submit Phase 1 as parallel SLURM jobs and wait for completion."""
        from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
        from ops_utils.data.feature_discovery import sanitize_signal_filename

        pca_cfg = self._build_pca_config()
        norm_method = self.config.normalization.get("method", "ntc")
        slurm = self.config.slurm

        slurm_params = {
            "timeout_min": slurm.get("time_minutes", 10),
            "mem": slurm.get("memory", "100GB"),
            "cpus_per_task": slurm.get("cpus", 16),
            "slurm_partition": slurm.get("partition", "cpu,gpu"),
        }

        jobs = []
        for signal, pairs in signal_groups.items():
            sig_safe = sanitize_signal_filename(signal)[:40]
            jobs.append(
                {
                    "name": f"pca_opt_{sig_safe}",
                    "func": _process_signal_group,
                    "kwargs": {
                        "signal": signal,
                        "exp_channel_pairs": pairs,
                        "output_dir": str(output_dir),
                        "base_dir": self.config.base_dir,
                        "feature_dir": self.config.feature_dir,
                        "pca_config": pca_cfg,
                        "downsampling_config": downsampling_config,
                        "norm_method": norm_method,
                        "preserve_batch": self.config.preserve_batch,
                        "no_pca": self.config.no_pca,
                        "cell_filter": cell_filter,
                    },
                }
            )

        logger.info(f"Submitting {len(jobs)} SLURM Phase 1 jobs...")
        result = submit_parallel_jobs(
            jobs_to_submit=jobs,
            experiment="pca_optimization",
            slurm_params=slurm_params,
            log_dir="pca_optimization",
            manifest_prefix="pca_opt",
            wait_for_completion=True,
        )

        failed = result.get("failed", [])
        if failed:
            logger.warning(f"{len(failed)} Phase 1 job(s) failed: {failed}")

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------

    def _compute_embeddings(self, adata: "ad.AnnData", embedding_config) -> None:
        """Compute UMAP and/or PHATE directly (not via scanpy) and store in obsm."""
        import numpy as np

        X = adata.X.astype(np.float32)
        n_obs = adata.n_obs

        if embedding_config.umap:
            try:
                from umap import UMAP

                nn = min(embedding_config.n_neighbors, n_obs - 1)
                if nn >= 2:
                    logger.info(f"Computing UMAP ({n_obs} obs, n_neighbors={nn})...")
                    coords = UMAP(
                        n_components=2, n_neighbors=nn, random_state=42
                    ).fit_transform(X)
                    adata.obsm["X_umap"] = coords.astype(np.float32)
                    logger.info("  UMAP complete.")
                else:
                    logger.warning(f"Skipping UMAP: too few observations ({n_obs})")
            except ImportError:
                logger.warning("UMAP skipped: install umap-learn")
            except Exception as e:
                logger.warning(f"UMAP failed: {e}")

        if embedding_config.phate:
            try:
                import phate

                knn = min(15 if n_obs > 2000 else 10, n_obs - 1)
                if knn >= 2:
                    logger.info(f"Computing PHATE ({n_obs} obs, knn={knn})...")
                    coords = phate.PHATE(
                        n_components=2,
                        knn=knn,
                        decay=15,
                        t="auto",
                        n_jobs=-1,
                        random_state=42,
                        verbose=0,
                    ).fit_transform(X)
                    adata.obsm["X_phate"] = coords.astype(np.float32)
                    logger.info("  PHATE complete.")
                else:
                    logger.warning(f"Skipping PHATE: too few observations ({n_obs})")
            except ImportError:
                logger.warning("PHATE skipped: install phate")
            except Exception as e:
                logger.warning(f"PHATE failed: {e}")

    def _run_phase2(self, output_dir: Path) -> Tuple["ad.AnnData", "ad.AnnData"]:
        """Load per-signal guide h5ads, hconcat, NTC normalize, aggregate, embed."""
        import numpy as np
        from ops_model.features.anndata_utils import (
            hconcat_by_perturbation,
            normalize_guide_adata,
            aggregate_to_level,
        )

        per_channel_dir = output_dir / "per_channel"
        guide_files = sorted(per_channel_dir.glob("*_guide.h5ad"))

        if not guide_files:
            raise FileNotFoundError(
                f"No per-signal guide h5ads found in {per_channel_dir}. "
                "Ensure Phase 1 completed successfully before running Phase 2."
            )

        logger.info(f"Phase 2: loading {len(guide_files)} per-signal guide files...")
        guide_blocks = []
        for gf in guide_files:
            g = ad.read_h5ad(gf)
            sig = g.uns.get("signal", gf.stem.replace("_guide", ""))
            if sig == "unknown" or sig.startswith("(unmapped:"):
                logger.warning(f"  Skipping {gf.name}: unmapped signal ({sig!r})")
                continue
            guide_blocks.append(g)
            logger.info(f"  {sig}: {g.n_obs} guides × {g.n_vars} PCs")

        if not guide_blocks:
            raise ValueError("No valid per-signal guide blocks loaded for Phase 2.")

        logger.info("Concatenating per-signal blocks horizontally...")
        cell_type = guide_blocks[0].uns.get("cell_type", "cell")
        embedding_type = guide_blocks[0].uns.get(
            "embedding_type", self.config.feature_type
        )

        # Build pca_optimized_metadata before consuming guide_blocks
        biological_groups = {}
        feature_slices = {}
        offset = 0
        for g in guide_blocks:
            sig = g.uns["signal"]
            n_feat = g.n_vars
            biological_groups[sig] = {
                "biological_signal": sig,
                "aggregation_type": "pooled_pca",
                "experiments": g.uns.get("experiments", "").split(","),
                "channels": g.uns.get("channels", []),
                "n_cells_per_experiment": g.uns.get("exp_cell_counts", {}),
                "n_cells_total": g.uns.get("n_cells_pooled", 0),
                "n_cells_used": g.uns.get("n_cells", 0),
                "n_features_raw": g.uns.get("n_features_raw", 0),
                "n_pcs": g.uns.get("n_pcs", n_feat),
                "pca_threshold": g.uns.get("pca_threshold", None),
                "explained_variance": g.uns.get("explained_variance", None),
                "feature_range": [offset, offset + n_feat],
                "n_features": n_feat,
            }
            feature_slices[sig] = {
                "start": offset,
                "end": offset + n_feat,
                "n_features": n_feat,
            }
            offset += n_feat

        pca_optimized_metadata = {
            "strategy": "pca_optimized",
            "feature_type": self.config.feature_type,
            "aggregation_level": "guide",
            "n_biological_signals": len(biological_groups),
            "biological_groups": biological_groups,
            "feature_slices": feature_slices,
        }

        adata_guide = hconcat_by_perturbation(guide_blocks, "guide")
        del guide_blocks

        norm_method = self.config.normalization.get("method", "ntc")
        logger.info(f"NTC normalizing at guide level (method={norm_method!r})...")
        adata_guide = normalize_guide_adata(adata_guide, norm_method)
        adata_guide.X = adata_guide.X.astype(np.float32)

        logger.info("Aggregating guide → gene...")
        adata_gene = aggregate_to_level(
            adata_guide,
            "gene",
            preserve_batch_info=False,
            subsample_controls=False,
        )
        logger.info(f"  Guide: {adata_guide.n_obs} obs × {adata_guide.n_vars} features")
        logger.info(f"  Gene:  {adata_gene.n_obs} obs × {adata_gene.n_vars} features")

        # Stamp required metadata fields (inferred from per-signal intermediates)
        gene_metadata = {**pca_optimized_metadata, "aggregation_level": "gene"}
        for adata, meta in [
            (adata_guide, pca_optimized_metadata),
            (adata_gene, gene_metadata),
        ]:
            adata.uns["cell_type"] = cell_type
            adata.uns["embedding_type"] = embedding_type
            adata.uns["comprehensive_metadata"] = meta
            adata.uns["aggregation_method"] = "mean"

        # Validate before returning
        from ops_model.post_process.anndata_processing.anndata_validator import (
            AnndataValidator,
        )

        _validator = AnndataValidator()
        for _adata, _level in [(adata_guide, "guide"), (adata_gene, "gene")]:
            _report = _validator.validate(_adata, level=_level, strict=False)
            if not _report.is_valid:
                logger.warning(
                    f"Phase 2 {_level}-level AnnData failed validation:\n{_report.summary()}"
                )
            else:
                logger.info(
                    f"Phase 2 {_level}-level AnnData passed validation ({_report.get_warning_count()} warnings)"
                )

        # Embeddings on gene level
        embedding_config = self.config.embeddings.get(
            "gene_level"
        ) or self.config.embeddings.get("guide_level")
        if embedding_config is not None and embedding_config.compute_embeddings:
            self._compute_embeddings(adata_gene, embedding_config)

        return adata_guide, adata_gene

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def combine(self) -> Tuple["ad.AnnData", "ad.AnnData"]:
        """Run the full two-phase pipeline and return (adata_guide, adata_gene)."""
        output_dir = Path(self.config.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Resolve experiments and group by biological signal
        pairs = self._resolve_experiments()
        signal_groups = self._group_by_signal(pairs)

        # 2. Resolve downsampling target (auto pre-scan happens here if needed)
        downsampling_config = dict(self.config.downsampling)
        if (
            downsampling_config.get("enabled", False)
            and downsampling_config.get("target_n_cells", "auto") == "auto"
        ):
            downsampling_config["target_n_cells"] = self._compute_auto_target(
                signal_groups
            )

        # 3. Phase 1: PCA sweep per signal group
        from .cell_filters import build_cell_filter

        cell_filter = build_cell_filter(self.config.cell_filters)

        slurm_enabled = self.config.slurm.get("enabled", False)
        if slurm_enabled:
            self._run_phase1_slurm(signal_groups, output_dir, downsampling_config, cell_filter)
        else:
            self._run_phase1_local(signal_groups, output_dir, downsampling_config, cell_filter)

        # 4. Phase 2: aggregate, normalize, embed
        if self.config.preserve_batch or self.config.no_pca:
            logger.info(
                f"Skipping Phase 2 aggregation (preserve_batch={self.config.preserve_batch}, no_pca={self.config.no_pca})."
            )
            return None, None

        logger.info("Starting Phase 2 aggregation...")
        adata_guide, adata_gene = self._run_phase2(output_dir)

        logger.info("PCA-optimized combination complete.")
        return adata_guide, adata_gene
