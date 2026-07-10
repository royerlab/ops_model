"""Phase 1: pooled PCA sweep per biological signal group.

``pca_sweep_pooled_signal`` is the picklable submitit worker fired once
per signal group. It pools cells across experiments sharing the same
biological signal, downsamples, optionally per-experiment z-scores,
fits PCA (with a passthrough subsample-fit / chunked-transform path
when n_cells > ``PCA_FIT_CAP``), sweeps variance thresholds, and writes
the canonical ``per_signal/{signal}_{guide,gene,cells}.h5ad`` outputs.

In-module dependencies on ``pca_optimization`` (constants and the
sweep-core helpers) are imported lazily inside the function body so
this module and its parent can re-import each other at load time
without a circular dependency.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd

from ops_utils.analysis.pca import fit_pca, n_pcs_for_threshold
from ops_utils.data.feature_discovery import (
    find_cell_h5ad_path,
    get_channel_maps_path,
    get_storage_roots,
    load_attribution_config,
    load_cell_h5ad,
    sanitize_signal_filename,
)


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
    agg_method: str = "mean",
    apply_iss_sidecar: bool = False,
    cell_paths: Optional[Dict[Tuple[str, str], str]] = None,
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
    from ops_model.post_process.combination.pca_optimization import (
        DEFAULT_SWEEP_THRESHOLDS,
        DUD_GUIDES,
        MIN_PCS,
        PCA_FIT_CAP,
    )
    from ops_model.post_process.combination.pca_optimization.sweep_core import (
        _init_sweep_logger,
        _run_threshold_sweep,
        _save_raw_outputs,
        _save_sweep_outputs,
    )

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

    def _resolve(exp_: str, ch_: str):
        """Explicit-path override (external mode) else standard discovery."""
        if cell_paths and (exp_, ch_) in cell_paths:
            _pp = Path(cell_paths[(exp_, ch_)])
            return _pp if _pp.exists() else None
        return find_cell_h5ad_path(exp_, ch_, storage_roots, feature_dir, maps_path)

    # --- Pass 1: Lightweight pre-scan for cell counts (no full matrix load) ---
    import h5py

    exp_cell_counts = {}  # (exp, ch) -> n_cells
    for exp, ch in exp_channel_pairs:
        cell_file = _resolve(exp, ch)
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
            path = _resolve(exp, ch)
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

        if apply_iss_sidecar:
            # Load via the ISS-drift-fix sidecar so obs[perturbation]/[sgRNA]
            # reflect the current 3-assembly linked_pheno_iss.csv calls, not
            # the stale frozen models/link_csvs/ snapshot. Orphans (cells the
            # ISS re-run dropped entirely) are dropped here so they don't
            # carry stale labels into PCA + aggregation. See
            # ops_model.data.iss_drift_fix for how the sidecars are built.
            from ops_model.features.anndata_utils import load_features_corrected
            cell_path = _resolve(exp, ch)
            adata = (
                load_features_corrected(cell_path, drop_orphans=True)
                if cell_path is not None else None
            )
        else:
            _cp = _resolve(exp, ch)
            adata = ad.read_h5ad(_cp) if _cp is not None else None
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
            agg_method=agg_method,
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
            k: result.get(k)
            for k in ("peak_act_t", "peak_dist_t", "peak_ebi_t")
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
        agg_method=agg_method,
    )

    elapsed = time.time() - t_start
    _logger.info(
        f"  Done: {signal} in {elapsed:.0f}s — {selected_n} PCs @ {selected_t:.0%}"
    )

    return f"SUCCESS: {signal} — {selected_n} PCs @ {selected_t:.0%}, {selected_r:.1%} active ({n_exps} exps, {n_cells}/{n_cells_pooled} cells)"
