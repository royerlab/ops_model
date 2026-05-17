"""OrganelleProfiler mode for pca_optimization.

Reads consolidated per-marker ``all_cells_*.h5ad`` files (output of
``organelle_profiler.feature_extraction.consolidate_all_cells``) and runs
the same PCA threshold sweep as the dino/CP path. Each h5ad is treated
as one signal group (cells are already pooled across experiments).

Two public functions:

* ``_discover_op_files(op_root)`` — list ``(viz_channel, path)`` pairs
  under ``op_root``, reading the canonical marker label from each file's
  ``obs.viz_channel``.
* ``pca_sweep_op_signal(...)`` — top-level submitit-picklable worker
  that fits PCA on one OP file and writes the standard ``per_signal/``
  outputs.

In-module helpers and constants from ``pca_optimization`` (``DEFAULT_SWEEP_THRESHOLDS``,
``DUD_GUIDES``, ``MIN_PCS``, ``PCA_FIT_CAP``, ``_init_sweep_logger``,
``_save_sweep_outputs``, ``_save_raw_outputs``, ``_run_threshold_sweep``)
are imported lazily inside ``pca_sweep_op_signal`` to avoid a circular
import.
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
    load_attribution_config,
    sanitize_signal_filename,
)


def _discover_op_files(
    op_root: str, paper_v1_path: Optional[str] = None,
) -> List[Tuple[str, Path]]:
    """List all ``all_cells_*.h5ad`` files under ``op_root``.

    Returns ``[(viz_channel, path), ...]`` where ``viz_channel`` is the
    canonical marker label read from each file's ``obs.viz_channel`` (first
    row). Falls back to deriving from the filename if the column is absent.

    ``paper_v1_path`` is accepted for symmetry with the dino/CP path but is
    not enforced here — OP h5ads are already constructed against a fixed
    cohort (see consolidate_all_cells's ``--paper-v1``).
    """
    op_root = Path(op_root)
    if not op_root.exists():
        raise FileNotFoundError(f"--op-root does not exist: {op_root}")
    files = sorted(op_root.glob("all_cells_*.h5ad"))
    if not files:
        raise FileNotFoundError(f"No all_cells_*.h5ad in {op_root}")
    pairs: List[Tuple[str, Path]] = []
    for path in files:
        # Pull viz_channel from obs (first row) for the canonical label
        try:
            backed = ad.read_h5ad(path, backed="r")
            viz = (
                str(backed.obs["viz_channel"].iloc[0])
                if "viz_channel" in backed.obs.columns
                else path.stem.replace("all_cells_phase", "Phase").replace(
                    "all_cells_fluor_", ""
                )
            )
            backed.file.close()
        except Exception:
            viz = path.stem.replace("all_cells_phase", "Phase").replace(
                "all_cells_fluor_", ""
            )
        pairs.append((viz, path))
    return pairs


def pca_sweep_op_signal(
    signal: str,
    op_path: str,
    output_dir: str,
    target_n_cells: int,
    norm_method: str = "ntc",
    sweep_thresholds: Optional[List[float]] = None,
    random_seed: int = 42,
    distance: str = "cosine",
    fixed_threshold: Optional[float] = None,
    preserve_batch: bool = False,
    no_pca: bool = False,
    zscore_per_experiment: bool = False,
    exclude_dud_guides: bool = True,
    agg_method: str = "mean",
) -> str:
    """OrganelleProfiler variant of :func:`pca_sweep_pooled_signal`.

    Reads a single ``all_cells_*.h5ad`` (cells already pooled across all
    paper_v1 experiments for one viz_channel), normalizes obs, optionally
    z-scores per experiment, fits PCA, sweeps thresholds, and writes the same
    ``per_signal/`` outputs the rest of the pipeline expects.

    Top-level (picklable) so submitit can dispatch one job per OP file.
    """
    # Lazy import: pca_optimization re-imports pca_sweep_op_signal at the
    # top of its module, so importing back from it at top-level here would
    # cycle. By the time pca_sweep_op_signal() is actually called, the
    # parent module is fully loaded.
    from ops_model.post_process.combination.pca_optimization import (
        DEFAULT_SWEEP_THRESHOLDS,
        DUD_GUIDES,
        MIN_PCS,
        PCA_FIT_CAP,
        _init_sweep_logger,
        _run_threshold_sweep,
        _save_raw_outputs,
        _save_sweep_outputs,
    )

    _logger = _init_sweep_logger()
    t_start = time.time()
    output_dir = Path(output_dir)
    op_path = Path(op_path)
    rng = np.random.RandomState(random_seed)

    thresholds = (
        DEFAULT_SWEEP_THRESHOLDS
        if fixed_threshold is not None
        else (sweep_thresholds or DEFAULT_SWEEP_THRESHOLDS)
    )

    _logger.info(f"OP signal {signal}: reading {op_path.name}")
    adata = ad.read_h5ad(op_path)
    _logger.info(f"  Loaded {adata.n_obs:,} cells × {adata.n_vars} features")

    # --- Normalize obs to match downstream expectations ---
    obs = adata.obs

    # Strip experiment to short name (ops0146 not ops0146_20260402)
    if "experiment" in obs.columns:
        obs["experiment"] = obs["experiment"].astype(str).str.split("_").str[0]
    else:
        _logger.warning("  No 'experiment' column — using a single batch")
        obs["experiment"] = "op_pooled"

    # Ensure perturbation column. OP h5ads store NTC cells with gene="NTC"
    # but gene_name="" (blank), and KO cells get both columns set identically.
    # MUST prefer `gene` over `gene_name` so NTC controls survive — otherwise
    # scoring has no null reference and every metric returns 0.
    if "perturbation" not in obs.columns:
        for fallback in ("gene", "gene_name"):
            if fallback in obs.columns:
                obs["perturbation"] = obs[fallback].astype(str)
                break
    if "perturbation" not in obs.columns:
        return f"FAILED: {signal} — no perturbation / gene / gene_name column"

    # Drop rows with blank perturbation (defensive: NTC cells with empty
    # gene_name in some files, mislabeled rows, etc.) so the null reference
    # isn't polluted.
    blank_mask = obs["perturbation"].isin({"", "nan", "None"})
    if blank_mask.any():
        n_blank = int(blank_mask.sum())
        adata = adata[~blank_mask.values].copy()
        obs = adata.obs
        _logger.warning(
            f"  Dropped {n_blank:,} cells with blank perturbation label"
        )

    if "label_str" not in obs.columns:
        obs["label_str"] = obs["perturbation"]

    if exclude_dud_guides and "sgRNA" in obs.columns:
        n_before = adata.n_obs
        keep = ~obs["sgRNA"].isin(DUD_GUIDES)
        if (~keep).any():
            adata = adata[keep].copy()
            obs = adata.obs
            _logger.info(
                f"  Dropped {n_before - adata.n_obs:,} dud-guide cells "
                f"({n_before:,} → {adata.n_obs:,})"
            )

    # --- Subsample to target_n_cells (proportional across experiments) ---
    n_total = adata.n_obs
    if n_total > target_n_cells:
        exps = obs["experiment"].values
        exp_counts = pd.Series(exps).value_counts()
        kept_idx = []
        for exp_id, cnt in exp_counts.items():
            mask_idx = np.where(exps == exp_id)[0]
            fraction = cnt / n_total
            n_take = max(1, int(round(fraction * target_n_cells)))
            n_take = min(n_take, len(mask_idx))
            picked = rng.choice(mask_idx, n_take, replace=False)
            kept_idx.append(picked)
        kept_idx = np.sort(np.concatenate(kept_idx))
        adata = adata[kept_idx].copy()
        obs = adata.obs
        _logger.info(
            f"  Subsampled {n_total:,} → {adata.n_obs:,} cells "
            f"(proportional across {len(exp_counts)} experiments)"
        )

    n_cells_pooled = int(n_total)
    n_cells = adata.n_obs
    loaded_exps = sorted(obs["experiment"].unique().tolist())
    n_exps = len(loaded_exps)

    X_raw = np.asarray(adata.X, dtype=np.float32)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    feature_names = list(adata.var_names)
    n_feats = X_raw.shape[1]
    del adata

    score_cols = [c for c in ["sgRNA", "perturbation", "label_str"] if c in obs.columns]
    obs_df_full = obs[score_cols + ["experiment"]].reset_index(drop=True)
    obs_df = obs_df_full[score_cols].copy()

    # Per-experiment z-score before PCA
    if zscore_per_experiment:
        from sklearn.preprocessing import StandardScaler

        experiments = obs_df_full["experiment"].values
        for exp_id in np.unique(experiments):
            mask = experiments == exp_id
            X_raw[mask] = StandardScaler().fit_transform(X_raw[mask])
        _logger.info(
            f"  Per-experiment z-score applied ({len(np.unique(experiments))} experiments)"
        )

    # --- No-PCA early exit ---
    if no_pca:
        output_suffix = "_nopca" + ("_batch" if preserve_batch else "")
        file_prefix = sanitize_signal_filename(signal)
        _save_raw_outputs(
            X_raw=X_raw,
            obs_df=obs_df_full,
            feature_names=feature_names,
            signal=signal,
            uns_metadata={
                "experiment": ",".join(loaded_exps),
                "channel": signal,
                "n_cells": int(n_cells),
                "n_cells_pooled": n_cells_pooled,
                "n_experiments": n_exps,
                "n_features_raw": int(n_feats),
                "source": str(op_path),
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
        return (
            f"SUCCESS: {signal} — no PCA, {n_feats} raw features "
            f"({n_exps} exps, {n_cells}/{n_cells_pooled} cells) in {elapsed:.0f}s"
        )

    # --- Fit PCA (single shot — OP files are at most a few M cells after capping) ---
    t_pca = time.time()
    if n_cells > PCA_FIT_CAP:
        fit_idx = rng.choice(n_cells, PCA_FIT_CAP, replace=False)
        fit_idx.sort()
        _logger.info(
            f"  Fitting PCA on {PCA_FIT_CAP:,}/{n_cells:,} subsampled cells..."
        )
        _, cumvar, pca_model = fit_pca(X_raw[fit_idx])
        del fit_idx
        chunk_size = 2_000_000
        X_pcs_chunks = []
        for i in range(0, n_cells, chunk_size):
            chunk = np.asarray(X_raw[i : i + chunk_size], dtype=np.float64)
            chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
            X_pcs_chunks.append(pca_model.transform(chunk).astype(np.float32))
        X_pcs = np.vstack(X_pcs_chunks)
        del X_pcs_chunks
    else:
        _logger.info(f"  Fitting PCA on {n_cells:,} × {n_feats} matrix...")
        X_pcs, cumvar, pca_model = fit_pca(X_raw)

    _logger.info(
        f"  PCA done in {time.time() - t_pca:.0f}s — {X_pcs.shape[1]} components"
    )
    pca_components = pca_model.components_.copy()
    pca_mean = pca_model.mean_.copy() if getattr(pca_model, "mean_", None) is not None else None
    del X_raw, pca_model

    # --- Threshold sweep ---
    if preserve_batch:
        attr_config = load_attribution_config()
        variance_cutoff = attr_config.get("pca", {}).get("variance_cutoff", 0.80)
        selected_t = variance_cutoff
        selected_n = n_pcs_for_threshold(cumvar, variance_cutoff)
        selected_r, selected_a = 0.0, 0.0
        sweep_rows: List[Dict] = []
        metric_peaks: Dict = {}
        sweep_peak_t = variance_cutoff
    else:
        t_sweep = time.time()
        _logger.info(f"  Threshold sweep ({len(thresholds)} thresholds)...")
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
        sweep_peak_t = consensus_t
        selected_t, selected_n, selected_r, selected_a = (
            consensus_t, consensus_n, consensus_r, consensus_a,
        )
        if fixed_threshold is not None:
            fixed_n = n_pcs_for_threshold(cumvar, fixed_threshold)
            fixed_row = next(
                (r for r in sweep_rows if r["threshold"] == fixed_threshold), None
            )
            selected_r = fixed_row["activity"] if fixed_row else consensus_r
            selected_a = fixed_row["auc"] if fixed_row else consensus_a
            _logger.info(
                f"  Fixed threshold override: {fixed_threshold:.0%} → {fixed_n} PCs "
                f"(consensus was {consensus_t:.0%})"
            )
            selected_t, selected_n = fixed_threshold, fixed_n

    # --- Save ---
    file_prefix = sanitize_signal_filename(signal)
    exps_str = ", ".join(loaded_exps[:5]) + (f" +{len(loaded_exps)-5} more" if len(loaded_exps) > 5 else "")
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
            "experiment": ",".join(loaded_exps),
            "channel": signal,
            "n_cells": int(n_cells),
            "n_cells_pooled": int(n_cells_pooled),
            "n_experiments": int(n_exps),
            "n_features_raw": int(n_feats),
            "pca_components": pca_components[:selected_n].tolist(),
            "pca_feature_names": feature_names,
            "pca_mean": pca_mean.tolist() if pca_mean is not None else None,
            "source": str(op_path),
        },
        output_dir=output_dir,
        subdir="per_signal",
        file_prefix=file_prefix,
        suptitle=f"{signal} ({n_exps} exps: {exps_str}) — {n_cells:,}/{n_cells_pooled:,} cells, {n_feats} features",
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
    return (
        f"SUCCESS: {signal} — {selected_n} PCs @ {selected_t:.0%}, "
        f"{selected_r:.1%} active ({n_exps} exps, {n_cells}/{n_cells_pooled} cells) "
        f"in {elapsed:.0f}s"
    )
