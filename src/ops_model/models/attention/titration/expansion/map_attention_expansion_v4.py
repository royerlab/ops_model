"""v4 attention expansion sweep — selection + bin scoring via v3's _score_bin.

Reads PCA-projected per-experiment h5ads (95-D, built by fit_v4_pca.py)
with all 3 head ranks attached, selects cells per bin, calls v3's
``_score_bin`` for NTC z-score + 4 mAP metrics.

The 14 direction-types × K bins:
- Sweep A — per-head top/low: {ebi, chad, geneko} × {top, low} × K_BINS  (6 × K)
- Sweep A — head-agnostic random: 1 × K_BINS                              (1 × K)
- Sweep B — EBI ∩ geneKO bottom-P% intersection-removed random: 6 × K_BINS (6 × K)
- Baseline — all_cells: 1 (K-agnostic)

K_BINS = [100, 500, 1k, 5k, 10k, 15k, 20k, 50k, 100k]
P-percentile thresholds (Sweep B) = {10, 20, 25, 30, 40, 50}

Outputs
-------
    /hpc/projects/icd.fast.ops/models/alex_lin_attention/v4/expansion_v1/
        bin_results/<bin_id>.json     one shard per bin
        bin_results.csv               aggregated table
        plots/{sweep_a,sweep_b}_*.{png,pdf,svg}

Usage
-----
    # One bin locally (debug):
    uv run python map_attention_expansion_v4.py --bin sweep_a:ebi:top:5000

    # SLURM-parallel + aggregate + plot:
    uv run python map_attention_expansion_v4.py --submit-slurm
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import anndata as ad
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

V4 = Path("/hpc/projects/icd.fast.ops/models/alex_lin_attention/v4")
import os as _os
_EXP_SUBDIR = _os.environ.get("V4_EXPANSION_SUBDIR", "expansion_v1")
EXPANSION = V4 / _EXP_SUBDIR
PCA_DIR = EXPANSION / "per_experiment_v4_pca"
GUIDE_DIR = EXPANSION / "bin_guide_means"   # Phase 1 output: small guide_a.h5ad per bin
BIN_DIR = EXPANSION / "bin_results"
PLOT_DIR = EXPANSION / "plots"

K_BINS = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 12500, 15000, 20000, 50000, 100000]
PERCENTILES = [10, 20, 25, 30, 40, 50]
HEADS = ("ebi", "chad", "geneko", "set_accuracy", "set_accuracy_ebi")


# ---------------------------------------------------------------------------
# v3 imports — direct reuse of the proven units
# ---------------------------------------------------------------------------
# map_attention_decay lives in the sibling decay/ subdir
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "decay"))
from map_attention_decay import _score_bin  # noqa: E402  (reused below for EBI+CHAD+dist)

CHAD_YAML = Path("/hpc/projects/icd.fast.ops/configs/gene_clusters/chad_positive_controls_v4.yml")


# ---------------------------------------------------------------------------
# Cell selection
# ---------------------------------------------------------------------------

def _open_concat(exps: Optional[List[str]] = None) -> ad.AnnData:
    """Read every per-exp PCA h5ad and concat into one in-memory AnnData."""
    paths = sorted(PCA_DIR.glob("*.h5ad"))
    if exps is not None:
        wanted = set(exps)
        paths = [p for p in paths if p.stem in wanted]
    if not paths:
        raise FileNotFoundError(f"no PCA h5ads in {PCA_DIR}")
    adatas = [ad.read_h5ad(p) for p in paths]
    return ad.concat(adatas, axis=0, join="outer", merge="first", index_unique=None)


def _uncovered_gene_cells(obs: pd.DataFrame, rank_col: str,
                            group_col: str = "perturbation") -> np.ndarray:
    """Positions of cells whose gene is NOT in the head's coverage (i.e. no
    cell anywhere in obs has a rank for that gene under this head).

    ``group_col`` is which column defines "gene identity" for the head. For
    set_accuracy_ebi we pass Alex's authoritative gene name column
    (``set_accuracy_ebi_gene``) instead of ``perturbation`` because Alex's
    sgRNA→gene lookup differs from the h5ad's for some cells (NTC reassignments,
    barcode-mapping drift).
    """
    has = obs[rank_col].notna()
    covered_genes = set(obs.loc[has, group_col].astype(str).unique())
    keep = (
        (obs["perturbation"] != "NTC")
        & (~obs[group_col].astype(str).isin(covered_genes))
    )
    return np.flatnonzero(keep.to_numpy())


def _select_top_per_gene(obs: pd.DataFrame, rank_col: str, K: int,
                          group_col: str = "perturbation",
                          include_uncovered: bool = True) -> np.ndarray:
    """Top-K cells per gene for head-covered genes.

    When ``include_uncovered`` is True (default for ebi/chad/geneko heads),
    ALL cells of genes outside the head's panel are also included so the
    curve saturates to the all-cells baseline at high K. For set_accuracy
    heads, we set it False so K=10 really means "10 cells per gene, period."
    """
    obs = obs.reset_index(drop=True)
    mask = obs[rank_col].notna() & (obs[f"{rank_col}_type"] == "top")
    if mask.any():
        sub = obs.loc[mask].sort_values(rank_col)
        top_idx = sub.groupby(group_col).head(K).index.to_numpy(dtype=np.int64)
    else:
        top_idx = np.array([], dtype=np.int64)
    if include_uncovered:
        uncovered_idx = _uncovered_gene_cells(obs, rank_col, group_col=group_col)
        return np.unique(np.concatenate([top_idx, uncovered_idx]))
    return np.unique(top_idx)


def _select_bottom_per_gene(obs: pd.DataFrame, rank_col: str, K: int,
                             group_col: str = "perturbation",
                             include_uncovered: bool = True) -> np.ndarray:
    """Bottom-K per gene (largest rank values). See ``_select_top_per_gene``
    for ``include_uncovered`` semantics.
    """
    obs = obs.reset_index(drop=True)
    mask = obs[rank_col].notna()
    if mask.any():
        sub = obs.loc[mask].sort_values(rank_col, ascending=False)
        bot_idx = sub.groupby(group_col).head(K).index.to_numpy(dtype=np.int64)
    else:
        bot_idx = np.array([], dtype=np.int64)
    if include_uncovered:
        uncovered_idx = _uncovered_gene_cells(obs, rank_col, group_col=group_col)
        return np.unique(np.concatenate([bot_idx, uncovered_idx]))
    return np.unique(bot_idx)


def _select_random_per_gene(obs: pd.DataFrame, K: int, seed: int = 0) -> np.ndarray:
    """For each REAL gene (non-NTC): random K cells. Returns positional ints."""
    obs = obs.reset_index(drop=True)
    real_pos = np.flatnonzero((obs["perturbation"] != "NTC").to_numpy())
    if real_pos.size == 0:
        return np.array([], dtype=np.int64)
    pert = obs["perturbation"].to_numpy()
    rng = np.random.default_rng(seed)
    parts = []
    # Group positional indices by perturbation
    real_pert = pert[real_pos]
    for g in np.unique(real_pert):
        positions = real_pos[real_pert == g]
        n = min(K, positions.size)
        if n == 0:
            continue
        parts.append(rng.choice(positions, n, replace=False))
    return np.concatenate(parts).astype(np.int64) if parts else np.array([], dtype=np.int64)


def _bottom_p_set_positions(obs: pd.DataFrame, rank_col: str, P: int) -> set:
    """Per gene: positions of cells in the bottom-P% by rank."""
    obs = obs.reset_index(drop=True)
    mask_has_rank = obs[rank_col].notna()
    if not mask_has_rank.any():
        return set()
    max_per_gene = obs.loc[mask_has_rank].groupby("perturbation")[rank_col].transform("max")
    rank_vals = obs.loc[mask_has_rank, rank_col]
    threshold = max_per_gene * (1.0 - P / 100.0)
    bottom_pos = rank_vals.index[rank_vals > threshold]
    return set(int(p) for p in bottom_pos)


def _select_intersection_removed(obs: pd.DataFrame, P: int, K: int,
                                   seed: int = 0) -> np.ndarray:
    """Random K per gene with cells in (EBI bottom-P%) ∩ (geneKO bottom-P%) dropped."""
    obs = obs.reset_index(drop=True)
    drop = _bottom_p_set_positions(obs, "ebi_rank",    P) & \
           _bottom_p_set_positions(obs, "geneko_rank", P)
    drop_arr = np.fromiter(drop, dtype=np.int64) if drop else np.array([], dtype=np.int64)
    keep_mask = (obs["perturbation"] != "NTC").to_numpy()
    if drop_arr.size:
        keep_mask[drop_arr] = False
    real_pos = np.flatnonzero(keep_mask)
    if real_pos.size == 0:
        return np.array([], dtype=np.int64)
    pert = obs["perturbation"].to_numpy()
    real_pert = pert[real_pos]
    rng = np.random.default_rng(seed)
    parts = []
    for g in np.unique(real_pert):
        positions = real_pos[real_pert == g]
        n = min(K, positions.size)
        if n == 0:
            continue
        parts.append(rng.choice(positions, n, replace=False))
    return np.concatenate(parts).astype(np.int64) if parts else np.array([], dtype=np.int64)


# ---------------------------------------------------------------------------
# Bin enumeration
# ---------------------------------------------------------------------------

def enumerate_bins() -> List[str]:
    bins = ["all_cells"]
    for head in HEADS:
        for direction in ("top", "low"):
            for K in K_BINS:
                bins.append(f"sweep_a:{head}:{direction}:{K}")
    for K in K_BINS:
        bins.append(f"sweep_a:random:{K}")
    for P in PERCENTILES:
        for K in K_BINS:
            bins.append(f"sweep_b:p{P}:{K}")
    return bins


# ---------------------------------------------------------------------------
# Phase 1: per-bin streaming cell selection + guide aggregation
# ---------------------------------------------------------------------------

def _bin_id_to_meta(bin_id: str) -> dict:
    parts = bin_id.split(":")
    if parts[0] == "all_cells":
        return {"sweep": "baseline", "direction": "all_cells",
                "head": None, "K": None}
    if parts[0] == "sweep_a" and parts[1] == "random":
        return {"sweep": "A", "direction": "random",
                "head": None, "K": int(parts[2])}
    if parts[0] == "sweep_a":
        return {"sweep": "A", "direction": parts[2],
                "head": parts[1], "K": int(parts[3])}
    if parts[0] == "sweep_b":
        return {"sweep": "B", "direction": "intersection_removed",
                "percentile": int(parts[1].lstrip("p")), "head": None,
                "K": int(parts[2])}
    raise ValueError(f"bad bin_id: {bin_id}")


def _select_for_bin(obs: pd.DataFrame, bin_id: str, seed: int = 0) -> np.ndarray:
    """Apply bin-specific selection to one experiment's obs. Returns positions
    in this obs (0..N-1). NTC always included on top."""
    obs = obs.reset_index(drop=True)
    meta = _bin_id_to_meta(bin_id)
    if meta["direction"] == "all_cells":
        return np.arange(len(obs), dtype=np.int64)
    if meta["direction"] == "random":
        sel = _select_random_per_gene(obs, meta["K"], seed=seed)
    elif meta["direction"] in ("top", "low"):
        rank_col = f"{meta['head']}_rank"
        # For set_accuracy heads, group by Alex's authoritative gene name
        # (attached alongside the rank), not the h5ad's perturbation. Alex's
        # sgRNA→gene lookup differs from the h5ad's for some cells (NTC
        # reassignments, barcode-mapping drift).
        #
        # include_uncovered semantics per head:
        # - set_accuracy (geneKO): Alex covers ~all 1000 real genes → uncovered
        #   pool is only ~1M unranked cells that would dilute low-K. Off.
        # - set_accuracy_ebi: Alex covers 311 genes (EBI panel). To keep the
        #   Distinctiveness mAP measured over all 1001 genes (comparable to
        #   the geneKO row), include the 690 non-EBI genes' full cells so
        #   their guide means populate at baseline. On.
        if meta["head"] == "set_accuracy_ebi":
            group_col = "set_accuracy_ebi_gene"
            include_uncovered = True
        elif meta["head"] == "set_accuracy":
            group_col = "set_accuracy_gene"
            include_uncovered = False
        else:
            group_col = "perturbation"
            include_uncovered = True
        if meta["direction"] == "top":
            sel = _select_top_per_gene(obs, rank_col, meta["K"],
                                        group_col=group_col,
                                        include_uncovered=include_uncovered)
        else:
            sel = _select_bottom_per_gene(obs, rank_col, meta["K"],
                                           group_col=group_col,
                                           include_uncovered=include_uncovered)
    elif meta["direction"] == "intersection_removed":
        sel = _select_intersection_removed(obs, P=meta["percentile"],
                                             K=meta["K"], seed=seed)
    else:
        raise ValueError(f"unknown direction: {meta['direction']}")
    return sel.astype(np.int64)


def build_bin_guide_means(bin_id: str, guide_dir: Path = GUIDE_DIR) -> Optional[Path]:
    """Phase 1: two-pass build.

      Pass A — build a global obs table (across all 77 experiments) with the
      head rank cols + an __exp_idx + __cell_pos. Cheap (no X loaded).
      Bin selection is computed GLOBALLY here so K is K-per-gene-total, not
      K-per-(gene, experiment).

      Pass B — group selected rows by experiment, stream X loads per-exp,
      aggregate sums+counts per (gene, sgRNA). Peak mem ~one experiment.
    """
    guide_dir.mkdir(parents=True, exist_ok=True)
    out_path = guide_dir / f"{bin_id.replace(':', '_')}.h5ad"

    paths = sorted(PCA_DIR.glob("*.h5ad"))
    if not paths:
        raise FileNotFoundError(f"no per-exp PCA h5ads in {PCA_DIR}")

    # ---- Pass A: global obs table ----
    obs_cols = ["perturbation", "sgRNA",
                "ebi_rank", "chad_rank", "geneko_rank",
                "ebi_rank_type", "chad_rank_type", "geneko_rank_type",
                "set_accuracy_rank", "set_accuracy_rank_type",
                "set_accuracy_gene",
                "set_accuracy_ebi_rank", "set_accuracy_ebi_rank_type",
                "set_accuracy_ebi_gene"]
    obs_blocks = []
    for exp_idx, p in enumerate(paths):
        a = ad.read_h5ad(p, backed="r")
        ob = a.obs[obs_cols].copy()
        ob["__exp_idx"]  = exp_idx
        ob["__cell_pos"] = np.arange(len(ob), dtype=np.int64)
        obs_blocks.append(ob)
        a.file.close()
    all_obs = pd.concat(obs_blocks, ignore_index=True)
    del obs_blocks
    logger.info("[%s] global obs table: %d cells", bin_id, len(all_obs))

    # ---- Global selection ----
    sel_pos = _select_for_bin(all_obs, bin_id)

    # Subsample NTC to ntc_per_exp cells per experiment (v3 default: 1500).
    # Without this cap the NTC reference dwarfs the real-gene sgRNA means
    # (~25 cells/sgRNA at K=100), making NTC look pathologically stable and
    # inflating activity / consistency mAPs to ~1.0.
    NTC_PER_EXP = 1500
    ntc_pos_all = np.flatnonzero((all_obs["perturbation"] == "NTC").to_numpy())
    if ntc_pos_all.size > 0:
        rng = np.random.default_rng(seed=hash(bin_id) & 0xFFFF_FFFF)
        ntc_exp_idx = all_obs["__exp_idx"].to_numpy()[ntc_pos_all]
        keep_ntc = []
        for ei in np.unique(ntc_exp_idx):
            pool = ntc_pos_all[ntc_exp_idx == ei]
            if pool.size > NTC_PER_EXP:
                pool = rng.choice(pool, NTC_PER_EXP, replace=False)
            keep_ntc.append(pool)
        ntc_pos = np.concatenate(keep_ntc)
    else:
        ntc_pos = ntc_pos_all
    logger.info("[%s] NTC subsample: kept %d/%d cells (cap=%d per experiment)",
                bin_id, ntc_pos.size, ntc_pos_all.size, NTC_PER_EXP)

    final_pos = np.unique(np.concatenate([
        np.asarray(sel_pos, dtype=np.int64),
        np.asarray(ntc_pos, dtype=np.int64),
    ]))
    if final_pos.size == 0:
        logger.warning("[%s] no cells selected globally", bin_id)
        return None
    selected_obs = all_obs.iloc[final_pos]
    logger.info("[%s] selected %d cells globally  (incl %d NTC)",
                bin_id, len(selected_obs), int(ntc_pos.size))

    # ---- Pass B: per-experiment X loads + aggregate ----
    sums: dict = {}
    counts: dict = {}
    for exp_idx, grp in selected_obs.groupby("__exp_idx"):
        p = paths[exp_idx]
        a = ad.read_h5ad(p)
        pos = grp["__cell_pos"].to_numpy(dtype=np.int64)
        # h5py/anndata accepts sorted fancy index; do that to avoid the slow path
        order = np.argsort(pos)
        sorted_pos = pos[order]
        X_sorted = np.asarray(a.X[sorted_pos], dtype=np.float64)
        # Unsort back to grp order
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        X = X_sorted[inv]
        pert = grp["perturbation"].to_numpy()
        sg   = grp["sgRNA"].astype(str).to_numpy()
        for u_pert, u_sg, vec in zip(pert, sg, X):
            key = (str(u_pert), str(u_sg))
            if key in sums:
                sums[key] += vec
                counts[key] += 1
            else:
                sums[key] = vec.copy()
                counts[key] = 1
        del a, X, X_sorted, pert, sg, pos
    if not sums:
        logger.warning("[%s] no cells aggregated", bin_id)
        return None

    # Build guide-level matrix from weighted means
    keys = list(sums.keys())
    X_guide = np.stack([sums[k] / counts[k] for k in keys]).astype(np.float32)
    obs_df = pd.DataFrame({
        "perturbation": [k[0] for k in keys],
        "sgRNA":        [k[1] for k in keys],
        "n_cells":      [counts[k] for k in keys],
    })
    var_df = pd.DataFrame(index=[f"PC{i}" for i in range(X_guide.shape[1])])
    guide_a = ad.AnnData(X=X_guide, obs=obs_df, var=var_df)
    guide_a.uns["bin_id"] = bin_id
    guide_a.uns["meta"] = _bin_id_to_meta(bin_id)

    tmp = out_path.with_suffix(".h5ad.tmp")
    guide_a.write_h5ad(tmp)
    tmp.replace(out_path)
    n_ntc = int((guide_a.obs["perturbation"] == "NTC").sum())
    logger.info("[%s] guide_a: %d guides (%d NTC) → %s",
                bin_id, guide_a.n_obs, n_ntc, out_path.name)
    return out_path


# ---------------------------------------------------------------------------
# Phase 2: per-(bin, metric) mAP from guide_a
# ---------------------------------------------------------------------------

METRICS = ("ebi_consistency", "chad_consistency",
           "distinctiveness", "phenotypic_activity")


def score_one_metric(bin_id: str, metric: str,
                       guide_dir: Path = GUIDE_DIR,
                       bin_dir: Path = BIN_DIR,
                       null_size: int = 10_000) -> Optional[dict]:
    """Phase 2: read the bin's guide_a.h5ad, NTC z-score, compute ONE mAP metric."""
    from ops_model.features.anndata_utils import (
        aggregate_to_level, normalize_guide_adata,
    )
    bin_dir.mkdir(parents=True, exist_ok=True)
    out_path = bin_dir / f"{bin_id.replace(':', '_')}__{metric}.json"

    guide_path = guide_dir / f"{bin_id.replace(':', '_')}.h5ad"
    if not guide_path.exists():
        raise FileNotFoundError(f"missing guide_a for {bin_id}")
    guide_a = ad.read_h5ad(guide_path)
    meta = _bin_id_to_meta(bin_id)
    n_ntc = int((guide_a.obs["perturbation"].astype(str) == "NTC").sum())
    if n_ntc < 2:
        raise RuntimeError(f"[{bin_id}] need >=2 NTC guides, got {n_ntc}")
    guide_a = normalize_guide_adata(guide_a, norm_method="ntc")
    n_cells_total = int(guide_a.obs["n_cells"].sum())
    n_genes = int(guide_a.obs.loc[guide_a.obs["perturbation"] != "NTC",
                                    "perturbation"].nunique())

    # The 4 phenotypic_* functions return (map_df, ratio). v3's pca_optimization
    # canonical CSV reports BOTH ``<metric>_map_mean`` (= map_df['mean_average_precision'].mean())
    # AND ``<metric>_ratio`` (= fraction passing corrected_p threshold). My earlier
    # code used the 2nd return value (ratio) — that's NOT the "mAP" people quote
    # in v3's titration violins. Now compute and store BOTH.
    map_mean = float("nan")
    ratio = float("nan")
    if metric == "phenotypic_activity":
        from ops_utils.analysis.map_scores import phenotypic_activity_assesment
        map_df, ratio = phenotypic_activity_assesment(
            guide_a, plot_results=False, null_size=null_size,
        )
    elif metric == "distinctiveness":
        from ops_utils.analysis.map_scores import phenotypic_distinctivness
        map_df, ratio = phenotypic_distinctivness(
            guide_a, plot_results=False, null_size=null_size,
        )
    elif metric in ("ebi_consistency", "chad_consistency"):
        # Both need guide→gene aggregation
        gene_a = aggregate_to_level(guide_a, "gene", method="mean",
                                      preserve_batch_info=False)
        X = gene_a.X
        nan_mask = np.isnan(X).any(axis=1)
        if hasattr(nan_mask, "A1"):
            nan_mask = nan_mask.A1
        if nan_mask.any():
            gene_a = gene_a[~nan_mask].copy()
        if metric == "ebi_consistency":
            from ops_utils.analysis.map_scores import phenotypic_consistency_ebi
            map_df, ratio = phenotypic_consistency_ebi(
                gene_a, plot_results=False, null_size=null_size,
                cache_similarity=False,
            )
        else:
            from ops_utils.analysis.map_scores import (
                phenotypic_consistency_manual_annotation,
            )
            map_df, ratio = phenotypic_consistency_manual_annotation(
                gene_a, plot_results=False, null_size=null_size,
                cache_similarity=False, annotation_path=str(CHAD_YAML),
            )
    else:
        raise ValueError(f"unknown metric: {metric}")

    if map_df is not None and len(map_df) and "mean_average_precision" in map_df.columns:
        map_mean = float(map_df["mean_average_precision"].astype(float).mean())
    ratio = float(ratio) if ratio is not None else float("nan")

    row = {**meta, "metric": metric,
           "value": map_mean,                          # canonical mAP (mean per pert)
           "map_mean": map_mean, "ratio": ratio,        # both stored
           "n_guides": int(guide_a.n_obs), "n_genes": n_genes,
           "n_cells": n_cells_total, "bin_id": bin_id}
    out_path.write_text(json.dumps(row, default=float))
    logger.info("[%s/%s] map_mean=%.4f  ratio=%.4f  (n_guides=%d  n_cells=%d)",
                bin_id, metric, map_mean, ratio, guide_a.n_obs, n_cells_total)
    return row


# ---------------------------------------------------------------------------
# Legacy in-memory path (kept for --bin debug)
# ---------------------------------------------------------------------------

def run_one_bin(bin_id: str, bin_dir: Path = BIN_DIR, seed: int = 0) -> dict:
    bin_dir.mkdir(parents=True, exist_ok=True)
    out_path = bin_dir / f"{bin_id.replace(':', '_')}.json"

    # Load ALL cells once per bin worker (~50M cells × 95 dim = 19 GB float32)
    adata = _open_concat()
    # Work in positional-int space throughout — AnnData concat may leave
    # string obs_names that break np.unique([int_arr, str_arr]).
    obs = adata.obs.reset_index(drop=True)   # local copy with int RangeIndex
    logger.info("[%s] loaded %d cells", bin_id, len(obs))

    parts = bin_id.split(":")
    if parts[0] == "all_cells":
        selected_pos = np.arange(len(obs), dtype=np.int64)
        meta = {"sweep": "baseline", "direction": "all_cells", "head": None, "K": None}
    elif parts[0] == "sweep_a" and parts[1] == "random":
        K = int(parts[2])
        selected_pos = _select_random_per_gene(obs, K, seed=seed)
        meta = {"sweep": "A", "direction": "random", "head": None, "K": K}
    elif parts[0] == "sweep_a":
        _, head, direction, K = parts; K = int(K)
        rank_col = f"{head}_rank"
        if direction == "top":
            selected_pos = _select_top_per_gene(obs, rank_col, K)
        elif direction == "low":
            selected_pos = _select_bottom_per_gene(obs, rank_col, K)
        else:
            raise ValueError(f"bad direction: {direction}")
        meta = {"sweep": "A", "direction": direction, "head": head, "K": K}
    elif parts[0] == "sweep_b":
        _, p_tag, K = parts
        P = int(p_tag.lstrip("p")); K = int(K)
        selected_pos = _select_intersection_removed(obs, P=P, K=K, seed=seed)
        meta = {"sweep": "B", "direction": "intersection_removed",
                "percentile": P, "head": None, "K": K}
    else:
        raise ValueError(f"bad bin_id: {bin_id}")

    # Include ALL NTC cells (reference for NTC z-score). Cap optional but we
    # leave uncapped here since the bin uses a single big in-memory AnnData.
    ntc_pos = np.flatnonzero((obs["perturbation"] == "NTC").to_numpy())
    selected_pos = np.asarray(selected_pos, dtype=np.int64)
    ntc_pos = np.asarray(ntc_pos, dtype=np.int64)
    final_pos = np.unique(np.concatenate([selected_pos, ntc_pos]))
    if final_pos.size == 0:
        logger.warning("[%s] no cells selected", bin_id)
        return {}

    sub = adata[final_pos].copy()
    # _score_bin reads obs["perturbation"]/"sgRNA". Already there from Stage 0.
    # It also reads obs["rank_type"] for filtering but we already pre-filtered
    # cells before passing → set rank_type = "selected" to satisfy v3's
    # NTC vs "selected" partition.
    sub.obs["rank_type"] = np.where(sub.obs["perturbation"] == "NTC", "NTC", "selected")
    sub.obs["rank"] = np.zeros(sub.n_obs, dtype=np.int64)

    # Call v3's _score_bin with CHAD yaml: produces EBI + CHAD + distinctiveness.
    try:
        result = _score_bin(sub, lo=1, hi=int(meta.get("K") or 0) or 1,
                              rank_type="selected", null_size=10_000,
                              chad_yaml=CHAD_YAML)
    except Exception as e:
        logger.exception("[%s] _score_bin failed: %s", bin_id, e)
        result = None

    # _score_bin doesn't compute activity. Re-build guide_a (cheap — same
    # aggregation v3 already did internally) and call activity directly.
    activity = float("nan")
    try:
        from ops_model.features.anndata_utils import (
            aggregate_to_level, normalize_guide_adata,
        )
        from ops_utils.analysis.map_scores import phenotypic_activity_assesment
        guide_a = aggregate_to_level(sub, "guide", method="mean",
                                       preserve_batch_info=False)
        guide_a = normalize_guide_adata(guide_a, norm_method="ntc")
        _, activity = phenotypic_activity_assesment(
            guide_a, plot_results=False, null_size=10_000,
        )
        activity = float(activity)
    except Exception as e:
        logger.warning("[%s] phenotypic_activity_assesment failed: %s", bin_id, e)

    n_genes = int(sub.obs.loc[sub.obs["perturbation"] != "NTC", "perturbation"].nunique())
    row = {**meta, "n_cells": int(sub.n_obs), "n_genes": n_genes}
    if result:
        row.update({
            "ebi_consistency":     result.get("mean_map"),
            "chad_consistency":    result.get("mean_map_chad"),
            "distinctiveness":     result.get("mean_map_dist"),
        })
    row["phenotypic_activity"] = activity
    out_path.write_text(json.dumps(row, default=float))
    logger.info("[%s] n_cells=%d  n_genes=%d  ebi=%.4f  chad=%.4f  dist=%.4f  act=%.4f",
                bin_id, sub.n_obs, n_genes,
                row.get("ebi_consistency", float("nan")) or float("nan"),
                row.get("chad_consistency", float("nan")) or float("nan"),
                row.get("distinctiveness", float("nan")) or float("nan"),
                row.get("phenotypic_activity", float("nan")) or float("nan"))
    return row


# ---------------------------------------------------------------------------
# SLURM driver + aggregator
# ---------------------------------------------------------------------------

def submit_phase1(force: bool = False) -> None:
    """Phase 1: per-bin streaming guide aggregation. 118 SLURM jobs, each ~48GB.

    Memory peaks: 49M-row global obs DataFrame (~6 GB) + groupby/sort over
    the head's rank column (49M for geneko since it covers all 1001 genes).
    16GB OOMs on geneko top/low; 48GB has comfortable headroom.
    """
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

    bins = enumerate_bins()
    todo = [b for b in bins
            if force or not (GUIDE_DIR / f"{b.replace(':','_')}.h5ad").exists()]
    logger.info("PHASE 1: submitting %d / %d bins", len(todo), len(bins))
    if not todo:
        return

    op_dir = str(Path(__file__).resolve().parents[3])
    ko_shap_dir = str(Path(__file__).resolve().parent)
    slurm_params = {
        "timeout_min": 30, "mem": "48GB", "cpus_per_task": 2,
        "slurm_partition": "cpu",
        "slurm_setup": [
            f"export PYTHONPATH={op_dir}:{ko_shap_dir}:$PYTHONPATH",
            "export OMP_NUM_THREADS=1",
        ],
    }
    jobs = [
        {"name": f"v4gm_{b.replace(':','_')}",
         "func": build_bin_guide_means,
         "kwargs": {"bin_id": b, "guide_dir": GUIDE_DIR},
         "metadata": {"type": "v4_phase1", "bin_id": b}}
        for b in todo
    ]
    submit_parallel_jobs(
        jobs_to_submit=jobs, experiment="v4_phase1",
        slurm_params=slurm_params,
        log_dir="slurm_step_logs/v4_phase1",
        manifest_prefix="v4_phase1",
        wait_for_completion=True, verbose=True,
    )


def submit_phase2(force: bool = False) -> None:
    """Phase 2: per-(bin × metric) mAP scoring. 4 × 118 = 472 small SLURM jobs."""
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

    bins = enumerate_bins()
    op_dir = str(Path(__file__).resolve().parents[3])
    ko_shap_dir = str(Path(__file__).resolve().parent)
    slurm_params = {
        "timeout_min": 30, "mem": "8GB", "cpus_per_task": 2,
        "slurm_partition": "cpu",
        "slurm_setup": [
            f"export PYTHONPATH={op_dir}:{ko_shap_dir}:$PYTHONPATH",
            "export OMP_NUM_THREADS=1",
        ],
    }
    jobs = []
    for b in bins:
        guide_path = GUIDE_DIR / f"{b.replace(':','_')}.h5ad"
        if not guide_path.exists():
            continue
        for m in METRICS:
            out_p = BIN_DIR / f"{b.replace(':','_')}__{m}.json"
            if not force and out_p.exists():
                continue
            jobs.append({
                "name": f"v4m_{b.replace(':','_')}_{m[:3]}",
                "func": score_one_metric,
                "kwargs": {"bin_id": b, "metric": m, "guide_dir": GUIDE_DIR,
                            "bin_dir": BIN_DIR},
                "metadata": {"type": "v4_phase2", "bin_id": b, "metric": m},
            })
    logger.info("PHASE 2: submitting %d / %d (bin × metric)",
                len(jobs), len(bins) * len(METRICS))
    if not jobs:
        return
    submit_parallel_jobs(
        jobs_to_submit=jobs, experiment="v4_phase2",
        slurm_params=slurm_params,
        log_dir="slurm_step_logs/v4_phase2",
        manifest_prefix="v4_phase2",
        wait_for_completion=True, verbose=True,
    )


def submit_slurm(force: bool = False) -> None:
    """Submit Phase 1 then Phase 2."""
    submit_phase1(force=force)
    submit_phase2(force=force)


def aggregate(out_csv: Path = EXPANSION / "bin_results.csv") -> pd.DataFrame:
    """Pivot per-(bin, metric) JSON files into one row per bin × 4 metric columns."""
    rows = [json.loads(p.read_text()) for p in sorted(BIN_DIR.glob("*__*.json"))]
    if not rows:
        logger.warning("no per-metric JSON shards found")
        return pd.DataFrame()
    long_df = pd.DataFrame(rows)
    # Pivot: index = bin_id, columns = metric, value = value. Keep meta cols.
    meta_cols = [c for c in ["sweep", "direction", "head", "K", "percentile",
                              "n_guides", "n_genes", "n_cells"]
                 if c in long_df.columns]
    pivot = long_df.pivot_table(index="bin_id", columns="metric",
                                  values="value", aggfunc="first").reset_index()
    meta_df = long_df.groupby("bin_id")[meta_cols].first().reset_index()
    df = meta_df.merge(pivot, on="bin_id", how="left")
    df.to_csv(out_csv, index=False)
    logger.info("aggregated %d (bin × metric) → %d bins → %s",
                len(long_df), len(df), out_csv)
    return df


def _fmt_int_short(n: float) -> str:
    """Shorthand cell count: 3M, 500K, 1.2K, ..."""
    n = float(n)
    if n >= 1e6:
        return f"{n / 1e6:.1f}M".replace(".0M", "M")
    if n >= 1e3:
        return f"{n / 1e3:.0f}K"
    return f"{int(n)}"


def plot(df: pd.DataFrame, plot_dir: Path = PLOT_DIR) -> None:
    import matplotlib
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    baseline = df[df["sweep"] == "baseline"].iloc[0] if (df["sweep"] == "baseline").any() else None

    metrics = [("ebi_consistency",     "EBI consistency"),
               ("chad_consistency",    "CHAD consistency"),
               ("distinctiveness",     "Distinctiveness"),
               ("phenotypic_activity", "Phenotypic activity")]

    # For Sweep B legends: total real-gene cell pool — drop = baseline_real - n_cells_at_high_K.
    # Estimate from `random (no removal)` at the highest K we have.
    sa_all = df[df["sweep"] == "A"].copy()
    rnd_full = sa_all[sa_all["direction"] == "random"].sort_values("K")
    n_cells_no_removal = (
        int(rnd_full["n_cells"].max()) if not rnd_full.empty else int(baseline.get("n_cells", 0))
    )

    # ---- Sweep A: per-head top/low + head-agnostic random ----
    sa = df[df["sweep"] == "A"].copy()
    for col, label in metrics:
        fig, ax = plt.subplots(figsize=(7, 5))
        for head in HEADS:
            for direction, ls in [("top", "-"), ("low", "--")]:
                sub = sa[(sa["head"] == head) & (sa["direction"] == direction)].sort_values("K")
                if sub.empty:
                    continue
                peak = float(sub[col].max())
                ax.plot(sub["K"], sub[col], marker="o", linestyle=ls,
                        label=f"{head}/{direction}  (peak {peak:.3f})")
        rnd = sa[sa["direction"] == "random"].sort_values("K")
        if not rnd.empty:
            peak = float(rnd[col].max())
            ax.plot(rnd["K"], rnd[col], marker="s", color="black", linestyle=":",
                    label=f"random (head-agnostic, peak {peak:.3f})")
        if baseline is not None and pd.notna(baseline.get(col)):
            ax.axhline(baseline[col], color="red", linestyle="--", linewidth=1,
                       label=f"all-cells ({baseline[col]:.3f})")
        ax.set_xscale("log"); ax.set_xlabel("K cells / gene"); ax.set_ylabel(label)
        ax.set_title(f"Sweep A — {label}")
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
        ax.grid(alpha=0.3); fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(plot_dir / f"sweep_a_{col}.{ext}", dpi=120, bbox_inches="tight")
        plt.close(fig)

    # ---- Sweep B: intersection-removed at P ∈ {10,20,25,30,40,50} ----
    # Viridis: low-P (few cells removed) → dark, high-P (many removed) → bright.
    # Start above 0.0 to stay clear of the black "no removal" control.
    sb = df[df["sweep"] == "B"].copy()
    sorted_P = sorted(PERCENTILES)
    viridis = plt.get_cmap("viridis")
    p_colors = {P: viridis(0.15 + 0.85 * i / max(1, len(sorted_P) - 1))
                for i, P in enumerate(sorted_P)}
    for col, label in metrics:
        fig, ax = plt.subplots(figsize=(7, 5))
        for P in sorted_P:
            sub = sb[sb.get("percentile") == P].sort_values("K")
            if sub.empty:
                continue
            peak = float(sub[col].max())
            # n_dropped = baseline pool − cells kept at largest K (intersection drop is fixed across K)
            n_kept_high_K = int(sub["n_cells"].max())
            n_dropped = max(0, n_cells_no_removal - n_kept_high_K)
            ax.plot(sub["K"], sub[col], marker="o", color=p_colors[P],
                    label=f"P={P}% (drop {_fmt_int_short(n_dropped)}, peak {peak:.3f})")
        if not sa.empty:
            rnd = sa[sa["direction"] == "random"].sort_values("K")
            if not rnd.empty:
                peak = float(rnd[col].max())
                ax.plot(rnd["K"], rnd[col], marker="s", color="black",
                        linestyle=":",
                        label=f"random no removal (peak {peak:.3f})")
        if baseline is not None and pd.notna(baseline.get(col)):
            ax.axhline(baseline[col], color="red", linestyle="--", linewidth=1,
                       label=f"all-cells ({baseline[col]:.3f})")
        ax.set_xscale("log"); ax.set_xlabel("K cells / gene"); ax.set_ylabel(label)
        ax.set_title(f"Sweep B — {label}  (EBI ∩ geneKO bottom-P% removed)")
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
        ax.grid(alpha=0.3); fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(plot_dir / f"sweep_b_{col}.{ext}", dpi=120, bbox_inches="tight")
        plt.close(fig)

    logger.info("wrote %d plots to %s", 8, plot_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bin", help="Run one bin id locally")
    p.add_argument("--submit-slurm", action="store_true",
                   help="Submit all bins via SLURM, wait, aggregate, plot")
    p.add_argument("--force", action="store_true")
    p.add_argument("--aggregate-only", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if args.bin:
        run_one_bin(args.bin)
        return 0
    if args.submit_slurm:
        submit_slurm(force=args.force)
    if args.aggregate_only or args.submit_slurm:
        df = aggregate()
        plot(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
