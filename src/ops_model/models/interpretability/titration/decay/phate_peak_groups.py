"""UMAP + PHATE + EBI-overlay embeddings for arbitrary attention groups.

Reusable wrapper around `pca_optimization._compute_and_plot_embeddings`
(UMAP with Max's scanpy recipe, PHATE knn=8/decay=10, EBI-complex
overlays, CHAD positive-controls grid).

Each group is specified as `direction:K` where `direction` is one of
`top`, `bottom`, `random`, `random_low_removed`, `mixed`, `mixed_union`.
The script submits one self-contained SLURM job per group (each computes
its own exact-K thresholds + filters cells + aggregates to guide level +
NTC-normalizes + embeds) plus an optional sgRNA-coverage job.

Example:
    # Two peak groups + sgRNA coverage:
    python organelle_profiler/scripts/ko_shap/phate_peak_groups.py \\
        --groups top:18000 random_low_removed:20000 --sgrna-coverage

    # Single group, custom source/K:
    python organelle_profiler/scripts/ko_shap/phate_peak_groups.py \\
        --groups mixed_union:5000 \\
        --out-dir /hpc/mydata/$USER/phate_mixed_union

    # Local debug (no SLURM):
    python organelle_profiler/scripts/ko_shap/phate_peak_groups.py \\
        --groups top:1000 --local
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd

# Helpers inlined (not imported from map_attention_decay) so the SLURM
# workers can unpickle this module without needing the script dir on
# sys.path. Each helper mirrors the equivalent in map_attention_decay.py;
# keep them in sync if that file is updated.
_ALEX = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention"
)
DEFAULT_PMA_PARQUET_CHAD = (
    _ALEX / "v3" / "attention_v3" / "pma_top_phase_cells_chad_v1.parquet"
)


def _cell_hashes(experiment_arr, segmentation_arr) -> np.ndarray:
    """Deterministic uint64 hash per (experiment, segmentation) cell."""
    ids = (pd.Series(experiment_arr, dtype="object").astype(str) + "_"
           + pd.Series(segmentation_arr).astype("int64").astype(str))
    return pd.util.hash_array(ids.values)


def _read_pma(pma_path: Path,
               columns: Optional[list] = None,
               experiment: Optional[str] = None) -> pd.DataFrame:
    """Read a PMA table (CSV or parquet), optionally column-pruned and/or
    filtered to one experiment via pyarrow predicate pushdown.
    """
    p = str(pma_path)
    if p.endswith(".parquet"):
        import pyarrow.parquet as papq
        filters = [("experiment", "=", experiment)] if experiment else None
        return papq.read_table(p, columns=columns, filters=filters).to_pandas()
    df = pd.read_csv(p, usecols=columns)
    if experiment is not None:
        df = df[df["experiment"].astype(str) == experiment].copy()
    return df


def _read_chad_rank_for_exp(chad_pma_parquet: Path,
                              experiment: str) -> dict:
    """{(pma_gene, segmentation): chad_rank} restricted to top cohort."""
    import pyarrow.parquet as papq
    t = papq.read_table(
        str(chad_pma_parquet),
        columns=["gene", "segmentation", "rank", "rank_type"],
        filters=[("experiment", "=", experiment),
                 ("rank_type", "=", "top")],
    )
    df = t.to_pandas()
    return {(str(g), int(s)): int(r)
            for g, s, r in zip(df["gene"], df["segmentation"], df["rank"])}


def _compute_compacted_thresholds(per_exp_dir: Path,
                                    bin_ks: list,
                                    chad_pma_parquet: Optional[Path] = None,
                                    chad_max_bin: int = 5000) -> dict:
    """Per (gene, K) exact-K thresholds over CACHED cells. Mirrors
    map_attention_decay._compute_compacted_thresholds. See that file's
    docstring for details.
    """
    obs_blocks = []
    for p in sorted(per_exp_dir.glob("*.h5ad")):
        if p.stat().st_size == 0:
            continue
        try:
            ob = ad.read_h5ad(p, backed="r").obs
        except Exception:
            continue
        need = ["pma_gene", "pma_segmentation", "rank", "rank_type"]
        if not all(c in ob.columns for c in need):
            continue
        sub = ob[need].copy()
        sub["_exp"] = p.stem
        obs_blocks.append(sub)
    if not obs_blocks:
        raise RuntimeError(f"No usable per-experiment caches in {per_exp_dir}")
    df = pd.concat(obs_blocks, ignore_index=True)
    df = df[df["rank_type"].astype(str) == "top"]
    df["pma_gene"] = df["pma_gene"].astype(str)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce").astype("int64")
    df["_h"] = _cell_hashes(df["_exp"].values,
                              df["pma_segmentation"].astype("int64").values)

    if chad_pma_parquet is not None and chad_pma_parquet.exists():
        import pyarrow.parquet as papq
        ct = papq.read_table(
            str(chad_pma_parquet),
            columns=["gene", "experiment", "segmentation", "rank", "rank_type"],
            filters=[("rank_type", "=", "top")],
        )
        cdf = ct.to_pandas()
        cdf.rename(columns={"experiment": "_exp",
                              "segmentation": "pma_segmentation",
                              "gene": "pma_gene",
                              "rank": "chad_rank"}, inplace=True)
        cdf["pma_gene"] = cdf["pma_gene"].astype(str)
        cdf["pma_segmentation"] = cdf["pma_segmentation"].astype("int64")
        df = df.merge(
            cdf[["_exp", "pma_segmentation", "pma_gene", "chad_rank"]],
            on=["_exp", "pma_segmentation", "pma_gene"], how="left",
        )
        # 10**12 sentinel — int64.max round-trips through float64
        # to INT64_MIN and poisons combined_rank. See map_attention_decay.
        CHAD_MISSING_SENTINEL = 10**12
        df["chad_rank"] = df["chad_rank"].fillna(CHAD_MISSING_SENTINEL).astype("int64")
        df["combined_rank"] = np.minimum(
            df["rank"].astype("int64").values,
            df["chad_rank"].astype("int64").values,
        )

    out = {"top": {}, "bottom": {}, "random": {}, "mixed": {},
           "random_low_removed": {}, "upper_half_rank": {}}
    max_uint64 = int(np.iinfo(np.uint64).max)
    max_int64  = int(np.iinfo(np.int64).max)
    for gene, sub in df.groupby("pma_gene"):
        ranks_asc = np.sort(sub["rank"].values)
        ranks_desc = ranks_asc[::-1]
        hashes_asc = np.sort(sub["_h"].values)
        if "combined_rank" in sub.columns:
            mixed_asc = np.sort(sub["combined_rank"].values)
        else:
            mixed_asc = None
        n = len(ranks_asc)
        n_upper = max(1, n // 2)
        upper_half_thresh = int(ranks_asc[n_upper - 1])
        out["upper_half_rank"][gene] = upper_half_thresh
        upper_mask = sub["rank"].values <= upper_half_thresh
        upper_h_sorted = np.sort(sub["_h"].values[upper_mask])
        n_pool = len(upper_h_sorted)
        for K in bin_ks:
            K = int(K)
            if K >= n:
                out["top"][(gene, K)]    = int(ranks_asc[-1])
                out["bottom"][(gene, K)] = int(ranks_asc[0])
                out["random"][(gene, K)] = max_uint64
            else:
                out["top"][(gene, K)]    = int(ranks_asc[K - 1])
                out["bottom"][(gene, K)] = int(ranks_desc[K - 1])
                out["random"][(gene, K)] = int(hashes_asc[K - 1])
            if K >= n_pool:
                out["random_low_removed"][(gene, K)] = max_uint64
            else:
                out["random_low_removed"][(gene, K)] = int(upper_h_sorted[K - 1])
            if mixed_asc is not None and K <= chad_max_bin:
                if K >= n:
                    out["mixed"][(gene, K)] = max_int64
                else:
                    out["mixed"][(gene, K)] = int(mixed_asc[K - 1])
    return out

logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────────
_CDINO = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v3/attention_v3/cdino"
)
DEFAULT_PER_EXP_DIR_FAST = _CDINO / "per_experiment_filtered_all_fast"
DEFAULT_PER_EXP_PCA_DIR_FAST = _CDINO / "per_experiment_filtered_all_fast_pca"
DEFAULT_OUT_DIR = _CDINO / "phate_peak_groups"


# ── Cell filter (mirrors map_attention_decay.compute_bin_map_from_per_exp) ─
def _filter_cells(
    per_exp_pca_dir: Path,
    direction: str,
    K: int,
    compacted_thresholds: dict,
    chad_pma_parquet: Optional[Path] = None,
    ntc_per_exp: int = 1500,
) -> ad.AnnData:
    """Filter cached cells for one (direction, K). Returns a raw
    cell-level AnnData (X = PCA-projected features from the cache).
    Same logic as `map_attention_decay.compute_bin_map_from_per_exp`
    but stops before guide-level aggregation so the caller controls it.
    """
    import h5py

    X_blocks, obs_blocks, var_names = [], [], None
    K_int = int(K)
    d_thresh = compacted_thresholds.get(direction, {})
    upper_lookup = compacted_thresholds.get("upper_half_rank", {})

    for p in sorted(per_exp_pca_dir.glob("*.h5ad")):
        if p.stat().st_size == 0:
            continue
        try:
            obs = ad.read_h5ad(p, backed="r").obs.copy()
        except Exception:
            continue
        obs_rt = obs["rank_type"].astype(str)
        obs_g  = obs.get("pma_gene", obs.get("perturbation")).astype(str)
        obs_r  = pd.to_numeric(obs["rank"], errors="coerce").astype("int64")

        if direction == "top":
            thresh = obs_g.map(
                lambda g: int(d_thresh.get((g, K_int), 0))
            ).astype("int64").values
            rank_mask = (obs_rt == "top") & (
                pd.Series(obs_r.values <= thresh, index=obs.index)
            )
        elif direction == "bottom":
            thresh = obs_g.map(
                lambda g: int(d_thresh.get(
                    (g, K_int), np.iinfo(np.int64).max))
            ).astype("int64").values
            rank_mask = (obs_rt == "top") & (
                pd.Series(obs_r.values >= thresh, index=obs.index)
            )
        elif direction == "random":
            cell_h = _cell_hashes(
                np.full(len(obs), p.stem, dtype=object),
                obs["pma_segmentation"].astype("int64").values,
            )
            thresh = obs_g.map(
                lambda g: int(d_thresh.get((g, K_int), 0))
            ).astype("uint64").values
            rank_mask = (obs_rt == "top") & (
                pd.Series(cell_h <= thresh, index=obs.index)
            )
        elif direction == "random_low_removed":
            upper_thresh = obs_g.map(
                lambda g: int(upper_lookup.get(g, 0))
            ).astype("int64").values
            in_upper = obs_r.values <= upper_thresh
            cell_h = _cell_hashes(
                np.full(len(obs), p.stem, dtype=object),
                obs["pma_segmentation"].astype("int64").values,
            )
            thresh = obs_g.map(
                lambda g: int(d_thresh.get((g, K_int), 0))
            ).astype("uint64").values
            rank_mask = (obs_rt == "top") & (
                pd.Series(in_upper & (cell_h <= thresh), index=obs.index)
            )
        elif direction in ("mixed", "mixed_union"):
            if chad_pma_parquet is None:
                raise ValueError(
                    f"direction {direction!r} requires chad_pma_parquet"
                )
            chad_lookup = _read_chad_rank_for_exp(
                Path(chad_pma_parquet), p.stem
            )
            max_i64 = np.iinfo(np.int64).max
            chad_ranks = np.fromiter(
                (chad_lookup.get((g, int(s)), max_i64)
                 for g, s in zip(obs_g.values,
                                 obs["pma_segmentation"].astype("int64").values)),
                dtype=np.int64, count=len(obs),
            )
            gko_ranks = obs_r.astype("int64").values
            if direction == "mixed":
                combined = np.minimum(gko_ranks, chad_ranks)
                thresh = obs_g.map(
                    lambda g: int(d_thresh.get((g, K_int), 0))
                ).astype("int64").values
                rank_mask = (obs_rt == "top") & (
                    pd.Series(combined <= thresh, index=obs.index)
                )
            else:  # mixed_union
                in_either = (gko_ranks <= K_int) | (chad_ranks <= K_int)
                rank_mask = (obs_rt == "top") & (
                    pd.Series(in_either, index=obs.index)
                )
        else:
            raise ValueError(f"unsupported direction: {direction}")

        # NTC handling — same as bin worker.
        ntc_mask = (obs_rt == "NTC").to_numpy()
        rank_mask_np = rank_mask.to_numpy()
        ntc_pos = np.where(ntc_mask)[0]
        if len(ntc_pos) > ntc_per_exp:
            rng = np.random.default_rng(
                seed=hash((p.name, "ntc", direction, K_int)) & 0xFFFF_FFFF
            )
            keep_ntc = rng.choice(ntc_pos, ntc_per_exp, replace=False)
            ntc_keep_mask = np.zeros_like(ntc_mask)
            ntc_keep_mask[keep_ntc] = True
        else:
            ntc_keep_mask = ntc_mask
        mask = rank_mask_np | ntc_keep_mask
        if not mask.any():
            continue
        row_idx = np.where(mask)[0]
        obs_sub = obs.iloc[row_idx].reset_index(drop=True)
        with h5py.File(p, "r") as f:
            X_ds = f["X"]
            order = np.argsort(row_idx)
            sorted_idx = row_idx[order]
            sorted_slab = X_ds[sorted_idx, :]
            slab = np.empty_like(sorted_slab)
            slab[order] = sorted_slab
            if var_names is None:
                var_names = [s.decode() if isinstance(s, bytes) else s
                              for s in f["var"]["_index"][:]]
        X_blocks.append(np.asarray(slab, dtype=np.float32))
        obs_blocks.append(obs_sub)

    if not X_blocks:
        raise RuntimeError(f"No cells matched for {direction} K={K}")
    return ad.AnnData(
        X=np.vstack(X_blocks),
        obs=pd.concat(obs_blocks, ignore_index=True),
        var=pd.DataFrame(index=var_names) if var_names else None,
    )


# ── Per-group worker: self-contained (computes its own thresholds) ──────
def _group_worker(
    per_exp_dir_s: str,
    per_exp_pca_dir_s: str,
    direction: str,
    K: int,
    out_dir_s: str,
    chad_pma_parquet_s: Optional[str] = None,
) -> str:
    """SLURM worker: compute thresholds (self-contained), filter cells,
    aggregate to guide + NTC-norm, run UMAP + PHATE + EBI overlay.
    Saves guide.h5ad, gene.h5ad, and plots.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ops_model.features.anndata_utils import (
        aggregate_to_level, normalize_guide_adata,
    )
    from ops_model.post_process.combination.pca_optimization.embeddings import (
        _compute_and_plot_embeddings,
    )

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    out_dir = Path(out_dir_s) / f"{direction}_K{int(K)}"
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{direction} K={K}] computing exact-K thresholds…")
    chad_pq = (Path(chad_pma_parquet_s)
                 if chad_pma_parquet_s and direction in ("mixed", "mixed_union")
                 else None)
    thresholds = _compute_compacted_thresholds(
        Path(per_exp_dir_s), [int(K)], chad_pma_parquet=chad_pq,
    )
    logger.info(f"[{direction} K={K}] filtering cells…")
    sub = _filter_cells(
        Path(per_exp_pca_dir_s), direction, int(K), thresholds,
        chad_pma_parquet=chad_pq,
    )
    logger.info(f"[{direction} K={K}] {sub.n_obs:,} cells × {sub.n_vars} feats")

    guide_a = aggregate_to_level(sub, "guide", method="mean",
                                  preserve_batch_info=False)
    n_ntc = int((guide_a.obs["perturbation"].astype(str).str.upper()
                  == "NTC").sum())
    if n_ntc < 2:
        raise RuntimeError(f"need ≥2 NTC guide rows, got {n_ntc}")
    guide_a = normalize_guide_adata(guide_a, norm_method="ntc")
    guide_a.obsm["X_pca"] = np.asarray(guide_a.X, dtype=np.float32)
    logger.info(f"[{direction} K={K}] guide-level: {guide_a.n_obs:,} sgRNAs")

    metric_lookup: dict = {}
    adata_gene_embed = _compute_and_plot_embeddings(
        adata_guide=guide_a,
        metric_lookup=metric_lookup,
        plots_dir=plots_dir,
        plt=plt,
        _logger=logger,
        random_seed=42,
        chromosome_csv=None,
        umap_type="max",
    )

    guide_a.write_h5ad(out_dir / "guide.h5ad")
    if adata_gene_embed is not None:
        adata_gene_embed.write_h5ad(out_dir / "gene.h5ad")
    return f"wrote {out_dir}"


# ── sgRNA coverage worker — same thresholds, metadata-only scan ─────────
def _sgrna_coverage_worker(
    per_exp_dir_s: str,
    out_csv_s: str,
) -> str:
    """SLURM worker: report per-gene sgRNA coverage of the bottom-50%
    attention filter (= the pool that random_low_removed samples from).

    For each gene, counts:
      n_sgrnas_all_cells: distinct sgRNAs in the cache
      n_sgrnas_upper_half: distinct sgRNAs with ≥1 cell in upper-half-
                            by-attention (RLR pool)
      sgrnas_lost: comma-list of sgRNAs absent from the upper half
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    per_exp_dir = Path(per_exp_dir_s)

    # We only need ONE K to compute upper_half_rank — the thresholds
    # dict carries it under "upper_half_rank" independent of K.
    logger.info("[sgRNA] computing upper_half thresholds…")
    thresholds = _compute_compacted_thresholds(per_exp_dir, [1])
    upper_lookup = thresholds.get("upper_half_rank", {})
    if not upper_lookup:
        raise RuntimeError("missing upper_half_rank in thresholds")

    blocks = []
    for p in sorted(per_exp_dir.glob("*.h5ad")):
        if p.stat().st_size == 0:
            continue
        try:
            ob = ad.read_h5ad(p, backed="r").obs
        except Exception:
            continue
        col_sg = None
        for cand in ("sgRNA", "guide", "perturbation_guide"):
            if cand in ob.columns:
                col_sg = cand
                break
        if col_sg is None:
            continue
        # Group by `perturbation` (TRUE KO), NOT `pma_gene` (PMA spatial-
        # NN match assignment). ~20% of cells have pma_gene ≠
        # perturbation because nearest-PMA-row within 5 px isn't always
        # the cell's own KO row — using pma_gene would mix sgRNAs across
        # different actual KOs and inflate per-gene sgRNA counts.
        need = ["perturbation", "pma_gene", col_sg, "rank", "rank_type"]
        df = ob[need].copy()
        df.columns = ["perturbation", "pma_gene", "sgRNA", "rank", "rank_type"]
        df = df[df["rank_type"].astype(str) == "top"].copy()
        df["perturbation"] = df["perturbation"].astype(str)
        df["pma_gene"]     = df["pma_gene"].astype(str)
        df["sgRNA"]        = df["sgRNA"].astype(str)
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce").astype("int64")
        # Upper-half filter applies the threshold of the cell's pma_gene
        # (which is what the actual bin filter uses), even though we
        # group/report by true perturbation.
        df["upper_thresh"] = df["pma_gene"].map(
            lambda g: int(upper_lookup.get(g, 0))
        ).astype("int64")
        df["in_upper"] = df["rank"] <= df["upper_thresh"]
        blocks.append(df[["perturbation", "sgRNA", "in_upper"]])

    big = pd.concat(blocks, ignore_index=True)
    rows = []
    for gene, sub in big.groupby("perturbation"):
        all_sg   = set(sub["sgRNA"].unique())
        upper_sg = set(sub.loc[sub["in_upper"], "sgRNA"].unique())
        lost = sorted(all_sg - upper_sg)
        rows.append({
            "gene":                gene,
            "n_sgrnas_all_cells":  len(all_sg),
            "n_sgrnas_upper_half": len(upper_sg),
            "sgrnas_lost":         "|".join(lost),
            "n_sgrnas_lost":       len(lost),
        })
    cov = pd.DataFrame(rows).sort_values("n_sgrnas_lost", ascending=False)
    out_csv = Path(out_csv_s)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cov.to_csv(out_csv, index=False)

    summary = cov["n_sgrnas_lost"].value_counts().sort_index().to_dict()
    print(f"[sgRNA] {len(cov)} genes total")
    print(f"[sgRNA] sgRNAs-lost distribution: {summary}")
    n_lost = int((cov["n_sgrnas_lost"] > 0).sum())
    print(f"[sgRNA] {n_lost}/{len(cov)} genes lost ≥1 sgRNA in RLR filter")
    print(f"[sgRNA] full table: {out_csv}")
    return f"wrote {out_csv}"


# ── CLI ─────────────────────────────────────────────────────────────────
def _parse_group(spec: str) -> tuple:
    if ":" not in spec:
        raise argparse.ArgumentTypeError(
            f"group spec {spec!r} must be 'direction:K' (e.g. top:18000)"
        )
    direction, K_str = spec.split(":", 1)
    valid = {"top", "bottom", "random", "random_low_removed",
              "mixed", "mixed_union"}
    if direction not in valid:
        raise argparse.ArgumentTypeError(
            f"direction {direction!r} not in {sorted(valid)}"
        )
    try:
        K = int(K_str)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"K {K_str!r} must be an integer"
        )
    return (direction, K)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--groups", nargs="+", type=_parse_group,
        default=[("top", 18000), ("random_low_removed", 20000)],
        help="One or more group specs as `direction:K`. Default: "
             "`top:18000 random_low_removed:20000` (the observed "
             "distinctiveness peak + its random-from-upper-half control).",
    )
    ap.add_argument("--per-exp-dir", type=Path,
                    default=DEFAULT_PER_EXP_DIR_FAST,
                    help="Per-experiment raw-feature cache dir (obs is "
                         "sgRNA-rich; used for threshold compute + "
                         "sgRNA coverage scan).")
    ap.add_argument("--per-exp-pca-dir", type=Path,
                    default=DEFAULT_PER_EXP_PCA_DIR_FAST,
                    help="Per-experiment PCA-projected cache dir. "
                         "Embeddings + aggregation operate on these "
                         "(97-d, ~10× smaller than raw).")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--chad-parquet", type=Path,
                    default=DEFAULT_PMA_PARQUET_CHAD,
                    help="CHAD PMA parquet (only needed for mixed* groups).")
    ap.add_argument("--sgrna-coverage", action="store_true",
                    help="Also submit a SLURM job that reports per-gene "
                         "sgRNA coverage of the bottom-50% filter (= "
                         "the random_low_removed pool).")
    ap.add_argument("--local", action="store_true",
                    help="Run inline (no SLURM).")
    ap.add_argument("--slurm-mem", default="256GB")
    ap.add_argument("--slurm-timeout-min", type=int, default=60)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[plan] {len(args.groups)} group(s) → {args.out_dir}")
    for direction, K in args.groups:
        print(f"  - {direction} @ K={K}")
    if args.sgrna_coverage:
        print(f"  + sgRNA coverage report")

    if args.local:
        for direction, K in args.groups:
            print(f"\n========== {direction} K={K} ==========")
            print(_group_worker(
                per_exp_dir_s=str(args.per_exp_dir),
                per_exp_pca_dir_s=str(args.per_exp_pca_dir),
                direction=direction, K=int(K),
                out_dir_s=str(args.out_dir),
                chad_pma_parquet_s=str(args.chad_parquet),
            ))
        if args.sgrna_coverage:
            print(f"\n========== sgRNA coverage ==========")
            print(_sgrna_coverage_worker(
                per_exp_dir_s=str(args.per_exp_dir),
                out_csv_s=str(args.out_dir / "rlr_sgrna_coverage.csv"),
            ))
        return

    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
    jobs = [
        {
            "name": f"phate_{direction}_K{int(K)}",
            "func": _group_worker,
            "kwargs": dict(
                per_exp_dir_s=str(args.per_exp_dir),
                per_exp_pca_dir_s=str(args.per_exp_pca_dir),
                direction=direction, K=int(K),
                out_dir_s=str(args.out_dir),
                chad_pma_parquet_s=str(args.chad_parquet),
            ),
            "metadata": {"direction": direction, "K": int(K)},
        }
        for direction, K in args.groups
    ]
    if args.sgrna_coverage:
        jobs.append({
            "name": "sgrna_coverage",
            "func": _sgrna_coverage_worker,
            "kwargs": dict(
                per_exp_dir_s=str(args.per_exp_dir),
                out_csv_s=str(args.out_dir / "rlr_sgrna_coverage.csv"),
            ),
            "metadata": {"kind": "sgrna_coverage"},
        })

    print(f"\n[slurm] submitting {len(jobs)} jobs in parallel…")
    submit_parallel_jobs(
        jobs_to_submit=jobs,
        experiment="phate_peak_groups",
        slurm_params={
            "timeout_min":     args.slurm_timeout_min,
            "slurm_mem":       args.slurm_mem,
            "cpus_per_task":   8,
            "slurm_partition": "cpu",
        },
        log_dir="phate_peak_groups",
        manifest_prefix="phate_peak_groups",
        wait_for_completion=True,
    )


if __name__ == "__main__":
    main()
