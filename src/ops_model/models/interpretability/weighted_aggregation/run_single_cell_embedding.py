"""Single-cell UMAP+PHATE of the top-1k cells per class from the v5 ebifb
(EBI-feedback 1000-gene SetTransformer) set-accuracy ranking.

Two phases:
  1. gather (parallel, 1 job/experiment): read one V5_PER_EXP h5ad, filter
     ``obs["ebifb_rank"] <= top_k``, write a shard parquet + npy.
  2. embed (single big-mem job): concat shards, dedup by (exp, well, seg),
     run UMAP + landmark-PHATE on BOTH raw CellDINO and PCA-reduced features
     (fit at ``--pca-var`` variance on the top-1k cells), save cells.parquet
     with all four sets of coords + plots.

Driver ``launch_single_cell_embedding.py`` orchestrates both via
``submit_parallel_jobs``.

Selection: ``obs["ebifb_rank"] <= 1000`` per class across the 88 V5_PER_EXP h5ads
(1001 classes × ~1000 cells ≈ 1M rows; dedup by (exp, well, seg) → unique cells).
Features: raw CellDINO (``.X``, 1024-dim). No per-exp z-score (unlikely to
matter at cell-level for this comparison).
Embed: UMAP (n_neighbors=8, min_dist=0.25, euclidean) + PHATE (knn=8, decay=10,
n_landmark=2000). Two variants produced side-by-side: ``raw`` vs ``pca``.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

V5_PER_EXP = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v5/expansion_v1/per_experiment_v5"
)
EBI_YAML = Path(
    "/hpc/projects/icd.fast.ops/configs/gene_clusters/EBI_complexes_v1_old_gene_names.yaml"
)
DEFAULT_OUT = Path(
    "/hpc/projects/icd.fast.ops/analysis/figure4_embedding/single_cell_embedding_ebifb_top1k"
)

SHARD_META_COLS = [
    "experiment", "well", "segmentation_id",
    "ebifb_gene", "ebifb_rank", "ebifb_rank_type",
    "gene_name", "sgRNA", "x_position", "y_position",
]


def _logger(out_dir: Path, name: str) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    for h in [logging.StreamHandler(), logging.FileHandler(out_dir / f"{name}.log")]:
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        lg.addHandler(h)
    return lg


def _load_ebi_map() -> dict[str, str]:
    import yaml
    with open(EBI_YAML) as f:
        hier = yaml.safe_load(f) or {}
    gene_to_complex: dict[str, str] = {}
    for _id, cluster in hier.items():
        if not isinstance(cluster, dict) or "name" not in cluster:
            continue
        for gene in cluster.get("genes", []):
            gene_to_complex.setdefault(gene, cluster["name"])
    return gene_to_complex


# ---------- phase 1: gather one shard --------------------------------------

def gather_shard(h5ad_path: str, top_k: int, shard_dir: str) -> dict:
    """Read one V5_PER_EXP h5ad, keep rows with ``ebifb_rank <= top_k``, and
    write a shard parquet (obs subset) + npy (CellDINO) named after the
    experiment stem."""
    hp = Path(h5ad_path); sd = Path(shard_dir); sd.mkdir(parents=True, exist_ok=True)
    exp = hp.stem
    t0 = time.time()
    a = ad.read_h5ad(hp)
    mask = (a.obs["ebifb_rank"] <= top_k).to_numpy()
    n_keep = int(mask.sum())
    if n_keep == 0:
        (sd / f"{exp}.EMPTY").touch()
        return {"experiment": exp, "n_kept": 0, "elapsed_s": time.time() - t0}
    X = np.asarray(a.X[mask], dtype=np.float32)
    obs = a.obs.loc[mask, SHARD_META_COLS].copy()
    obs.to_parquet(sd / f"{exp}.parquet")
    np.save(sd / f"{exp}.npy", X.astype(np.float16))
    return {"experiment": exp, "n_kept": n_keep, "elapsed_s": time.time() - t0,
            "n_total": int(a.n_obs)}


# ---------- phase 2: embed -------------------------------------------------

def _load_shards(shard_dir: Path, lg: logging.Logger) -> tuple[np.ndarray, pd.DataFrame]:
    parquets = sorted(shard_dir.glob("*.parquet"))
    lg.info(f"loading {len(parquets)} shards from {shard_dir}")
    obs_parts, X_parts = [], []
    t0 = time.time()
    for i, p in enumerate(parquets, 1):
        o = pd.read_parquet(p)
        x = np.load(p.with_suffix(".npy"))
        obs_parts.append(o); X_parts.append(x)
        lg.info(f"  [{i:>3}/{len(parquets)}] {p.stem}: {len(o):,} rows")
    obs = pd.concat(obs_parts, ignore_index=True)
    X = np.concatenate(X_parts, axis=0)
    lg.info(f"pre-dedup: {len(obs):,} rows, X={X.shape} in {time.time()-t0:.1f}s")
    return X, obs


def _dedup(X: np.ndarray, obs: pd.DataFrame, lg: logging.Logger) -> tuple[np.ndarray, pd.DataFrame]:
    obs = obs.reset_index(drop=True)
    obs["_row"] = np.arange(len(obs))
    keep_idx = (
        obs.sort_values("ebifb_rank", kind="mergesort")
        .drop_duplicates(subset=["experiment", "well", "segmentation_id"], keep="first")["_row"]
        .to_numpy()
    )
    keep_idx.sort()
    X = X[keep_idx]
    obs = obs.iloc[keep_idx].drop(columns=["_row"]).reset_index(drop=True)
    lg.info(f"post-dedup by (exp,well,seg): {len(obs):,} unique cells, X={X.shape}")
    return X, obs


def _fit_umap(X: np.ndarray, seed: int, lg: logging.Logger,
              use_gpu: bool = False) -> np.ndarray:
    lg.info(f"UMAP fit: n={X.shape[0]:,} d={X.shape[1]} "
            f"(n_neighbors=8, min_dist=0.25, euclidean, gpu={use_gpu})")
    t0 = time.time()
    if use_gpu:
        from cuml.manifold import UMAP as cuUMAP
        coords = cuUMAP(
            n_neighbors=8, min_dist=0.25, n_components=2, metric="euclidean",
            random_state=seed, verbose=True,
        ).fit_transform(X)
        coords = np.asarray(coords)
    else:
        import umap
        coords = umap.UMAP(
            n_neighbors=8, min_dist=0.25, n_components=2, metric="euclidean",
            random_state=seed, low_memory=True, verbose=True,
        ).fit_transform(X)
    lg.info(f"UMAP done in {time.time()-t0:.0f}s")
    return coords


def _fit_phate(X: np.ndarray, seed: int, n_landmark: int, lg: logging.Logger) -> np.ndarray:
    import phate
    n_pca = min(50, X.shape[1] - 1)
    lg.info(f"PHATE fit: n={X.shape[0]:,} (knn=8, decay=10, t=auto, "
            f"n_landmark={n_landmark}, n_pca={n_pca}, mds_solver=smacof)")
    t0 = time.time()
    coords = phate.PHATE(
        n_components=2, knn=8, decay=10, t="auto",
        n_landmark=n_landmark, n_pca=n_pca,
        mds_solver="smacof",
        n_jobs=-1, random_state=seed, verbose=1,
    ).fit_transform(X)
    lg.info(f"PHATE done in {time.time()-t0:.0f}s")
    return coords


def _plot(coords_by_name: dict[str, np.ndarray], color_labels: pd.Series,
          out_png: Path, title: str, gray_labels: set[str] | None = None,
          save_svg: Path | None = None, dot_size: float = 4.0,
          highlight_mask: np.ndarray | None = None,
          highlight_label: str = "NTC") -> None:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt

    uniq = pd.Series(color_labels).astype(str)
    keep = uniq if gray_labels is None else uniq.where(~uniq.isin(gray_labels), other=None)
    unique_vals = pd.unique(keep.dropna())
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(unique_vals), 1)))
    color_of = {v: cmap[i % len(cmap)] for i, v in enumerate(unique_vals)}

    n_axes = len(coords_by_name)
    fig, axes = plt.subplots(1, n_axes, figsize=(8 * n_axes, 8), squeeze=False)
    axes = axes.ravel()
    for ax, (name, coords) in zip(axes, coords_by_name.items()):
        gray_mask = uniq.isin(gray_labels) if gray_labels else pd.Series(False, index=uniq.index)
        gm = gray_mask.to_numpy()
        # Exclude NTC (rendered as overlay markers below) from the normal scatter,
        # so its color box doesn't wash out cluster colors underneath.
        if highlight_mask is not None:
            gm = gm & (~highlight_mask)
        if gm.any():
            ax.scatter(coords[gm, 0], coords[gm, 1], s=dot_size * 0.6,
                       c="lightgray", alpha=0.5, linewidths=0, rasterized=True)
        cm = ~gm
        if highlight_mask is not None:
            cm = cm & (~highlight_mask)
        colors = np.array([color_of.get(v, (0.7, 0.7, 0.7, 0.5)) for v in uniq[cm]])
        ax.scatter(coords[cm, 0], coords[cm, 1], s=dot_size, c=colors,
                   alpha=0.75, linewidths=0, rasterized=True)
        if highlight_mask is not None and highlight_mask.any():
            ax.scatter(coords[highlight_mask, 0], coords[highlight_mask, 1],
                       s=dot_size * 10, marker="x", c="#8B0000",
                       linewidths=2.2, alpha=0.95, zorder=10,
                       label=f"{highlight_label} (n={int(highlight_mask.sum())})",
                       rasterized=True)
            ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax.set_title(f"{name}  (n={len(coords):,})")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    if save_svg is not None:
        fig.savefig(save_svg, bbox_inches="tight")
    plt.close(fig)


def _fit_pca(X: np.ndarray, var: float, lg: logging.Logger) -> np.ndarray:
    """Fit PCA on the top-1k cell CellDINO features and keep enough PCs to
    hit `var` cumulative explained variance (matches pca_optimization's
    fixed-threshold logic)."""
    from sklearn.decomposition import PCA
    lg.info(f"PCA fit (var={var}): n={X.shape[0]:,} d={X.shape[1]}")
    t0 = time.time()
    p = PCA(n_components=min(X.shape) - 1, svd_solver="randomized",
            random_state=0)
    Xp = p.fit_transform(X)
    cum = np.cumsum(p.explained_variance_ratio_)
    n_pcs = int(np.searchsorted(cum, var) + 1)
    n_pcs = max(n_pcs, 10)
    lg.info(f"PCA: kept {n_pcs} PCs at cum={cum[n_pcs-1]:.4f} in {time.time()-t0:.0f}s")
    return Xp[:, :n_pcs]


def _dot_size_for(n: int) -> float:
    """Auto-scale dot size to cell count: 4pt at 200k, 20pt at 20k, capped."""
    return float(max(3, min(25, int(400_000 / max(n, 1)))))


def _run_embed_variant(X: np.ndarray, obs: pd.DataFrame, tag: str, out_dir: Path,
                       seed: int, n_landmark: int, skip_phate: bool,
                       lg: logging.Logger, use_gpu: bool = False) -> None:
    """Fit UMAP + PHATE on X, attach coords to obs with `tag` suffix, plot."""
    umap_xy = _fit_umap(X, seed, lg, use_gpu=use_gpu)
    obs[f"umap_x_{tag}"] = umap_xy[:, 0]; obs[f"umap_y_{tag}"] = umap_xy[:, 1]
    coords_by_name = {"UMAP": umap_xy}
    if not skip_phate:
        phate_xy = _fit_phate(X, seed, n_landmark, lg)
        obs[f"phate_x_{tag}"] = phate_xy[:, 0]; obs[f"phate_y_{tag}"] = phate_xy[:, 1]
        coords_by_name["PHATE"] = phate_xy
    ds = _dot_size_for(len(obs))
    lg.info(f"[{tag}] dot_size={ds} for n={len(obs):,} cells")
    ntc_mask = (obs["ebifb_gene"] == "NTC").to_numpy()
    _plot(coords_by_name, obs["ebifb_gene"].astype(str),
          out_dir / f"embedding_{tag}_gene.png",
          title=f"v5 ebifb top-1k / class [{tag}] — colored by gene-KO",
          dot_size=ds, highlight_mask=ntc_mask)
    _plot(coords_by_name, obs["ebi_complex"].astype(str),
          out_dir / f"embedding_{tag}_ebi.png",
          title=f"v5 ebifb top-1k / class [{tag}] — colored by EBI complex",
          gray_labels={"NA"},
          save_svg=out_dir / f"embedding_{tag}_ebi.svg",
          dot_size=ds, highlight_mask=ntc_mask)
    if "supercategory" in obs.columns:
        _plot(coords_by_name, obs["supercategory"].astype(str),
              out_dir / f"embedding_{tag}_super.png",
              title=f"v5 ebifb top-1k / class [{tag}] — colored by supercategory",
              gray_labels={"NA"},
              save_svg=out_dir / f"embedding_{tag}_super.svg",
              dot_size=ds, highlight_mask=ntc_mask)
    obs.to_parquet(out_dir / "cells.parquet")
    lg.info(f"[{tag}] plots + cells.parquet written")


def embed(out_dir: Path, seed: int, n_landmark: int, pca_var: float = 0.8,
          skip_phate: bool = False, skip_pca: bool = False,
          skip_raw: bool = False, use_gpu: bool = False,
          per_class_cap: int | None = None,
          ntc_normalize: bool = False) -> None:
    lg = _logger(out_dir, "embed")
    shard_dir = out_dir / "shards"
    if not shard_dir.exists():
        raise FileNotFoundError(f"no shards in {shard_dir}; run phase 1 first")
    X, obs = _load_shards(shard_dir, lg)
    X, obs = _dedup(X, obs, lg)

    if ntc_normalize:
        ntc_mask = (obs["ebifb_gene"] == "NTC").to_numpy()
        if ntc_mask.sum() < 10:
            raise RuntimeError(
                f"ntc_normalize=True but only {int(ntc_mask.sum())} NTC cells "
                "in shards — refusing to normalize on a noisy centroid"
            )
        # Z-score against NTC: subtract NTC mean, divide by NTC std.
        # Matches ops_utils.analysis.normalization.zscore_normalize(method='ntc').
        ntc_X = X[ntc_mask].astype(np.float64)
        ntc_mean = np.nanmean(ntc_X, axis=0)
        ntc_std = np.nanstd(ntc_X, axis=0, ddof=1)
        ntc_std[ntc_std == 0] = 1.0
        X = ((X.astype(np.float64) - ntc_mean[None, :]) / ntc_std[None, :]).astype(np.float32)
        lg.info(f"NTC z-scored: {int(ntc_mask.sum())} NTC cells "
                f"(mean range {float(ntc_mean.min()):.3f}..{float(ntc_mean.max()):.3f}, "
                f"std range {float(ntc_std.min()):.3f}..{float(ntc_std.max()):.3f})")

    if per_class_cap is not None and per_class_cap > 0:
        mask = (obs["ebifb_rank"] <= per_class_cap).to_numpy()
        X = X[mask]; obs = obs.loc[mask].reset_index(drop=True)
        lg.info(f"per_class_cap={per_class_cap}: kept {len(obs):,} cells")

    ebi_map = _load_ebi_map()
    obs["ebi_complex"] = obs["ebifb_gene"].astype(str).map(ebi_map).fillna("NA")
    try:
        from ops_model.post_process.combination.analysis.embedding_overlays import (
            load_overlay_maps,
        )
        super_map = load_overlay_maps().get("super") or {}
        obs["supercategory"] = obs["ebifb_gene"].astype(str).map(super_map).fillna("NA")
        lg.info(f"supercategory: {obs['supercategory'].nunique()} categories "
                f"({(obs['supercategory']!='NA').sum():,}/{len(obs):,} cells mapped)")
    except Exception as exc:
        lg.warning(f"supercategory load failed ({exc}); skipping super panel")

    X = X.astype(np.float32, copy=False)
    if not skip_raw:
        _run_embed_variant(X, obs, "raw", out_dir, seed, n_landmark, skip_phate, lg,
                           use_gpu=use_gpu)
    if not skip_pca:
        Xp = _fit_pca(X, pca_var, lg)
        _run_embed_variant(Xp, obs, "pca", out_dir, seed, n_landmark, skip_phate, lg,
                           use_gpu=use_gpu)

    obs.to_parquet(out_dir / "cells.parquet")
    lg.info(f"wrote {out_dir / 'cells.parquet'} ({len(obs):,} rows)")
    lg.info("done")


# ---------- CLI ------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="phase", required=True)

    g = sub.add_parser("gather", help="Filter one h5ad → shard files")
    g.add_argument("--h5ad", required=True)
    g.add_argument("--top-k", type=int, default=1000)
    g.add_argument("--shard-dir", required=True)

    e = sub.add_parser("embed", help="Concat shards + UMAP + PHATE + plots")
    e.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    e.add_argument("--n-landmark", type=int, default=2000)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--pca-var", type=float, default=0.8)
    e.add_argument("--skip-phate", action="store_true")
    e.add_argument("--skip-pca", action="store_true")
    e.add_argument("--skip-raw", action="store_true")
    e.add_argument("--use-gpu", action="store_true",
                   help="Use cuml UMAP (requires GPU node + cuml)")
    e.add_argument("--per-class-cap", type=int, default=None,
                   help="Downsample to top-K per class (e.g. 200 for a 5x faster run)")
    e.add_argument("--ntc-normalize", action="store_true",
                   help="Subtract NTC centroid before UMAP/PHATE")

    args = ap.parse_args()
    if args.phase == "gather":
        res = gather_shard(args.h5ad, args.top_k, args.shard_dir)
        print(res)
    else:
        embed(args.out_dir, args.seed, args.n_landmark, args.pca_var,
              args.skip_phate, args.skip_pca, args.skip_raw, args.use_gpu,
              args.per_class_cap, args.ntc_normalize)


if __name__ == "__main__":
    main()
