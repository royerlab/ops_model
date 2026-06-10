"""Regenerate UMAP with custom params + emit interactive Plotly HTMLs.

For each peak-group dir (e.g. top_K18000, random_low_removed_K20000):
  1. Re-fit UMAP at user-chosen (n_neighbors, min_dist) — overwrites
     the existing obsm["X_umap"] in guide.h5ad + gene.h5ad.
  2. Keep existing obsm["X_phate"] (already knn=8/decay=10 from the
     original PHATE run; no need to re-fit).
  3. Re-emit static UMAP overlays (PNG via plot_embedding_overlay).
  4. Emit Plotly interactive HTMLs via
     `embedding_overlays.save_interactive_html` — one per (level,
     embedding) with hover metadata + dropdown for super/CHAD/CORUM
     overlay coloring.

Default params (`--n-neighbors 15 --min-dist 0.5`) ≠ the 'gav' default
(0.25); use `--n-neighbors`/`--min-dist` to switch.

Run:
    python organelle_profiler/scripts/ko_shap/regen_umap_html.py
    # custom params:
    python organelle_profiler/scripts/ko_shap/regen_umap_html.py \\
        --n-neighbors 25 --min-dist 0.1
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_BASE_DIR = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v3/attention_v3/cdino/phate_peak_groups"
)
DEFAULT_GROUPS = ["top_K18000", "random_low_removed_K20000"]


def _fit_umap(X: np.ndarray, n_obs: int, *,
               n_neighbors: int, min_dist: float, seed: int = 42):
    from umap import UMAP
    nn = min(n_neighbors, n_obs - 1)
    if nn < 2:
        return None, {}
    model = UMAP(
        n_components=2, n_neighbors=nn, min_dist=float(min_dist),
        random_state=seed,
    )
    coords = model.fit_transform(X)
    params = {
        "n_neighbors": int(nn),
        "min_dist":    float(min_dist),
        "random_state": int(seed),
        "metric":      "euclidean",
        "umap_type":   "custom",
    }
    return coords, params


def _fit_phate(X: np.ndarray, n_obs: int, *,
                knn: int, decay: float, seed: int = 42):
    """Refit PHATE at user-chosen params. Same defaults shape as the
    pca_optimization recipe (`t="auto"`, `n_jobs=-1`)."""
    import phate
    knn_eff = min(knn, n_obs - 1)
    if knn_eff < 2:
        return None, {}
    coords = phate.PHATE(
        n_components=2, knn=knn_eff, decay=float(decay), t="auto",
        n_jobs=-1, random_state=seed, verbose=0,
    ).fit_transform(X)
    params = {
        "knn":          int(knn_eff),
        "decay":        float(decay),
        "t":            "auto",
        "random_state": int(seed),
    }
    return coords, params


def process_group(group_dir: Path,
                   n_neighbors: int, min_dist: float,
                   phate_knn: int, phate_decay: float,
                   refit_phate: bool,
                   seed: int = 42):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ops_model.features.anndata_utils import split_ntc_for_embedding
    from ops_utils.analysis.embedding_plots import (
        clean_X_for_embedding,
        get_perts_col,
        plot_embedding_overlay,
    )
    from ops_model.post_process.combination.pca_optimization.aggregation import (
        _annotate_genes_from_panel,
    )
    from ops_model.post_process.combination.embedding_overlays import (
        load_overlay_maps, save_interactive_html,
    )

    plots_dir = group_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{group_dir.name}] loading guide.h5ad…")
    guide = ad.read_h5ad(group_dir / "guide.h5ad")
    has_phate = "X_phate" in guide.obsm
    logger.info(
        f"  guide.h5ad: {guide.n_obs} guides × {guide.n_vars} feats, "
        f"existing obsm: {list(guide.obsm.keys())}"
    )

    # Rebuild gene-level adata (same as _compute_and_plot_embeddings).
    gene = split_ntc_for_embedding(guide, random_seed=seed)
    if "perturbation" in gene.obs.columns:
        gene.obs_names = gene.obs["perturbation"].astype(str).values
        gene.obs_names_make_unique()
    _annotate_genes_from_panel(gene, logger)
    gene.obsm["X_pca"] = np.asarray(gene.X, dtype=np.float32)
    if "pca" in guide.uns:
        gene.uns["pca"] = guide.uns["pca"]

    # Copy X_phate from guide → gene level via the standard pipeline?
    # No — gene-level phate is independent. We need PHATE at gene level
    # too. Reuse the existing gene.h5ad's X_phate if present.
    gene_existing_p = group_dir / "gene.h5ad"
    if gene_existing_p.exists():
        try:
            gene_existing = ad.read_h5ad(gene_existing_p, backed="r")
            if "X_phate" in gene_existing.obsm:
                # The existing gene.h5ad may have a different obs order;
                # match by perturbation index.
                old_phate = pd.DataFrame(
                    np.asarray(gene_existing.obsm["X_phate"]),
                    index=gene_existing.obs.index,
                    columns=["x", "y"],
                )
                # Reindex to current gene obs order; missing rows → NaN.
                aligned = old_phate.reindex(gene.obs.index)
                gene.obsm["X_phate"] = aligned.values.astype(np.float32)
                logger.info(
                    f"  carried over existing gene-level X_phate "
                    f"({aligned.notna().all(axis=1).sum()}/{len(aligned)} aligned)"
                )
        except Exception as e:
            logger.warning(f"  couldn't carry gene X_phate: {e}")

    # Re-fit UMAP (+ optionally PHATE) at chosen params for BOTH levels.
    metric_lookup: dict = {}
    for level_name, ad_level in [("guide", guide), ("gene", gene)]:
        X = clean_X_for_embedding(ad_level)
        # UMAP
        logger.info(
            f"[{group_dir.name}] {level_name} UMAP "
            f"(n_neighbors={n_neighbors}, min_dist={min_dist}): "
            f"{ad_level.n_obs} obs × {ad_level.n_vars} features"
        )
        coords, params = _fit_umap(
            X, ad_level.n_obs,
            n_neighbors=n_neighbors, min_dist=min_dist, seed=seed,
        )
        if coords is not None:
            ad_level.obsm["X_umap"] = coords.astype(np.float32)
            ad_level.uns["umap"] = {"params": params}
            perts = get_perts_col(ad_level)
            fname = plot_embedding_overlay(
                coords, perts, metric_lookup, level_name, "UMAP",
                plots_dir, ad_level.n_obs, ad_level.n_vars, plt,
            )
            logger.info(f"  wrote plots/{fname}")
            df = pd.DataFrame(coords, columns=["UMAP1", "UMAP2"])
            df.insert(0, "perturbation",
                       perts.values if hasattr(perts, "values") else perts)
            df.to_csv(plots_dir / f"{level_name}_umap_coords.csv", index=False)
        # PHATE refit (when requested)
        if refit_phate:
            logger.info(
                f"[{group_dir.name}] {level_name} PHATE "
                f"(knn={phate_knn}, decay={phate_decay}): "
                f"{ad_level.n_obs} obs"
            )
            pcoords, pparams = _fit_phate(
                X, ad_level.n_obs,
                knn=phate_knn, decay=phate_decay, seed=seed,
            )
            if pcoords is not None:
                ad_level.obsm["X_phate"] = pcoords.astype(np.float32)
                ad_level.uns["phate"] = {"params": pparams}
                perts = get_perts_col(ad_level)
                fname = plot_embedding_overlay(
                    pcoords, perts, metric_lookup, level_name, "PHATE",
                    plots_dir, ad_level.n_obs, ad_level.n_vars, plt,
                )
                logger.info(f"  wrote plots/{fname}")
                df = pd.DataFrame(pcoords, columns=["PHATE1", "PHATE2"])
                df.insert(0, "perturbation",
                           perts.values if hasattr(perts, "values") else perts)
                df.to_csv(plots_dir / f"{level_name}_phate_coords.csv",
                            index=False)

    # Plotly interactive HTMLs (one per level × embedding present).
    # Loaded overlay maps stay constant across groups — load once and pass.
    logger.info(f"[{group_dir.name}] loading overlay maps (super/CHAD/CORUM)…")
    overlay_maps = load_overlay_maps()
    logger.info(
        f"  overlay sizes: super={len(overlay_maps['super'])}, "
        f"chad={len(overlay_maps['chad'])}, "
        f"corum={len(overlay_maps['corum'])}"
    )
    save_interactive_html(
        adata_guide=guide,
        adata_gene_embed=gene,
        overlay_maps=overlay_maps,
        leiden_results={},
        activity_map=None,
        dist_map=None,
        corum_map=None,
        chad_map=None,
        plots_dir=plots_dir,
        _logger=logger,
    )


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    ap.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS)
    ap.add_argument("--n-neighbors", type=int, default=15,
                    help="UMAP n_neighbors")
    ap.add_argument("--min-dist", type=float, default=0.5,
                    help="UMAP min_dist")
    ap.add_argument("--phate-knn", type=int, default=8,
                    help="PHATE knn (default: 8, GRASSP-canonical)")
    ap.add_argument("--phate-decay", type=float, default=10.0,
                    help="PHATE decay (default: 10, GRASSP-canonical)")
    ap.add_argument("--refit-phate", action="store_true",
                    help="Re-fit PHATE at --phate-knn/--phate-decay. "
                         "Otherwise keeps the existing X_phate (faster).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for name in args.groups:
        gd = args.base_dir / name
        if not (gd / "guide.h5ad").exists():
            logger.warning(f"skipping {name}: no guide.h5ad in {gd}")
            continue
        process_group(
            gd,
            n_neighbors=args.n_neighbors, min_dist=args.min_dist,
            phate_knn=args.phate_knn, phate_decay=args.phate_decay,
            refit_phate=args.refit_phate,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
