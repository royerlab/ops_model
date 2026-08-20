"""Regenerate ONLY the UMAP plots for existing PHATE peak-groups output
dirs, using the 'gav' UMAP recipe from pca_optimization (umap-learn
direct, n_neighbors=min(10, n-1), min_dist=0.25). Leaves PHATE plots
and positive_controls overlays untouched.

Reads saved guide.h5ad (from the earlier `phate_peak_groups.py` run),
rebuilds gene-level NTC-split + panel annotation, runs UMAP-gav on
guide + gene, then re-renders only the UMAP overlays via
`plot_embedding_overlay`.

Default groups match `phate_peak_groups.py`:
    python regen_umap_gav.py
or arbitrary groups (subdir names under --base-dir):
    python regen_umap_gav.py --groups top_K18000 random_low_removed_K20000
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_BASE_DIR = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v3/attention_v3/cdino/phate_peak_groups"
)
DEFAULT_GROUPS = ["top_K18000", "random_low_removed_K20000"]


def _umap_gav(X: np.ndarray, n_obs: int, random_seed: int = 42):
    """The 'gav' UMAP recipe (legacy umap-learn direct) — see
    pca_optimization.embeddings._make_embedder.
    """
    from umap import UMAP
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
        "umap_type": "gav",
    }
    return coords, params


def regen_group(group_dir: Path, random_seed: int = 42):
    """Regenerate UMAP-gav for one (group_dir/guide.h5ad + gene.h5ad)."""
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

    plots_dir = group_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{group_dir.name}] loading guide.h5ad…")
    adata_guide = ad.read_h5ad(group_dir / "guide.h5ad")

    # Rebuild gene-level adata (same as _compute_and_plot_embeddings does).
    adata_gene_embed = split_ntc_for_embedding(adata_guide, random_seed=random_seed)
    if "perturbation" in adata_gene_embed.obs.columns:
        adata_gene_embed.obs_names = (
            adata_gene_embed.obs["perturbation"].astype(str).values
        )
        adata_gene_embed.obs_names_make_unique()
    _annotate_genes_from_panel(adata_gene_embed, logger)
    adata_gene_embed.obsm["X_pca"] = np.asarray(adata_gene_embed.X, dtype=np.float32)
    if "pca" in adata_guide.uns:
        adata_gene_embed.uns["pca"] = adata_guide.uns["pca"]

    # Pass-through: no per-pert metric overlay (color falls back to EBI
    # complex membership from the panel annotation).
    metric_lookup: dict = {}

    for level_name, adata_level in [("guide", adata_guide),
                                       ("gene",  adata_gene_embed)]:
        logger.info(
            f"[{group_dir.name}] {level_name} UMAP-gav: "
            f"{adata_level.n_obs} obs, {adata_level.n_vars} features"
        )
        X_clean = clean_X_for_embedding(adata_level)
        coords, params = _umap_gav(X_clean, adata_level.n_obs, random_seed)
        if coords is None:
            logger.warning(f"  skipped (too few observations)")
            continue
        # Overwrite obsm/uns with the new gav UMAP.
        adata_level.obsm["X_umap"] = coords.astype(np.float32)
        adata_level.uns["umap"] = {"params": params}
        perts = get_perts_col(adata_level)
        fname = plot_embedding_overlay(
            coords, perts, metric_lookup, level_name, "UMAP",
            plots_dir, adata_level.n_obs, adata_level.n_vars, plt,
        )
        logger.info(f"  wrote plots/{fname}")
        # Save coords CSV.
        df = pd.DataFrame(coords, columns=["UMAP1", "UMAP2"])
        df.insert(0, "perturbation",
                    perts.values if hasattr(perts, "values") else perts)
        csv_name = f"{level_name}_umap_coords.csv"
        df.to_csv(plots_dir / csv_name, index=False)
        logger.info(f"  wrote plots/{csv_name}")


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR,
                    help="Parent dir containing per-group subdirs.")
    ap.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS,
                    help="Per-group subdir names (e.g. top_K18000).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for name in args.groups:
        group_dir = args.base_dir / name
        if not (group_dir / "guide.h5ad").exists():
            logger.warning(f"skipping {name}: no guide.h5ad in {group_dir}")
            continue
        regen_group(group_dir, random_seed=args.seed)


if __name__ == "__main__":
    main()
