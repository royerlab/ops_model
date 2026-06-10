"""Embedding + scoring helpers for pca_optimization.

* ``_compute_and_plot_embeddings`` — UMAP + PHATE for guide/gene levels,
  plus the metric overlays, positive-controls grid, and (optionally)
  chromosome-arm colored gene-level scatter.
* ``_score_distinctiveness`` / ``_score_consistency`` — final scoring of
  the aggregated guide/gene h5ads at the end of Phase 2.

CHAD/EBI annotation paths are imported lazily inside ``_score_consistency``
so the mutable module globals (set by ``main()`` from
``--chad-annotation`` / ``--ebi-annotation``) are picked up at call
time.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ops_model.features.anndata_utils import split_ntc_for_embedding
from ops_utils.analysis.embedding_plots import (
    clean_X_for_embedding,
    get_perts_col,
    plot_embedding_overlay,
)
from ops_utils.analysis.map_scores import plot_map_scatter
from ops_utils.data.positive_controls import plot_positive_controls_grid

from ops_model.post_process.combination.pca_optimization.aggregation import (
    _annotate_genes_from_panel,
)
from ops_model.post_process.combination.pipeline_add_ons.chromosome import (
    _load_chromosome_map,
    _plot_chromosome_overlay,
    _plot_chromosome_overlay_html,
)


def _compute_and_plot_embeddings(
    adata_guide,
    metric_lookup,
    plots_dir,
    plt,
    _logger,
    random_seed: int = 42,
    chromosome_csv: Optional[str] = None,
    umap_type: str = "max",
):
    """Compute UMAP + PHATE embeddings for guide/gene levels, plot overlays + positive controls.

    Returns adata_gene_embed with embeddings stored in obsm — caller should save it.
    The same ``random_seed`` is threaded into ``split_ntc_for_embedding``, UMAP,
    and PHATE so a given seed deterministically reproduces the same embedding.

    ``chromosome_csv``: optional CSV (columns include ``perturbation``,
    ``chromosome``, ``chromosome_arm``). When provided, also writes a
    chromosome-arm-colored gene-level scatter for each embedding.

    ``umap_type``: which UMAP recipe to use.

      * ``"max"`` (default, Max's settings): scanpy-driven
        ``sc.pp.neighbors(n_neighbors=8, use_rep="X_pca")`` followed by
        ``sc.tl.umap(min_dist=0.25, alpha=1.0, gamma=1.5, maxiter=2000,
        init_pos=X_pca[:, :2])``. PCA-anchored initialization gives a more
        stable, biology-aware layout that converges quickly.
      * ``"gav"`` (legacy): umap-learn ``UMAP(n_neighbors=min(10, n-1),
        min_dist=0.25, random_state=seed)`` fit directly on the feature
        matrix with default spectral init.

    Both write the same ``obsm["X_umap"]`` and ``uns["umap"]["params"]``;
    the params dict records the chosen ``umap_type`` so downstream
    consumers can tell which one produced the layout.
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
        """Return ``fit_fn(X, n_obs, level, adata)`` → ``(coords, params)``,
        or ``(None, {})`` if the inputs can't be embedded. The ``adata``
        argument is only used by the ``umap_type="max"`` UMAP recipe which
        needs ``obsm["X_pca"]`` for its PCA-anchored init.
        """
        if name == "UMAP":
            if umap_type == "max":
                import scanpy as sc

                def _fit(X, n_obs, level, adata=None):
                    nn = min(8, max(n_obs - 1, 2))
                    if nn < 2:
                        return None, {}
                    if adata is None or "X_pca" not in adata.obsm:
                        _logger.warning(
                            f"  umap_type='max' needs adata.obsm['X_pca']; "
                            f"falling back to umap-learn for {level}"
                        )
                        from umap import UMAP as _UMAP
                        model = _UMAP(n_components=2, n_neighbors=nn,
                                      min_dist=0.25, random_state=random_seed)
                        return model.fit_transform(X), {
                            "n_neighbors": nn, "min_dist": 0.25,
                            "random_state": random_seed, "umap_type": "max",
                            "fallback": "no_X_pca",
                        }
                    # Scanpy path — see Max's recipe.
                    adata_tmp = adata.copy()
                    init_pos = adata_tmp.obsm["X_pca"][:, :2].copy()
                    sc.pp.neighbors(adata_tmp, n_neighbors=nn, use_rep="X_pca")
                    sc.tl.umap(
                        adata_tmp,
                        min_dist=0.25,
                        random_state=random_seed,
                        alpha=1.0, gamma=1.5, maxiter=2000,
                        init_pos=init_pos,
                    )
                    coords = adata_tmp.obsm["X_umap"]
                    return coords, {
                        "n_neighbors": int(nn), "min_dist": 0.25,
                        "random_state": int(random_seed),
                        "alpha": 1.0, "gamma": 1.5, "maxiter": 2000,
                        "init_pos": "X_pca[:, :2]",
                        "umap_type": "max",
                    }

                return _fit

            # umap_type == "gav" — legacy umap-learn-direct path.
            from umap import UMAP

            def _fit(X, n_obs, level, adata=None):
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

            def _fit(X, n_obs, level, adata=None):
                # GRASSP-canonical PHATE: knn=8, decay=10. For tiny levels
                # (n_obs <= 8) we fall back to (n_obs - 1) so PHATE still fits.
                knn = min(8, n_obs - 1)
                if knn < 2:
                    return None, {}
                decay = 10
                coords = phate.PHATE(
                    n_components=2,
                    knn=knn,
                    decay=decay,
                    t="auto",
                    n_jobs=-1,
                    random_state=random_seed,
                    verbose=0,
                ).fit_transform(X)
                params = {"knn": knn, "decay": decay, "t": "auto", "random_state": random_seed}
                return coords, params

            return _fit

    # Load chromosome map once (None if not requested or file is unreadable)
    chrom_df = (
        _load_chromosome_map(chromosome_csv, _logger) if chromosome_csv else None
    )

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
                coords, embed_params = fit_fn(X_clean, adata_level.n_obs, level_name, adata_level)
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
                # Chromosome-arm overlay (gene level only — guide level has 4×
                # the points and the same chr_arm per perturbation, so the
                # plot would just be a denser copy).
                if chrom_df is not None and level_name == "gene":
                    try:
                        chrom_stem = plots_dir / f"{level_name}_{embed_name.lower()}_chromosome"
                        _plot_chromosome_overlay(
                            coords, perts, chrom_df, embed_name,
                            chrom_stem, plt, _logger,
                        )
                        _plot_chromosome_overlay_html(
                            coords, perts, chrom_df, embed_name,
                            chrom_stem, _logger,
                        )
                    except Exception as chr_err:
                        _logger.warning(
                            f"  Chromosome overlay ({embed_name}) failed: {chr_err}"
                        )
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
    """Run CORUM + CHAD + EBI consistency scoring, save CSVs and plots.

    Returns ``(corum_map, corum_ratio, chad_map, chad_ratio,
    ebi_map, ebi_ratio)`` — six values now that EBI is a permanent third
    consistency metric. Failure mode is six zeros so callers can keep
    unpacking with one shape.

    NOTE: ``phenotypic_consistency_*`` is called WITHOUT ``activity_map``, so
    consistency is computed over all genes regardless of the ``suffix``
    argument (which only affects output filenames).
    """
    from ops_model.post_process.combination.pca_optimization import (
        CHAD_ANNOTATION_PATH,
        EBI_ANNOTATION_PATH,
    )

    label = "all geneKOs"
    if activity_map is None:
        return None, 0.0, None, 0.0, None, 0.0
    try:
        from ops_utils.analysis.map_scores import (
            phenotypic_consistency_corum,
            phenotypic_consistency_ebi,
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

        _logger.info(f"Running EBI consistency ({label})...")
        consistency_ebi_map, consistency_ebi_ratio = phenotypic_consistency_ebi(
            adata_gene,
            plot_results=False,
            null_size=100_000,
            cache_similarity=True,
            distance=distance,
            annotation_path=EBI_ANNOTATION_PATH,
        )
        consistency_ebi_map.to_csv(
            metrics_dir / f"phenotypic_consistency_ebi{suffix}.csv", index=False
        )
        _logger.info(f"  EBI ({label}): {consistency_ebi_ratio:.1%}")

        # 1×3 panel: CORUM + CHAD + EBI scatter (existing style)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7))
        plot_map_scatter(ax1, consistency_corum_map,
                         f"Consistency CORUM ({label})",
                         consistency_corum_ratio, show_ntc=False)
        plot_map_scatter(ax2, consistency_manual_map,
                         f"Consistency CHAD ({label})",
                         consistency_manual_ratio, show_ntc=False)
        plot_map_scatter(ax3, consistency_ebi_map,
                         f"Consistency EBI ({label})",
                         consistency_ebi_ratio, show_ntc=False)
        fig.suptitle(
            f"Consistency Metrics ({label}) — {total_feats} features",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(
            plots_dir / f"map_consistency{suffix}.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)
        _logger.info(f"  Saved plots/map_consistency{suffix}.png")

        # Standalone EBI panel using the canonical map-scatter helper —
        # same style as the activity / distinctiveness mAP scatters.
        try:
            fig, ax = plt.subplots(figsize=(8, 7))
            plot_map_scatter(
                ax, consistency_ebi_map,
                f"Consistency EBI ({label})",
                consistency_ebi_ratio, show_ntc=False,
            )
            fig.tight_layout()
            fig.savefig(plots_dir / f"map_ebi_volcano{suffix}.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
            _logger.info(f"  Saved plots/map_ebi_volcano{suffix}.png")
        except Exception as exc:
            _logger.warning(f"  EBI volcano plot failed: {exc}")

        return (
            consistency_corum_map, consistency_corum_ratio,
            consistency_manual_map, consistency_manual_ratio,
            consistency_ebi_map, consistency_ebi_ratio,
        )
    except Exception as exc:
        _logger.error(f"  Consistency metrics ({label}) failed: {exc}")
        return None, 0.0, None, 0.0, None, 0.0
