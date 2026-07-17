"""CLI mode handlers for pca_optimization.

One function per ``main()`` dispatch branch — each implements a
mutually-exclusive CLI mode flag (``--chad-umap-only``, ``--umap-only``,
``--overlays-only``, ``--sweep-seed``, ``--aggregate-only``,
``--second-pca``-only paths, ``--organelle-profiler``, and the default
``--downsampled`` flow).

These functions are the call-site surface area for orchestrating Phase 1
+ Phase 2 (delegating to ``phase1.pca_sweep_pooled_signal`` /
``phase2.aggregate_channels`` / ``phase2.apply_second_pass_pca`` via
either an in-process call or a SLURM submission helper).

Module globals from ``pca_optimization`` (CHAD_ANNOTATION_PATH,
DEFAULT_SWEEP_THRESHOLDS_CP, PCA_FIT_CAP) are imported lazily inside
the handlers that read them so the parent and this module can
re-import each other at load time without cycling.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import anndata as ad
import numpy as np
import pandas as pd

from ops_model.features.anndata_utils import aggregate_to_level
from ops_utils.analysis.embedding_plots import build_metric_lookup, get_perts_col
from ops_utils.data.feature_discovery import (
    build_signal_groups,
    count_cells_per_signal_group,
    discover_cellprofiler_experiments,
    discover_dino_experiments,
    find_cell_h5ad_path,
    get_channel_maps_path,
    get_storage_roots,
    load_attribution_config,
    sanitize_signal_filename,
)

from ops_model.post_process.combination.pca_optimization.aggregation import (
    _atomic_write_h5ad,
    _plot_chad_umap,
)
from ops_model.post_process.combination.pipeline_add_ons.chromosome import (
    _load_chromosome_map,
    _plot_chromosome_overlay,
    _plot_chromosome_overlay_html,
)
from ops_model.post_process.combination.pca_optimization.embeddings import (
    _compute_and_plot_embeddings,
)
from ops_model.post_process.combination.pipeline_add_ons.op_signal import (
    _discover_op_files,
    pca_sweep_op_signal,
)
from ops_model.post_process.combination.pca_optimization.phase1 import (
    pca_sweep_pooled_signal,
)
from ops_model.post_process.combination.pca_optimization.phase2 import (
    aggregate_channels,
    apply_second_pass_pca,
)
from ops_model.post_process.combination.pca_optimization.slurm import (
    _build_second_pca_kwargs,
    _make_agg_slurm_params,
    _make_slurm_params,
    _submit_aggregation_slurm,
    _submit_phase1_slurm,
)


def _discover_experiment_pairs(
    cp_override,
    include_cellpainting: bool = False,
    include_4i: bool = False,
    include_cp: bool = False,
    include_standard: bool = True,
    force_include: Optional[set] = None,
):
    """Common experiment discovery for SLURM modes. Returns (all_pairs, attr_config, storage_roots, feature_dir, maps_path).

    ``force_include`` (set of exp short names like ``{"ops0146"}``) bypasses
    discovery's bad-experiment / non-default-library filters for the listed
    experiments — used when the caller explicitly asked for them via
    ``--experiments``.
    """
    attr_config = load_attribution_config()
    storage_roots = get_storage_roots(attr_config)
    feature_dir = cp_override or attr_config.get("feature_dir", "dino_features")
    maps_path = get_channel_maps_path()
    if cp_override == "cell-profiler":
        all_pairs = discover_cellprofiler_experiments(
            storage_roots,
            include_cellpainting=include_cellpainting,
            include_4i=include_4i,
            include_cp=include_cp,
            include_standard=include_standard,
            force_include=force_include,
        )
    else:
        all_pairs = discover_dino_experiments(
            storage_roots,
            feature_dir,
            include_cellpainting=include_cellpainting,
            include_4i=include_4i,
            include_cp=include_cp,
            include_standard=include_standard,
            force_include=force_include,
        )
    # 4i nucleus (DAPI) channels are nucleus segmentation refs, not biological
    # signals — drop them from the pool. CP DAPI stays since CP uses it as a
    # bona-fide channel in the multiplex panel.
    if include_4i:
        before = len(all_pairs)
        dropped_4i_dapi = [e for e, c in all_pairs if c == "4i:DAPI"]
        all_pairs = [(e, c) for e, c in all_pairs if c != "4i:DAPI"]
        if dropped_4i_dapi:
            print(
                f"  WARNING: dropping 4i:DAPI nucleus channel — it's a "
                f"segmentation ref, not a biological signal "
                f"({len(dropped_4i_dapi)} pairs across "
                f"{len(set(dropped_4i_dapi))} experiments). "
                f"CP DAPI is kept."
            )
    return all_pairs, attr_config, storage_roots, feature_dir, maps_path


def _handle_chad_umap_only(args, output_dir):
    """Only regenerate the CHAD-colored UMAP from existing gene embeddings."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import yaml as _yaml

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _logger = logging.getLogger(__name__)

    embed_path = output_dir / "gene_embedding_pca_optimized.h5ad"
    if not embed_path.exists():
        print(f"ERROR: {embed_path} not found. Run --aggregate-only first.")
        return

    _logger.info(f"Loading {embed_path}...")
    adata = ad.read_h5ad(embed_path)
    if "X_umap" not in adata.obsm:
        print("ERROR: No X_umap in gene embedding.")
        return

    chad_path = args.chad_annotation or "/hpc/projects/icd.fast.ops/configs/gene_clusters/chad_positive_controls_v5_hierarchy.yml"
    with open(chad_path) as f:
        chad_clusters = _yaml.safe_load(f)

    # Optional cluster range filter
    if args.chad_cluster_range:
        lo, hi = map(int, args.chad_cluster_range.split("-"))
        chad_clusters = {k: v for k, v in chad_clusters.items() if isinstance(k, int) and lo <= k <= hi}
        _logger.info(f"Filtered to clusters {lo}-{hi} ({len(chad_clusters)} clusters)")

    gene_to_cluster = {}
    for cid, cdata in chad_clusters.items():
        name = cdata.get("name", f"cluster_{cid}")
        for gene in cdata.get("genes", []):
            gene_to_cluster[gene.strip()] = name

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.chad_umap_output or "umap_chad_clusters.png"
    out_path = plots_dir / out_name

    _plot_chad_umap(
        adata.obsm["X_umap"],
        adata.obs["perturbation"].values,
        gene_to_cluster,
        out_path,
        plt, _logger,
    )


def _handle_umap_only(args, output_dir):
    """Generate UMAP + PHATE embedding plots from existing optimized h5ads."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    _logger = logging.getLogger(__name__)

    umap_dir = output_dir
    guide_path = umap_dir / "guide_pca_optimized.h5ad"
    if not guide_path.exists():
        print(f"ERROR: {guide_path} not found. Run --aggregate-only first.")
        return

    _logger.info(f"Loading {guide_path}...")
    adata_guide = ad.read_h5ad(guide_path)

    # Load activity metrics for coloring
    activity_csv = umap_dir / "metrics" / "phenotypic_activity.csv"
    if activity_csv.exists():
        activity_map = pd.read_csv(activity_csv)
        metric_lookup = build_metric_lookup(activity_map)
        _logger.info(
            f"  Loaded activity metrics for {len(metric_lookup)} perturbations"
        )
    else:
        metric_lookup = {}
        _logger.warning(
            f"  No activity CSV found at {activity_csv}, UMAPs will be uncolored"
        )

    plots_dir = umap_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _compute_and_plot_embeddings(
        adata_guide, metric_lookup, plots_dir, plt, _logger,
        umap_type=getattr(args, "umap_type", "max"),
    )
    print("SUCCESS: Embedding plots saved")


def _stored_embedding_seed(adata, embed_name: str):
    """Pull the random_state used when an embedding (umap/phate) was last fit.
    Returns None if absent."""
    rec = (adata.uns.get(embed_name) or {}).get("params") or {}
    val = rec.get("random_state")
    try:
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _try_load_swept_umap(
    out_dir: Path, seed: int, _logger, nn: int = 15, md: float = 0.1,
):
    """Look for ``gene_umap_seed_sweep_coords_nn{nn}_md{md}.npz`` in
    ``out_dir`` and return the coords for ``seed`` if present. Otherwise
    return None. The cache is written by ``_run_seed_sweep`` and is the
    authoritative source for layouts users selected from the sweep canvas."""
    cache_path = Path(out_dir) / f"gene_umap_seed_sweep_coords_nn{nn}_md{md:g}.npz"
    if not cache_path.exists():
        return None
    try:
        z = np.load(cache_path)
        seeds = z["seeds"]
        coords_all = z["coords"]
        idx = np.where(seeds == int(seed))[0]
        if idx.size == 0:
            _logger.info(
                "  Sweep cache %s lacks seed=%d — falling back to refit",
                cache_path.name, seed,
            )
            return None
        _logger.info(
            "  Loaded gene UMAP coords for seed=%d from %s "
            "(matches --sweep-seed exactly)", seed, cache_path.name,
        )
        return np.asarray(coords_all[idx[0]], dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        _logger.warning("  Sweep cache unreadable (%s) — falling back to refit", exc)
        return None


def _recompute_embeddings_for_seed(
    adata,
    level_name: str,
    seed: int,
    _logger,
    out_dir=None,
    umap_n_neighbors: Optional[int] = None,
    umap_min_dist: Optional[float] = None,
    umap_type: str = "max",
):
    """Refit UMAP + PHATE in-place using ``seed``. Skips embeddings whose
    stored params already match. Returns the set of embedding names that
    were actually refit (so the caller knows whether to rewrite the h5ad).

    ``umap_n_neighbors`` / ``umap_min_dist`` are optional overrides — when
    set they bypass the sweep cache (the cache is keyed on default
    n_neighbors=15, min_dist=0.1) and force a refit at the requested params.

    ``umap_type``: 'max' (default) → scanpy with PCA-anchored init;
    'gav' → legacy umap-learn directly on the feature matrix.
    """
    refit: set = set()
    n_obs = adata.n_obs
    # Honor the umap_type defaults when the caller didn't pass overrides.
    if umap_n_neighbors is not None:
        nn = int(umap_n_neighbors)
    elif umap_type == "max":
        nn = min(8, n_obs - 1)
    else:
        nn = min(15, n_obs - 1)
    nn = min(nn, n_obs - 1)
    if nn < 2:
        _logger.warning(
            "  %s: too few obs (%d) — skipping embedding recompute", level_name, n_obs,
        )
        return refit
    if umap_min_dist is not None:
        md = float(umap_min_dist)
    elif umap_type == "max":
        md = 0.25
    else:
        md = 0.1

    if "X_pca" in adata.obsm:
        X = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
    else:
        X = np.asarray(adata.X, dtype=np.float32)

    stored_params = ((adata.uns.get("umap") or {}).get("params") or {})
    stored_seed = _stored_embedding_seed(adata, "umap")
    stored_nn = stored_params.get("n_neighbors")
    stored_md = stored_params.get("min_dist")
    stored_type = stored_params.get("umap_type")

    # Look up the cache for the (nn, md) the user requested — sweep writes
    # one cache per param combo, so there's a hit if and only if a sweep was
    # run at these same params.
    cache_coords = None
    if level_name == "gene" and out_dir is not None:
        cache_coords = _try_load_swept_umap(out_dir, int(seed), _logger, nn=nn, md=md)

    needs_umap = (
        cache_coords is not None
        or stored_seed != int(seed)
        or stored_type != umap_type
        or (umap_n_neighbors is not None and stored_nn != nn)
        or (umap_min_dist is not None and stored_md != md)
    )
    if needs_umap:
        coords = cache_coords
        if coords is None:
            if umap_type == "max" and "X_pca" in adata.obsm:
                _logger.info(
                    "  Refitting %s UMAP via scanpy (umap_type=max): "
                    "sc.pp.neighbors(n_neighbors=%d, use_rep='X_pca'); "
                    "sc.tl.umap(min_dist=%g, random_state=%d, alpha=1.0, "
                    "gamma=1.5, maxiter=2000, init_pos=X_pca[:, :2])  "
                    "(stored: seed=%s, nn=%s, md=%s, type=%s)",
                    level_name, nn, md, int(seed),
                    stored_seed, stored_nn, stored_md, stored_type,
                )
                try:
                    import scanpy as sc
                    adata_tmp = adata.copy()
                    init_pos = adata_tmp.obsm["X_pca"][:, :2].copy()
                    sc.pp.neighbors(adata_tmp, n_neighbors=nn, use_rep="X_pca")
                    sc.tl.umap(
                        adata_tmp, min_dist=md, random_state=int(seed),
                        alpha=1.0, gamma=1.5, maxiter=2000, init_pos=init_pos,
                    )
                    coords = adata_tmp.obsm["X_umap"]
                except Exception as exc:  # noqa: BLE001
                    _logger.warning("  %s scanpy UMAP refit failed: %s", level_name, exc)
                    coords = None
            else:
                _logger.info(
                    "  Refitting %s UMAP via umap-learn (umap_type=%s): "
                    "UMAP(n_neighbors=%d, min_dist=%g, random_state=%d)  "
                    "(stored: seed=%s, nn=%s, md=%s, type=%s)",
                    level_name, umap_type, nn, md, int(seed),
                    stored_seed, stored_nn, stored_md, stored_type,
                )
                from umap import UMAP
                try:
                    model = UMAP(
                        n_components=2, n_neighbors=nn, min_dist=md,
                        random_state=int(seed),
                    )
                    coords = model.fit_transform(X)
                except Exception as exc:  # noqa: BLE001
                    _logger.warning("  %s UMAP refit failed: %s", level_name, exc)
                    coords = None

        if coords is not None:
            adata.obsm["X_umap"] = np.asarray(coords, dtype=np.float32)
            params_dict = {
                "n_neighbors": nn,
                "min_dist": md,
                "random_state": int(seed),
                "metric": "euclidean",
                "umap_type": umap_type,
            }
            if umap_type == "max":
                params_dict.update({
                    "alpha": 1.0, "gamma": 1.5, "maxiter": 2000,
                    "init_pos": "X_pca[:, :2]",
                })
            adata.uns["umap"] = {"params": params_dict}
            refit.add("umap")

    if _stored_embedding_seed(adata, "phate") != int(seed):
        try:
            import phate

            knn = min(15 if n_obs > 2000 else 10, n_obs - 1)
            if knn >= 2:
                _logger.info(
                    "  Recomputing %s PHATE at seed=%d (was %s)",
                    level_name, seed, _stored_embedding_seed(adata, "phate"),
                )
                coords = phate.PHATE(
                    n_components=2, knn=knn, decay=15, t="auto",
                    n_jobs=-1, random_state=int(seed), verbose=0,
                ).fit_transform(X)
                adata.obsm["X_phate"] = coords.astype(np.float32)
                adata.uns["phate"] = {"params": {
                    "knn": knn, "decay": 15, "t": "auto", "random_state": int(seed),
                }}
                refit.add("phate")
        except ImportError:
            _logger.warning("  PHATE recompute skipped: install phate")
        except Exception as exc:
            _logger.warning("  PHATE recompute failed: %s", exc)

    return refit


def _run_overlays_only(
    output_dir: str,
    seed: int = 42,
    umap_n_neighbors: Optional[int] = None,
    umap_min_dist: Optional[float] = None,
    chromosome_csv: Optional[str] = None,
    chromosome_only: bool = False,
    umap_type: str = "max",
) -> str:
    """Picklable SLURM worker: regenerate HTML overlays from existing h5ads.

    If ``seed`` differs from the random_state stored in ``adata.uns["umap"]``
    (or ``["phate"]``), the corresponding embedding is refit in-place and the
    h5ad is rewritten so subsequent overlay regenerations stay consistent.
    """
    from ops_model.post_process.combination.pca_optimization import CHAD_ANNOTATION_PATH

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    _logger = logging.getLogger(__name__)

    out = Path(output_dir)
    guide_path = out / "guide_pca_optimized.h5ad"
    gene_path = out / "gene_embedding_pca_optimized.h5ad"

    if not guide_path.exists():
        return f"ERROR: {guide_path} not found. Run pipeline first."
    if not gene_path.exists():
        return f"ERROR: {gene_path} not found. Run pipeline first."

    _logger.info(f"Loading {guide_path}...")
    adata_guide = ad.read_h5ad(guide_path)
    _logger.info(f"Loading {gene_path}...")
    adata_gene_embed = ad.read_h5ad(gene_path)

    plots_dir = out / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if chromosome_only:
        # Skip the seed refit + save_extra_overlays. Just plot the chromosome
        # overlay using whatever X_umap / X_phate is already in the gene h5ad.
        if not chromosome_csv:
            return "ERROR: --chromosome-only requires --chromosome-csv"
        chrom_df = _load_chromosome_map(chromosome_csv, _logger)
        if chrom_df is None:
            return "ERROR: chromosome CSV unreadable or missing required columns"
        perts_gene = get_perts_col(adata_gene_embed)
        chrom_count = 0
        for embedding_name, obsm_key in (("UMAP", "X_umap"), ("PHATE", "X_phate")):
            coords = adata_gene_embed.obsm.get(obsm_key)
            if coords is None:
                _logger.warning(
                    f"  {obsm_key} missing on gene embedding — skipping {embedding_name}"
                )
                continue
            try:
                stem = plots_dir / f"gene_{embedding_name.lower()}_chromosome"
                _plot_chromosome_overlay(
                    np.asarray(coords), perts_gene, chrom_df,
                    embedding_name, stem, plt, _logger,
                )
                _plot_chromosome_overlay_html(
                    np.asarray(coords), perts_gene, chrom_df,
                    embedding_name, stem, _logger,
                )
                chrom_count += 1
            except Exception as chr_err:
                _logger.warning(
                    f"  Chromosome overlay ({embedding_name}) failed: {chr_err}"
                )
        return f"SUCCESS: regenerated {chrom_count} chromosome plot(s) only"

    # Guide level: don't pass UMAP param overrides (gene-level only).
    if _recompute_embeddings_for_seed(
        adata_guide, "guide", seed, _logger, out_dir=out, umap_type=umap_type,
    ):
        _logger.info(f"  Rewriting {guide_path} with refit embeddings")
        _atomic_write_h5ad(adata_guide, guide_path, _logger)
    if _recompute_embeddings_for_seed(
        adata_gene_embed, "gene", seed, _logger, out_dir=out,
        umap_n_neighbors=umap_n_neighbors, umap_min_dist=umap_min_dist,
        umap_type=umap_type,
    ):
        _logger.info(f"  Rewriting {gene_path} with refit embeddings")
        _atomic_write_h5ad(adata_gene_embed, gene_path, _logger)

    metrics_dir = out / "metrics"

    def _load_csv(name):
        path = metrics_dir / name
        if path.exists():
            _logger.info(f"  Loaded {name}")
            return pd.read_csv(path)
        _logger.warning(f"  {name} not found — skipping")
        return None

    activity_map = _load_csv("phenotypic_activity.csv")
    dist_map = _load_csv("phenotypic_distinctiveness.csv")
    corum_map = _load_csv("phenotypic_consistency_corum.csv")
    chad_map = _load_csv("phenotypic_consistency_manual.csv")

    from ops_model.post_process.combination.analysis.embedding_overlays import save_extra_overlays

    save_extra_overlays(
        adata_guide=adata_guide,
        adata_gene_embed=adata_gene_embed,
        plots_dir=plots_dir,
        plt=plt,
        activity_map=activity_map,
        dist_map=dist_map,
        corum_map=corum_map,
        chad_map=chad_map,
        chad_path_override=CHAD_ANNOTATION_PATH,
        _logger=_logger,
    )

    # Chromosome-arm overlays (gene level only) — uses the freshly loaded
    # X_umap / X_phate from gene_embedding_pca_optimized.h5ad.
    chrom_count = 0
    if chromosome_csv:
        chrom_df = _load_chromosome_map(chromosome_csv, _logger)
        if chrom_df is not None:
            perts_gene = get_perts_col(adata_gene_embed)
            for embedding_name, obsm_key in (("UMAP", "X_umap"), ("PHATE", "X_phate")):
                coords = adata_gene_embed.obsm.get(obsm_key)
                if coords is None:
                    _logger.warning(
                        f"  {obsm_key} missing on gene embedding — skipping "
                        f"chromosome overlay for {embedding_name}"
                    )
                    continue
                try:
                    stem = plots_dir / f"gene_{embedding_name.lower()}_chromosome"
                    _plot_chromosome_overlay(
                        np.asarray(coords), perts_gene, chrom_df,
                        embedding_name, stem, plt, _logger,
                    )
                    chrom_count += 1
                except Exception as chr_err:
                    _logger.warning(
                        f"  Chromosome overlay ({embedding_name}) failed: {chr_err}"
                    )

    # Persist obs additions from save_extra_overlays (CHAD / CORUM / supercategory
    # / leiden_r* / GO BP/CC / Reactome / KEGG / neighbors graph) onto disk.
    _atomic_write_h5ad(adata_guide, guide_path, _logger)
    _atomic_write_h5ad(adata_gene_embed, gene_path, _logger)
    chrom_note = f" + {chrom_count} chromosome plot(s)" if chrom_count else ""
    return f"SUCCESS: Overlays regenerated{chrom_note}"


def _fit_umap_one_seed(seed: int, X, nn: int, min_dist: float = 0.1):
    """Picklable worker: fit UMAP at one seed and return (seed, coords, err)."""
    try:
        from umap import UMAP

        model = UMAP(
            n_components=2,
            n_neighbors=nn,
            min_dist=float(min_dist),
            random_state=int(seed),
        )
        coords = model.fit_transform(X)
        return int(seed), np.asarray(coords, dtype=np.float32), None
    except Exception as exc:  # noqa: BLE001
        return int(seed), None, str(exc)


def _run_seed_sweep(
    output_dir: str,
    n_seeds: int = 36,
    base_seed: int = 0,
    n_workers: int = 1,
    umap_n_neighbors: Optional[int] = None,
    umap_min_dist: Optional[float] = None,
) -> str:
    """Fit UMAP at ``n_seeds`` consecutive seeds on the gene-level h5ad and
    write a single PNG canvas (sqrt(n)×sqrt(n) panels) so different layouts
    can be compared at a glance. NTCs render as faded red x's. Fits run in
    parallel with ``n_workers`` joblib workers (each pinned to 1 thread)."""
    import math
    import time
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    _logger = logging.getLogger(__name__)

    out = Path(output_dir)
    gene_path = out / "gene_embedding_pca_optimized.h5ad"
    if not gene_path.exists():
        return f"ERROR: {gene_path} not found. Run pipeline first."

    _logger.info(f"Loading {gene_path}...")
    adata = ad.read_h5ad(gene_path)
    if "X_pca" in adata.obsm:
        X = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
    else:
        X = np.asarray(adata.X, dtype=np.float32)

    n_obs = adata.n_obs
    nn = int(umap_n_neighbors) if umap_n_neighbors is not None else min(15, n_obs - 1)
    nn = min(nn, n_obs - 1)
    if nn < 2:
        return f"ERROR: too few obs ({n_obs}) for UMAP"
    md = float(umap_min_dist) if umap_min_dist is not None else 0.1

    perts = (
        adata.obs["perturbation"].astype(str).values
        if "perturbation" in adata.obs.columns
        else np.asarray(adata.obs_names.values, dtype=str)
    )
    is_ntc = np.array([p.startswith("NTC") for p in perts])

    seeds = list(range(int(base_seed), int(base_seed) + int(n_seeds)))
    n_workers = max(1, min(int(n_workers), len(seeds)))
    _logger.info(
        f"Fitting {len(seeds)} UMAPs (n_obs={n_obs}, n_features={X.shape[1]}, "
        f"n_neighbors={nn}, min_dist={md:g}) on {n_workers} worker(s)..."
    )

    t0 = time.time()
    if n_workers > 1:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_workers, backend="loky", verbose=5)(
            delayed(_fit_umap_one_seed)(s, X, nn, md) for s in seeds
        )
    else:
        results = [_fit_umap_one_seed(s, X, nn, md) for s in seeds]
    _logger.info(f"All UMAP fits done in {time.time() - t0:.1f}s")

    coords_by_seed = {s: c for s, c, _ in results if c is not None}
    for s, _, err in results:
        if err is not None:
            _logger.warning(f"  UMAP failed for seed={s}: {err}")

    n_cols = int(math.ceil(math.sqrt(len(seeds))))
    n_rows = int(math.ceil(len(seeds) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.5, n_rows * 2.5),
        squeeze=False,
    )
    for i, seed in enumerate(seeds):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]
        coords = coords_by_seed.get(seed)
        if coords is None:
            ax.set_title(f"seed={seed} (failed)", fontsize=8)
            ax.axis("off")
            continue
        ax.scatter(
            coords[~is_ntc, 0], coords[~is_ntc, 1],
            s=2, c="#3a7", alpha=0.5, linewidths=0,
        )
        if is_ntc.any():
            ax.scatter(
                coords[is_ntc, 0], coords[is_ntc, 1],
                s=6, c="#cc6060", marker="x", alpha=0.7, linewidths=0.4,
            )
        ax.set_title(f"seed={seed}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
    for i in range(len(seeds), n_rows * n_cols):
        r, c = divmod(i, n_cols)
        axes[r][c].axis("off")

    param_tag = f"nn{nn}_md{md:g}"
    plots_dir = out / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / (
        f"gene_umap_seed_sweep_{n_seeds}_base{base_seed}_{param_tag}.png"
    )
    fig.suptitle(
        f"Gene UMAP seed sweep — {n_seeds} seeds (base={base_seed}, "
        f"n_neighbors={nn}, min_dist={md:g})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Persist the exact coords each seed produced so --overlays-only --seed N
    # can load them directly instead of re-fitting (UMAP's parallel reductions
    # aren't bit-deterministic across runs even with random_state pinned).
    cache_path = out / f"gene_umap_seed_sweep_coords_{param_tag}.npz"
    np.savez(
        cache_path,
        seeds=np.asarray(list(coords_by_seed.keys()), dtype=np.int64),
        coords=np.stack(list(coords_by_seed.values()), axis=0).astype(np.float32),
        n_neighbors=np.int32(nn),
        min_dist=np.float32(md),
    )
    _logger.info(f"Saved per-seed coords cache → {cache_path}")
    return f"SUCCESS: {out_path}"


def _handle_sweep_seed(args, output_dir):
    """Submit / run the seed-sweep canvas for the gene-level UMAP."""
    if getattr(args, "second_pca_only", False):
        subdir = getattr(args, "second_pca_subdir", None)
        if subdir is None:
            threshold = getattr(args, "second_pca_threshold", 0.0)
            subdir = (
                "second_pca_consensus"
                if threshold == 0.0
                else f"second_pca_{int(round(threshold * 100))}"
            )
        output_dir = output_dir / subdir
        print(f"Second-pass seed-sweep mode: output → {output_dir}")

    n_seeds = int(getattr(args, "sweep_seed_n", 36))
    base_seed = int(getattr(args, "sweep_seed_base", 0))
    n_workers = int(getattr(args, "slurm_cpus", 36))
    umap_nn = getattr(args, "umap_n_neighbors", None)
    umap_md = getattr(args, "umap_min_dist", None)
    sweep_kwargs = {
        "output_dir": str(output_dir),
        "n_seeds": n_seeds,
        "base_seed": base_seed,
        "n_workers": n_workers,
        "umap_n_neighbors": int(umap_nn) if umap_nn is not None else None,
        "umap_min_dist": float(umap_md) if umap_md is not None else None,
    }

    if args.slurm:
        from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

        slurm_params = {
            "timeout_min": 10,
            "mem": "64GB",
            "cpus_per_task": n_workers,
            "slurm_partition": args.slurm_partition,
        }
        print(
            f"Submitting --sweep-seed as SLURM job "
            f"({slurm_params['mem']}, {slurm_params['timeout_min']}min, "
            f"{n_workers} CPUs, n={n_seeds}, base={base_seed}, "
            f"umap nn={sweep_kwargs['umap_n_neighbors']}, "
            f"md={sweep_kwargs['umap_min_dist']})..."
        )
        result = submit_parallel_jobs(
            jobs_to_submit=[
                {
                    "name": "pca_seed_sweep",
                    "func": _run_seed_sweep,
                    "kwargs": sweep_kwargs,
                }
            ],
            experiment="pca_seed_sweep",
            slurm_params=slurm_params,
            log_dir="pca_optimization",
            manifest_prefix="pca_seed_sweep",
            wait_for_completion=True,
        )
        if result.get("failed"):
            print("Seed sweep FAILED")
        else:
            print("Seed sweep complete")
    else:
        print(_run_seed_sweep(**sweep_kwargs))


def _handle_overlays_only(args, output_dir):
    """Re-generate interactive HTML overlays from existing optimised h5ads."""
    # When combined with --second-pca-only, descend into the second-pass subdir
    if getattr(args, "second_pca_only", False):
        subdir = getattr(args, "second_pca_subdir", None)
        if subdir is None:
            threshold = getattr(args, "second_pca_threshold", 0.0)
            if threshold == 0.0:
                subdir = "second_pca_consensus"
            else:
                subdir = f"second_pca_{int(round(threshold * 100))}"
            # Honor the chrom-arm-correction suffix used by run_chrom_arm_then_second_pca
            if getattr(args, "chrom_arm_correct", False):
                from ops_model.post_process.combination.pipeline_add_ons.guide_chrom_arm_correction import (
                    METHOD_SUFFIX,
                )
                method = getattr(args, "chrom_arm_method", "cohesion")
                subdir = subdir + "_chrom_arm_corr" + METHOD_SUFFIX.get(method, "")
        output_dir = output_dir / subdir
        print(f"Second-pass overlays mode: output → {output_dir}")

    # When --chrom-arm-correct is on but no --chromosome-csv was given,
    # auto-pipe the shared symbol→arm cache so the chr-arm overlay plot fires.
    if getattr(args, "chrom_arm_correct", False) and not getattr(args, "chromosome_csv", None):
        from ops_model.post_process.combination.pipeline_add_ons.guide_chrom_arm_correction import (
            SHARED_MAP_CSV_PATH,
        )
        if SHARED_MAP_CSV_PATH is not None and SHARED_MAP_CSV_PATH.is_file():
            args.chromosome_csv = str(SHARED_MAP_CSV_PATH)
            print(f"--chrom-arm-correct: auto-piping chromosome overlay from "
                  f"shared cache {SHARED_MAP_CSV_PATH}")

    seed = int(getattr(args, "seed", 42))
    umap_nn = getattr(args, "umap_n_neighbors", None)
    umap_md = getattr(args, "umap_min_dist", None)
    umap_kwargs = {
        "umap_n_neighbors": int(umap_nn) if umap_nn is not None else None,
        "umap_min_dist": float(umap_md) if umap_md is not None else None,
        "chromosome_csv": getattr(args, "chromosome_csv", None),
        "chromosome_only": getattr(args, "chromosome_only", False),
        "umap_type": getattr(args, "umap_type", "max"),
    }
    if args.slurm:
        from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

        slurm_params = {
            "timeout_min": 30,
            "mem": "64GB",
            "cpus_per_task": args.slurm_cpus,
            "slurm_partition": args.slurm_partition,
        }
        print(
            f"Submitting --overlays-only as SLURM job "
            f"({slurm_params['mem']}, {slurm_params['timeout_min']}min, "
            f"seed={seed}, umap nn={umap_kwargs['umap_n_neighbors']}, "
            f"md={umap_kwargs['umap_min_dist']})..."
        )
        result = submit_parallel_jobs(
            jobs_to_submit=[
                {
                    "name": "pca_overlays_only",
                    "func": _run_overlays_only,
                    "kwargs": {
                        "output_dir": str(output_dir), "seed": seed, **umap_kwargs,
                    },
                }
            ],
            experiment="pca_overlays_only",
            slurm_params=slurm_params,
            log_dir="pca_optimization",
            manifest_prefix="pca_overlays_only",
            wait_for_completion=True,
        )
        if result.get("failed"):
            print("Overlays regeneration FAILED")
        else:
            print("Overlays regeneration complete")
    else:
        print(_run_overlays_only(str(output_dir), seed=seed, **umap_kwargs))


def _handle_aggregate_only(args, output_dir):
    """Run only the aggregation step (Phase 2).

    When a non-default distance metric is used, the per_signal/ h5ads live in
    the *parent* directory (the original cosine sweep output).  We read from
    there but write results into the distance-specific subdirectory.

    The parent symlink fallback is only safe when the per-signal data is
    interchangeable across the path segment we descended through. That's true
    for distance metrics (they don't affect Phase 1 outputs) but **NOT** for
    ``--agg-method`` — median changes the cells→guide reduction, so a median
    aggregate-only run must point at a per_signal/ that was itself produced
    with median. We refuse the symlink in that case and tell the user to do a
    full run first.
    """
    agg_output = str(output_dir)
    agg_subdir = "per_signal"
    agg_method = getattr(args, "agg_method", "mean")

    per_signal_dir = Path(agg_output) / agg_subdir
    if not per_signal_dir.exists():
        parent_per_signal = Path(agg_output).parent / agg_subdir
        if agg_method != "mean":
            print(
                f"\nERROR: --aggregate-only with --agg-method={agg_method} requires "
                f"per-signal h5ads aggregated with method={agg_method}, but none "
                f"were found at:\n  {per_signal_dir}\n\n"
                f"Symlinking from the mean tree at {parent_per_signal} would give "
                f"wrong cells→guide aggregation (mean, not {agg_method}).\n"
                f"Run the full pipeline first (drop --aggregate-only) to produce "
                f"{agg_method}-aggregated per_signal/, then re-run --aggregate-only."
            )
            return
        if parent_per_signal.exists():
            # e.g. output_dir = .../all/euclidean, per_signal lives in .../all/per_signal
            source_subdir = str(parent_per_signal)
            print(f"Reading swept h5ads from {source_subdir}")
            print(f"Writing aggregated results to {agg_output}")
            # Symlink per_signal into the output dir so aggregate_channels can find it
            per_signal_dir.parent.mkdir(parents=True, exist_ok=True)
            per_signal_dir.symlink_to(parent_per_signal)

    second_pca_kwargs = _build_second_pca_kwargs(args)
    if args.slurm:
        print(
            f"Submitting aggregation as SLURM job ({args.slurm_agg_memory}, {args.slurm_agg_time}min)..."
        )
        if args.downsampled or getattr(args, "all_cells", False):
            print(f"  Mode: signal-group (reading from {agg_output}/per_signal/)")
        if second_pca_kwargs is not None:
            print("  Chained: 2nd-pass PCA will run after aggregation in the same SLURM job.")
        _submit_aggregation_slurm(
            agg_output,
            args.norm_method,
            agg_subdir,
            _make_agg_slurm_params(args),
            "pca_aggregation",
            "pca_agg",
            agg_method=getattr(args, "agg_method", "mean"),
            chromosome_csv=getattr(args, "chromosome_csv", None),
            distance=args.distance,
            second_pca_kwargs=second_pca_kwargs,
            random_seed=getattr(args, "seed", 42),
            umap_type=getattr(args, "umap_type", "max"),
            consensus_metrics=getattr(args, "second_pca_consensus_metrics", None),
            sweep_metric=getattr(args, "sweep_metric", "mean_map"),
        )
    else:
        result = aggregate_channels(
            output_dir=agg_output,
            norm_method=args.norm_method,
            per_unit_subdir=agg_subdir,
            distance=args.distance,
            random_seed=getattr(args, "seed", 42),
            agg_method=getattr(args, "agg_method", "mean"),
            chromosome_csv=getattr(args, "chromosome_csv", None),
            umap_type=getattr(args, "umap_type", "max"),
        )
        print(result)
        if second_pca_kwargs is not None:
            print("\nChaining 2nd-pass PCA on aggregate output...")
            _handle_second_pca(args, output_dir)


def run_second_pca_then_chrom_arm(
    output_dir: str,
    chrom_arm_kwargs: Dict,
    second_pca_kwargs: Dict,
) -> str:
    """Reverse-order SLURM wrapper: 2nd-pass PCA → chrom-arm correction.

    Step 1: run ``apply_second_pass_pca`` on the uncorrected guide h5ad
    (re-uses an existing ``second_pca_consensus/`` output when present,
    otherwise computes it fresh).

    Step 2: load the 2pca'd guide h5ad, apply the chrom-arm correction in
    the PC space (sPCs become the features being regressed), write the
    corrected version next to it.

    Step 3: call ``apply_second_pass_pca(..., skip_pca=True)`` on the
    corrected sPC h5ad to do scoring + aggregation + plots without
    re-running PCA. Outputs land in
    ``second_pca_consensus_then_chrom_arm_corr<method>/`` so this never
    clobbers the standard ``second_pca_consensus_chrom_arm_corr*/`` dirs
    produced by the original-order wrapper.
    """
    from ops_model.post_process.combination.pipeline_add_ons.guide_chrom_arm_correction import (
        run_chrom_arm_correction,
        METHOD_SUFFIX,
    )

    output_dir = Path(output_dir)
    method = chrom_arm_kwargs.get("method", "cohesion")
    method_suffix = METHOD_SUFFIX.get(method, "")

    # Step 1: 2nd-pass PCA on the uncorrected input. Use the default
    # second_pca_consensus subdir; if it already exists, reuse it.
    pca_subdir = output_dir / "second_pca_consensus"
    pca_guide = pca_subdir / "guide_pca_optimized.h5ad"
    if not pca_guide.exists():
        logger = logging.getLogger(__name__)
        logger.info(
            f"[2pca→chrom-arm] No existing {pca_subdir.name}/; running 2nd-pass PCA..."
        )
        spkw = dict(second_pca_kwargs)
        spkw["subdir"] = None  # force default 'second_pca_consensus'
        spkw["subdir_suffix"] = ""
        spkw["input_path"] = None
        spkw["skip_pca"] = False
        result = apply_second_pass_pca(**spkw)
        if "FAILED" in str(result):
            return result
    else:
        logging.getLogger(__name__).info(
            f"[2pca→chrom-arm] Reusing existing 2pca output at {pca_subdir}"
        )

    # Step 2: chrom-arm correction on the sPC matrix. Write the corrected
    # h5ad next to the 2pca'd one with a method-aware suffix so subsequent
    # methods don't overwrite each other.
    corrected_out = pca_subdir / (
        f"guide_pca_optimized_chrom_arm_corr{method_suffix}_after_2pca.h5ad"
    )
    corrected_path = run_chrom_arm_correction(
        pca_guide, output_path=corrected_out, **chrom_arm_kwargs,
    )

    # Step 3: skip-pca scoring on the corrected sPC matrix. Drop into a
    # then-corrected subdir under output_dir.
    spkw = dict(second_pca_kwargs)
    spkw["input_path"] = str(corrected_path)
    spkw["skip_pca"] = True
    spkw["subdir"] = (
        f"second_pca_consensus_then_chrom_arm_corr{method_suffix}"
    )
    spkw["subdir_suffix"] = ""  # we set subdir explicitly above
    return apply_second_pass_pca(**spkw)


def run_chrom_arm_then_second_pca(
    output_dir: str,
    chrom_arm_kwargs: Dict,
    second_pca_kwargs: Dict,
    skip_pca: bool = False,
) -> str:
    """SLURM-picklable wrapper: chrom-arm correct guide h5ad → 2nd-pass PCA.

    Reads ``<output_dir>/guide_pca_optimized.h5ad``, runs the chrom-arm
    correction (annotation + per-method correction step), writes the
    corrected guide h5ad next to the original (filename varies by method),
    then chains ``apply_second_pass_pca`` on that corrected h5ad. Output
    subdir gets a method-aware suffix so the cohesion run lands in
    ``second_pca_consensus_chrom_arm_corr/`` and the centering run in
    ``second_pca_consensus_chrom_arm_corr_centering/`` — never clobber each
    other.

    Keeping this as a top-level function (not a closure inside
    ``_handle_second_pca``) so submitit can pickle it for SLURM submission.
    """
    from ops_model.post_process.combination.pipeline_add_ons.guide_chrom_arm_correction import (
        run_chrom_arm_correction,
        METHOD_SUFFIX,
    )

    output_dir = Path(output_dir)
    guide_path = output_dir / "guide_pca_optimized.h5ad"
    if not guide_path.exists():
        return f"FAILED: {guide_path} not found — run aggregation first"

    method = chrom_arm_kwargs.get("method", "cohesion")
    corrected_path = run_chrom_arm_correction(guide_path, **chrom_arm_kwargs)

    spkw = dict(second_pca_kwargs)
    spkw["input_path"] = str(corrected_path)
    spkw["subdir_suffix"] = "_chrom_arm_corr" + METHOD_SUFFIX.get(method, "")
    spkw["skip_pca"] = bool(skip_pca)
    return apply_second_pass_pca(**spkw)


def _handle_second_pca(args, output_dir):
    """Run a second-pass PCA on the existing aggregated guide h5ad.

    With ``--chrom-arm-correct``, first regresses out per-arm cohesion effects
    from the guide-level h5ad and runs the 2nd-pass on the corrected output;
    the corrected h5ad and its 2nd-pass subdir both get a ``_chrom_arm_corr``
    suffix so they never clobber an existing untouched run.
    """
    sweep_thresholds = None
    if args.second_pca_sweep_thresholds:
        sweep_thresholds = [
            float(t) for t in args.second_pca_sweep_thresholds.split(",") if t.strip()
        ]

    # Auto-pipe the shared chrom-arm map into the chromosome overlay plot
    # whenever the user opts into --chrom-arm-correct but didn't pass an
    # explicit --chromosome-csv. Without this, the gene-level chromosome
    # UMAP/PHATE plot silently skips — _compute_and_plot_embeddings only
    # generates it when chromosome_csv is non-None.
    chromosome_csv = getattr(args, "chromosome_csv", None)
    if (
        not chromosome_csv
        and getattr(args, "chrom_arm_correct", False)
    ):
        from ops_model.post_process.combination.pipeline_add_ons.guide_chrom_arm_correction import (
            SHARED_MAP_CSV_PATH,
        )
        if SHARED_MAP_CSV_PATH is not None and SHARED_MAP_CSV_PATH.is_file():
            chromosome_csv = str(SHARED_MAP_CSV_PATH)
            print(
                f"--chrom-arm-correct: auto-piping chromosome overlay from "
                f"shared cache {SHARED_MAP_CSV_PATH}"
            )

    second_pca_kwargs = dict(
        output_dir=str(output_dir),
        threshold=args.second_pca_threshold,
        distance=args.distance,
        norm_method=args.norm_method,
        subdir=args.second_pca_subdir,
        run_sweep=not args.second_pca_no_sweep,
        sweep_thresholds=sweep_thresholds,
        random_seed=getattr(args, "seed", 42),
        agg_method=getattr(args, "agg_method", "mean"),
        chromosome_csv=chromosome_csv,
        umap_type=getattr(args, "umap_type", "max"),
        consensus_metrics=getattr(args, "second_pca_consensus_metrics", None),
        sweep_metric=getattr(args, "sweep_metric", "mean_map"),
    )

    chrom_arm = bool(getattr(args, "chrom_arm_correct", False))
    if chrom_arm:
        chrom_arm_method = getattr(args, "chrom_arm_method", "cohesion")
        chrom_arm_skip_2pca = bool(getattr(args, "chrom_arm_skip_second_pca", False))
        chrom_arm_after_2pca = bool(getattr(args, "chrom_arm_after_second_pca", False))
        if chrom_arm_skip_2pca and chrom_arm_after_2pca:
            raise SystemExit(
                "--chrom-arm-skip-second-pca and --chrom-arm-after-second-pca "
                "are mutually exclusive (the first runs correction without "
                "2pca; the second runs 2pca then correction)."
            )
        chrom_arm_kwargs = dict(
            method=chrom_arm_method,
            k_nn=int(getattr(args, "chrom_arm_knn", 15)),
            pv_threshold=float(getattr(args, "chrom_arm_qval", 0.01)),
            min_genes_for_regression=int(getattr(args, "chrom_arm_min_genes", 10)),
            map_csv_path=getattr(args, "chrom_arm_map_csv", None),
        )
        method_suffix = "" if chrom_arm_method == "cohesion" else f"_{chrom_arm_method}"
        if chrom_arm_after_2pca:
            wrapped_kwargs = dict(
                output_dir=str(output_dir),
                chrom_arm_kwargs=chrom_arm_kwargs,
                second_pca_kwargs=second_pca_kwargs,
            )
            job_name = f"second_pca_then_chrom_arm_corr{method_suffix}"
            func = run_second_pca_then_chrom_arm
            kwargs_for_run = wrapped_kwargs
        else:
            wrapped_kwargs = dict(
                output_dir=str(output_dir),
                chrom_arm_kwargs=chrom_arm_kwargs,
                second_pca_kwargs=second_pca_kwargs,
                skip_pca=chrom_arm_skip_2pca,
            )
            prefix = "chrom_arm_corr_only" if chrom_arm_skip_2pca else "pca_second_pass_chrom_arm_corr"
            job_name = f"{prefix}{method_suffix}"
            func = run_chrom_arm_then_second_pca
            kwargs_for_run = wrapped_kwargs
    else:
        job_name = "pca_second_pass"
        func = apply_second_pass_pca
        kwargs_for_run = second_pca_kwargs

    if args.slurm:
        from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

        slurm_params = _make_agg_slurm_params(args)
        suffix = " (chrom-arm corrected)" if chrom_arm else ""
        print(
            f"Submitting --second-pca{suffix} as SLURM job "
            f"({slurm_params.get('mem')}, {slurm_params.get('timeout_min')}min)..."
        )
        agg_result = submit_parallel_jobs(
            jobs_to_submit=[
                {
                    "name": job_name,
                    "func": func,
                    "kwargs": kwargs_for_run,
                }
            ],
            experiment=job_name,
            slurm_params=slurm_params,
            log_dir="pca_optimization",
            manifest_prefix=job_name,
            wait_for_completion=True,
        )
        if agg_result.get("failed"):
            print(f"Second-pass PCA{suffix} FAILED")
        else:
            print(f"Second-pass PCA{suffix} complete")
    else:
        print(func(**kwargs_for_run))


def _handle_op(args, output_dir):
    """OrganelleProfiler mode: one PCA-sweep job per all_cells_*.h5ad file.

    Mirrors :func:`_handle_downsampled` minus the experiment-discovery step —
    OP h5ads are already one-per-marker with cells pooled inside, so we just
    list the files and dispatch one ``pca_sweep_op_signal`` per file.
    """
    op_root = args.op_root
    ds_output_dir = output_dir
    ds_output_dir.mkdir(parents=True, exist_ok=True)

    if getattr(args, "clean", False):
        import shutil

        per_signal_dir = ds_output_dir / "per_signal"
        if per_signal_dir.exists():
            print(f"--clean: removing {per_signal_dir}")
            shutil.rmtree(per_signal_dir)

    try:
        pairs = _discover_op_files(op_root)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return
    print(f"OP mode: discovered {len(pairs)} marker h5ads under {op_root}")

    target_n_cells = int(getattr(args, "target_n_cells", 0) or 0)
    if target_n_cells == 0:
        # OP files are already capped during consolidation; use a generous cap
        # so we keep ~all of each file (caps in consolidate_all_cells already
        # bound this to ~2-8M cells / file).
        target_n_cells = 10_000_000

    skip_phase2 = getattr(args, "preserve_batch", False) or getattr(args, "no_pca", False)

    def _op_job_kwargs(signal: str, path: Path) -> Dict:
        kwargs = dict(
            signal=signal,
            op_path=str(path),
            output_dir=str(ds_output_dir),
            target_n_cells=target_n_cells,
            norm_method=args.norm_method,
            random_seed=getattr(args, "seed", 42),
            distance=args.distance,
            zscore_per_experiment=getattr(args, "zscore_per_experiment", False),
            exclude_dud_guides=getattr(args, "exclude_dud_guides", True),
        )
        if args.fixed_threshold is not None and args.fixed_threshold > 0:
            kwargs["fixed_threshold"] = args.fixed_threshold
        if getattr(args, "preserve_batch", False):
            kwargs["preserve_batch"] = True
        if getattr(args, "no_pca", False):
            kwargs["no_pca"] = True
        if getattr(args, "agg_method", "mean") != "mean":
            kwargs["agg_method"] = args.agg_method
        return kwargs

    if not args.slurm:
        print("\nRunning locally (sequential)...")
        for signal, path in pairs:
            result = pca_sweep_op_signal(**_op_job_kwargs(signal, path))
            print(f"  {result}")
        if not skip_phase2:
            result = aggregate_channels(
                output_dir=str(ds_output_dir),
                norm_method=args.norm_method,
                per_unit_subdir="per_signal",
                distance=args.distance,
                random_seed=getattr(args, "seed", 42),
                agg_method=getattr(args, "agg_method", "mean"),
                chromosome_csv=getattr(args, "chromosome_csv", None),
                umap_type=getattr(args, "umap_type", "max"),
            )
            print(result)
            if getattr(args, "second_pca", False):
                print("\nChaining 2nd-pass PCA on aggregate output...")
                _handle_second_pca(args, ds_output_dir)
        return

    # SLURM mode — split into high-memory (>4M cells, e.g. Phase) and
    # standard-memory batches, matching the CP/DINO _handle_downsampled flow.
    # Cell counts are peeked cheaply via h5py shape[0] (no full load).
    import h5py
    from ops_utils.hpc.slurm_batch_utils import (
        submit_parallel_jobs,
        wait_for_multiple_job_arrays,
    )

    # OP signal jobs sweep ~17 PCA thresholds, each running aggregate_to_level
    # + copairs scoring on 1-4M cells. Observed runtime: ~11 min for the
    # mid-size reporter signals — 20 min gives a buffer. Phase still gets
    # the 360-min minimum below.
    args.slurm_time = max(args.slurm_time, 20)

    HIGH_MEMORY_CELL_THRESHOLD = 4_000_000

    high_mem_jobs = []
    other_jobs = []
    for signal, path in pairs:
        sig_safe = sanitize_signal_filename(signal)[:40]
        try:
            with h5py.File(path, "r") as f:
                n_obs = int(f["X"].shape[0])
        except Exception:
            n_obs = 0
        job = {
            "name": f"pca_op_{sig_safe}",
            "func": pca_sweep_op_signal,
            "kwargs": _op_job_kwargs(signal, path),
            "metadata": {"signal": signal, "source": str(path), "n_cells": n_obs},
        }
        if n_obs > HIGH_MEMORY_CELL_THRESHOLD:
            high_mem_jobs.append(job)
        else:
            other_jobs.append(job)

    slurm_params = _make_slurm_params(args)
    job_arrays = []

    if other_jobs:
        print(
            f"\nSubmitting {len(other_jobs)} OP signal jobs "
            f"({slurm_params['mem']} each, {slurm_params['timeout_min']}min)..."
        )
        result_other = submit_parallel_jobs(
            jobs_to_submit=other_jobs,
            experiment="pca_op",
            slurm_params=slurm_params,
            log_dir="pca_optimization",
            manifest_prefix="pca_op",
            wait_for_completion=False,
        )
        if result_other.get("submitted_jobs"):
            job_arrays.append({
                "submitted_jobs": result_other["submitted_jobs"],
                "base_job_id": result_other["base_job_id"],
                "label": "reporters",
                "slurm_params": slurm_params,
            })

    if high_mem_jobs:
        phase_memory = getattr(args, "phase_memory", "600GB")
        high_mem_slurm_params = {
            **slurm_params,
            "mem": phase_memory,
            "timeout_min": max(slurm_params.get("timeout_min", 60), 360),
        }
        high_mem_names = [j["metadata"]["signal"] for j in high_mem_jobs]
        print(
            f"\nSubmitting {len(high_mem_jobs)} high-memory OP signal job(s) "
            f"({phase_memory}, >4M cells): {', '.join(high_mem_names)}"
        )
        result_high = submit_parallel_jobs(
            jobs_to_submit=high_mem_jobs,
            experiment="pca_op_high_mem",
            slurm_params=high_mem_slurm_params,
            log_dir="pca_optimization",
            manifest_prefix="pca_op_high",
            wait_for_completion=False,
        )
        if result_high.get("submitted_jobs"):
            job_arrays.append({
                "submitted_jobs": result_high["submitted_jobs"],
                "base_job_id": result_high["base_job_id"],
                "label": "high_mem",
                "slurm_params": high_mem_slurm_params,
            })

    result = {"failed": []}
    if job_arrays:
        wait_result = wait_for_multiple_job_arrays(
            job_arrays,
            experiment="pca_op",
        )
        result["failed"] = wait_result.get("failed", []) or []

    if result.get("failed"):
        print(f"\n{len(result['failed'])} OP signal jobs failed")
        return
    print(f"\nAll {len(other_jobs) + len(high_mem_jobs)} OP signal jobs complete")

    if skip_phase2:
        print("  Phase 2 aggregation skipped (--preserve-batch or --no-pca mode)")
        return

    sp_kwargs = _build_second_pca_kwargs(args)
    msg = "aggregation SLURM job"
    if sp_kwargs is not None:
        msg += " (chained with 2nd-pass PCA)"
    print(f"\nSubmitting {msg}...")
    _submit_aggregation_slurm(
        str(ds_output_dir),
        args.norm_method,
        "per_signal",
        _make_agg_slurm_params(args),
        "pca_op_aggregation",
        "pca_op_agg",
        distance=args.distance,
        second_pca_kwargs=sp_kwargs,
        random_seed=getattr(args, "seed", 42),
        agg_method=getattr(args, "agg_method", "mean"),
        chromosome_csv=getattr(args, "chromosome_csv", None),
        umap_type=getattr(args, "umap_type", "max"),
        consensus_metrics=getattr(args, "second_pca_consensus_metrics", None),
    )


def _handle_external(args, output_dir):
    """External mode: combine explicit per-signal h5ads from config ``signal_paths``.

    ``args.signal_paths`` maps a signal-group name -> one h5ad path or a list of
    paths. Each h5ad must have the same schema as the discovery
    ``features_processed_*.h5ad`` (obs with sgRNA / perturbation / experiment;
    ``X`` = embedding). Multiple paths under one signal are pooled. Experiment
    discovery is bypassed; the standard ``pca_sweep_pooled_signal`` worker (with
    an explicit-path override) + Phase 2 aggregation are reused unchanged.
    """
    ds_output_dir = output_dir
    ds_output_dir.mkdir(parents=True, exist_ok=True)

    if getattr(args, "clean", False):
        import shutil

        per_signal_dir = ds_output_dir / "per_signal"
        if per_signal_dir.exists():
            print(f"--clean: removing {per_signal_dir}")
            shutil.rmtree(per_signal_dir)

    spec = args.signal_paths
    if not isinstance(spec, dict) or not spec:
        print("ERROR: signal_paths must be a non-empty mapping of signal -> path(s).")
        return

    # Build signal groups + an explicit (exp_label, channel) -> path override.
    # Each file becomes its own synthetic "experiment" batch (channel == signal),
    # so per-experiment z-scoring treats each file independently.
    signal_groups: Dict[str, list] = {}
    cell_path_map: Dict[tuple, str] = {}
    missing: list = []
    for signal, paths in spec.items():
        if isinstance(paths, (str, Path)):
            paths = [paths]
        pairs = []
        for p in paths:
            p = Path(p)
            if not p.exists():
                missing.append(str(p))
                continue
            exp_label = p.stem
            pairs.append((exp_label, signal))
            cell_path_map[(exp_label, signal)] = str(p)
        if pairs:
            signal_groups[signal] = pairs

    if missing:
        print("ERROR: signal_paths references missing file(s):")
        for m in missing:
            print(f"  - {m}")
        return
    if not signal_groups:
        print("ERROR: no usable signal_paths entries.")
        return

    print(f"External mode: {len(signal_groups)} signal group(s) from explicit paths:")
    for sig, pairs in signal_groups.items():
        print(f"  {sig}: {len(pairs)} file(s)")

    if getattr(args, "dry_run", False):
        print("\n--dry-run: not processing.")
        return

    # External files are user-provided; default to keeping all cells unless the
    # user opts into downsampling (--target-cells / --downsampled).
    target_n_cells = int(getattr(args, "target_cells", 0) or 0) or 10_000_000
    skip_phase2 = getattr(args, "preserve_batch", False) or getattr(args, "no_pca", False)

    def _job_kwargs(signal, pairs):
        kwargs = dict(
            signal=signal,
            exp_channel_pairs=pairs,
            output_dir=str(ds_output_dir),
            target_n_cells=target_n_cells,
            norm_method=args.norm_method,
            random_seed=getattr(args, "seed", 42),
            distance=args.distance,
            zscore_per_experiment=getattr(args, "zscore_per_experiment", False),
            exclude_dud_guides=getattr(args, "exclude_dud_guides", True),
            cell_paths=cell_path_map,
        )
        if args.fixed_threshold is not None and args.fixed_threshold > 0:
            kwargs["fixed_threshold"] = args.fixed_threshold
        if getattr(args, "preserve_batch", False):
            kwargs["preserve_batch"] = True
        if getattr(args, "no_pca", False):
            kwargs["no_pca"] = True
        if getattr(args, "agg_method", "mean") != "mean":
            kwargs["agg_method"] = args.agg_method
        if getattr(args, "downsample_per_guide", False):
            kwargs["downsample_per_guide"] = True
            kwargs["cells_per_guide"] = getattr(args, "cells_per_guide", 250)
        return kwargs

    if not args.slurm:
        print("\nRunning locally (sequential)...")
        for signal, pairs in signal_groups.items():
            print(f"  {pca_sweep_pooled_signal(**_job_kwargs(signal, pairs))}")
        if not skip_phase2:
            print(
                aggregate_channels(
                    output_dir=str(ds_output_dir),
                    norm_method=args.norm_method,
                    per_unit_subdir="per_signal",
                    distance=args.distance,
                    random_seed=getattr(args, "seed", 42),
                    agg_method=getattr(args, "agg_method", "mean"),
                    chromosome_csv=getattr(args, "chromosome_csv", None),
                    umap_type=getattr(args, "umap_type", "max"),
                )
            )
            if getattr(args, "second_pca", False):
                print("\nChaining 2nd-pass PCA on aggregate output...")
                _handle_second_pca(args, ds_output_dir)
        return

    # SLURM: one job per signal group, wait, then a chained aggregation job.
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

    jobs = [
        {
            "name": f"pca_ext_{sanitize_signal_filename(sig)[:40]}",
            "func": pca_sweep_pooled_signal,
            "kwargs": _job_kwargs(sig, pairs),
            "metadata": {"signal": sig, "n_files": len(pairs)},
        }
        for sig, pairs in signal_groups.items()
    ]
    slurm_params = _make_slurm_params(args)
    print(
        f"\nSubmitting {len(jobs)} external signal job(s) "
        f"({slurm_params['mem']} each, {slurm_params['timeout_min']}min)..."
    )
    res = submit_parallel_jobs(
        jobs_to_submit=jobs,
        experiment="pca_ext",
        slurm_params=slurm_params,
        log_dir="pca_optimization",
        manifest_prefix="pca_ext",
        wait_for_completion=True,
    )
    if res.get("failed"):
        print(f"\n{len(res['failed'])} external signal job(s) failed. Check logs.")
        return
    print(f"\nAll {len(jobs)} external signal job(s) complete")

    if skip_phase2:
        print("  Phase 2 aggregation skipped (--preserve-batch or --no-pca mode)")
        return

    sp_kwargs = _build_second_pca_kwargs(args)
    print("\nSubmitting aggregation SLURM job...")
    _submit_aggregation_slurm(
        str(ds_output_dir),
        args.norm_method,
        "per_signal",
        _make_agg_slurm_params(args),
        "pca_ext_aggregation",
        "pca_ext_agg",
        distance=args.distance,
        second_pca_kwargs=sp_kwargs,
        random_seed=getattr(args, "seed", 42),
        agg_method=getattr(args, "agg_method", "mean"),
        chromosome_csv=getattr(args, "chromosome_csv", None),
        umap_type=getattr(args, "umap_type", "max"),
        consensus_metrics=getattr(args, "second_pca_consensus_metrics", None),
    )


def _handle_downsampled(args, output_dir, cp_override):
    """Pool cells by signal group, downsample, PCA sweep (local or SLURM)."""
    from ops_model.post_process.combination.pca_optimization import (
        DEFAULT_SWEEP_THRESHOLDS_CP,
        PCA_FIT_CAP,
    )

    ds_output_dir = output_dir
    ds_output_dir.mkdir(parents=True, exist_ok=True)

    # --clean: wipe stale per-signal h5ads so Phase 1 runs from scratch
    if getattr(args, "clean", False):
        import shutil

        per_signal_dir = ds_output_dir / "per_signal"
        if per_signal_dir.exists():
            print(f"--clean: removing {per_signal_dir}")
            shutil.rmtree(per_signal_dir)

    # When the user explicitly named experiments via --experiments, let them
    # through discovery's bad-experiment / non-default-library filters.
    exp_whitelist = getattr(args, "experiments", None)
    force_set: Optional[set] = None
    if exp_whitelist:
        force_set = {e.strip() for e in exp_whitelist.split(",") if e.strip()}

    all_pairs, attr_config, storage_roots, feature_dir, maps_path = (
        _discover_experiment_pairs(
            cp_override,
            include_cellpainting=getattr(args, "include_cellpainting", False),
            include_4i=getattr(args, "include_4i", False),
            include_cp=getattr(args, "include_cp", False),
            include_standard=getattr(args, "include_standard", True),
            force_include=force_set,
        )
    )
    if not all_pairs:
        print("No experiment-channel pairs found!")
        return

    paper_v1_path = getattr(args, "paper_v1", None) or getattr(args, "paper_v2", None)
    paper_flag = "--paper-v2" if getattr(args, "paper_v2", None) else "{paper_flag}"
    if paper_v1_path:
        import yaml as _yaml

        v1_path = Path(paper_v1_path)
        if not v1_path.exists():
            print(f"ERROR: {paper_flag} manifest not found: {v1_path}")
            return
        with open(v1_path) as f:
            v1_doc = _yaml.safe_load(f) or {}
        v1_map = v1_doc.get("experiments_channels", {}) or {}
        if not v1_map:
            print(f"ERROR: {paper_flag} manifest has empty 'experiments_channels': {v1_path}")
            return
        # YAML keys are full experiment names (e.g. ops0094_20251217). Match by full name.
        expected_full = set(v1_map.keys())
        expected_short = {e.split("_")[0] for e in expected_full}
        before = len(all_pairs)
        all_pairs = [(exp, ch) for exp, ch in all_pairs if exp in expected_full]
        print(
            f"{paper_flag}: {before} → {len(all_pairs)} pairs "
            f"(restricted to {len(expected_full)} experiments from {v1_path.name})"
        )
        discovered_full = {exp for exp, _ in all_pairs}
        missing = sorted(expected_full - discovered_full)
        # --only-4i / --only-cp legitimately scope to a sub-cohort that lacks
        # most paper_v1 experiments; relax the strict-equality check there.
        only_subset = getattr(args, "only_4i", False) or getattr(args, "only_cp", False)
        if missing and not only_subset:
            print(
                f"\nERROR: {paper_flag} expects {len(expected_full)} experiments but "
                f"{len(missing)} are MISSING from discovery (must be present, no more, no less):"
            )
            for m in missing:
                print(f"  - {m}")
            return
        if missing and only_subset:
            print(
                f"  --only-4i/--only-cp scope: {len(missing)}/{len(expected_full)} paper_v1 "
                f"experiments have no data in this sub-cohort and are skipped."
            )
        extra_short = {exp.split("_")[0] for exp, _ in all_pairs} - expected_short
        if extra_short:
            print(
                f"\nERROR: {paper_flag} saw {len(extra_short)} extra experiments after filtering "
                f"(this should not happen): {sorted(extra_short)}"
            )
            return

        # Per-experiment channel allow-list from the YAML. Discovery names
        # channels by the sanitized label (e.g. "SEC61B", "no_label", "Phase");
        # the YAML lists fluorophores (Phase, GFP, mCherry, Cy5). Walk the
        # channel_maps once to invert label→channel_name so we can compare.
        from ops_utils.data.feature_metadata import FeatureMetadata
        from ops_utils.data.filesystem import extract_ops_key

        meta = FeatureMetadata()
        marker_to_fluor: Dict[str, Dict[str, str]] = {
            exp_short: {
                FeatureMetadata.sanitize_label(e["label"]): e["channel_name"]
                for e in (entries or [])
                if isinstance(e, dict) and e.get("channel_name") and e.get("label")
            }
            for exp_short, entries in meta.metadata.items()
        }
        exp_to_allowed_fluors: Dict[str, set] = {
            exp: {meta.normalize_channel_name(str(c).strip())
                  for c in (chs or []) if str(c).strip()}
            for exp, chs in v1_map.items()
        }

        before_ch = len(all_pairs)
        kept, excluded, unresolved = [], [], []
        for exp, ch in all_pairs:
            allowed = exp_to_allowed_fluors.get(exp) or set()
            if not allowed:
                kept.append((exp, ch)); continue
            exp_short = extract_ops_key(exp) or exp.split("_")[0]
            fluor = (marker_to_fluor.get(exp_short) or {}).get(ch)
            if fluor is None:
                kept.append((exp, ch)); unresolved.append((exp, ch))
            elif fluor in allowed:
                kept.append((exp, ch))
            else:
                excluded.append((exp, ch, fluor))
        all_pairs = kept
        print(
            f"{paper_flag} channel filter: {before_ch} → {len(all_pairs)} pairs "
            f"({len(excluded)} excluded by per-experiment channel allow-list)"
        )

        def _print_groups(title: str, groups: Dict[Tuple[str, ...], List[str]]) -> None:
            print(title)
            for key, exps in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
                exps_sorted = sorted(set(exps))
                sample = ", ".join(exps_sorted[:3])
                more = f" ... (+{len(exps_sorted) - 3} more)" if len(exps_sorted) > 3 else ""
                label = key[0] if len(key) == 1 else f"{key[0]} → {key[1]}"
                print(f"    {label}: {len(exps_sorted)} experiments — {sample}{more}")

        if excluded:
            grouped = {}
            for exp, ch, fluor in excluded:
                grouped.setdefault((ch, fluor or "?"), []).append(exp)
            _print_groups("  Excluded (marker → fluorophore, not in allow-list):", grouped)
        if unresolved:
            grouped = {}
            for exp, ch in unresolved:
                grouped.setdefault((ch,), []).append(exp)
            _print_groups("  WARNING: kept (marker not resolvable via channel_maps):", grouped)
        if not all_pairs:
            print(f"ERROR: no experiment-channel pairs remain after {paper_flag} channel filter.")
            return

    exp_whitelist = getattr(args, "experiments", None)
    if getattr(args, "match_v02", False):
        v02_manifest = Path("/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.2/dino/all/consensus_sweep/cosine/downsampled_manifest.csv")
        if not v02_manifest.exists():
            print(f"ERROR: --match-v02 requires {v02_manifest}")
            return
        v02_df = pd.read_csv(v02_manifest)
        allowed = set()
        for exps_str in v02_df["experiments"].dropna():
            allowed.update(e.strip() for e in exps_str.split(",") if e.strip())
        before = len(all_pairs)
        all_pairs = [(exp, ch) for exp, ch in all_pairs if exp.split("_")[0] in allowed]
        print(f"--match-v02: {before} → {len(all_pairs)} pairs (matched {len(allowed)} experiments from v0.2)")
        if not all_pairs:
            print("ERROR: no experiment-channel pairs remain after --match-v02 filter.")
            return
    elif exp_whitelist:
        allowed = {e.strip() for e in exp_whitelist.split(",") if e.strip()}
        before = len(all_pairs)
        all_pairs = [(exp, ch) for exp, ch in all_pairs if exp.split("_")[0] in allowed]
        print(f"--experiments filter: {before} → {len(all_pairs)} pairs (allowed {len(allowed)} experiments)")
        if not all_pairs:
            print("ERROR: no experiment-channel pairs remain after --experiments filter.")
            return

    from ops_utils.data.feature_metadata import FeatureMetadata

    fm = FeatureMetadata(metadata_path=maps_path)
    signal_groups = build_signal_groups(all_pairs, fm)

    # Apply phase filter (--phase-only / --no-phase) when running in pooled signal mode
    phase_filter = getattr(args, "phase_filter", None)
    if phase_filter == "phase_only":
        signal_groups = {s: p for s, p in signal_groups.items() if s == "Phase"}
        if not signal_groups:
            print(
                "ERROR: --phase-only found no Phase signal group in the discovered channels."
            )
            return
    elif phase_filter == "no_phase":
        signal_groups = {s: p for s, p in signal_groups.items() if s != "Phase"}

    # Apply --signals filter (retry-failed-shards mode). The matcher
    # normalizes both sides by collapsing all non-alphanumeric runs into a
    # single underscore and lowercasing, so any of these all match the same
    # canonical label "actin filament, FastAct_SPY555 Live Cell Dye":
    #   - canonical channel-map label
    #   - sanitized filename form
    #   - SLURM job-name form
    #   - any space/underscore/comma mix the user happens to type
    signals_arg = getattr(args, "signals", None)
    if signals_arg:
        import re as _re

        def _norm(s: str) -> str:
            return _re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

        wanted_raw = [s.strip() for s in signals_arg.split(",") if s.strip()]
        wanted_norm = {_norm(s): s for s in wanted_raw}
        before = len(signal_groups)

        sig_match: Dict[str, str] = {}  # canonical signal → user input that matched
        for sig in signal_groups.keys():
            for key in (_norm(sig), _norm(sanitize_signal_filename(sig))):
                if key in wanted_norm:
                    sig_match[sig] = wanted_norm[key]
                    break

        matched_inputs = set(sig_match.values())
        missing = [w for w in wanted_raw if w not in matched_inputs]
        if missing:
            print(
                f"WARNING: --signals listed {len(missing)} unknown signal(s) "
                f"(not in discovered groups): {missing[:5]}\n"
                f"  Tip: matcher accepts canonical labels, sanitized filenames, "
                f"and SLURM job-name forms — but the marker name itself must "
                f"appear in the discovered signal groups."
            )
        signal_groups = {s: p for s, p in signal_groups.items() if s in sig_match}
        print(
            f"--signals filter: {before} → {len(signal_groups)} signal groups "
            f"({sorted(signal_groups.keys())[:5]}...)"
        )
        if not signal_groups:
            print("ERROR: no signal groups remain after --signals filter.")
            return

    n_signals = len(signal_groups)

    mode_label = (
        "CellProfiler"
        if cp_override == "cell-profiler"
        else ("Cell-DINO" if cp_override == "cell_dino_features"
              else ("Downsampled" if args.downsampled else "All-Cells"))
    )
    print(
        f"\n{mode_label} PCA Optimization: {len(all_pairs)} channels -> {n_signals} signal groups"
    )
    print(f"Output: {ds_output_dir}")

    # Pre-scan cell counts and compute per-signal target
    print("\nPre-scanning cell counts per signal group...")
    cell_counts = count_cells_per_signal_group(
        signal_groups, storage_roots, feature_dir, maps_path
    )

    if getattr(args, "downsample_per_guide", False):
        # Per-guide cap happens inside pca_sweep_pooled_signal — outer target is
        # just the total cell count (so target_n_cells doesn't artificially trim).
        per_signal_target = dict(cell_counts)
        cap = int(getattr(args, "cells_per_guide", 250))
        # Quick pre-scan: count unique sgRNAs per signal (backed-mode obs reads only)
        print(f"\nPre-scanning unique sgRNAs per signal (for post-cap estimate)...")
        sgrna_counts: Dict[str, int] = {}
        for signal, pairs in signal_groups.items():
            seen = set()
            for exp, ch in pairs:
                path = find_cell_h5ad_path(exp, ch, storage_roots, feature_dir, maps_path)
                if path is None:
                    continue
                try:
                    a = ad.read_h5ad(path, backed="r")
                    if "sgRNA" in a.obs.columns:
                        seen.update(a.obs["sgRNA"].astype(str).unique())
                    a.file.close()
                except Exception:
                    pass
            sgrna_counts[signal] = len(seen)
        print(f"\nSignal group manifest (per-sgRNA cap = {cap:,} cells/guide applied inside each job):")
        print(f"  {'Signal':<45} {'Exps':>5} {'Cells':>10} {'sgRNAs':>7} {'-> Expected':>15}")
        print(f"  {'-'*45} {'-'*5} {'-'*10} {'-'*7} {'-'*15}")
    elif args.downsampled:
        # Equalise: all groups target the smallest group (floor 750k), unless
        # --target-cells is set, in which case use that exact value.
        forced = getattr(args, "target_cells", None)
        if forced is not None and forced > 0:
            global_target = int(forced)
            print(f"\n  --target-cells override: using {global_target:,} cells/signal")
        else:
            MIN_CELLS_FLOOR = 750_000
            global_target = max(min(cell_counts.values()), MIN_CELLS_FLOOR)
        per_signal_target = {s: global_target for s in cell_counts}
        small_groups = {s: n for s, n in cell_counts.items() if n < global_target}
        if small_groups:
            print(
                f"\n  {len(small_groups)} signal group(s) have fewer than {global_target:,} cells (will use all available):"
            )
            for s, n in sorted(small_groups.items(), key=lambda x: x[1]):
                print(f"    {s}: {n:,} cells")
        print(f"\nSignal group manifest (downsampling all to {global_target:,} cells):")
        print(f"  {'Signal':<45} {'Exps':>5} {'Cells':>10} {'-> Downsampled':>15}")
        print(f"  {'-'*45} {'-'*5} {'-'*10} {'-'*15}")
    else:
        # All-cells: load every cell; pca_sweep_pooled_signal uses passthrough PCA for >5M
        per_signal_target = dict(cell_counts)
        print(
            f"\nSignal group manifest (all cells — PCA fit subsampled at >{PCA_FIT_CAP:,}):"
        )
        print(f"  {'Signal':<45} {'Exps':>5} {'Cells':>10}")
        print(f"  {'-'*45} {'-'*5} {'-'*10}")

    manifest_rows = []
    for signal in sorted(signal_groups.keys()):
        pairs = signal_groups[signal]
        n_cells = cell_counts[signal]
        t = per_signal_target[signal]
        if getattr(args, "downsample_per_guide", False):
            n_sg = sgrna_counts.get(signal, 0)
            expected = min(n_cells, n_sg * int(getattr(args, "cells_per_guide", 250)))
            print(f"  {signal:<45} {len(pairs):>5} {n_cells:>10,} {n_sg:>7,} -> {expected:>12,}")
        elif args.downsampled:
            print(f"  {signal:<45} {len(pairs):>5} {n_cells:>10,} -> {t:>15,}")
        else:
            pca_note = f" (passthrough PCA)" if n_cells > PCA_FIT_CAP else ""
            print(f"  {signal:<45} {len(pairs):>5} {n_cells:>10,}{pca_note}")
        manifest_rows.append(
            {
                "signal": signal,
                "n_experiments": len(pairs),
                "n_cells_pooled": n_cells,
                "n_cells_used": t,
                "experiments": ",".join(e.split("_")[0] for e, c in pairs),
            }
        )
    n_unique_exps = len({exp.split("_")[0] for pairs in signal_groups.values() for exp, _ in pairs})
    if getattr(args, "downsample_per_guide", False):
        total_expected = sum(
            min(cell_counts[s], sgrna_counts.get(s, 0) * int(getattr(args, "cells_per_guide", 250)))
            for s in cell_counts
        )
        print(
            f"\n  Total: {n_signals} signal groups, {n_unique_exps} experiments, "
            f"{sum(cell_counts.values()):,} cells → {total_expected:,} expected after cap"
        )
    else:
        print(
            f"\n  Total: {n_signals} signal groups, {n_unique_exps} experiments, "
            f"{sum(cell_counts.values()):,} total cells"
        )

    if getattr(args, "dry_run", False):
        print("\n--dry-run: exiting before processing.")
        return

    pd.DataFrame(manifest_rows).to_csv(
        ds_output_dir / "downsampled_manifest.csv", index=False
    )

    skip_phase2 = (
        getattr(args, "preserve_batch", False)
        or getattr(args, "no_pca", False)
        or attr_config.get("preserve_batch", False)
    )

    # Build common kwargs for signal-group jobs
    def _signal_job_kwargs(signal, pairs):
        kwargs = dict(
            signal=signal,
            exp_channel_pairs=pairs,
            output_dir=str(ds_output_dir),
            target_n_cells=per_signal_target[signal],
            norm_method=args.norm_method,
            distance=args.distance,
        )
        if cp_override:
            kwargs["feature_dir_override"] = cp_override
            if cp_override == "cell-profiler":
                kwargs["sweep_thresholds"] = DEFAULT_SWEEP_THRESHOLDS_CP
        if args.fixed_threshold is not None and args.fixed_threshold > 0:
            kwargs["fixed_threshold"] = args.fixed_threshold
        if getattr(args, "preserve_batch", False):
            kwargs["preserve_batch"] = True
        if getattr(args, "no_pca", False):
            kwargs["no_pca"] = True
        if getattr(args, "zscore_per_experiment", False):
            kwargs["zscore_per_experiment"] = True
        if not getattr(args, "exclude_dud_guides", True):
            kwargs["exclude_dud_guides"] = False
        if getattr(args, "downsample_per_guide", False):
            kwargs["downsample_per_guide"] = True
            kwargs["cells_per_guide"] = int(getattr(args, "cells_per_guide", 250))
        if getattr(args, "agg_method", "mean") != "mean":
            kwargs["agg_method"] = args.agg_method
        if getattr(args, "apply_iss_sidecar", False):
            kwargs["apply_iss_sidecar"] = True
        return kwargs

    if not args.slurm:
        print("\nRunning locally (sequential)...")
        for signal, pairs in signal_groups.items():
            result = pca_sweep_pooled_signal(**_signal_job_kwargs(signal, pairs))
            print(f"  {result}")
        if not skip_phase2:
            result = aggregate_channels(
                output_dir=str(ds_output_dir),
                norm_method=args.norm_method,
                per_unit_subdir="per_signal",
                distance=args.distance,
                random_seed=getattr(args, "seed", 42),
                agg_method=getattr(args, "agg_method", "mean"),
                chromosome_csv=getattr(args, "chromosome_csv", None),
                umap_type=getattr(args, "umap_type", "max"),
            )
            print(result)
            if getattr(args, "second_pca", False):
                print("\nChaining 2nd-pass PCA on aggregate output...")
                _handle_second_pca(args, ds_output_dir)
        else:
            print("  Phase 2 aggregation skipped (--preserve-batch or --no-pca mode)")
        return

    # SLURM mode: one job per signal group — split into high-memory (>4M cells)
    # and standard-memory batches
    HIGH_MEMORY_CELL_THRESHOLD = 4_000_000
    high_mem_jobs = []
    other_jobs = []
    for signal, pairs in signal_groups.items():
        sig_safe = sanitize_signal_filename(signal)[:40]
        job = {
            "name": f"pca_ds_{sig_safe}",
            "func": pca_sweep_pooled_signal,
            "kwargs": _signal_job_kwargs(signal, pairs),
            "metadata": {"signal": signal, "n_experiments": len(pairs)},
        }
        if cell_counts.get(signal, 0) > HIGH_MEMORY_CELL_THRESHOLD:
            high_mem_jobs.append(job)
        else:
            other_jobs.append(job)

    # Signal-group jobs need more time than per-channel (copairs scoring is slow)
    args.slurm_time = max(args.slurm_time, 60)

    from ops_utils.hpc.slurm_batch_utils import (
        submit_parallel_jobs,
        wait_for_multiple_job_arrays,
    )

    slurm_params = _make_slurm_params(args)
    agg_slurm_params = _make_agg_slurm_params(args)

    # Submit both batches without waiting — they run in parallel on SLURM
    job_arrays = []

    if other_jobs:
        print(
            f"\nSubmitting {len(other_jobs)} non-Phase signal-group SLURM jobs ({slurm_params.get('mem', '?')} each)..."
        )
        result_other = submit_parallel_jobs(
            jobs_to_submit=other_jobs,
            experiment="pca_ds_optimization",
            slurm_params=slurm_params,
            log_dir="pca_optimization",
            manifest_prefix="pca_ds_opt",
            wait_for_completion=False,
        )
        if result_other.get("submitted_jobs"):
            job_arrays.append(
                {
                    "submitted_jobs": result_other["submitted_jobs"],
                    "base_job_id": result_other["base_job_id"],
                    "label": "reporters",
                    "slurm_params": slurm_params,
                }
            )

    if high_mem_jobs:
        phase_memory = getattr(args, "phase_memory", "600GB")
        high_mem_slurm_params = {
            **slurm_params,
            "mem": phase_memory,
            "timeout_min": max(slurm_params.get("timeout_min", 60), 360),
        }
        high_mem_names = [j["metadata"]["signal"] for j in high_mem_jobs]
        print(
            f"\nSubmitting {len(high_mem_jobs)} high-memory SLURM job(s) "
            f"({phase_memory}, >4M cells): {', '.join(high_mem_names)}"
        )
        result_high = submit_parallel_jobs(
            jobs_to_submit=high_mem_jobs,
            experiment="pca_ds_optimization_high_mem",
            slurm_params=high_mem_slurm_params,
            log_dir="pca_optimization",
            manifest_prefix="pca_ds_high_opt",
            wait_for_completion=False,
        )
        if result_high.get("submitted_jobs"):
            job_arrays.append(
                {
                    "submitted_jobs": result_high["submitted_jobs"],
                    "base_job_id": result_high["base_job_id"],
                    "label": "high_mem",
                    "slurm_params": high_mem_slurm_params,
                }
            )

    # Wait for ALL arrays with unified progress monitoring
    if job_arrays:
        wait_result = wait_for_multiple_job_arrays(
            job_arrays,
            experiment="pca_ds_optimization",
        )
        if wait_result.get("failed"):
            print(f"\nWarning: {len(wait_result['failed'])} jobs failed")
            for name in wait_result["failed"]:
                print(f"  - {name}")

    # Chain aggregation after all Phase 1 jobs complete (skipped for --preserve-batch and --no-pca)
    if skip_phase2:
        print(
            f"\nAll signal-group jobs complete. Phase 2 aggregation skipped (--preserve-batch or --no-pca mode)."
        )
    else:
        sp_kwargs = _build_second_pca_kwargs(args)
        msg = "aggregation SLURM job"
        if sp_kwargs is not None:
            msg += " (chained with 2nd-pass PCA)"
        print(f"\nAll signal-group jobs complete. Submitting {msg}...")
        _submit_aggregation_slurm(
            str(ds_output_dir),
            args.norm_method,
            "per_signal",
            agg_slurm_params,
            "pca_ds_aggregation",
            "pca_ds_agg",
            distance=args.distance,
            second_pca_kwargs=sp_kwargs,
            random_seed=getattr(args, "seed", 42),
            agg_method=getattr(args, "agg_method", "mean"),
            chromosome_csv=getattr(args, "chromosome_csv", None),
            umap_type=getattr(args, "umap_type", "max"),
        )


