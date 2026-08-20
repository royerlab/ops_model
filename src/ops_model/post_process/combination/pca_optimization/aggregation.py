"""Per-reporter scoring + cross-channel concat helpers for pca_optimization.

These functions implement the Phase 2 aggregation primitives that run
after Phase 1's per-signal h5ads are on disk:

* ``_load_per_unit_blocks`` — read every ``per_signal/<signal>_guide.h5ad`` +
  matching gene h5ad, score all 4 mAP metrics per reporter, return the
  loaded blocks + report rows + per-reporter mAP DataFrames.
* ``_concat_and_normalize`` — horizontal concat across reporters, NTC
  normalize, re-aggregate to gene level.
* ``_score_activity_aggregated`` — final scoring of the concatenated
  guide-level h5ad.
* ``_score_single_reporter_metrics`` — per-reporter mAP scoring building
  block called by ``_load_per_unit_blocks``.
* ``_save_aggregated_h5ads`` / ``_save_per_reporter_metric_matrices`` —
  write out the canonical ``guide_pca_optimized.h5ad`` +
  ``gene_pca_optimized.h5ad`` and the per-(reporter × entity) matrices.
* ``_atomic_write_h5ad`` — write h5ad to a .tmp then os.replace so a
  failed write doesn't leave a half-overwritten target.
* ``_annotate_genes_from_panel`` — merge the canonical annotated gene
  panel CSV into ``adata_gene.obs``.
* ``_plot_chad_umap`` — CHAD-cluster colored UMAP renderer.

Public surface unchanged: every symbol that lived on
``pca_optimization`` is re-imported there for back-compat.

``CHAD_ANNOTATION_PATH`` (mutable module global, set by main() from
``--chad-annotation``) is imported lazily inside the function bodies so
its current value is read each call.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from ops_model.features.anndata_utils import (
    aggregate_to_level,
    hconcat_by_perturbation,
    normalize_guide_adata,
)
from ops_utils.analysis.map_scores import (
    compute_auc_score,
    phenotypic_activity_assesment,
)

from ops_model.post_process.combination.pca_optimization.sweep_core import (
    _prepare_for_copairs,
)


ANNOTATED_GENE_PANEL_PATH = Path(
    "/hpc/projects/icd.fast.ops/configs/annotated_gene_panel_July2025.csv"
)


def _plot_chad_umap(umap_coords, genes, gene_to_cluster, out_path, plt, _logger):
    """Plot UMAP colored by CHAD cluster with gene labels."""
    import seaborn as sns
    from itertools import product

    cats = [gene_to_cluster.get(g, "Uncategorized") for g in genes]
    is_ntc = np.array([str(g).startswith("NTC") for g in genes])
    unique_cats = sorted(set(c for c in cats if c != "Uncategorized"))

    # 10 dark colors x 6 markers = 60 unique combos for 50+ clusters
    # 20 colors x 10 markers = 200 unique combos
    colors_20 = sns.color_palette("dark", 10) + sns.color_palette("Set1", 9) + sns.color_palette("Set2", 1)
    markers_10 = ["o", "s", "D", "^", "v", "P", "p", "h", "*", "X"]
    combos = list(product(colors_20, markers_10))
    cat_to_color = {cat: combos[i % len(combos)][0] for i, cat in enumerate(unique_cats)}
    cat_to_marker = {cat: combos[i % len(combos)][1] for i, cat in enumerate(unique_cats)}

    # Square UMAP on left, legend fills right
    fig = plt.figure(figsize=(30, 12))
    ax = fig.add_axes([0.04, 0.04, 0.50, 0.94])  # [left, bottom, width, height]
    ax.set_box_aspect(0.7)

    # Pin axis limits to full UMAP extent so every plot has identical framing
    x_min, x_max = umap_coords[:, 0].min(), umap_coords[:, 0].max()
    y_min, y_max = umap_coords[:, 1].min(), umap_coords[:, 1].max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # Uncategorized background
    uncat_mask = np.array([c == "Uncategorized" for c in cats]) & ~is_ntc
    if uncat_mask.any():
        ax.scatter(umap_coords[uncat_mask, 0], umap_coords[uncat_mask, 1],
                   c=[(0.85, 0.85, 0.85)], s=40, alpha=0.2, edgecolors="none", label="Uncategorized")

    # Categorized genes — unique color+marker per cluster
    for cat in unique_cats:
        if cat == "OR controls":
            continue  # Plotted separately with special marker
        mask = np.array([c == cat for c in cats]) & ~is_ntc
        if mask.any():
            # Truncate long labels for legend readability
            label = cat if len(cat) <= 60 else cat[:57] + "..."
            ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       c=[cat_to_color[cat]], marker=cat_to_marker[cat],
                       s=150, alpha=0.85, edgecolors="white", linewidths=0.3, label=label)

    # OR controls — bright red X, larger than NTCs
    is_or = np.array([gene_to_cluster.get(g, "") == "OR controls" for g in genes])
    if is_or.any():
        ax.scatter(umap_coords[is_or, 0], umap_coords[is_or, 1],
                   c="#FF0000", marker="X", s=285, alpha=0.7, edgecolors="#CC0000",
                   linewidths=0.6, label="OR controls", zorder=11)

    # NTCs
    if is_ntc.any():
        ax.scatter(umap_coords[is_ntc, 0], umap_coords[is_ntc, 1],
                   c="#e08080", marker="X", s=195, alpha=0.4, edgecolors="#b05050",
                   linewidths=0.3, label="NTC", zorder=10)

    # Gene labels — only annotated genes, radial offset to avoid overlap
    rng = np.random.RandomState(42)
    for i, gene in enumerate(genes):
        if str(gene).startswith("NTC"):
            continue
        if gene_to_cluster.get(gene, "Uncategorized") == "Uncategorized":
            continue
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(40, 80)
        dx = radius * np.cos(angle)
        dy = radius * np.sin(angle)
        color = "#FF0000" if gene_to_cluster.get(gene) == "OR controls" else cat_to_color.get(gene_to_cluster.get(gene, ""), "black")
        ax.annotate(gene, xy=(umap_coords[i, 0], umap_coords[i, 1]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=14, alpha=0.75, ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", color=color, alpha=0.5, lw=0.5))

    # Use 1 column by default; only split to 2 if too many items to fit vertically
    # At fontsize 13 with labelspacing 0.4, ~40 rows fit in 18in figure
    ncol = 2 if len(unique_cats) > 40 else 1
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=16,
              framealpha=0.9, ncol=ncol, columnspacing=0.6, handletextpad=0.3,
              labelspacing=0.4)
    ax.set_title("Gene UMAP -- colored by CHAD cluster", fontsize=32, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=24)
    ax.set_ylabel("UMAP 2", fontsize=24)
    ax.tick_params(labelsize=18)
    # Fixed axes position so legend size doesn't distort UMAP aspect across plots
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    _logger.info(f"  Saved CHAD UMAP: {out_path}")


def _score_single_reporter_metrics(
    g_raw, norm_method, _logger, null_size=100_000, distance="cosine"
):
    """Score all 4 phenotypic metrics for one reporter's guide h5ad.

    Uses a smaller null_size than the aggregate run for speed.
    Returns a dict with:
      Scalars: activity, auc, distinctiveness, corum, chad, plus
        unfiltered variants distinctiveness_all/corum_all/chad_all
        (NaN on failure).
      Per-gene/per-complex DataFrames (NEW, for downstream
      reuse — atlas etc.):
        activity_df         : per-perturbation activity mAP
        distinctiveness_df  : per-perturbation distinctiveness mAP
        corum_df            : per-CORUM-complex consistency mAP
        chad_df             : per-CHAD-complex consistency mAP
      Each DataFrame is None if its scoring step failed.
    """
    from ops_model.post_process.combination.pca_optimization import CHAD_ANNOTATION_PATH, EBI_ANNOTATION_PATH
    import math

    result = {
        k: math.nan
        for k in (
            "activity",
            "auc",
            "distinctiveness",
            "corum",
            "chad",
            "ebi_plus",
            "distinctiveness_all",
            "corum_all",
            "chad_all",
            "ebi_plus_all",
        )
    }
    # Per-gene / per-complex DataFrames (raw mAP scores per entity for
    # this reporter). The 4 mAP-scoring helpers below all return
    # `(df, scalar_ratio)` — previously we kept only the scalar; now
    # we also stash the df so callers can pivot to a per-(reporter,
    # entity) matrix and save it.
    result["activity_df"] = None
    result["distinctiveness_df"] = None
    result["corum_df"] = None
    result["chad_df"] = None
    result["ebi_plus_df"] = None
    try:
        from ops_utils.analysis.map_scores import (
            phenotypic_activity_assesment,
            phenotypic_distinctivness,
            phenotypic_consistency_corum,
            phenotypic_consistency_manual_annotation,
            phenotypic_ebi_plus,
        )

        g_norm = normalize_guide_adata(g_raw.copy(), norm_method)
        g_norm = _prepare_for_copairs(g_norm)

        activity_map, active_ratio = phenotypic_activity_assesment(
            g_norm,
            plot_results=False,
            null_size=null_size,
            distance=distance,
        )
        result["activity"] = float(active_ratio)
        result["auc"] = float(compute_auc_score(activity_map))
        result["activity_df"] = activity_map

        dist_df, dist_ratio = phenotypic_distinctivness(
            g_norm,
            plot_results=False,
            null_size=null_size,
            distance=distance,
        )
        result["distinctiveness"] = float(dist_ratio)
        result["distinctiveness_all"] = result["distinctiveness"]
        result["distinctiveness_df"] = dist_df

        ebi_plus_df, ebi_plus_ratio = phenotypic_ebi_plus(
            g_norm,
            plot_results=False,
            null_size=null_size,
            distance=distance,
        )
        result["ebi_plus"] = float(ebi_plus_ratio)
        result["ebi_plus_all"] = result["ebi_plus"]
        result["ebi_plus_df"] = ebi_plus_df

        e_norm = aggregate_to_level(
            g_norm, "gene", preserve_batch_info=False, subsample_controls=False
        )
        e_norm = _prepare_for_copairs(e_norm)

        corum_df, corum_ratio = phenotypic_consistency_corum(
            e_norm,
            plot_results=False,
            null_size=null_size,
            cache_similarity=True,
            distance=distance,
        )
        result["corum"] = float(corum_ratio)
        result["corum_all"] = result["corum"]
        result["corum_df"] = corum_df

        chad_df, chad_ratio = phenotypic_consistency_manual_annotation(
            e_norm,
            plot_results=False,
            null_size=null_size,
            cache_similarity=True,
            distance=distance,
            annotation_path=CHAD_ANNOTATION_PATH,
        )
        result["chad"] = float(chad_ratio)
        result["chad_all"] = result["chad"]
        result["chad_df"] = chad_df

        # EBI Complex Portal consistency per reporter (over ALL geneKOs, like corum/chad above)
        ebi_df, ebi_ratio = phenotypic_consistency_manual_annotation(
            e_norm,
            plot_results=False,
            null_size=null_size,
            cache_similarity=True,
            distance=distance,
            annotation_path=EBI_ANNOTATION_PATH,
        )
        result["ebi"] = float(ebi_ratio)
        result["ebi_all"] = result["ebi"]
        result["ebi_df"] = ebi_df

    except Exception as exc:
        _logger.warning(f"  Per-reporter metrics scoring failed: {exc}")
    return result


def _load_per_unit_blocks(per_unit_dir, norm_method, _logger, distance="cosine"):
    """Load per-channel/per-signal guide+gene h5ads, return blocks +
    report rows + per-reporter metric DataFrames.

    Returns a 5-tuple:
        (guide_blocks, gene_blocks, report_rows, total_cells,
         per_reporter_metric_dfs)
    where per_reporter_metric_dfs is
        {signal_name: {"activity": df, "distinctiveness": df,
                       "corum": df, "chad": df}}
    Used by the caller to pivot into a per-(reporter, entity) matrix
    and save it alongside `pca_report.csv` so atlas / downstream
    consumers can read all 4 mAP scores per reporter.
    """
    guide_files = sorted(per_unit_dir.glob("*_guide.h5ad"))
    if not guide_files:
        return None, None, [], 0, {}

    _logger.info(f"Found {len(guide_files)} per-unit guide files")
    guide_blocks, gene_blocks, report_rows = [], [], []
    per_reporter_metric_dfs = {}
    total_cells = 0

    for gf in guide_files:
        file_prefix = gf.stem.replace("_guide", "")
        gene_file = per_unit_dir / f"{file_prefix}_gene.h5ad"
        if not gene_file.exists():
            _logger.warning(f"  Skipping {file_prefix}: no gene file")
            continue

        g = ad.read_h5ad(gf)
        sig = g.uns.get("signal", file_prefix)
        if sig == "unknown" or sig.startswith("(unmapped:"):
            _logger.warning(
                f"  Skipping {file_prefix}: unmapped channel (signal={sig!r})"
            )
            continue

        e = ad.read_h5ad(gene_file)
        guide_blocks.append(g)
        gene_blocks.append(e)
        n_cells = int(g.uns.get("n_cells", 0))
        total_cells += n_cells

        _logger.info(f"  {sig}: scoring all 4 metrics per-reporter...")
        reporter_metrics = _score_single_reporter_metrics(
            g, norm_method, _logger, distance=distance
        )

        # Stash the per-gene/per-complex DataFrames keyed by signal so
        # the caller can pivot them into per-(reporter, entity)
        # matrices and save. Skip None values (failed scoring step).
        per_reporter_metric_dfs[sig] = {
            "activity": reporter_metrics.get("activity_df"),
            "distinctiveness": reporter_metrics.get("distinctiveness_df"),
            "corum": reporter_metrics.get("corum_df"),
            "chad": reporter_metrics.get("chad_df"),
            "ebi": reporter_metrics.get("ebi_df"),
            "ebi_plus": reporter_metrics.get("ebi_plus_df"),
        }

        report_rows.append(
            {
                "experiment": g.uns.get("experiment", ""),
                "channel": g.uns.get("channel", ""),
                "signal": sig,
                "n_cells": n_cells,
                "n_features_raw": int(g.uns.get("n_features_raw", 0)),
                "peak_threshold": float(g.uns.get("pca_threshold", 0)),
                "n_pcs": int(g.uns.get("n_pcs", 0)),
                "explained_variance": float(g.uns.get("explained_variance", 0)),
                "activity": reporter_metrics["activity"],
                "auc": reporter_metrics["auc"],
                "distinctiveness": reporter_metrics["distinctiveness"],
                "corum": reporter_metrics["corum"],
                "chad": reporter_metrics["chad"],
                "ebi_plus": reporter_metrics["ebi_plus"],
                "distinctiveness_all": reporter_metrics["distinctiveness_all"],
                "corum_all": reporter_metrics["corum_all"],
                "chad_all": reporter_metrics["chad_all"],
                "ebi_plus_all": reporter_metrics["ebi_plus_all"],
            }
        )
        _logger.info(
            f"  {sig}: {g.n_obs} guides x {g.n_vars} PCs @ {g.uns.get('pca_threshold', '?')} | "
            f"act={reporter_metrics['activity']:.1%} dist={reporter_metrics['distinctiveness']:.1%} "
            f"ebi+={reporter_metrics['ebi_plus']:.1%} "
            f"corum={reporter_metrics['corum']:.1%} chad={reporter_metrics['chad']:.1%} | "
            f"all: dist={reporter_metrics['distinctiveness_all']:.1%} "
            f"corum={reporter_metrics['corum_all']:.1%} chad={reporter_metrics['chad_all']:.1%}"
        )

    return (guide_blocks or None, gene_blocks, report_rows, total_cells,
            per_reporter_metric_dfs)


def _concat_and_normalize(guide_blocks, gene_blocks, norm_method, _logger, agg_method: str = "mean"):
    """Horizontal concat, NTC normalize, re-aggregate to gene, strip obs for copairs.

    ``agg_method`` controls the guide→gene reduction (default ``mean``).
    """
    adata_guide = hconcat_by_perturbation(guide_blocks, "guide")
    adata_gene = hconcat_by_perturbation(gene_blocks, "gene")
    del guide_blocks, gene_blocks

    _logger.info(
        f"Concatenated: {adata_guide.n_obs} guides, {adata_guide.n_vars} features"
    )
    _logger.info(f"NTC normalizing at guide level...")
    adata_guide = normalize_guide_adata(adata_guide, norm_method)
    adata_guide.X = np.asarray(adata_guide.X, dtype=np.float32)

    _logger.info(f"Aggregating guide→gene with method={agg_method!r}")
    adata_gene = aggregate_to_level(
        adata_guide,
        "gene",
        method=agg_method,
        preserve_batch_info=False,
        subsample_controls=False,
    )
    _logger.info(f"  Guide: {adata_guide.n_obs} obs, {adata_guide.n_vars} features")
    _logger.info(f"  Gene: {adata_gene.n_obs} obs, {adata_gene.n_vars} features")

    # Strip obs to copairs-required columns (extra string cols cause isnan error)
    adata_guide = _prepare_for_copairs(adata_guide)
    adata_gene = _prepare_for_copairs(adata_gene)

    return adata_guide, adata_gene


def _score_activity_aggregated(adata_guide, metrics_dir, _logger, distance="cosine"):
    """Run phenotypic activity scoring on aggregated data. Returns (activity_map, ratio, auc)."""
    _logger.info(f"Running activity scoring...")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    try:
        activity_map, active_ratio = phenotypic_activity_assesment(
            adata_guide,
            plot_results=False,
            null_size=100_000,
            distance=distance,
        )
        activity_map.to_csv(metrics_dir / "phenotypic_activity.csv", index=False)
        auc = compute_auc_score(activity_map)
        _logger.info(f"  Activity: {active_ratio:.1%} active, AUC={auc:.4f}")

        inactive = activity_map[~activity_map["below_corrected_p"]]
        inactive = inactive[inactive["perturbation"] != "NTC"]
        if len(inactive) > 0:
            names = sorted(inactive["perturbation"].tolist())
            _logger.info(f"  Inactive genes ({len(names)}): {', '.join(names)}")
        else:
            _logger.info(f"  All non-NTC perturbations are active")
        return activity_map, active_ratio, auc
    except Exception as exc:
        _logger.error(f"  Activity scoring failed: {exc}")
        return None, 0.0, 0.0


def _save_per_reporter_metric_matrices(
    per_reporter_metric_dfs,
    output_dir,
    _logger,
):
    """Pivot the per-reporter mAP DataFrames collected by
    `_load_per_unit_blocks` into 4 per-(reporter × entity) matrices and
    save them alongside `pca_report.csv`. Outputs land in
    `<output_dir>/plots/marker_overlay/`:

        gene_reporter_activity_raw.csv         (rows = perturbations)
        gene_reporter_distinctiveness_raw.csv  (rows = perturbations)
        complex_reporter_corum_consistency.csv (rows = CORUM complexes)
        complex_reporter_chad_consistency.csv  (rows = CHAD complexes)

    The distinctiveness CSV mirrors what
    `gene_best_marker_assignment.compute_all_scores` produces (same
    schema), so atlas / heatmap consumers can pick it up by either
    path. The CHAD/CORUM matrices are NEW — they're the per-(complex,
    reporter) consistency mAP that previously had no precomputed
    artifact (only per-complex scalars in `pca_report.csv`).
    """
    if not per_reporter_metric_dfs:
        return
    overlay_dir = Path(output_dir) / "plots" / "marker_overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    def _pivot(metric_key, key_col, value_col, out_csv):
        """Build {entity_id: {reporter: mAP}} → DataFrame, save."""
        pivot = {}
        for sig, dfs in per_reporter_metric_dfs.items():
            df = dfs.get(metric_key)
            if df is None or len(df) == 0:
                continue
            if key_col not in df.columns or value_col not in df.columns:
                _logger.warning(
                    f"  [{metric_key}] {sig}: missing {key_col!r}/{value_col!r}"
                    f" in returned df — skipping"
                )
                continue
            pivot[sig] = dict(zip(df[key_col].astype(str),
                                  df[value_col].astype(float)))
        if not pivot:
            _logger.warning(f"  [{metric_key}] no data to save")
            return
        all_entities = sorted({k for r in pivot.values() for k in r})
        out_df = pd.DataFrame(
            {sig: [pivot[sig].get(e, float("nan")) for e in all_entities]
             for sig in sorted(pivot)},
            index=all_entities,
        )
        out_df.index.name = key_col
        out_df.to_csv(out_csv)
        _logger.info(
            f"  Saved {out_csv.name}: {out_df.shape[0]} entities × "
            f"{out_df.shape[1]} reporters"
        )

    _pivot(
        "activity", "perturbation", "mean_average_precision",
        overlay_dir / "gene_reporter_activity_raw.csv",
    )
    _pivot(
        "distinctiveness", "perturbation", "mean_average_precision",
        overlay_dir / "gene_reporter_distinctiveness_raw.csv",
    )
    _pivot(
        "ebi_plus", "ebi_group", "mean_average_precision",
        overlay_dir / "group_reporter_ebi_plus_raw.csv",
    )
    _pivot(
        "corum", "complex_num", "mean_average_precision",
        overlay_dir / "complex_reporter_corum_consistency.csv",
    )
    _pivot(
        "chad", "complex_num", "mean_average_precision",
        overlay_dir / "complex_reporter_chad_consistency.csv",
    )
    _pivot(
        "ebi", "complex_num", "mean_average_precision",
        overlay_dir / "complex_reporter_ebi_consistency.csv",
    )


def _save_aggregated_h5ads(
    adata_guide,
    adata_gene,
    report_rows,
    output_dir,
    r,
    a,
    norm_method,
    total_cells,
    _logger,
):
    """Write guide/gene h5ads and report CSV with metadata."""
    adata_guide.uns["pca_optimized"] = True
    adata_guide.uns["pca_report"] = pd.DataFrame(report_rows).to_dict(orient="list")
    adata_guide.uns["baseline_activity"] = float(r)
    adata_guide.uns["baseline_auc"] = float(a)
    adata_guide.uns["norm_method"] = norm_method
    adata_guide.uns["total_cells"] = total_cells

    adata_gene.uns["pca_optimized"] = True
    adata_gene.uns["norm_method"] = norm_method

    adata_guide.write_h5ad(output_dir / "guide_pca_optimized.h5ad")
    adata_gene.write_h5ad(output_dir / "gene_pca_optimized.h5ad")
    pd.DataFrame(report_rows).to_csv(output_dir / "pca_report.csv", index=False)
    _logger.info(
        f"  Saved guide_pca_optimized.h5ad, gene_pca_optimized.h5ad, pca_report.csv"
    )


def _atomic_write_h5ad(adata: ad.AnnData, path: Path, _logger) -> None:
    """Write h5ad to a sibling .tmp path, then os.replace into target.

    AnnData's write_h5ad is NOT atomic — a mid-write failure (e.g. unsupported
    obs dtype) leaves the target file partially overwritten and unreadable.
    This helper writes to a temp path first and only moves on success.

    Also duplicates the in-pipeline ``perturbation`` column as ``geneKO_name``
    for the saved file (matches reference gene-level h5ad layout; viewers pick
    it up as the default ID variable) while keeping ``perturbation`` so
    downstream readers (e.g. 2nd-pass PCA) still find it.
    """
    import os

    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    # Coerce object obs columns with mixed/NaN values to strings so h5py's
    # vlen-string writer doesn't choke (matches what existing pipelines do).
    for col in list(adata.obs.columns):
        if adata.obs[col].dtype == object:
            adata.obs[col] = adata.obs[col].astype(str).replace({"nan": ""})

    added_geneko = False
    if "perturbation" in adata.obs.columns and "geneKO_name" not in adata.obs.columns:
        adata.obs["geneKO_name"] = adata.obs["perturbation"]
        added_geneko = True
    try:
        adata.write_h5ad(tmp)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    finally:
        if added_geneko:
            adata.obs = adata.obs.drop(columns=["geneKO_name"])


def _annotate_genes_from_panel(adata_gene: ad.AnnData, _logger) -> None:
    """Merge annotated_gene_panel_July2025.csv metadata into adata_gene.obs by gene symbol.

    Adds columns like LongName, NCBI_ID, Funk/Ramezani/Replogle map coords,
    In_corum_complexes, funk_cluster, Gene_Category, ... — matching the reference
    gene-level h5ad layout (cropseq_gene_level.h5ad).
    """
    if not ANNOTATED_GENE_PANEL_PATH.exists():
        _logger.warning(f"  Gene panel not found at {ANNOTATED_GENE_PANEL_PATH}, skipping annotation")
        return
    panel = pd.read_csv(ANNOTATED_GENE_PANEL_PATH, index_col=0)
    panel = panel.set_index("Gene.name")
    # Preserve existing obs columns; only add panel columns that aren't already present
    cols_to_add = [c for c in panel.columns if c not in adata_gene.obs.columns]
    joined = adata_gene.obs.join(panel[cols_to_add], how="left")
    adata_gene.obs = joined
    n_matched = panel.index.intersection(adata_gene.obs_names).size
    _logger.info(
        f"  Annotated {n_matched}/{adata_gene.n_obs} genes from panel "
        f"({len(cols_to_add)} columns added)"
    )
