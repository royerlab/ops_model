"""Extra overlay plots on top of the per-aggregate UMAP/PHATE embeddings.

Three additions on top of ``_compute_and_plot_embeddings``:

1. ``save_supercluster_overlays`` — static matplotlib PNGs duplicating each
   existing UMAP/PHATE plot, recolored by gene super-category (CHAD-boosted)
   and by direct CHAD cluster name.
2. ``save_leiden_overlays`` — runs scanpy Leiden at several resolutions and
   saves per-resolution static PNGs plus a CSV of (gene, cluster) per
   resolution. Outputs land under ``plots/leiden/{level}/{png|csv}/`` so
   guide/gene live in their own folders and PNGs/CSVs are split.
3. ``save_interactive_html`` — Plotly HTML (one per level x embedding) with
   hover showing perturbation, all 4 mAP metrics, super-category, CHAD
   cluster, CORUM complex, and a dropdown to switch the color overlay
   between super-category, CHAD cluster, CORUM complex, and each Leiden
   resolution.

All helpers are best-effort: a missing dependency or scoring map degrades
gracefully (logged warning, partial output saved).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


DEFAULT_LEIDEN_RESOLUTIONS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 20.0, 30.0, 50.0, 100.0)
DEFAULT_ENRICHR_LIBRARIES = (
    "GO_Biological_Process_2025",
    "GO_Cellular_Component_2025",
    "Reactome_2022",
    "KEGG_2026",
)
DEFAULT_SUPERCATEGORY_CONFIG = Path(
    "/home/gav.sturm/linked_folders/mydata/ops_mono/organelle_profiler/configs/gene_supercategory_mapping.yaml"
)


# =============================================================================
# Annotation loading
# =============================================================================


def load_overlay_maps(
    supercategory_config_path: Optional[Path] = None,
) -> Dict[str, Dict[str, str]]:
    """Build the three gene → annotation dicts used for overlay coloring.

    Returns ``{"super": ..., "chad": ..., "corum": ...}`` where each value is
    a ``gene_name -> annotation`` dict. Genes without an annotation are not
    present in the returned dict (callers default them to ``"Uncategorized"``).
    """
    out: Dict[str, Dict[str, str]] = {"super": {}, "chad": {}, "corum": {}}

    cfg_path = Path(supercategory_config_path or DEFAULT_SUPERCATEGORY_CONFIG)
    if not cfg_path.exists():
        logger.warning(
            "Super-category config not found at %s — supercluster overlays will be empty",
            cfg_path,
        )
    else:
        try:
            import yaml

            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            from ops_utils.analysis.gene_supercategories import (
                build_gene_supercategory_map,
            )

            out["super"] = build_gene_supercategory_map(cfg, boosted=True)
            logger.info(
                "  Loaded super-category map: %d genes assigned across %d categories",
                len(out["super"]),
                len(set(out["super"].values())),
            )
        except Exception as exc:
            logger.warning("  Super-category map load failed: %s", exc)

    try:
        # CHAD cluster (from heatmap pattern)
        from ops_utils.analysis.gene_supercategories import (
            _load_chad_hierarchy,
            DEFAULT_CHAD_PATH,
        )

        chad_path = (
            Path(cfg.get("chad_hierarchy_path", str(DEFAULT_CHAD_PATH)))
            if cfg_path.exists()
            else DEFAULT_CHAD_PATH
        )
        chad = _load_chad_hierarchy(chad_path)
        gene_to_cluster: Dict[str, str] = {}
        for _id, cluster in chad.items():
            if not isinstance(cluster, dict) or "name" not in cluster:
                continue
            cname = cluster["name"]
            for gene in cluster.get("genes", []):
                if gene not in gene_to_cluster:
                    gene_to_cluster[gene] = cname
        out["chad"] = gene_to_cluster
        logger.info(
            "  Loaded CHAD cluster map: %d genes across %d clusters",
            len(gene_to_cluster),
            len(set(gene_to_cluster.values())),
        )
    except Exception as exc:
        logger.warning("  CHAD cluster map load failed: %s", exc)

    try:
        out["corum"] = _build_corum_map()
        logger.info("  Loaded CORUM complex map: %d genes", len(out["corum"]))
    except Exception as exc:
        logger.warning("  CORUM map load failed: %s", exc)

    return out


def _build_corum_map() -> Dict[str, str]:
    """Build gene → CORUM complex name from the annotated gene panel CSV."""
    import ast

    panel_path = Path(
        "/hpc/projects/intracellular_dashboard/ops/configs/annotated_gene_panel_July2025.csv"
    )
    if not panel_path.exists():
        return {}
    df = pd.read_csv(panel_path)
    gene_to_complex: Dict[str, str] = {}
    if "Gene.name" not in df.columns or "In_same_complex_with" not in df.columns:
        return {}
    for _, row in df.iterrows():
        gene = row.get("Gene.name")
        if not isinstance(gene, str) or not gene.strip():
            continue
        raw = row.get("In_same_complex_with")
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            members = ast.literal_eval(raw) or []
        except Exception:
            continue
        members = [str(g) for g in members if isinstance(g, str)]
        if not members:
            continue
        # Use a deterministic label: sorted members joined (truncated for display)
        label = ",".join(sorted(set(members + [gene])))
        if len(label) > 60:
            label = label[:57] + "..."
        gene_to_complex[gene] = label
    return gene_to_complex


# =============================================================================
# Leiden clustering at multiple resolutions
# =============================================================================


_LIBRARY_OBS_COL = {
    "GO_Biological_Process_2025": "go_bp_term",
    "GO_Cellular_Component_2025": "go_cc_term",
    "Reactome_2022": "reactome_term",
    "KEGG_2026": "kegg_term",
}

# Significance threshold for surfacing an Enrichr term as the "top" term in obs.
# Terms with adj_pvalue > this are treated as not significant → NaN in obs.
# Standard BH-FDR cutoff.
ENRICHMENT_ADJ_P_THRESHOLD = 0.05


def _significant_term(rec: Optional[Dict], thresh: float = ENRICHMENT_ADJ_P_THRESHOLD) -> Optional[str]:
    """Return the term string if its adj_pvalue is below ``thresh``, else None."""
    if not rec:
        return None
    term = str(rec.get("term", "") or "")
    if not term:
        return None
    p = rec.get("adj_pvalue")
    try:
        if p is None or float(p) > thresh:
            return None
    except (TypeError, ValueError):
        return None
    return term


def _apply_enrichment_to_adata(
    adata,
    leiden_results: Dict[str, np.ndarray],
    enrichment_per_res: Dict[str, Dict[str, Dict]],
    canonical_resolution: str = "leiden_r4",
    _logger=logger,
) -> None:
    """For each gene, attach the top enriched term per ontology library at the
    gene's cluster (at ``canonical_resolution``) as obs columns.

    Adds: go_bp_term, go_cc_term, reactome_term, kegg_term. Genes whose
    cluster has no enrichment record get an empty string.
    """
    if canonical_resolution not in leiden_results:
        # Fall back to any available resolution (prefer mid-range)
        candidates = [r for r in ("leiden_r4", "leiden_r3", "leiden_r5", "leiden_r2") if r in leiden_results]
        if not candidates:
            return
        canonical_resolution = candidates[0]
    if canonical_resolution not in enrichment_per_res:
        return

    cluster_labels = np.asarray(leiden_results[canonical_resolution]).astype(str)
    if len(cluster_labels) != adata.n_obs:
        return

    enrich = enrichment_per_res[canonical_resolution]
    # Build per-library cluster_id -> top_term lookup (only terms with
    # adj_pvalue <= ENRICHMENT_ADJ_P_THRESHOLD survive; rest become "")
    lib_lookup: Dict[str, Dict[str, str]] = {lib: {} for lib in _LIBRARY_OBS_COL}
    # Best-across-libraries cluster_id -> (term, library) lookup (lowest adj p-value)
    top_lookup: Dict[str, Tuple[str, str]] = {}
    for cid, rec in (enrich or {}).items():
        by_lib = (rec or {}).get("by_library") or {}
        for lib, terms in by_lib.items():
            if lib in lib_lookup and terms:
                term = _significant_term(terms[0])
                if term:
                    lib_lookup[lib][str(cid)] = term
        top1 = (rec or {}).get("top1") or {}
        term = _significant_term(top1)
        if term:
            top_lookup[str(cid)] = (term, str(top1.get("library", "")))

    # Enrichment keys are stored as "cluster_{label}" — match that convention.
    cluster_keys = [f"cluster_{c}" for c in cluster_labels]
    for lib, col_name in _LIBRARY_OBS_COL.items():
        terms_per_gene = [lib_lookup[lib].get(c) for c in cluster_keys]
        adata.obs[col_name] = pd.Categorical(terms_per_gene)
        n_assigned = sum(1 for t in terms_per_gene if t)
        _logger.info(
            f"  Annotated {n_assigned}/{adata.n_obs} genes with {col_name} "
            f"(from {canonical_resolution}, adj_p<={ENRICHMENT_ADJ_P_THRESHOLD})"
        )

    # Single best-across-libraries term per gene's cluster (lowest adj p-value)
    top_pairs = [top_lookup.get(c) for c in cluster_keys]
    top_terms = [p[0] if p else None for p in top_pairs]
    top_libs = [p[1] if p else None for p in top_pairs]
    adata.obs["top_ontology"] = pd.Categorical(top_terms)
    adata.obs["top_ontology_library"] = pd.Categorical(top_libs)
    n_assigned = sum(1 for t in top_terms if t)
    _logger.info(
        f"  Annotated {n_assigned}/{adata.n_obs} genes with top_ontology "
        f"(best adj-p across {len(_LIBRARY_OBS_COL)} libraries, from {canonical_resolution})"
    )

    # Per-resolution top_ontology (term only) for every resolution we have
    # enrichment for — lets users pick the granularity for downstream views.
    for res_col, res_enrich in enrichment_per_res.items():
        if res_col not in leiden_results:
            continue
        labels = np.asarray(leiden_results[res_col]).astype(str)
        if len(labels) != adata.n_obs:
            continue
        res_top: Dict[str, str] = {}
        for cid, rec in (res_enrich or {}).items():
            term = _significant_term((rec or {}).get("top1"))
            if term:
                res_top[str(cid)] = term
        terms = [res_top.get(f"cluster_{l}") for l in labels]
        # Strip "leiden_" prefix; trailing dot stays for "r0.5" etc.
        suffix = res_col.replace("leiden_", "")
        col_name = f"top_ontology_{suffix}"
        adata.obs[col_name] = pd.Categorical(terms)
        n_assigned = sum(1 for t in terms if t)
        _logger.info(
            f"  Annotated {n_assigned}/{adata.n_obs} genes with {col_name}"
        )


def _apply_overlay_maps_to_adata(
    adata,
    overlay_maps: Dict[str, Dict[str, str]],
    _logger=logger,
) -> None:
    """Add CHAD cluster, CORUM complex, and super-category as obs columns.

    Uses ``perturbation`` to look up annotations; missing genes get NaN.
    Stored as categoricals so downstream viewers handle them naturally.
    """
    pert_col = "perturbation" if "perturbation" in adata.obs.columns else None
    if pert_col is None:
        return
    perts = adata.obs[pert_col].astype(str).values
    col_map = [
        ("chad_cluster", overlay_maps.get("chad") or {}),
        ("corum_complex", overlay_maps.get("corum") or {}),
        ("supercategory", overlay_maps.get("super") or {}),
    ]
    for col_name, lookup in col_map:
        if not lookup:
            continue
        labels = pd.Series([lookup.get(p) for p in perts], index=adata.obs.index)
        adata.obs[col_name] = pd.Categorical(labels)
        n_assigned = labels.notna().sum()
        _logger.info(f"  Annotated {n_assigned}/{adata.n_obs} genes with {col_name}")


def _apply_leiden_to_adata(
    adata,
    leiden_results: Dict[str, np.ndarray],
    _logger=logger,
    n_neighbors: int = 15,
) -> None:
    """Apply cached leiden labels back onto adata.obs and ensure the neighbors
    graph (obsp.connectivities, obsp.distances, uns.neighbors) is populated.

    Used on cache hit so the in-memory adata matches what a fresh run produces,
    which lets downstream callers persist these fields to h5ad.
    """
    if "neighbors" not in adata.uns:
        try:
            import scanpy as sc

            knn = min(n_neighbors, max(2, adata.n_obs - 1))
            sc.pp.neighbors(adata, n_neighbors=knn, use_rep="X_pca")
        except Exception as exc:
            _logger.warning("  sc.pp.neighbors (cache replay) failed: %s", exc)
    for col, labels in leiden_results.items():
        if len(labels) != adata.n_obs:
            continue
        adata.obs[col] = pd.Categorical(np.asarray(labels).astype(str))


def run_leiden_clustering(
    adata,
    resolutions: Tuple[float, ...] = DEFAULT_LEIDEN_RESOLUTIONS,
    n_neighbors: int = 15,
) -> Dict[str, np.ndarray]:
    """Run scanpy Leiden at each resolution. Returns {col_name: labels_array}.

    Stores results in ``adata.obs[col_name]`` as well. Falls back to an empty
    dict if scanpy or the leiden backend is unavailable.
    """
    try:
        import scanpy as sc
    except Exception as exc:
        logger.warning("  Leiden skipped: scanpy not available (%s)", exc)
        return {}

    if adata.n_obs < 10:
        logger.info("  Leiden skipped: only %d observations", adata.n_obs)
        return {}

    knn = min(n_neighbors, max(2, adata.n_obs - 1))
    try:
        sc.pp.neighbors(adata, n_neighbors=knn, use_rep="X_pca")
    except Exception as exc:
        logger.warning("  sc.pp.neighbors failed: %s — leiden skipped", exc)
        return {}

    out: Dict[str, np.ndarray] = {}
    for r in resolutions:
        col = f"leiden_r{r:g}"
        try:
            sc.tl.leiden(adata, resolution=float(r), key_added=col, flavor="igraph", n_iterations=2, directed=False)
            out[col] = np.asarray(adata.obs[col].values)
            n_clusters = len(set(out[col]))
            logger.info("  Leiden r=%g: %d clusters", r, n_clusters)
        except TypeError:
            # Older scanpy may not accept flavor="igraph"
            try:
                sc.tl.leiden(adata, resolution=float(r), key_added=col)
                out[col] = np.asarray(adata.obs[col].values)
                logger.info("  Leiden r=%g: %d clusters (fallback)", r, len(set(out[col])))
            except Exception as exc:
                logger.warning("  Leiden r=%g failed: %s", r, exc)
        except Exception as exc:
            logger.warning("  Leiden r=%g failed: %s", r, exc)
    return out


# =============================================================================
# Static supercluster overlay PNGs
# =============================================================================


def _categorical_palette(categories: List[str]) -> Dict[str, tuple]:
    """Build a deterministic color mapping that varies across categories.

    "Uncategorized" / "NTC" / "OR controls" get fixed special colors.
    """
    import seaborn as sns

    # 39 distinct colors total; cycles for very large category sets.
    colors = (
        sns.color_palette("tab20", 20)
        + sns.color_palette("Set1", 9)
        + sns.color_palette("Set2", 8)
        + sns.color_palette("Dark2", 8)
    )

    palette: Dict[str, tuple] = {
        "Uncategorized": (0.85, 0.85, 0.85),
        "NTC": (0.88, 0.50, 0.50),
        "OR controls": (1.0, 0.0, 0.0),
    }
    sorted_cats = [c for c in sorted(categories) if c not in palette]
    for i, cat in enumerate(sorted_cats):
        palette[cat] = colors[i % len(colors)]
    return palette


def _categorical_markers(categories: List[str]) -> Dict[str, str]:
    """Marker mapping for high-cardinality overlays. Marker cycles every N
    categories so adjacent (color-rotating) categories also get different
    markers, maximizing visual separability for >40 categories.
    """
    markers = ["o", "s", "D", "^", "v", "P", "p", "h", "*", "X"]
    palette_size = 39  # must match _categorical_palette length
    out: Dict[str, str] = {}
    for i, cat in enumerate(sorted(categories)):
        # Marker bumps every full cycle of the color palette.
        out[cat] = markers[(i // palette_size) % len(markers)]
    return out


def _plot_categorical_overlay(
    coords: np.ndarray,
    perts: np.ndarray,
    gene_to_cat: Dict[str, str],
    title: str,
    out_path: Path,
    plt,
    label_genes: bool = True,
    use_markers: bool = False,
):
    """Save a single static PNG colored by a gene→category mapping."""
    cats = np.array([gene_to_cat.get(p, "Uncategorized") for p in perts])
    is_ntc = np.array([str(p).startswith("NTC") for p in perts])
    cats[is_ntc] = "NTC"

    unique_cats = sorted({c for c in cats if c not in {"Uncategorized", "NTC"}})
    palette = _categorical_palette(unique_cats)
    markers = _categorical_markers(unique_cats) if use_markers else None

    fig = plt.figure(figsize=(20, 11))
    ax = fig.add_axes([0.04, 0.06, 0.62, 0.88])

    # Pin axes to full extent
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    uncat_mask = cats == "Uncategorized"
    if uncat_mask.any():
        ax.scatter(
            coords[uncat_mask, 0], coords[uncat_mask, 1],
            c=[palette["Uncategorized"]], s=40, alpha=0.18,
            edgecolors="none", label="Uncategorized",
        )

    for cat in unique_cats:
        m = (cats == cat)
        if not m.any():
            continue
        marker = markers[cat] if markers else "o"
        size = 130 if not markers else 150
        ax.scatter(
            coords[m, 0], coords[m, 1],
            c=[palette[cat]], marker=marker, s=size, alpha=0.85,
            edgecolors="white", linewidths=0.3,
            label=cat if len(cat) <= 60 else cat[:57] + "...",
        )

    if is_ntc.any():
        ax.scatter(
            coords[is_ntc, 0], coords[is_ntc, 1],
            c=[palette["NTC"]], marker="X", s=140, alpha=0.5,
            edgecolors="#b05050", linewidths=0.3, label="NTC", zorder=10,
        )

    if label_genes:
        rng = np.random.RandomState(42)
        for i, gene in enumerate(perts):
            if str(gene).startswith("NTC"):
                continue
            cat = gene_to_cat.get(gene, "Uncategorized")
            if cat == "Uncategorized":
                continue
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(35, 70)
            dx = radius * np.cos(angle)
            dy = radius * np.sin(angle)
            ax.annotate(
                gene, xy=(coords[i, 0], coords[i, 1]),
                xytext=(dx, dy), textcoords="offset points",
                fontsize=10, alpha=0.65, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=palette[cat], alpha=0.4, lw=0.4),
            )

    ax.set_title(title, fontsize=22, fontweight="bold")
    ax.set_xlabel("Dim 1", fontsize=16)
    ax.set_ylabel("Dim 2", fontsize=16)
    ax.tick_params(labelsize=12)
    ncol = 2 if len(unique_cats) > 30 else 1
    ax.legend(
        bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=11,
        framealpha=0.9, ncol=ncol, columnspacing=0.6, handletextpad=0.3,
        labelspacing=0.4,
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_supercluster_overlays(
    adata_guide,
    adata_gene_embed,
    overlay_maps: Dict[str, Dict[str, str]],
    plots_dir: Path,
    plt,
    _logger=logger,
) -> None:
    """For each (level, embedding) pair already computed, save a PNG colored
    by super-category and one colored by CHAD cluster.
    """
    plots_dir = Path(plots_dir)
    pairs = []
    for level_name, ad_obj in (("guide", adata_guide), ("gene", adata_gene_embed)):
        if ad_obj is None:
            continue
        perts = (
            ad_obj.obs["perturbation"].values
            if "perturbation" in ad_obj.obs.columns
            else ad_obj.obs_names.values
        )
        for embed_key, embed_label in (("X_umap", "umap"), ("X_phate", "phate")):
            if embed_key in ad_obj.obsm:
                coords = np.asarray(ad_obj.obsm[embed_key])
                pairs.append((level_name, embed_label, coords, perts))

    if not pairs:
        _logger.info("  No embeddings found for supercluster overlays")
        return

    super_map = overlay_maps.get("super", {}) or {}
    chad_map = overlay_maps.get("chad", {}) or {}

    for level_name, embed_label, coords, perts in pairs:
        if super_map:
            out = plots_dir / f"{level_name}_{embed_label}_supercluster.png"
            try:
                _plot_categorical_overlay(
                    coords, perts, super_map,
                    title=f"{level_name.capitalize()} {embed_label.upper()} — super-category",
                    out_path=out, plt=plt,
                    label_genes=False,
                    use_markers=False,
                )
                _logger.info("  Saved %s", out.name)
            except Exception as exc:
                _logger.warning("  Supercluster overlay failed (%s): %s", out.name, exc)

        if chad_map and level_name == "gene":
            out = plots_dir / f"{level_name}_{embed_label}_chad_clusters.png"
            try:
                _plot_categorical_overlay(
                    coords, perts, chad_map,
                    title=f"{level_name.capitalize()} {embed_label.upper()} — CHAD cluster",
                    out_path=out, plt=plt,
                    label_genes=False,
                    use_markers=True,
                )
                _logger.info("  Saved %s", out.name)
            except Exception as exc:
                _logger.warning("  CHAD overlay failed (%s): %s", out.name, exc)


# =============================================================================
# Leiden per-resolution overlays + gene lists
# =============================================================================


def _try_import_speedenrich():
    """Load ``speedenrich`` from maayanlab-bioinformatics, bypassing its
    top-level ``__init__`` (which imports pydeseq2). Returns the function or
    None if the package isn't installed.
    """
    try:
        import importlib.util
        import sys

        for search_path in sys.path:
            candidate = Path(search_path) / "maayanlab_bioinformatics" / "api" / "speedrichr.py"
            if candidate.exists():
                spec = importlib.util.spec_from_file_location("_speedrichr", candidate)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod.speedenrich
    except Exception:
        pass
    return None


def _run_cluster_enrichment(
    cluster_to_genes: Dict[str, List[str]],
    background_genes: Optional[List[str]],
    libraries: Tuple[str, ...] = DEFAULT_ENRICHR_LIBRARIES,
    max_workers: int = 6,
    min_genes: int = 5,
    top_n_terms: int = 10,
    _logger=logger,
) -> Dict[str, Dict]:
    """Run Enrichr on each cluster's gene list (Maayan Lab speedrichr API).

    Returns ``{cluster_id: {top1, terms}}`` where ``top1`` is the single best
    record (used for the legend label) and ``terms`` is a list of up to
    ``top_n_terms`` records ranked by combined_score. Each record has fields:
    ``term, library, adj_pvalue, combined_score, overlap, n_genes``. Clusters
    with no results get ``{"top1": None, "terms": []}``. Empty dict
    if maayanlab-bioinformatics isn't installed.
    """
    speedenrich = _try_import_speedenrich()
    if speedenrich is None:
        _logger.info(
            "  GO enrichment skipped: maayanlab-bioinformatics not installed "
            "(install with: pip install 'maayanlab-bioinformatics@git+https://github.com/MaayanLab/maayanlab-bioinformatics.git')"
        )
        return {}

    from concurrent.futures import ThreadPoolExecutor, as_completed

    libs = list(libraries)

    def _record(row, n_genes):
        return {
            "term": str(row["term"]),
            "library": str(row.get("library", "")),
            "adj_pvalue": float(row["adj pvalue"]),
            "combined_score": float(row.get("combined score", float("nan"))),
            "overlap": str(row.get("overlap", "")),
            "n_genes": int(n_genes),
        }

    def _one(cluster_id, genes):
        genes = [g for g in genes if g and not str(g).startswith("NTC")]
        empty = {"top1": None, "terms": [], "by_library": {}}
        if len(genes) < min_genes:
            return cluster_id, empty
        try:
            df = speedenrich(userlist=genes, libraries=libs, background=background_genes)
            if df is None or df.empty:
                return cluster_id, empty
            # rank by adj p-value (most significant first); combined score shown as bar
            top_overall = df.nsmallest(top_n_terms, "adj pvalue").reset_index(drop=True)
            terms_overall = [_record(r, len(genes)) for _, r in top_overall.iterrows()]
            # Per-library top-N by adj p-value
            by_library: Dict[str, List[Dict]] = {}
            if "library" in df.columns:
                for lib in libs:
                    lib_df = df[df["library"] == lib].nsmallest(top_n_terms, "adj pvalue")
                    if not lib_df.empty:
                        by_library[lib] = [_record(r, len(genes)) for _, r in lib_df.iterrows()]
            return cluster_id, {
                "top1": terms_overall[0],
                "terms": terms_overall,
                "by_library": by_library,
            }
        except Exception as exc:
            _logger.debug("Enrichr failed for %s: %s", cluster_id, exc)
            return cluster_id, empty

    out: Dict[str, Dict] = {c: {"top1": None, "terms": [], "by_library": {}} for c in cluster_to_genes}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_one, c, g) for c, g in cluster_to_genes.items()]
        for fut in as_completed(futures):
            cid, rec = fut.result()
            out[cid] = rec
    return out


def _enriched_cluster_label(cluster_id: str, rec: Optional[Dict]) -> str:
    """Format a cluster legend label, optionally with its top GO term.

    ``rec`` may be either the legacy single-term dict OR the new
    ``{"top1": ..., "terms": [...]}`` wrapper.
    """
    if isinstance(rec, dict) and "top1" in rec:
        rec = rec.get("top1")
    if not rec or not rec.get("term"):
        return cluster_id
    term = rec["term"]
    if " (GO:" in term:
        term = term.split(" (GO:")[0]
    if len(term) > 38:
        term = term[:35] + "..."
    score = rec.get("combined_score", float("nan"))
    if np.isnan(score):
        return f"{cluster_id}: {term}"
    return f"{cluster_id}: {term} (s={score:.1f})"


def save_leiden_overlays(
    adata_gene_embed,
    leiden_results: Dict[str, np.ndarray],
    plots_dir: Path,
    plt,
    _logger=logger,
    enrichment_per_res: Optional[Dict[str, Dict[str, Dict]]] = None,
) -> None:
    """Static PNGs (gene level) + per-resolution cluster/enrichment CSVs.

    Layout:
      ``plots/leiden/gene/png/{embed}_{col}.png`` (this function),
      ``plots/leiden/gene/csv/{col}_{clusters,enrichment}.csv``
      (delegated to ``save_leiden_csvs``).

    Static PNGs no longer use offset gene labels (use the HTMLs to identify
    dots by hover).
    """
    if adata_gene_embed is None or not leiden_results:
        return
    sub_dir = Path(plots_dir) / "leiden"
    png_dir = sub_dir / "gene" / "png"
    png_dir.mkdir(parents=True, exist_ok=True)

    perts = (
        adata_gene_embed.obs["perturbation"].values
        if "perturbation" in adata_gene_embed.obs.columns
        else adata_gene_embed.obs_names.values
    )
    enrichment_per_res = enrichment_per_res or {}

    for embed_key, embed_label in (("X_umap", "umap"), ("X_phate", "phate")):
        if embed_key not in adata_gene_embed.obsm:
            continue
        coords = np.asarray(adata_gene_embed.obsm[embed_key])

        for col, labels in leiden_results.items():
            res_str = col.replace("leiden_r", "")
            enrichment = enrichment_per_res.get(col, {})

            # Build enriched gene_to_cluster (label-only) for the static PNG legend
            gene_to_cluster_enriched = {}
            for i, p in enumerate(perts):
                cid = f"cluster_{labels[i]}"
                gene_to_cluster_enriched[str(p)] = _enriched_cluster_label(
                    cid, enrichment.get(cid)
                )

            png_out = png_dir / f"{embed_label}_{col}.png"
            try:
                _plot_categorical_overlay(
                    coords, perts, gene_to_cluster_enriched,
                    title=f"Gene {embed_label.upper()} — Leiden r={res_str}",
                    out_path=png_out, plt=plt,
                    label_genes=False, use_markers=True,
                )
                _logger.info("  Saved leiden/gene/png/%s", png_out.name)
            except Exception as exc:
                _logger.warning("  Leiden overlay failed (%s): %s", png_out.name, exc)

    save_leiden_csvs(
        adata_gene_embed, leiden_results, sub_dir, "gene",
        enrichment_per_res=enrichment_per_res, _logger=_logger,
    )


def save_leiden_csvs(
    adata,
    leiden_results: Dict[str, np.ndarray],
    sub_dir: Path,
    level_name: str,
    enrichment_per_res: Optional[Dict[str, Dict[str, Dict]]] = None,
    _logger=logger,
) -> None:
    """Write per-resolution cluster + enrichment CSVs for any level.

    Output files: ``{sub_dir}/{level_name}/csv/{col}_clusters.csv`` (one row
    per obs, with perturbation/sgRNA + cluster) and
    ``{sub_dir}/{level_name}/csv/{col}_enrichment.csv``. Guide-level CSVs
    include the sgRNA (obs index) alongside the target gene so guide →
    cluster assignments are recoverable.
    """
    if adata is None or not leiden_results:
        return
    csv_dir = Path(sub_dir) / level_name / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    enrichment_per_res = enrichment_per_res or {}

    perts = (
        adata.obs["perturbation"].astype(str).values
        if "perturbation" in adata.obs.columns
        else adata.obs_names.values
    )
    # sgRNA column for guide level (obs.index typically holds the guide id).
    extra_cols: Dict[str, np.ndarray] = {}
    if level_name == "guide":
        extra_cols["sgRNA"] = adata.obs_names.values

    for col, labels in leiden_results.items():
        df = pd.DataFrame({**extra_cols, "perturbation": perts, "cluster": labels})
        sort_cols = ["cluster"] + (["perturbation", "sgRNA"] if "sgRNA" in df.columns
                                   else ["perturbation"])
        df = df.sort_values(sort_cols)
        df.to_csv(csv_dir / f"{col}_clusters.csv", index=False)

        enrich_rows = []
        for cid, rec in (enrichment_per_res.get(col) or {}).items():
            for term in (rec.get("terms") or []):
                enrich_rows.append({
                    "cluster": cid, "term": term["term"],
                    "library": term["library"], "adj_pvalue": term["adj_pvalue"],
                    "combined_score": term["combined_score"], "overlap": term["overlap"],
                    "n_genes_in_cluster": term["n_genes"],
                })
        if enrich_rows:
            pd.DataFrame(enrich_rows).to_csv(
                csv_dir / f"{col}_enrichment.csv", index=False,
            )
    _logger.info(
        "  Saved %s leiden CSVs (%d resolutions) to %s",
        level_name, len(leiden_results), csv_dir,
    )


def _save_leiden_html(
    coords: np.ndarray,
    perts: np.ndarray,
    labels: np.ndarray,
    embed_label: str,
    res_str: str,
    out_path: Path,
    go,
    n_cells: Optional[np.ndarray] = None,
    enrichment: Optional[Dict[str, Dict]] = None,
) -> None:
    """Two-panel Plotly HTML: scatter on left + per-cluster GO bar chart on
    right with dropdown to select which cluster's bars to display.

    Hover on the scatter shows perturbation, n_cells, cluster, cluster size,
    and the cluster's top GO term + score. The legend uses enriched cluster
    labels (top GO term).
    """
    enrichment = enrichment or {}
    cluster_strs = np.asarray([f"cluster_{l}" for l in labels])
    unique = sorted(set(cluster_strs), key=lambda s: int(s.split("_")[1]))
    sizes = {c: int((cluster_strs == c).sum()) for c in unique}

    def _fmt_n_cells(idx: int) -> str:
        if n_cells is None:
            return ""
        try:
            return f"<br>n_cells: {int(n_cells[idx]):,}"
        except (TypeError, ValueError):
            return ""

    def _fmt_top_term(cid: str) -> str:
        rec = (enrichment.get(cid) or {}).get("top1")
        if not rec or not rec.get("term"):
            return ""
        return (
            f"<br>{rec['term']}<br>"
            f"score={rec['combined_score']:.1f}, "
            f"adj_p={rec['adj_pvalue']:.1e}"
        )

    # ----- LEFT: scatter, one trace per cluster -----
    scatter_traces = []
    for c in unique:
        idxs = np.where(cluster_strs == c)[0]
        cluster_perts = perts[idxs]
        top_term_str = _fmt_top_term(c)
        hover_text = [
            f"<b>{cluster_perts[k]}</b>"
            f"{_fmt_n_cells(idxs[k])}"
            f"<br>{c}<br>cluster size: {sizes[c]}"
            f"{top_term_str}"
            for k in range(len(idxs))
        ]
        legend_label = _enriched_cluster_label(c, enrichment.get(c))
        # Append cluster size if not already in label
        if " (n=" not in legend_label:
            legend_label = f"{legend_label} (n={sizes[c]})"
        scatter_traces.append(go.Scattergl(
            x=coords[idxs, 0],
            y=coords[idxs, 1],
            mode="markers",
            name=legend_label,
            marker=dict(size=9, opacity=0.85, line=dict(width=0.4, color="white")),
            text=hover_text,
            hoverinfo="text",
            showlegend=True,
            legendgroup=c,
            xaxis="x1", yaxis="y1",
        ))

    # ----- RIGHT: one bar trace per cluster (only one visible at a time) -----
    bar_traces = []
    cluster_has_bars = []
    for c in unique:
        terms = (enrichment.get(c) or {}).get("terms") or []
        if terms:
            terms = terms[::-1]  # least significant at bottom, most significant on top
            term_names = [t["term"] for t in terms]
            adj_ps = [t["adj_pvalue"] for t in terms]
            raw_scores = [t.get("combined_score") for t in terms]
            bar_x = [-np.log10(max(p, 1e-300)) for p in adj_ps]
            hover = [
                f"<b>{n}</b><br>-log10(p): {x:.2f}<br>adj p: {p:.2e}"
                + (f"<br>combined score: {s:.1f}" if s is not None and not np.isnan(s) else "")
                + f"<br>library: {t.get('library','')}"
                for n, x, p, s, t in zip(term_names, bar_x, adj_ps, raw_scores, terms)
            ]
            bar = go.Bar(
                x=bar_x, y=term_names, orientation="h",
                marker=dict(color=bar_x, colorscale="Viridis", showscale=False),
                text=[f"p={p:.1e}" for p in adj_ps],
                textposition="inside", insidetextanchor="start",
                hovertext=hover, hoverinfo="text",
                name=c, visible=False, showlegend=False,
                xaxis="x2", yaxis="y2",
            )
            cluster_has_bars.append(c)
        else:
            bar = go.Bar(
                x=[0], y=[f"({c}: no terms)"], orientation="h",
                marker=dict(color="lightgray"), hoverinfo="skip",
                visible=False, showlegend=False, xaxis="x2", yaxis="y2",
            )
        bar_traces.append(bar)

    if bar_traces:
        bar_traces[0].visible = True

    n_scatter = len(scatter_traces)
    n_bars = len(bar_traces)

    buttons = []
    for i, c in enumerate(unique):
        vis = [True] * n_scatter + [False] * n_bars
        vis[n_scatter + i] = True
        buttons.append(dict(
            label=_enriched_cluster_label(c, enrichment.get(c)) + f" (n={sizes[c]})",
            method="update",
            args=[
                {"visible": vis},
                {
                    "title": (
                        f"Gene {embed_label.upper()} — Leiden r={res_str} ({len(unique)} clusters) — "
                        f"showing GO terms for {c}"
                    ),
                },
            ],
        ))

    axis_label = embed_label.upper()
    fig = go.Figure(data=list(scatter_traces) + list(bar_traces))

    title_init = (
        f"Gene {axis_label} — Leiden r={res_str} ({len(unique)} clusters) — "
        f"showing GO terms for {unique[0]}"
        if unique else f"Gene {axis_label} — Leiden r={res_str}"
    )

    fig.update_layout(
        title=title_init,
        plot_bgcolor="white",
        height=860,
        width=1700,
        # left scatter panel
        xaxis=dict(
            domain=[0.00, 0.55], title=f"{axis_label} 1",
            showgrid=True, gridcolor="rgba(180,180,180,0.18)",
            zeroline=False, showline=True, linecolor="rgba(80,80,80,0.4)",
        ),
        yaxis=dict(
            domain=[0.00, 1.00], title=f"{axis_label} 2",
            scaleanchor="x", scaleratio=1,
            showgrid=True, gridcolor="rgba(180,180,180,0.18)",
            zeroline=False, showline=True, linecolor="rgba(80,80,80,0.4)",
        ),
        # right bar panel
        xaxis2=dict(
            domain=[0.74, 0.97], title="-log10(adj p)",
            showgrid=True, gridcolor="rgba(180,180,180,0.18)",
            zeroline=False, showline=True, linecolor="rgba(80,80,80,0.4)",
            anchor="y2",
        ),
        yaxis2=dict(
            domain=[0.00, 0.78], automargin=True,
            anchor="x2", showgrid=False,
            tickfont=dict(size=13),
        ),
        legend=dict(
            x=0.56, y=1.00, xanchor="left", yanchor="top",
            itemsizing="constant", tracegroupgap=2,
            bgcolor="rgba(255,255,255,0.9)", bordercolor="lightgray", borderwidth=0,
            font=dict(size=10),
        ),
        updatemenus=[dict(
            type="dropdown", buttons=buttons,
            x=0.74, y=1.00, xanchor="left", yanchor="bottom",
            bgcolor="white", bordercolor="lightgray",
            pad=dict(t=2, b=2, l=4, r=4),
        )],
        annotations=[
            dict(
                text="Top GO terms (select cluster ↓)",
                xref="paper", yref="paper",
                x=0.74, y=1.05, xanchor="left", yanchor="bottom",
                showarrow=False, font=dict(size=18, color="#444"),
            ),
        ],
        margin=dict(l=60, r=40, t=90, b=60),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)


# =============================================================================
# Interactive Plotly HTML
# =============================================================================


def _build_hover_records(
    perts: np.ndarray,
    activity_lookup: Dict[str, Dict[str, float]],
    overlay_maps: Dict[str, Dict[str, str]],
    leiden_results: Dict[str, np.ndarray],
    n_cells: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Hover-data table aligned with ``perts``.

    Columns: perturbation, n_cells, activity, auc, distinctiveness, corum,
    chad, super, chad_cluster, corum_complex, plus one per Leiden resolution.
    """
    super_map = overlay_maps.get("super", {}) or {}
    chad_map = overlay_maps.get("chad", {}) or {}
    corum_map = overlay_maps.get("corum", {}) or {}

    records = []
    for i, p in enumerate(perts):
        rec = {"perturbation": str(p)}
        if n_cells is not None:
            try:
                rec["n_cells"] = int(n_cells[i])
            except (TypeError, ValueError):
                rec["n_cells"] = None
        m = activity_lookup.get(str(p), {})
        for key in ("activity", "auc", "distinctiveness", "corum", "chad"):
            rec[key] = m.get(key, float("nan"))
        rec["super"] = super_map.get(str(p), "Uncategorized")
        rec["chad_cluster"] = chad_map.get(str(p), "Uncategorized")
        rec["corum_complex"] = corum_map.get(str(p), "")
        for col, labels in leiden_results.items():
            rec[col] = f"cluster_{labels[i]}"
        records.append(rec)
    return pd.DataFrame(records)


def _build_activity_lookup(
    activity_map: Optional[pd.DataFrame],
    dist_map: Optional[pd.DataFrame],
    corum_map: Optional[pd.DataFrame],
    chad_map: Optional[pd.DataFrame],
) -> Dict[str, Dict[str, float]]:
    """Per-perturbation lookup of all 4 mAP scores."""
    out: Dict[str, Dict[str, float]] = {}

    def _add(df, col_score, col_pert, key):
        if df is None:
            return
        if col_pert not in df.columns or col_score not in df.columns:
            return
        for _, row in df.iterrows():
            pert = str(row[col_pert])
            out.setdefault(pert, {})[key] = float(row[col_score])

    if activity_map is not None and "average_precision" in activity_map.columns:
        _add(activity_map, "average_precision", "perturbation", "activity")
        if "below_corrected_p" in activity_map.columns:
            for _, row in activity_map.iterrows():
                pert = str(row["perturbation"])
                out.setdefault(pert, {})["active"] = bool(row["below_corrected_p"])

    if dist_map is not None and "average_precision" in dist_map.columns:
        _add(dist_map, "average_precision", "perturbation", "distinctiveness")

    # CORUM/CHAD maps are per-complex, not per-gene; skip per-gene join here
    # (the per-gene CHAD/CORUM identity is in overlay_maps).
    return out


def save_interactive_html(
    adata_guide,
    adata_gene_embed,
    overlay_maps: Dict[str, Dict[str, str]],
    leiden_results: Dict[str, np.ndarray],
    activity_map: Optional[pd.DataFrame],
    dist_map: Optional[pd.DataFrame],
    corum_map: Optional[pd.DataFrame],
    chad_map: Optional[pd.DataFrame],
    plots_dir: Path,
    _logger=logger,
    enrichment_per_res: Optional[Dict[str, Dict[str, Dict]]] = None,
    leiden_guide_results: Optional[Dict[str, np.ndarray]] = None,
    enrichment_guide_per_res: Optional[Dict[str, Dict[str, Dict]]] = None,
) -> None:
    """Plotly HTML per (level, embedding). Hover shows all metadata; a dropdown
    switches the color overlay between super-category, CHAD cluster, CORUM
    complex, and each Leiden resolution.

    Guide-level Leiden + GO enrichment (computed in ``save_extra_overlays``
    using each guide's target-gene identity, deduplicated per cluster) is
    threaded through ``leiden_guide_results`` / ``enrichment_guide_per_res``.
    """
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        _logger.warning("  Interactive HTML skipped: plotly not available (%s)", exc)
        return

    plots_dir = Path(plots_dir)
    activity_lookup = _build_activity_lookup(activity_map, dist_map, corum_map, chad_map)

    leiden_by_level: Dict[str, Dict[str, np.ndarray]] = {
        "gene": leiden_results or {},
        "guide": leiden_guide_results or {},
    }
    enrichment_by_level: Dict[str, Optional[Dict[str, Dict[str, Dict]]]] = {
        "gene": enrichment_per_res,
        "guide": enrichment_guide_per_res,
    }

    levels = []
    if adata_guide is not None:
        levels.append(("guide", adata_guide))
    if adata_gene_embed is not None:
        levels.append(("gene", adata_gene_embed))

    for level_name, ad_obj in levels:
        perts = (
            ad_obj.obs["perturbation"].values
            if "perturbation" in ad_obj.obs.columns
            else ad_obj.obs_names.values
        )
        n_cells = (
            ad_obj.obs["n_cells"].values
            if "n_cells" in ad_obj.obs.columns
            else None
        )
        leiden_for_level = leiden_by_level.get(level_name) or {}
        enrichment_for_level = enrichment_by_level.get(level_name)
        hover_df = _build_hover_records(
            perts, activity_lookup, overlay_maps, leiden_for_level, n_cells=n_cells,
        )

        for embed_key, embed_label in (("X_umap", "umap"), ("X_phate", "phate")):
            if embed_key not in ad_obj.obsm:
                continue
            coords = np.asarray(ad_obj.obsm[embed_key])
            out = plots_dir / f"{level_name}_{embed_label}_interactive.html"
            try:
                _save_one_interactive(
                    coords, hover_df, leiden_for_level,
                    title=f"{level_name.capitalize()} {embed_label.upper()} — interactive",
                    out_path=out, go=go,
                    embed_label=embed_label,
                    enrichment_per_res=enrichment_for_level,
                    level_name=level_name,
                )
                _logger.info("  Saved %s", out.name)
            except Exception as exc:
                _logger.warning("  Interactive HTML failed (%s): %s", out.name, exc)


def _save_one_interactive(
    coords: np.ndarray,
    hover_df: pd.DataFrame,
    leiden_results: Dict[str, np.ndarray],
    title: str,
    out_path: Path,
    go,
    embed_label: str = "umap",
    enrichment_per_res: Optional[Dict[str, Dict[str, Dict]]] = None,
    level_name: str = "gene",
) -> None:
    """Plotly figure with two-panel layout:
      * Left: scatter, recolored by the active color mode (Super-category /
        CHAD / CORUM / Leiden_rX). Color-mode dropdown restyles only scatter
        traces.
      * Right (gene level only): horizontal bar chart of the top-N enriched
        GO/Reactome/KEGG terms for the *currently-selected leiden cluster*.
        Cluster dropdown restyles only bar traces; it lists every
        ``(resolution, cluster)`` combo across all leiden resolutions.
    """
    enrichment_per_res = enrichment_per_res or {}

    color_modes: List[Tuple[str, str]] = []
    color_modes.append(("Super-category", "super"))
    color_modes.append(("CHAD cluster", "chad_cluster"))
    color_modes.append(("CORUM complex", "corum_complex"))
    for col in sorted(leiden_results.keys()):
        color_modes.append((f"Leiden r={col.replace('leiden_r','')}", col))

    # Hover string assembled per row
    def _hover_text(row) -> str:
        lines = [f"<b>{row['perturbation']}</b>"]
        if "n_cells" in row.index and row.get("n_cells") not in (None, "", float("nan")):
            try:
                lines.append(f"n_cells: {int(row['n_cells']):,}")
            except (TypeError, ValueError):
                pass
        if not pd.isna(row.get("activity", float("nan"))):
            lines.append(f"Activity: {float(row['activity']):.3f}")
        if not pd.isna(row.get("auc", float("nan"))):
            lines.append(f"AUC: {float(row['auc']):.3f}")
        if not pd.isna(row.get("distinctiveness", float("nan"))):
            lines.append(f"Distinctiveness: {float(row['distinctiveness']):.3f}")
        if row.get("super", ""):
            lines.append(f"Super: {row['super']}")
        if row.get("chad_cluster", ""):
            lines.append(f"CHAD: {row['chad_cluster']}")
        if row.get("corum_complex", ""):
            lines.append(f"CORUM: {row['corum_complex']}")
        for k in row.index:
            if k.startswith("leiden_r"):
                lines.append(f"{k}: {row[k]}")
        return "<br>".join(lines)

    hover_texts = [_hover_text(row) for _, row in hover_df.iterrows()]

    # ---- Build SCATTER traces (one per category per color mode) ----
    # Highlight visuals: highlighted points pop with big size + full opacity;
    # dimmed points stay at default size but fade to 25% opacity (still
    # readable). NTCs render as a separate always-visible faded red X overlay.
    DEFAULT_OPACITY = 0.8
    DIMMED_OPACITY = 0.25
    DEFAULT_SIZE = 8
    HIGHLIGHTED_SIZE = 16
    NTC_OPACITY = 0.4
    NTC_SIZE = 10

    perts_arr = hover_df["perturbation"].astype(str).values
    is_ntc_mask = np.array([p.startswith("NTC") for p in perts_arr])
    n_ntc = int(is_ntc_mask.sum())
    has_ntc_overlay = n_ntc > 0

    main_idx = np.where(~is_ntc_mask)[0]
    ntc_idx_arr = np.where(is_ntc_mask)[0]
    coords_main = coords[main_idx]
    hover_df_main = hover_df.iloc[main_idx].reset_index(drop=True)
    hover_texts_main = [hover_texts[i] for i in main_idx]

    scatter_groups: List[Tuple[str, str, List]] = []  # (mode_label, mode_col, [traces], [cat_names])
    scatter_categories: Dict[str, List[str]] = {}
    for mode_label, col in color_modes:
        if col not in hover_df_main.columns:
            continue
        labels = hover_df_main[col].astype(str).fillna("Uncategorized").values
        unique_labels = sorted(set(labels))
        traces_for_mode = []
        for cat in unique_labels:
            mask = labels == cat
            traces_for_mode.append(go.Scattergl(
                x=coords_main[mask, 0],
                y=coords_main[mask, 1],
                mode="markers",
                name=str(cat) if len(str(cat)) <= 50 else str(cat)[:47] + "...",
                marker=dict(size=DEFAULT_SIZE, opacity=DEFAULT_OPACITY, line=dict(width=0.4, color="white")),
                text=[hover_texts_main[i] for i in np.where(mask)[0]],
                hoverinfo="text",
                legendgroup=mode_label,
                showlegend=True,
                visible=False,
                xaxis="x1", yaxis="y1",
            ))
        scatter_groups.append((mode_label, col, traces_for_mode))
        scatter_categories[mode_label] = unique_labels

    # NTC overlay: always visible, never dimmed/highlighted.
    ntc_overlay_trace = None
    if has_ntc_overlay:
        ntc_hover = [hover_texts[i] for i in ntc_idx_arr]
        ntc_overlay_trace = go.Scattergl(
            x=coords[ntc_idx_arr, 0], y=coords[ntc_idx_arr, 1],
            mode="markers",
            name=f"NTC (n={n_ntc})",
            marker=dict(symbol="x", color="#cc6060", size=NTC_SIZE,
                        opacity=NTC_OPACITY,
                        line=dict(width=0.5, color="#882020")),
            text=ntc_hover, hoverinfo="text",
            legendgroup="__ntc__",
            showlegend=True, visible=True,
            xaxis="x1", yaxis="y1",
        )

    if not scatter_groups:
        return

    n_main_scatter = sum(len(t) for _, _, t in scatter_groups)
    scatter_offsets: Dict[str, int] = {}  # mode_label -> first main-scatter trace index
    cursor = 0
    for mode_label, _, traces in scatter_groups:
        scatter_offsets[mode_label] = cursor
        cursor += len(traces)
    ntc_overlay_idx = n_main_scatter if has_ntc_overlay else None
    n_scatter = n_main_scatter + (1 if has_ntc_overlay else 0)

    # ---- Build BAR traces (one per leiden_resolution × cluster × library) ----
    # Discover ontology panels actually present in any enrichment record.
    libs_present: List[str] = []
    if enrichment_per_res:
        seen_libs: set = set()
        for res_data in enrichment_per_res.values():
            for cid_data in (res_data or {}).values():
                if cid_data and cid_data.get("by_library"):
                    seen_libs.update(cid_data["by_library"].keys())
        # Preserve a stable, readable order (GO BP first, then GO CC, Reactome, KEGG).
        priority = [
            "GO_Biological_Process_2025", "GO_Cellular_Component_2025",
            "Reactome_2022", "KEGG_2026",
        ]
        libs_present = [L for L in priority if L in seen_libs]
        libs_present += sorted(L for L in seen_libs if L not in priority)
    n_panels = len(libs_present)

    # Pre-compute per-(resolution, cluster, library) bar payloads as a JS lookup.
    # Only n_libs bar traces (one per ontology) are baked into the figure; their
    # data is restyled in JS on cluster select. Keeps trace count constant in
    # cluster count, avoiding the O(clusters²) HTML payload growth that came
    # from one trace per (cluster × library).
    def _bar_payload_for_terms(terms: List[Dict], lib: str) -> Dict:
        if not terms:
            return {"x": [0], "y": ["(no terms)"], "text": [""], "hovertext": [""], "marker_color": [0]}
        terms = terms[::-1]
        term_names = [t["term"][:60] for t in terms]
        adj_ps = [t["adj_pvalue"] for t in terms]
        raw_scores = [t.get("combined_score") for t in terms]
        bar_x = [-np.log10(max(p, 1e-300)) for p in adj_ps]
        hovs = [
            f"<b>{t['term']}</b><br>-log10(p): {x:.2f}<br>adj p: {p:.2e}"
            + (f"<br>combined score: {s:.1f}" if s is not None and not np.isnan(s) else "")
            + f"<br>library: {lib}"
            for x, p, s, t in zip(bar_x, adj_ps, raw_scores, terms)
        ]
        return {
            "x": bar_x, "y": term_names,
            "text": [f"p={p:.1e}" for p in adj_ps],
            "hovertext": hovs, "marker_color": bar_x,
        }

    bar_data_by_mode_cluster: Dict[str, Dict[str, List[Dict]]] = {}
    for res_col, clusters in enrichment_per_res.items():
        clusters = clusters or {}
        by_cluster: Dict[str, List[Dict]] = {}
        for cid, rec in clusters.items():
            rec = rec or {}
            by_lib = rec.get("by_library") or {}
            all_terms = rec.get("terms") or []
            per_lib_data: List[Dict] = []
            for lib in libs_present:
                terms = by_lib.get(lib) or [t for t in all_terms if t.get("library") == lib]
                per_lib_data.append(_bar_payload_for_terms(terms, lib))
            by_cluster[cid] = per_lib_data
        bar_data_by_mode_cluster[res_col] = by_cluster

    # First cluster id per leiden resolution — used to seed bar traces and as
    # the default when entering a leiden color mode.
    first_cluster_by_mode: Dict[str, str] = {}
    for res_col in sorted(enrichment_per_res.keys()):
        clusters = enrichment_per_res[res_col] or {}
        if clusters:
            cids = sorted(clusters.keys(), key=lambda s: int(s.split("_")[1]))
            first_cluster_by_mode[res_col] = cids[0]

    seed_mode = next(iter(sorted(first_cluster_by_mode.keys())), None)
    seed_cid = first_cluster_by_mode.get(seed_mode) if seed_mode else None
    seed_per_lib = (
        bar_data_by_mode_cluster[seed_mode][seed_cid]
        if seed_mode and seed_cid else None
    )

    bar_traces: List = []
    for lib_idx, lib in enumerate(libs_present):
        d = (
            seed_per_lib[lib_idx]
            if seed_per_lib
            else {"x": [0], "y": ["(no terms)"], "text": [""], "hovertext": [""], "marker_color": [0]}
        )
        bar_traces.append(go.Bar(
            x=d["x"], y=d["y"], orientation="h",
            marker=dict(color=d["marker_color"], colorscale="Viridis", cmin=0, showscale=False),
            text=d["text"],
            textposition="inside", insidetextanchor="start",
            hovertext=d["hovertext"], hoverinfo="text",
            name=lib,
            visible=False, showlegend=False,
            xaxis=f"x{lib_idx + 2}", yaxis=f"y{lib_idx + 2}",
        ))

    n_bars = len(bar_traces)  # == n_panels (one trace per library)
    bar_offset = n_scatter

    all_traces = [t for _, _, traces in scatter_groups for t in traces]
    if ntc_overlay_trace is not None:
        all_traces.append(ntc_overlay_trace)
    all_traces.extend(bar_traces)
    fig = go.Figure(data=all_traces)

    # ---- Helpers used by buttons ----
    def _bar_visibility_for(res_col: Optional[str], cid: Optional[str] = None) -> List[bool]:
        """Bar visibility now depends only on whether we're in a leiden color
        mode — the n_libs bar traces stay visible while their DATA is restyled
        in JS on cluster select. ``cid`` is accepted for call-site symmetry but
        ignored."""
        is_leiden = res_col is not None and res_col in enrichment_per_res
        return [is_leiden] * n_bars

    def _scatter_visibility_for(mode_label: str) -> List[bool]:
        vis = [False] * n_scatter
        start = scatter_offsets[mode_label]
        end = start + len(scatter_categories[mode_label])
        for i in range(start, end):
            vis[i] = True
        if ntc_overlay_idx is not None:
            vis[ntc_overlay_idx] = True
        return vis

    def _first_cluster_for(res_col: str) -> Optional[str]:
        clusters = enrichment_per_res.get(res_col) or {}
        cluster_ids = sorted(clusters.keys(), key=lambda s: int(s.split("_")[1]))
        return cluster_ids[0] if cluster_ids else None

    # ---- Initial state: first color mode active ----
    first_mode_label, first_mode_col, _ = scatter_groups[0]
    init_scatter_vis = _scatter_visibility_for(first_mode_label)
    if first_mode_col in enrichment_per_res:
        init_bars_vis = _bar_visibility_for(first_mode_col, _first_cluster_for(first_mode_col))
    else:
        init_bars_vis = [False] * n_bars
    full_init = init_scatter_vis + init_bars_vis
    for i, t in enumerate(fig.data):
        t.visible = full_init[i]

    # Default per-trace marker.opacity/size arrays (length n_scatter + n_bars).
    # NTC overlay and bar traces hold their initial values so the picker never
    # dims or resizes them — only main-scatter traces toggle on highlight.
    def _default_opacity_array() -> List:
        arr: List = [DEFAULT_OPACITY] * n_main_scatter
        if has_ntc_overlay:
            arr.append(NTC_OPACITY)
        arr += [None] * n_bars
        return arr

    def _default_size_array() -> List:
        arr: List = [DEFAULT_SIZE] * n_main_scatter
        if has_ntc_overlay:
            arr.append(NTC_SIZE)
        arr += [None] * n_bars
        return arr

    # ---- Gene list per (mode, category) ---------------------------------
    # Used to populate the gene-list annotation that lives below the legend.
    GENE_LIST_MAX = 100
    GENE_LIST_PLACEHOLDER = "<i>(select a category)</i>"

    def _format_gene_list(genes: List[str]) -> str:
        if not genes:
            return GENE_LIST_PLACEHOLDER
        sorted_genes = sorted(set(genes))
        truncated = sorted_genes[:GENE_LIST_MAX]
        # Wrap each gene in a span so JS can scroll-to-gene by data-gene attr.
        parts: List[str] = [
            f'<span class="gene-item" data-gene="{g}">{g}</span>'
            for g in truncated
        ]
        if len(sorted_genes) > GENE_LIST_MAX:
            parts.append(f"<i>... ({len(sorted_genes) - GENE_LIST_MAX} more)</i>")
        return "<br>".join(parts)

    def _genes_in(mode_col: str, category: str) -> List[str]:
        if mode_col not in hover_df_main.columns:
            return []
        col_vals = hover_df_main[mode_col].astype(str).fillna("Uncategorized")
        mask = col_vals == str(category)
        if not mask.any():
            return []
        return list(hover_df_main.loc[mask, "perturbation"].astype(str).values)

    # Pre-compute gene-list HTML per (mode, category) so the JS overlay div
    # can update its content without re-formatting on every click.
    gene_list_map: Dict[str, Dict[str, str]] = {}
    for mode_label, mode_col, _traces in scatter_groups:
        cat_to_genes: Dict[str, str] = {}
        for cat in scatter_categories[mode_label]:
            cat_to_genes[str(cat)] = _format_gene_list(_genes_in(mode_col, str(cat)))
        gene_list_map[mode_label] = cat_to_genes

    # Gene → category map per color mode. Categories are stored truncated to
    # match the trace-name truncation used by ``click_button_map`` so the JS
    # search box can reuse the click code path.
    def _trace_cat(c) -> str:
        s = str(c)
        return s if len(s) <= 50 else s[:47] + "..."

    gene_to_category_by_mode: Dict[str, Dict[str, str]] = {}
    for mode_label, mode_col, _traces in scatter_groups:
        if mode_col not in hover_df_main.columns:
            continue
        col_vals = hover_df_main[mode_col].astype(str).fillna("Uncategorized")
        perts_col = hover_df_main["perturbation"].astype(str)
        g_to_cat: Dict[str, str] = {}
        for p, c in zip(perts_col.values, col_vals.values):
            g_to_cat[p] = _trace_cat(c)
        gene_to_category_by_mode[mode_label] = g_to_cat

    # Sorted unique gene/perturbation list for autocomplete suggestions.
    autocomplete_genes = sorted(
        {str(p) for p in hover_df_main["perturbation"].astype(str).values}
    )

    annotations_pre: List[Dict] = []  # gene list lives in the HTML overlay (post_script)

    # ---- Unified picker: one cascading dropdown per color mode that BOTH
    # highlights the selected category AND (for leiden modes) shows that
    # cluster's GO bars in the right panel. Only the dropdown for the active
    # color mode is visible at any time; the color-mode button toggles which
    # one is shown.
    update_menus: List[Dict] = []
    color_dropdown_idx = 0
    update_menus.append({})  # placeholder for color-mode dropdown

    picker_menu_idx_for_mode: Dict[str, int] = {}
    for mode_label, mode_col, _traces in scatter_groups:
        cats = scatter_categories[mode_label]
        start = scatter_offsets[mode_label]
        is_leiden = mode_col in enrichment_per_res

        # "Show all": reset highlight, hide bars (or show first cluster's bars
        # for leiden modes — keeps the panel populated).
        opacities_all = _default_opacity_array()
        sizes_all = _default_size_array()
        if is_leiden:
            bars_show = _bar_visibility_for(mode_col, _first_cluster_for(mode_col))
        else:
            bars_show = [False] * n_bars
        # When showing all, scatter visibility already matches mode (set by
        # color-mode button), so we just keep visible == True for this mode's
        # scatter and bars_show for bars.
        full_vis_show_all = _scatter_visibility_for(mode_label) + bars_show

        buttons = [dict(
            label=f"Show all ({mode_label})", method="update",
            args=[
                {
                    "visible": full_vis_show_all,
                    "marker.opacity": opacities_all,
                    "marker.size": sizes_all,
                },
                {"title": f"{title} — {mode_label}"},
            ],
        )]
        for j, cat in enumerate(cats):
            tgt = start + j
            # Build dim-everywhere-except-target arrays for main scatter only.
            opacities = [DIMMED_OPACITY] * n_main_scatter
            sizes = [DEFAULT_SIZE] * n_main_scatter
            opacities[tgt] = 1.0
            sizes[tgt] = HIGHLIGHTED_SIZE
            if has_ntc_overlay:
                opacities.append(NTC_OPACITY)
                sizes.append(NTC_SIZE)
            opacities += [None] * n_bars
            sizes += [None] * n_bars

            # Bar visibility: if leiden mode AND cat is a cluster_X label, show
            # that cluster's bars across all libraries.
            if is_leiden and isinstance(cat, str) and cat.startswith("cluster_"):
                bar_vis = _bar_visibility_for(mode_col, cat)
            else:
                bar_vis = [False] * n_bars

            full_vis = _scatter_visibility_for(mode_label) + bar_vis

            cat_label = str(cat) if len(str(cat)) <= 50 else str(cat)[:47] + "..."
            # Append the top-1 GO term for leiden clusters so the picker is
            # informative without expanding.
            if is_leiden and isinstance(cat, str) and cat.startswith("cluster_"):
                rec = (enrichment_per_res.get(mode_col) or {}).get(cat) or {}
                top1 = rec.get("top1")
                if top1 and top1.get("term"):
                    term = top1["term"]
                    if " (GO:" in term:
                        term = term.split(" (GO:")[0]
                    if len(term) > 32:
                        term = term[:29] + "..."
                    cat_label = f"{cat}: {term}"

            buttons.append(dict(
                label=cat_label, method="update",
                args=[
                    {
                        "visible": full_vis,
                        "marker.opacity": opacities,
                        "marker.size": sizes,
                    },
                    {"title": f"{title} — {mode_label} — {cat}"},
                ],
            ))

        update_menus.append(dict(
            type="dropdown", buttons=buttons,
            x=0.45, y=1.10, xanchor="left", yanchor="bottom",
            visible=(mode_label == first_mode_label),
            bgcolor="white", bordercolor="lightgray", showactive=True,
        ))
        picker_menu_idx_for_mode[mode_label] = len(update_menus) - 1

    # 0: Color mode dropdown — flips scatter coloring AND swaps which unified
    # picker is visible. Resets highlight visuals and bar visibility.
    color_buttons = []
    for mode_label, mode_col, _traces in scatter_groups:
        layout_updates: Dict = {}
        for ml, mi in picker_menu_idx_for_mode.items():
            # Show only the matching picker, and reset its active button to 0
            # ("Show all") so the highlight from a previous mode doesn't carry
            # over into the new one.
            layout_updates[f"updatemenus[{mi}].visible"] = (ml == mode_label)
            if ml == mode_label:
                layout_updates[f"updatemenus[{mi}].active"] = 0
        layout_updates["title"] = f"{title} — {mode_label}"

        scatter_vis = _scatter_visibility_for(mode_label)
        if mode_col in enrichment_per_res:
            bar_vis = _bar_visibility_for(mode_col, _first_cluster_for(mode_col))
        else:
            bar_vis = [False] * n_bars

        color_buttons.append(dict(
            label=mode_label,
            method="update",
            args=[
                {
                    "visible": scatter_vis + bar_vis,
                    "marker.opacity": _default_opacity_array(),
                    "marker.size": _default_size_array(),
                },
                layout_updates,
            ],
        ))

    update_menus[color_dropdown_idx] = dict(
        type="dropdown", buttons=color_buttons,
        x=0.00, y=1.10, xanchor="left", yanchor="bottom",
        bgcolor="white", bordercolor="lightgray", showactive=True,
    )

    # ---- Layout ----
    axis_label = embed_label.upper()
    layout_dict: Dict = dict(
        title=f"{title} — {first_mode_label}",
        plot_bgcolor="white",
        height=900,
        width=1700 if n_panels > 0 else 1300,
        legend=dict(
            # Shifted ~40px left of the previous 0.56 paper-coord anchor so
            # the legend sits closer to the UMAP and farther from the bar panels.
            x=0.535 if n_panels > 0 else 1.02,
            y=0.78,
            xanchor="left", yanchor="top",
            itemsizing="constant", tracegroupgap=4,
            bgcolor="rgba(255,255,255,0.55)", bordercolor="lightgray", borderwidth=0,
            font=dict(size=10),
        ),
        margin=dict(l=60, r=20 if n_panels > 0 else 260, t=140, b=60),
    )

    # Gene-list annotation lives at index GENE_LIST_ANNOT_IDX (=0); rest follow.
    annotations = list(annotations_pre)
    annotations.append(dict(
        text="<b>Color overlay</b>", xref="paper", yref="paper",
        x=0.00, y=1.16, xanchor="left", yanchor="bottom",
        showarrow=False, font=dict(size=11, color="#444"),
    ))
    if picker_menu_idx_for_mode:
        annotations.append(dict(
            text="<b>Highlight + GO terms</b>", xref="paper", yref="paper",
            x=0.45, y=1.16, xanchor="left", yanchor="bottom",
            showarrow=False, font=dict(size=11, color="#444"),
        ))

    if n_panels > 0:
        # Scatter panel: leave a bit more left margin for axis ticks
        layout_dict["xaxis"] = dict(
            domain=[0.00, 0.50], title=f"{axis_label} 1",
            showgrid=True, gridcolor="rgba(180,180,180,0.18)",
            zeroline=False, showline=True, linecolor="rgba(80,80,80,0.4)",
        )
        layout_dict["yaxis"] = dict(
            domain=[0.00, 1.00], title=f"{axis_label} 2",
            scaleanchor="x", scaleratio=1,
            showgrid=True, gridcolor="rgba(180,180,180,0.18)",
            zeroline=False, showline=True, linecolor="rgba(80,80,80,0.4)",
        )
        # Bar panels: stacked vertically on the right (~22% wide)
        bar_x_domain = [0.78, 0.97]
        gap = 0.055  # extra room so titles don't overlap adjacent panels
        total = 1.0
        panel_h = (total - gap * (n_panels - 1)) / n_panels
        for lib_idx, lib in enumerate(libs_present):
            y_top = total - lib_idx * (panel_h + gap)
            y_bot = max(0.0, y_top - panel_h)
            ax_x = f"xaxis{lib_idx + 2}"
            ax_y = f"yaxis{lib_idx + 2}"
            layout_dict[ax_x] = dict(
                domain=bar_x_domain,
                anchor=f"y{lib_idx + 2}",
                title=("-log10(adj p)" if lib_idx == n_panels - 1 else None),
                showgrid=True, gridcolor="rgba(180,180,180,0.18)",
                zeroline=False, showline=True, linecolor="rgba(80,80,80,0.4)",
                tickfont=dict(size=8),
            )
            layout_dict[ax_y] = dict(
                domain=[y_bot, y_top],
                anchor=f"x{lib_idx + 2}",
                automargin=True, showgrid=False,
                tickfont=dict(size=9),
            )
            # Title sits just above each panel (gap gives enough clearance)
            pretty = lib.replace("_2025", "").replace("_2026", "").replace("_2022", "").replace("_", " ")
            annotations.append(dict(
                text=f"<b>{pretty}</b>",
                xref="paper", yref="paper",
                x=bar_x_domain[0], y=y_top + 0.008,
                xanchor="left", yanchor="bottom",
                showarrow=False, font=dict(size=14, color="#555"),
            ))
    else:
        layout_dict["xaxis_title"] = f"{axis_label} 1"
        layout_dict["yaxis_title"] = f"{axis_label} 2"

    layout_dict["updatemenus"] = update_menus
    layout_dict["annotations"] = annotations

    fig.update_layout(**layout_dict)

    if n_panels == 0:
        fig.update_xaxes(
            showgrid=True, gridcolor="rgba(180,180,180,0.18)",
            zeroline=False, showline=True, linecolor="rgba(80,80,80,0.4)",
        )
        fig.update_yaxes(
            scaleanchor="x", scaleratio=1,
            showgrid=True, gridcolor="rgba(180,180,180,0.18)",
            zeroline=False, showline=True, linecolor="rgba(80,80,80,0.4)",
        )

    # Click-to-pick: clicking a scatter point triggers the picker button
    # matching that point's category. Implemented by attaching a plotly_click
    # handler at HTML render time. The handler uses fullData.legendgroup (mode)
    # and fullData.name (category) to look up the right button index.
    click_button_map: Dict[str, Dict[str, int]] = {}
    for mode_label, _mode_col, _traces in scatter_groups:
        cats = scatter_categories[mode_label]
        button_map: Dict[str, int] = {}
        for j, cat in enumerate(cats):
            cat_name = str(cat) if len(str(cat)) <= 50 else str(cat)[:47] + "..."
            button_map[cat_name] = j + 1  # +1 because button 0 is "Show all"
        click_button_map[mode_label] = button_map

    # mode_label → mode_col so JS can resolve the active leiden resolution from
    # the visible picker dropdown.
    mode_col_by_label: Dict[str, str] = {ml: mc for ml, mc, _ in scatter_groups}

    import json as _json
    # Gene search is enabled at every level — guide-level hover_df_main also
    # carries the per-row target gene name in the "perturbation" column, so the
    # autocomplete + click-to-pick wiring works identically. (At guide level,
    # selecting a gene jumps to one of its guides' categories — the last one
    # encountered when building gene_to_category_by_mode.)
    show_gene_search = True
    post_script = f"""
(function() {{
  var pickerMenuIdxForMode = {_json.dumps(picker_menu_idx_for_mode)};
  var clickButtonMap = {_json.dumps(click_button_map)};
  var geneListMap = {_json.dumps(gene_list_map)};
  var GENE_LIST_PLACEHOLDER = {_json.dumps(GENE_LIST_PLACEHOLDER)};
  var barDataByModeCluster = {_json.dumps(bar_data_by_mode_cluster)};
  var firstClusterByMode = {_json.dumps(first_cluster_by_mode)};
  var modeColByLabel = {_json.dumps(mode_col_by_label)};
  var geneToCategoryByMode = {_json.dumps(gene_to_category_by_mode)};
  var autocompleteGenes = {_json.dumps(autocomplete_genes)};
  var SHOW_GENE_SEARCH = {_json.dumps(show_gene_search)};
  var nScatter = {n_scatter};
  var nLibs = {n_panels};
  var barTraceIndices = [];
  for (var __i = 0; __i < nLibs; __i++) {{ barTraceIndices.push(nScatter + __i); }}

  function _updateBars(modeCol, clusterId) {{
    if (!modeCol || !clusterId || nLibs === 0) return;
    var modeData = barDataByModeCluster[modeCol];
    if (!modeData) return;
    var clusterData = modeData[clusterId];
    if (!clusterData) return;
    var update = {{x: [], y: [], text: [], hovertext: [], 'marker.color': []}};
    for (var i = 0; i < clusterData.length; i++) {{
      update.x.push(clusterData[i].x);
      update.y.push(clusterData[i].y);
      update.text.push(clusterData[i].text);
      update.hovertext.push(clusterData[i].hovertext);
      update['marker.color'].push(clusterData[i].marker_color);
    }}
    Plotly.restyle(gd, update, barTraceIndices);
  }}

  function _clusterIdFromLabel(btnLabel) {{
    if (!btnLabel) return null;
    var m = btnLabel.match(/^(cluster_\\d+)/);
    return m ? m[1] : null;
  }}

  var gds = document.getElementsByClassName('plotly-graph-div');
  if (gds.length === 0) return;
  var gd = gds[gds.length - 1];
  if (!gd.on) return;

  // ---- Build scrollable gene-list overlay anchored to the graph div ----
  // Position is in pixel offsets relative to the plotly graph div which has a
  // fixed size (e.g. 1700×900). Pixel values are computed from the figure
  // width so the panel always lands in the gutter between scatter and bars.
  var figWidth = (gd.layout && gd.layout.width) || 1700;
  var figHeight = (gd.layout && gd.layout.height) || 900;
  // Plot domain assumed: scatter x in [0, 0.50] of plot area; legend at x≈0.56;
  // bar panels at x in [0.78, 0.97]. Margin: l=60 (left), r=20 (right when bar
  // panels exist) or r=260 (when not — guide level, no GO panels).
  var hasBars = nLibs > 0;
  var plotW = figWidth - 60 - (hasBars ? 20 : 260);
  var plotH = figHeight - 200;  // approximate (margin t≈140, b≈60)
  // Panel sits under the legend (legend is at y=0.78 down to ~0.40 worst case).
  // Gene level: between legend and bar panels (in the gutter).
  // Guide level: no bar panels, scatter takes the whole plot area, so push the
  // panel into the right margin (where the legend lives at paper x≈1.02) so it
  // doesn't overlap the UMAP.
  var panelLeft = hasBars
    ? 60 + plotW * 0.56 - 60
    : figWidth - 220;
  var panelTop = 200 + plotH * 0.50;       // rough: below legend bottom
  var panelHeight = plotH * 0.45;

  var parent = gd.parentNode || gd;
  if (parent && getComputedStyle(parent).position === 'static') {{
    parent.style.position = 'relative';
  }}

  var panel = document.createElement('div');
  panel.id = 'gene-list-panel';
  panel.style.cssText =
    'position:absolute;' +
    ' left:' + panelLeft + 'px; top:' + panelTop + 'px;' +
    ' width:200px; height:' + panelHeight + 'px;' +
    ' overflow-y:auto; background:rgba(255,255,255,0.55);' +
    ' border:1px solid rgba(170,170,170,0.6); border-radius:3px;' +
    ' font-family:Menlo, Consolas, monospace; font-size:9px;' +
    ' line-height:1.25; padding:6px 8px; z-index:50;' +
    ' box-shadow:0 1px 4px rgba(0,0,0,0.04);';
  var header = document.createElement('div');
  header.style.cssText = 'font-weight:bold; font-family:sans-serif; font-size:11px; color:#444; margin-bottom:4px;';
  header.textContent = 'Genes in selected category';
  var body = document.createElement('div');
  body.id = 'gene-list-body';
  body.innerHTML = GENE_LIST_PLACEHOLDER;
  panel.appendChild(header);
  panel.appendChild(body);
  parent.appendChild(panel);

  // ---- Gene search box (gene-level only) -------------------------------
  var searchInput = null;
  var searchSuggest = null;
  var searchStatus = null;
  if (SHOW_GENE_SEARCH && autocompleteGenes.length > 0) {{
    var searchBox = document.createElement('div');
    var searchTop = Math.max(panelTop - 92, 100);
    searchBox.id = 'gene-search-box';
    searchBox.style.cssText =
      'position:absolute;' +
      ' left:' + panelLeft + 'px; top:' + searchTop + 'px;' +
      ' width:200px; z-index:55;' +
      ' font-family:sans-serif;';
    var searchHeader = document.createElement('div');
    searchHeader.style.cssText = 'font-weight:bold; font-size:11px; color:#444; margin-bottom:4px;';
    searchHeader.textContent = 'Search gene';
    searchInput = document.createElement('input');
    searchInput.type = 'text';
    searchInput.autocomplete = 'off';
    searchInput.spellcheck = false;
    searchInput.placeholder = 'e.g. MYC, TP53...';
    searchInput.style.cssText =
      'width:100%; box-sizing:border-box; padding:4px 6px;' +
      ' font-size:11px; border:1px solid #aaa; border-radius:3px;' +
      ' background:rgba(255,255,255,0.95);';
    searchSuggest = document.createElement('div');
    searchSuggest.id = 'gene-search-suggest';
    searchSuggest.style.cssText =
      'position:absolute; left:0; right:0; top:100%;' +
      ' background:rgba(255,255,255,0.98);' +
      ' border:1px solid #aaa; border-top:none; border-radius:0 0 3px 3px;' +
      ' max-height:180px; overflow-y:auto; font-size:11px;' +
      ' display:none; z-index:56; box-shadow:0 2px 6px rgba(0,0,0,0.08);';
    searchStatus = document.createElement('div');
    searchStatus.style.cssText = 'font-size:10px; color:#666; margin-top:3px; min-height:14px;';
    var inputWrap = document.createElement('div');
    inputWrap.style.cssText = 'position:relative;';
    inputWrap.appendChild(searchInput);
    inputWrap.appendChild(searchSuggest);
    searchBox.appendChild(searchHeader);
    searchBox.appendChild(inputWrap);
    searchBox.appendChild(searchStatus);
    parent.appendChild(searchBox);
  }}

  function _setGeneList(modeLabel, btnLabel) {{
    if (!btnLabel || /^Show all/.test(btnLabel)) {{
      body.innerHTML = GENE_LIST_PLACEHOLDER;
      return;
    }}
    var category = btnLabel;
    var m = btnLabel.match(/^(cluster_\\d+)/);
    if (m) category = m[1];
    var modeMap = geneListMap[modeLabel] || {{}};
    var html = modeMap[category];
    if (!html) {{
      Object.keys(modeMap).forEach(function(k) {{
        if (btnLabel.indexOf(k) === 0) html = modeMap[k];
      }});
    }}
    body.innerHTML = html || GENE_LIST_PLACEHOLDER;
    body.scrollTop = 0;
  }}

  function _scrollToGene(geneName) {{
    if (!geneName) return;
    var el = body.querySelector('[data-gene="' + geneName + '"]');
    if (!el) return;
    var bodyRect = body.getBoundingClientRect();
    var elRect = el.getBoundingClientRect();
    var current = body.scrollTop;
    var elOffset = (elRect.top - bodyRect.top) + current;
    body.scrollTop = elOffset - (bodyRect.height / 2) + (elRect.height / 2);
    var prevBg = el.style.background;
    var prevColor = el.style.color;
    el.style.background = '#ffe680';
    el.style.color = '#000';
    el.style.fontWeight = 'bold';
    setTimeout(function() {{
      el.style.background = prevBg;
      el.style.color = prevColor;
      el.style.fontWeight = '';
    }}, 1800);
  }}

  function _extractGeneFromHover(text) {{
    if (!text) return null;
    var m = text.match(/<b>([^<]+)<\\/b>/);
    return m ? m[1] : null;
  }}

  function _modeForMenu(menuIdx) {{
    var found = null;
    Object.keys(pickerMenuIdxForMode).forEach(function(ml) {{
      if (pickerMenuIdxForMode[ml] === menuIdx) found = ml;
    }});
    return found;
  }}

  function _activeModeLabel() {{
    var modeLabel = null;
    Object.keys(pickerMenuIdxForMode).forEach(function(ml) {{
      var idx = pickerMenuIdxForMode[ml];
      var menus = (gd.layout && gd.layout.updatemenus) || [];
      if (menus[idx] && menus[idx].visible !== false) modeLabel = ml;
    }});
    return modeLabel;
  }}

  // Apply the same picker-button as a click on the plot would, given a mode
  // and a (truncated) trace category. Returns true on success.
  function _pickCategory(modeLabel, category, geneToFlash) {{
    if (!modeLabel || !category) return false;
    var menuIdx = pickerMenuIdxForMode[modeLabel];
    if (menuIdx === undefined) return false;
    var btnMap = clickButtonMap[modeLabel];
    if (!btnMap) return false;
    var btnIdx = btnMap[category];
    if (btnIdx === undefined) return false;
    var menu = gd.layout.updatemenus[menuIdx];
    if (!menu) return false;
    var btn = menu.buttons[btnIdx];
    if (!btn) return false;
    var layoutUpd = Object.assign({{}}, btn.args[1] || {{}});
    layoutUpd['updatemenus[' + menuIdx + '].active'] = btnIdx;
    Plotly.update(gd, btn.args[0], layoutUpd);
    _setGeneList(modeLabel, category);
    var modeColP = modeColByLabel[modeLabel];
    if (modeColP && barDataByModeCluster[modeColP]) {{
      var cidP = _clusterIdFromLabel(category);
      if (cidP) _updateBars(modeColP, cidP);
    }}
    if (geneToFlash) {{
      setTimeout(function() {{ _scrollToGene(geneToFlash); }}, 0);
    }}
    return true;
  }}

  // ---- Gene search + autocomplete -----------------------------------------
  if (SHOW_GENE_SEARCH && searchInput) {{
    var SEARCH_LIMIT = 8;
    var suggestActive = -1;
    var lastSuggestions = [];

    function _rankMatches(q) {{
      if (!q) return [];
      var ql = q.toLowerCase();
      var pref = [], starts = [], contains = [];
      for (var i = 0; i < autocompleteGenes.length; i++) {{
        var g = autocompleteGenes[i];
        var gl = g.toLowerCase();
        if (gl === ql) pref.push(g);
        else if (gl.indexOf(ql) === 0) starts.push(g);
        else if (gl.indexOf(ql) !== -1) contains.push(g);
        if (pref.length + starts.length + contains.length >= SEARCH_LIMIT * 3) break;
      }}
      return pref.concat(starts).concat(contains).slice(0, SEARCH_LIMIT);
    }}

    function _renderSuggestions(items, query) {{
      if (!items.length) {{
        searchSuggest.style.display = 'none';
        searchSuggest.innerHTML = '';
        lastSuggestions = [];
        return;
      }}
      var q = (query || '').toLowerCase();
      var html = '';
      for (var i = 0; i < items.length; i++) {{
        var g = items[i];
        var gLow = g.toLowerCase();
        var hi = gLow.indexOf(q);
        var disp = g;
        if (q && hi >= 0) {{
          disp = g.substring(0, hi)
            + '<b>' + g.substring(hi, hi + q.length) + '</b>'
            + g.substring(hi + q.length);
        }}
        html += '<div class="g-sug" data-i="' + i + '"' +
                ' style="padding:4px 8px; cursor:pointer; ' +
                ((i === suggestActive) ? 'background:#e6f0ff;' : '') +
                '">' + disp + '</div>';
      }}
      searchSuggest.innerHTML = html;
      searchSuggest.style.display = 'block';
      lastSuggestions = items;
      var nodes = searchSuggest.querySelectorAll('.g-sug');
      for (var k = 0; k < nodes.length; k++) {{
        nodes[k].addEventListener('mousedown', (function(idx) {{
          return function(ev) {{
            ev.preventDefault();
            _commitGene(lastSuggestions[idx]);
          }};
        }})(k));
        nodes[k].addEventListener('mouseenter', (function(idx) {{
          return function() {{
            suggestActive = idx;
            _renderSuggestions(lastSuggestions, searchInput.value);
          }};
        }})(k));
      }}
    }}

    function _commitGene(gene) {{
      if (!gene) return;
      searchInput.value = gene;
      searchSuggest.style.display = 'none';
      suggestActive = -1;
      var modeLabel = _activeModeLabel();
      if (!modeLabel) {{
        searchStatus.textContent = 'No active mode';
        return;
      }}
      var modeMap = geneToCategoryByMode[modeLabel] || {{}};
      var category = modeMap[gene];
      if (!category) {{
        var lowered = gene.toLowerCase();
        var keys = Object.keys(modeMap);
        for (var k = 0; k < keys.length; k++) {{
          if (keys[k].toLowerCase() === lowered) {{
            category = modeMap[keys[k]];
            gene = keys[k];
            break;
          }}
        }}
      }}
      if (!category) {{
        searchStatus.textContent = '"' + gene + '" not found in ' + modeLabel;
        return;
      }}
      var ok = _pickCategory(modeLabel, category, gene);
      searchStatus.textContent = ok
        ? gene + ' → ' + category
        : 'Could not select ' + category;
    }}

    searchInput.addEventListener('input', function() {{
      suggestActive = -1;
      var items = _rankMatches(searchInput.value);
      _renderSuggestions(items, searchInput.value);
    }});

    searchInput.addEventListener('keydown', function(e) {{
      if (e.key === 'ArrowDown') {{
        e.preventDefault();
        if (lastSuggestions.length) {{
          suggestActive = (suggestActive + 1) % lastSuggestions.length;
          _renderSuggestions(lastSuggestions, searchInput.value);
        }}
      }} else if (e.key === 'ArrowUp') {{
        e.preventDefault();
        if (lastSuggestions.length) {{
          suggestActive = (suggestActive - 1 + lastSuggestions.length) % lastSuggestions.length;
          _renderSuggestions(lastSuggestions, searchInput.value);
        }}
      }} else if (e.key === 'Enter') {{
        e.preventDefault();
        var gene = (suggestActive >= 0 && lastSuggestions[suggestActive])
          ? lastSuggestions[suggestActive]
          : (lastSuggestions.length ? lastSuggestions[0] : searchInput.value);
        _commitGene(gene);
      }} else if (e.key === 'Escape') {{
        searchSuggest.style.display = 'none';
      }}
    }});

    searchInput.addEventListener('blur', function() {{
      // Delay so a click on a suggestion still registers
      setTimeout(function() {{ searchSuggest.style.display = 'none'; }}, 150);
    }});
    searchInput.addEventListener('focus', function() {{
      if (searchInput.value) {{
        var items = _rankMatches(searchInput.value);
        _renderSuggestions(items, searchInput.value);
      }}
    }});
  }}

  gd.on('plotly_buttonclicked', function(event) {{
    if (!event || !event.button) return;
    var btnLabel = event.button.label;
    var menuX = (event.menu || {{}}).x;
    // Color-mode dropdown sits at x≈0 — reset gene list when mode switches,
    // and if the new mode is a leiden one, repopulate bars with its first
    // cluster's data (the data wasn't included in the button's args payload).
    if (menuX === undefined || menuX < 0.1) {{
      body.innerHTML = GENE_LIST_PLACEHOLDER;
      var modeColCM = modeColByLabel[btnLabel];
      if (modeColCM && barDataByModeCluster[modeColCM]) {{
        _updateBars(modeColCM, firstClusterByMode[modeColCM]);
      }}
      return;
    }}
    // Picker dropdown clicked — find the currently visible picker mode
    // (only one picker is visible at a time; visibility is toggled by the
    // color-mode button via relayout so gd.layout reflects the current state)
    var modeLabel = null;
    Object.keys(pickerMenuIdxForMode).forEach(function(ml) {{
      var idx = pickerMenuIdxForMode[ml];
      var menus = gd.layout.updatemenus || [];
      if (menus[idx] && menus[idx].visible !== false) modeLabel = ml;
    }});
    if (!modeLabel) return;
    _setGeneList(modeLabel, btnLabel);
    var modeColP = modeColByLabel[modeLabel];
    if (modeColP && barDataByModeCluster[modeColP]) {{
      if (/^Show all/.test(btnLabel)) {{
        _updateBars(modeColP, firstClusterByMode[modeColP]);
      }} else {{
        var cidP = _clusterIdFromLabel(btnLabel);
        if (cidP) _updateBars(modeColP, cidP);
      }}
    }}
  }});

  gd.on('plotly_click', function(eventData) {{
    if (!eventData.points || eventData.points.length === 0) return;
    var pt = eventData.points[0];
    if (!pt.fullData || pt.fullData.type !== 'scattergl') return;
    var modeLabel = pt.fullData.legendgroup;
    var category = pt.fullData.name;
    var menuIdx = pickerMenuIdxForMode[modeLabel];
    if (menuIdx === undefined) return;
    var btnMap = clickButtonMap[modeLabel];
    if (!btnMap) return;
    var btnIdx = btnMap[category];
    if (btnIdx === undefined) return;
    var menu = gd.layout.updatemenus[menuIdx];
    if (!menu || !menu.visible) return;
    var clearedToShowAll = false;
    if (typeof menu.active === 'number' && menu.active === btnIdx) {{
      btnIdx = 0;
      clearedToShowAll = true;
    }}
    var btn = menu.buttons[btnIdx];
    if (!btn) return;
    Plotly.update(gd, btn.args[0], btn.args[1] || {{}});
    var relayout = {{}};
    relayout['updatemenus[' + menuIdx + '].active'] = btnIdx;
    Plotly.relayout(gd, relayout);
    if (clearedToShowAll) {{
      body.innerHTML = GENE_LIST_PLACEHOLDER;
    }} else {{
      _setGeneList(modeLabel, category);
      var clickedGene = _extractGeneFromHover(pt.text);
      if (clickedGene) {{
        setTimeout(function() {{ _scrollToGene(clickedGene); }}, 0);
      }}
    }}
    var modeColCK = modeColByLabel[modeLabel];
    if (modeColCK && barDataByModeCluster[modeColCK]) {{
      if (clearedToShowAll) {{
        _updateBars(modeColCK, firstClusterByMode[modeColCK]);
      }} else {{
        var cidCK = _clusterIdFromLabel(category);
        if (cidCK) _updateBars(modeColCK, cidCK);
      }}
    }}
  }});
}})();
"""
    fig.write_html(
        str(out_path),
        include_plotlyjs="cdn",
        full_html=True,
        post_script=post_script,
    )


# =============================================================================
# Canonical embeddings — one PNG per (level, embedder), colored by Leiden +
# top-GO-term centroid labels.
# =============================================================================


def _top_term_per_cluster(
    enrichment: Optional[Dict[str, Dict]],
) -> Dict[str, str]:
    """For each cluster, pick the highest-combined-score term and strip the
    parenthetical GO accession. Clusters without enrichment hits are dropped."""
    out: Dict[str, str] = {}
    for cid, rec in (enrichment or {}).items():
        terms = rec.get("terms") or []
        if not terms:
            continue
        top = max(terms, key=lambda t: t.get("combined_score") or 0.0)
        term = top["term"]
        if " (GO:" in term:
            term = term.split(" (GO:")[0]
        out[cid] = term
    return out


# Boilerplate fragments that bloat GO term names without adding signal.
# Order matters — longest matches first so "positive regulation of" wins
# over "regulation of".
_TERM_BOILERPLATE_PREFIXES: Tuple[str, ...] = (
    "positive regulation of ",
    "negative regulation of ",
    "regulation of ",
    "response to ",
    "involved in ",
)
_TERM_BOILERPLATE_SUFFIXES: Tuple[str, ...] = (
    " process", " pathway", " biosynthetic", " metabolic process",
    " signaling pathway", " response",
)


def _condense_term(term: str, max_chars: int = 30) -> str:
    """Return a tighter form of a long GO term. Strip common low-information
    prefixes/suffixes; if still over ``max_chars``, soft-wrap with textwrap so
    the label sits on multiple short lines instead of one long line."""
    if len(term) <= max_chars:
        return term
    lower = term.lower()
    for p in _TERM_BOILERPLATE_PREFIXES:
        if lower.startswith(p):
            term = term[len(p):]
            break
    lower = term.lower()
    for s in _TERM_BOILERPLATE_SUFFIXES:
        if lower.endswith(s):
            term = term[: -len(s)]
            break
    if len(term) <= max_chars:
        return term
    import textwrap
    return textwrap.fill(term, width=max_chars, break_long_words=False)


def _repel_label_positions(
    positions: np.ndarray, min_dist: float,
    n_iter: int = 80, step: float = 0.5,
) -> np.ndarray:
    """Push labels apart with simple pairwise repulsion. ``positions`` is an
    (N, 2) array; pairs within ``min_dist`` get nudged along the line
    connecting them until they're at least ``min_dist`` apart, or until
    ``n_iter`` is exhausted."""
    pos = np.asarray(positions, dtype=float).copy()
    n = len(pos)
    if n < 2:
        return pos
    rng = np.random.default_rng(0)
    for _ in range(n_iter):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[j] - pos[i]
                dist = float(np.linalg.norm(d))
                if dist >= min_dist:
                    continue
                if dist < 1e-9:
                    d = rng.standard_normal(2)
                    dist = float(np.linalg.norm(d))
                direction = d / dist
                push = (min_dist - dist) / 2.0 * step
                pos[j] += direction * push
                pos[i] -= direction * push
                moved = True
        if not moved:
            break
    return pos


def _save_canonical_panel_png(
    coords: np.ndarray, labels: np.ndarray, perts: np.ndarray,
    cluster_to_term: Dict[str, str],
    level: str, embedder: str, resolution_col: str,
    out_path: Path, plt,
) -> None:
    """One PNG: scatter of the embedding colored by leiden cluster, with one
    text label per unique top-GO term placed near the centroid of all points
    whose cluster maps to that term. Labels are pairwise-repelled, connected
    to their centroid by a thin guide line, and condensed/wrapped if long.
    NTCs render as a separate red-X overlay matching the rest of the suite."""
    from collections import defaultdict
    import matplotlib as _mpl

    coords = np.asarray(coords)
    labels = np.asarray(labels)
    perts = np.asarray(perts)

    is_ntc = np.array([str(p).startswith("NTC") for p in perts])
    nontnc = ~is_ntc

    unique = sorted(set(int(l) for l in labels))
    n_c = len(unique)
    cmap = _mpl.colormaps.get("gist_rainbow")
    cluster_to_color = {c: cmap(i / max(1, n_c - 1)) for i, c in enumerate(unique)}

    fig, ax = plt.subplots(figsize=(14, 14))

    # Non-NTC points colored by Leiden cluster
    if nontnc.any():
        nontnc_colors = np.array(
            [cluster_to_color[int(labels[i])] for i in np.where(nontnc)[0]]
        )
        ax.scatter(
            coords[nontnc, 0], coords[nontnc, 1],
            c=nontnc_colors, s=38, alpha=0.55, linewidths=0,
            zorder=2,
        )

    # NTC overlay — same red X marks as save_supercluster_overlays /
    # _plot_categorical_overlay (RGB (0.88, 0.50, 0.50), edge #b05050).
    if is_ntc.any():
        ax.scatter(
            coords[is_ntc, 0], coords[is_ntc, 1],
            c=[(0.88, 0.50, 0.50)], marker="X", s=175, alpha=0.5,
            edgecolors="#b05050", linewidths=0.3,
            label=f"NTC (n={int(is_ntc.sum())})", zorder=10,
        )

    # Group every point whose cluster's top term matches into one label group
    # and place a single label at that group's centroid.
    term_points: Dict[str, List[np.ndarray]] = defaultdict(list)
    for i, l in enumerate(labels):
        cid = f"cluster_{int(l)}"
        term = cluster_to_term.get(cid)
        if term:
            term_points[term].append(coords[i])

    if term_points:
        terms_in_order = list(term_points.keys())
        centroids = np.array(
            [np.asarray(term_points[t]).mean(axis=0) for t in terms_in_order]
        )
        spans = coords.max(axis=0) - coords.min(axis=0)
        # Push labels further apart than before (was 0.06 of max span) so they
        # actually clear each other after the bbox is rendered.
        min_dist = float(0.11 * max(spans))
        offsets = _repel_label_positions(
            centroids, min_dist=min_dist, n_iter=200, step=0.6,
        )

        # Nicer sans-serif stack — Helvetica/Arial if installed, else falls
        # back to matplotlib's bundled DejaVu Sans. semibold weight reads
        # better than plain bold in the small font sizes we use here.
        text_font_kw = dict(
            family="sans-serif",
            fontweight="semibold",
        )

        for term, centroid, offset in zip(terms_in_order, centroids, offsets):
            display = _condense_term(term, max_chars=22)
            if not np.allclose(centroid, offset):
                ax.plot(
                    [centroid[0], offset[0]], [centroid[1], offset[1]],
                    color="0.55", lw=0.5, alpha=0.5, zorder=3,
                )
            ax.text(
                offset[0], offset[1], display,
                fontsize=11, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="0.7",
                          alpha=0.35, lw=0.4),
                zorder=5,
                **text_font_kw,
            )

    # Axis titles + grid (no tick labels, but ticks still drive the grid)
    ax.set_xlabel(f"{embedder}1", fontsize=20, fontweight="semibold")
    ax.set_ylabel(f"{embedder}2", fontsize=20, fontweight="semibold")
    ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False, length=0)
    ax.grid(True, alpha=0.35, linewidth=0.6, color="0.55", zorder=0)
    ax.set_axisbelow(True)

    if is_ntc.any():
        ax.legend(loc="upper right", fontsize=12, framealpha=0.7)

    n_uniq = len(set(cluster_to_term.values()))
    ax.set_title(
        f"{level} {embedder} — colored by {resolution_col}\n"
        f"{n_c} clusters, {n_uniq} unique top GO terms (label near centroid)",
        fontsize=20, fontweight="bold", pad=12, family="sans-serif",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_canonical_leiden_panels(
    adata_guide,
    adata_gene_embed,
    leiden_results: Dict[str, np.ndarray],
    leiden_guide_results: Dict[str, np.ndarray],
    enrichment_per_res: Optional[Dict[str, Dict[str, Dict]]],
    enrichment_guide_per_res: Optional[Dict[str, Dict[str, Dict]]],
    plots_dir: Path,
    plt,
    _logger=logger,
) -> None:
    """For every Leiden resolution actually computed and every ``(level,
    embedder)``, save one PNG — guide UMAP/PHATE and gene UMAP/PHATE — colored
    by Leiden cluster with the top GO term per cluster placed near the
    centroid of all points sharing that term. NTCs render as red X marks.
    All PNGs land in ``plots_dir/canonical_leiden/{level}/png/``."""
    out_dir = Path(plots_dir) / "canonical_leiden"

    levels = (
        ("guide", adata_guide, leiden_guide_results or {}, enrichment_guide_per_res),
        ("gene",  adata_gene_embed, leiden_results or {}, enrichment_per_res),
    )
    n_saved = 0
    for level_name, ad_obj, leiden_dict, enrich_dict in levels:
        if ad_obj is None or not leiden_dict:
            continue
        png_dir = out_dir / level_name / "png"
        png_dir.mkdir(parents=True, exist_ok=True)
        perts = (
            ad_obj.obs["perturbation"].values
            if "perturbation" in ad_obj.obs.columns
            else ad_obj.obs_names.values
        )
        for resolution_col, labels in leiden_dict.items():
            cluster_to_term = _top_term_per_cluster(
                (enrich_dict or {}).get(resolution_col),
            )
            for embed_key, embed_label in (("X_umap", "UMAP"), ("X_phate", "PHATE")):
                if embed_key not in ad_obj.obsm:
                    continue
                out_path = png_dir / (
                    f"{embed_label.lower()}_{resolution_col}.png"
                )
                try:
                    _save_canonical_panel_png(
                        np.asarray(ad_obj.obsm[embed_key]),
                        labels,
                        perts,
                        cluster_to_term,
                        level=level_name,
                        embedder=embed_label,
                        resolution_col=resolution_col,
                        out_path=out_path,
                        plt=plt,
                    )
                    n_saved += 1
                    _logger.info(
                        "  Saved canonical_leiden/%s/png/%s",
                        level_name, out_path.name,
                    )
                except Exception as exc:
                    _logger.warning(
                        "  Canonical %s %s %s panel failed: %s",
                        level_name, embed_label, resolution_col, exc,
                    )
    if n_saved == 0:
        _logger.warning(
            "  Canonical leiden panels: no eligible (resolution, level, embedder) "
            "— wrote nothing",
        )


# =============================================================================
# Public entry-point used by aggregate_channels / apply_second_pass_pca
# =============================================================================


_LEIDEN_CACHE_VERSION = 1


def _leiden_cache_path(plots_dir: Path) -> Path:
    """Cache lives in the run dir (one level up from plots/) as a single
    pickle, so it survives wiping plots/ but tracks each h5ad pair."""
    return Path(plots_dir).parent / "leiden_cache.pkl"


def _load_leiden_cache(plots_dir: Path, _logger) -> Dict:
    p = _leiden_cache_path(plots_dir)
    if not p.exists():
        return {}
    try:
        import pickle
        with open(p, "rb") as f:
            cache = pickle.load(f)
        if not isinstance(cache, dict) or cache.get("version") != _LEIDEN_CACHE_VERSION:
            return {}
        return cache
    except Exception as exc:
        _logger.warning("  Leiden cache at %s unreadable (%s) — recomputing", p, exc)
        return {}


def _save_leiden_cache(plots_dir: Path, cache: Dict, _logger) -> None:
    p = _leiden_cache_path(plots_dir)
    try:
        import pickle
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        _logger.info("  Wrote leiden cache to %s", p)
    except Exception as exc:
        _logger.warning("  Failed to write leiden cache (%s)", exc)


def _cache_is_valid_for_level(
    cache_level: Optional[Dict],
    expected_n_obs: int,
    expected_resolutions: Tuple[float, ...],
) -> bool:
    """Cached level data must (a) match observation count and (b) cover every
    resolution requested. Extra cached resolutions are fine."""
    if not cache_level:
        return False
    if cache_level.get("n_obs") != int(expected_n_obs):
        return False
    cached_cols = set((cache_level.get("leiden") or {}).keys())
    needed_cols = {f"leiden_r{r:g}" for r in expected_resolutions}
    return needed_cols.issubset(cached_cols)


def _enrichment_for_gene_level(
    adata_gene_embed, leiden_results, enrichr_libraries, _logger,
) -> Dict[str, Dict[str, Dict]]:
    out: Dict[str, Dict[str, Dict]] = {}
    if not (leiden_results and adata_gene_embed is not None):
        return out
    perts_g = (
        adata_gene_embed.obs["perturbation"].values
        if "perturbation" in adata_gene_embed.obs.columns
        else adata_gene_embed.obs_names.values
    )
    background = [str(p) for p in perts_g if not str(p).startswith("NTC")]
    for col, labels in leiden_results.items():
        cluster_to_genes: Dict[str, List[str]] = {}
        for i, p in enumerate(perts_g):
            cluster_to_genes.setdefault(f"cluster_{labels[i]}", []).append(str(p))
        _logger.info(
            "  Running GO enrichment for %s (%d clusters)...",
            col, len(cluster_to_genes),
        )
        out[col] = _run_cluster_enrichment(
            cluster_to_genes, background,
            libraries=enrichr_libraries, _logger=_logger,
        )
    return out


def _enrichment_for_guide_level(
    adata_guide, leiden_guide_results, enrichr_libraries, _logger,
) -> Dict[str, Dict[str, Dict]]:
    out: Dict[str, Dict[str, Dict]] = {}
    if not (leiden_guide_results and adata_guide is not None):
        return out
    perts_guide = (
        adata_guide.obs["perturbation"].values
        if "perturbation" in adata_guide.obs.columns
        else adata_guide.obs_names.values
    )
    background_guide = sorted({str(p) for p in perts_guide if not str(p).startswith("NTC")})
    for col, labels in leiden_guide_results.items():
        cluster_to_genes: Dict[str, List[str]] = {}
        for i, p in enumerate(perts_guide):
            cluster_to_genes.setdefault(f"cluster_{labels[i]}", []).append(str(p))
        cluster_to_genes = {k: sorted(set(v)) for k, v in cluster_to_genes.items()}
        _logger.info(
            "  Running GO enrichment for guide-level %s (%d clusters, dedup'd to genes)...",
            col, len(cluster_to_genes),
        )
        out[col] = _run_cluster_enrichment(
            cluster_to_genes, background_guide,
            libraries=enrichr_libraries, _logger=_logger,
        )
    return out


def save_extra_overlays(
    adata_guide,
    adata_gene_embed,
    plots_dir: Path,
    plt,
    activity_map: Optional[pd.DataFrame] = None,
    dist_map: Optional[pd.DataFrame] = None,
    corum_map: Optional[pd.DataFrame] = None,
    chad_map: Optional[pd.DataFrame] = None,
    leiden_resolutions: Tuple[float, ...] = DEFAULT_LEIDEN_RESOLUTIONS,
    supercategory_config_path: Optional[Path] = None,
    enrichr_libraries: Tuple[str, ...] = DEFAULT_ENRICHR_LIBRARIES,
    use_cache: bool = True,
    _logger=logger,
) -> None:
    """One-shot helper called from aggregation/second-pass: produces all
    extra static overlays + Leiden plots + the interactive HTML.

    GO enrichment is computed once per Leiden resolution and reused by both
    the static Leiden PNGs (legend labels) and the main interactive HTMLs
    (per-cluster pathway bar charts).

    Leiden labels and Enrichr GO results are cached to ``<run_dir>/leiden_cache.pkl``
    after the first compute so re-runs (e.g. ``--overlays-only``) skip both
    steps. The cache is keyed by ``(level, n_obs)`` and validated against the
    requested resolutions; pass ``use_cache=False`` to force recomputation.
    """
    overlay_maps = load_overlay_maps(supercategory_config_path)

    # Persist gene → CHAD cluster / CORUM complex / super-category lookups onto
    # adata.obs so they survive into the saved h5ad alongside leiden columns.
    if adata_gene_embed is not None:
        _apply_overlay_maps_to_adata(adata_gene_embed, overlay_maps, _logger)
    if adata_guide is not None:
        _apply_overlay_maps_to_adata(adata_guide, overlay_maps, _logger)

    cache = _load_leiden_cache(plots_dir, _logger) if use_cache else {}
    cache_dirty = False
    cache.setdefault("version", _LEIDEN_CACHE_VERSION)

    # ----- Gene level -----
    leiden_results: Dict[str, np.ndarray] = {}
    enrichment_per_res: Dict[str, Dict[str, Dict]] = {}
    if adata_gene_embed is not None and "X_pca" in adata_gene_embed.obsm:
        cache_gene = cache.get("gene") or {}
        if _cache_is_valid_for_level(cache_gene, adata_gene_embed.n_obs, leiden_resolutions):
            _logger.info(
                "  Using cached gene Leiden + enrichment (%d resolutions)",
                len(cache_gene.get("leiden", {})),
            )
            leiden_results = dict(cache_gene["leiden"])
            enrichment_per_res = dict(cache_gene.get("enrichment", {}))
            _apply_leiden_to_adata(adata_gene_embed, leiden_results, _logger)
        else:
            _logger.info("  Running Leiden at resolutions %s on gene embedding...", list(leiden_resolutions))
            leiden_results = run_leiden_clustering(adata_gene_embed, leiden_resolutions)
            enrichment_per_res = _enrichment_for_gene_level(
                adata_gene_embed, leiden_results, enrichr_libraries, _logger,
            )
            cache["gene"] = {
                "n_obs": int(adata_gene_embed.n_obs),
                "leiden": leiden_results,
                "enrichment": enrichment_per_res,
            }
            cache_dirty = True
        # Per-gene top-term annotation from Enrichr (GO BP / GO CC / Reactome / KEGG)
        if enrichment_per_res:
            _apply_enrichment_to_adata(
                adata_gene_embed, leiden_results, enrichment_per_res, _logger=_logger
            )

    # ----- Guide level -----
    leiden_guide_results: Dict[str, np.ndarray] = {}
    enrichment_guide_per_res: Dict[str, Dict[str, Dict]] = {}
    if adata_guide is not None and "X_pca" in adata_guide.obsm:
        cache_guide = cache.get("guide") or {}
        if _cache_is_valid_for_level(cache_guide, adata_guide.n_obs, leiden_resolutions):
            _logger.info(
                "  Using cached guide Leiden + enrichment (%d resolutions)",
                len(cache_guide.get("leiden", {})),
            )
            leiden_guide_results = dict(cache_guide["leiden"])
            enrichment_guide_per_res = dict(cache_guide.get("enrichment", {}))
            _apply_leiden_to_adata(adata_guide, leiden_guide_results, _logger)
        else:
            _logger.info("  Running Leiden at resolutions %s on guide embedding...", list(leiden_resolutions))
            leiden_guide_results = run_leiden_clustering(adata_guide, leiden_resolutions)
            enrichment_guide_per_res = _enrichment_for_guide_level(
                adata_guide, leiden_guide_results, enrichr_libraries, _logger,
            )
            cache["guide"] = {
                "n_obs": int(adata_guide.n_obs),
                "leiden": leiden_guide_results,
                "enrichment": enrichment_guide_per_res,
            }
            cache_dirty = True

    if cache_dirty:
        _save_leiden_cache(plots_dir, cache, _logger)

    save_supercluster_overlays(
        adata_guide, adata_gene_embed, overlay_maps, plots_dir, plt, _logger=_logger,
    )
    save_leiden_overlays(
        adata_gene_embed, leiden_results, plots_dir, plt, _logger=_logger,
        enrichment_per_res=enrichment_per_res,
    )
    # Guide-level CSVs (no static PNGs — too many guides to plot meaningfully).
    save_leiden_csvs(
        adata_guide,
        leiden_guide_results,
        Path(plots_dir) / "leiden",
        "guide",
        enrichment_per_res=enrichment_guide_per_res,
        _logger=_logger,
    )
    save_interactive_html(
        adata_guide, adata_gene_embed, overlay_maps, leiden_results,
        activity_map, dist_map, corum_map, chad_map,
        plots_dir, _logger=_logger,
        enrichment_per_res=enrichment_per_res,
        leiden_guide_results=leiden_guide_results,
        enrichment_guide_per_res=enrichment_guide_per_res,
    )
    save_canonical_leiden_panels(
        adata_guide=adata_guide,
        adata_gene_embed=adata_gene_embed,
        leiden_results=leiden_results,
        leiden_guide_results=leiden_guide_results,
        enrichment_per_res=enrichment_per_res,
        enrichment_guide_per_res=enrichment_guide_per_res,
        plots_dir=plots_dir,
        plt=plt,
        _logger=_logger,
    )
