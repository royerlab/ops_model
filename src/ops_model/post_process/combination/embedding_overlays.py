"""Extra overlay plots on top of the per-aggregate UMAP/PHATE embeddings.

Three additions on top of ``_compute_and_plot_embeddings``:

1. ``save_supercluster_overlays`` — static matplotlib PNGs duplicating each
   existing UMAP/PHATE plot, recolored by gene super-category (CHAD-boosted)
   and by direct CHAD cluster name.
2. ``save_leiden_overlays`` — runs scanpy Leiden at several resolutions and
   saves per-resolution static PNGs plus a CSV of (gene, cluster) per
   resolution. One subdir ``plots/leiden/`` per aggregate.
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


DEFAULT_LEIDEN_RESOLUTIONS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0)
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
                    label_genes=(level_name == "gene"),
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
                    label_genes=True,
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
    adj_pvalue_cutoff: float = 0.25,
    top_n_terms: int = 10,
    _logger=logger,
) -> Dict[str, Dict]:
    """Run Enrichr on each cluster's gene list (Maayan Lab speedrichr API).

    Returns ``{cluster_id: {top1, terms}}`` where ``top1`` is the single best
    record (used for the legend label) and ``terms`` is a list of up to
    ``top_n_terms`` records used for the bar chart. Each record has fields:
    ``term, library, adj_pvalue, combined_score, overlap, n_genes``. Clusters
    with no significant term get ``{"top1": None, "terms": []}``. Empty dict
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
            df = df[df["adj pvalue"] < adj_pvalue_cutoff]
            if df.empty:
                return cluster_id, empty
            top_overall = df.nsmallest(top_n_terms, "adj pvalue").reset_index(drop=True)
            terms_overall = [_record(r, len(genes)) for _, r in top_overall.iterrows()]
            # Per-library top-N (so the bar panels can be split by ontology)
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
    """Static PNGs + per-resolution interactive HTMLs (gene level) +
    a CSV gene-list per resolution, all under ``plots/leiden/``.

    Static PNGs no longer use offset gene labels (use the HTMLs to identify
    dots by hover).
    """
    if adata_gene_embed is None or not leiden_results:
        return
    sub_dir = Path(plots_dir) / "leiden"
    sub_dir.mkdir(parents=True, exist_ok=True)

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

            png_out = sub_dir / f"gene_{embed_label}_{col}.png"
            try:
                _plot_categorical_overlay(
                    coords, perts, gene_to_cluster_enriched,
                    title=f"Gene {embed_label.upper()} — Leiden r={res_str}",
                    out_path=png_out, plt=plt,
                    label_genes=False, use_markers=True,
                )
                _logger.info("  Saved leiden/%s", png_out.name)
            except Exception as exc:
                _logger.warning("  Leiden overlay failed (%s): %s", png_out.name, exc)

    # CSVs: gene→cluster + enrichment per resolution
    for col, labels in leiden_results.items():
        rows = pd.DataFrame({
            "perturbation": perts,
            "cluster": labels,
        }).sort_values(["cluster", "perturbation"])
        rows.to_csv(sub_dir / f"gene_{col}_clusters.csv", index=False)

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
                sub_dir / f"gene_{col}_enrichment.csv", index=False,
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
            terms = terms[::-1]  # reverse so smallest p (most significant) ends up on top
            term_names = [t["term"] for t in terms]
            scores = [t.get("combined_score") or 0 for t in terms]
            adj_ps = [t["adj_pvalue"] for t in terms]
            hover = [
                f"<b>{name}</b><br>combined score: {sc:.1f}<br>adj p: {p:.2e}<br>library: {t.get('library','')}"
                for name, sc, p, t in zip(term_names, scores, adj_ps, terms)
            ]
            bar = go.Bar(
                x=scores, y=term_names,
                orientation="h",
                marker=dict(color=scores, colorscale="Viridis", showscale=False),
                text=[f"{sc:.1f}" for sc in scores],
                textposition="outside",
                hovertext=hover,
                hoverinfo="text",
                name=c,
                visible=False,
                showlegend=False,
                xaxis="x2", yaxis="y2",
            )
            cluster_has_bars.append(c)
        else:
            bar = go.Bar(
                x=[0], y=[f"({c}: no significant terms)"],
                orientation="h",
                marker=dict(color="lightgray"),
                hoverinfo="skip",
                visible=False, showlegend=False,
                xaxis="x2", yaxis="y2",
            )
        bar_traces.append(bar)

    # Default: show first cluster's bar chart
    if bar_traces:
        bar_traces[0].visible = True

    n_scatter = len(scatter_traces)
    n_bars = len(bar_traces)

    # Dropdown: each button toggles visibility of one bar trace, scatter traces always visible
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
            domain=[0.74, 0.97], title="Combined score (Enrichr)",
            showgrid=True, gridcolor="rgba(180,180,180,0.18)",
            zeroline=False, showline=True, linecolor="rgba(80,80,80,0.4)",
            anchor="y2",
        ),
        yaxis2=dict(
            domain=[0.00, 0.78], automargin=True,
            anchor="x2", showgrid=False,
            tickfont=dict(size=10),
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
                showarrow=False, font=dict(size=12, color="#444"),
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
) -> None:
    """Plotly HTML per (level, embedding). Hover shows all metadata; a dropdown
    switches the color overlay between super-category, CHAD cluster, CORUM
    complex, and each Leiden resolution.
    """
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        _logger.warning("  Interactive HTML skipped: plotly not available (%s)", exc)
        return

    plots_dir = Path(plots_dir)
    activity_lookup = _build_activity_lookup(activity_map, dist_map, corum_map, chad_map)

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
        # Leiden was computed at gene level; for guide level just leave it empty.
        leiden_for_level = leiden_results if level_name == "gene" else {}
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
                    enrichment_per_res=enrichment_per_res if level_name == "gene" else None,
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

    bar_traces: List = []
    bar_keys: List[Tuple[str, str, str]] = []  # parallel: (res_col, cluster_id, lib)
    for res_col in sorted(enrichment_per_res.keys()):
        clusters = enrichment_per_res[res_col] or {}
        for cid in sorted(clusters.keys(), key=lambda s: int(s.split("_")[1])):
            rec = clusters[cid] or {}
            by_lib = rec.get("by_library") or {}
            for lib_idx, lib in enumerate(libs_present):
                terms = (by_lib.get(lib) or [])[::-1]
                ax_x = "x1" if lib_idx == 0 else f"x{lib_idx + 2}"
                ax_y = "y1" if lib_idx == 0 else f"y{lib_idx + 2}"
                # libs_present[0] still uses x2/y2 (axis indexing starts at 2 for the first bar panel)
                ax_x = f"x{lib_idx + 2}"
                ax_y = f"y{lib_idx + 2}"
                if terms:
                    term_names = [t["term"][:60] for t in terms]
                    scores = [t.get("combined_score") or 0 for t in terms]
                    adj_ps = [t["adj_pvalue"] for t in terms]
                    hovs = [
                        f"<b>{t['term']}</b><br>combined score: {s:.1f}<br>adj p: {p:.2e}<br>library: {lib}"
                        for s, p, t in zip(scores, adj_ps, terms)
                    ]
                    bar = go.Bar(
                        x=scores, y=term_names,
                        orientation="h",
                        marker=dict(color=scores, colorscale="Viridis", showscale=False),
                        hovertext=hovs, hoverinfo="text",
                        name=f"{cid}/{lib}",
                        visible=False, showlegend=False,
                        xaxis=ax_x, yaxis=ax_y,
                    )
                else:
                    bar = go.Bar(
                        x=[0], y=["(no significant terms)"],
                        orientation="h",
                        marker=dict(color="lightgray"),
                        hoverinfo="skip",
                        name=f"{cid}/{lib}",
                        visible=False, showlegend=False,
                        xaxis=ax_x, yaxis=ax_y,
                    )
                bar_traces.append(bar)
                bar_keys.append((res_col, cid, lib))

    n_bars = len(bar_traces)
    bar_offset = n_scatter  # first bar trace index in the global trace list

    all_traces = [t for _, _, traces in scatter_groups for t in traces]
    if ntc_overlay_trace is not None:
        all_traces.append(ntc_overlay_trace)
    all_traces.extend(bar_traces)
    fig = go.Figure(data=all_traces)

    # ---- Helpers used by buttons ----
    def _bar_visibility_for(res_col: Optional[str], cid: Optional[str]) -> List[bool]:
        """Return a length-n_bars array, all False except bars matching (res_col, cid)
        across every ontology library."""
        vis = [False] * n_bars
        if res_col is None or cid is None:
            return vis
        for i, (rc, c, l) in enumerate(bar_keys):
            if rc == res_col and c == cid:
                vis[i] = True
        return vis

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
            x=0.56 if n_panels > 0 else 1.02,
            y=0.78,
            xanchor="left", yanchor="top",
            itemsizing="constant", tracegroupgap=4,
            bgcolor="rgba(255,255,255,0.9)", bordercolor="lightgray", borderwidth=0,
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
        gap = 0.025
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
                title=("Combined score" if lib_idx == n_panels - 1 else None),
                showgrid=True, gridcolor="rgba(180,180,180,0.18)",
                zeroline=False, showline=True, linecolor="rgba(80,80,80,0.4)",
                tickfont=dict(size=8),
            )
            layout_dict[ax_y] = dict(
                domain=[y_bot, y_top],
                anchor=f"x{lib_idx + 2}",
                automargin=True, showgrid=False,
                tickfont=dict(size=7),
            )
            # Pretty library label above each panel
            pretty = lib.replace("_2025", "").replace("_2026", "").replace("_2022", "").replace("_", " ")
            annotations.append(dict(
                text=f"<b>{pretty}</b>",
                xref="paper", yref="paper",
                x=bar_x_domain[0], y=min(1.0, y_top + 0.005),
                xanchor="left", yanchor="bottom",
                showarrow=False, font=dict(size=9, color="#555"),
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

    import json as _json
    post_script = f"""
(function() {{
  var pickerMenuIdxForMode = {_json.dumps(picker_menu_idx_for_mode)};
  var clickButtonMap = {_json.dumps(click_button_map)};
  var geneListMap = {_json.dumps(gene_list_map)};
  var GENE_LIST_PLACEHOLDER = {_json.dumps(GENE_LIST_PLACEHOLDER)};

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
  // panels exist). Plot area width = figWidth - 60 - 20 = figWidth - 80.
  var plotW = figWidth - 80;
  var plotH = figHeight - 200;  // approximate (margin t≈140, b≈60)
  // Panel sits under the legend (legend is at y=0.78 down to ~0.40 worst case).
  var panelLeft = 60 + plotW * 0.56;
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
    ' overflow-y:auto; background:rgba(255,255,255,0.94);' +
    ' border:1px solid #aaa; border-radius:3px;' +
    ' font-family:Menlo, Consolas, monospace; font-size:9px;' +
    ' line-height:1.25; padding:6px 8px; z-index:50;' +
    ' box-shadow:0 1px 4px rgba(0,0,0,0.06);';
  var header = document.createElement('div');
  header.style.cssText = 'font-weight:bold; font-family:sans-serif; font-size:11px; color:#444; margin-bottom:4px;';
  header.textContent = 'Genes in selected category';
  var body = document.createElement('div');
  body.id = 'gene-list-body';
  body.innerHTML = GENE_LIST_PLACEHOLDER;
  panel.appendChild(header);
  panel.appendChild(body);
  parent.appendChild(panel);

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

  gd.on('plotly_buttonclicked', function(event) {{
    if (!event || !event.menu) return;
    var menuIdx = event.menu._index;
    var btnLabel = event.button && event.button.label;
    if (menuIdx === 0) {{
      body.innerHTML = GENE_LIST_PLACEHOLDER;
      return;
    }}
    var modeLabel = _modeForMenu(menuIdx);
    if (!modeLabel) return;
    _setGeneList(modeLabel, btnLabel);
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
# Public entry-point used by aggregate_channels / apply_second_pass_pca
# =============================================================================


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
    _logger=logger,
) -> None:
    """One-shot helper called from aggregation/second-pass: produces all
    extra static overlays + Leiden plots + the interactive HTML.

    GO enrichment is computed once per Leiden resolution and reused by both
    the static Leiden PNGs (legend labels) and the main interactive HTMLs
    (per-cluster pathway bar charts).
    """
    overlay_maps = load_overlay_maps(supercategory_config_path)

    leiden_results: Dict[str, np.ndarray] = {}
    if adata_gene_embed is not None and "X_pca" in adata_gene_embed.obsm:
        _logger.info("  Running Leiden at resolutions %s on gene embedding...", list(leiden_resolutions))
        leiden_results = run_leiden_clustering(adata_gene_embed, leiden_resolutions)

    enrichment_per_res: Dict[str, Dict[str, Dict]] = {}
    if leiden_results and adata_gene_embed is not None:
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
            enrichment_per_res[col] = _run_cluster_enrichment(
                cluster_to_genes, background,
                libraries=enrichr_libraries, _logger=_logger,
            )

    save_supercluster_overlays(
        adata_guide, adata_gene_embed, overlay_maps, plots_dir, plt, _logger=_logger,
    )
    save_leiden_overlays(
        adata_gene_embed, leiden_results, plots_dir, plt, _logger=_logger,
        enrichment_per_res=enrichment_per_res,
    )
    save_interactive_html(
        adata_guide, adata_gene_embed, overlay_maps, leiden_results,
        activity_map, dist_map, corum_map, chad_map,
        plots_dir, _logger=_logger,
        enrichment_per_res=enrichment_per_res,
    )
