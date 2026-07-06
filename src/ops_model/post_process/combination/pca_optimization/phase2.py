"""Phase 2: cross-reporter aggregation + second-pass PCA.

Top-level (submitit-picklable) workers:

* ``aggregate_channels`` — read every ``per_signal/`` h5ad, concat
  across channels, NTC-normalize, score all 4 mAP metrics, compute
  UMAP+PHATE embeddings, write the canonical ``guide_pca_optimized.h5ad``
  + ``gene_embedding_pca_optimized.h5ad`` + ``pca_report.csv``.
* ``apply_second_pass_pca`` — read the aggregate guide h5ad, fit a
  second PCA on the horizontally-concatenated NTC-normalized features,
  retain top ``--second-pca-threshold`` of variance (or run a sweep and
  pick consensus peak), re-aggregate to gene level, re-score
  everything, and write to a ``second_pca_<subdir>/`` sibling.
* ``_save_pc_marker_contributions`` — helper that summarizes which
  per-marker source PCs each 2nd-pass PC inherits.

CHAD_ANNOTATION_PATH (mutable module global, set by main() from
``--chad-annotation``) is imported lazily inside the worker bodies so
each call reads the current value.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import anndata as ad
import numpy as np
import pandas as pd

from ops_model.features.anndata_utils import aggregate_to_level
from ops_utils.analysis.embedding_plots import build_metric_lookup
from ops_utils.analysis.map_scores import plot_map_scatter
from ops_utils.analysis.pca import fit_pca, n_pcs_for_threshold
from ops_utils.analysis.pca_sweep_plots import (
    plot_channel_peaks_bar,
    plot_metric_map_bar,
    plot_metric_violins,
    plot_pca_sweep,
    plot_sweep_curves_summary,
)

from ops_model.post_process.combination.pca_optimization.aggregation import (
    _atomic_write_h5ad,
    _concat_and_normalize,
    _load_per_unit_blocks,
    _plot_chad_umap,
    _save_aggregated_h5ads,
    _save_per_reporter_metric_matrices,
    _score_activity_aggregated,
)
from ops_model.post_process.combination.pca_optimization.embeddings import (
    _compute_and_plot_embeddings,
    _score_consistency,
    _score_distinctiveness,
    _score_ebi_plus,
)
from ops_model.post_process.combination.pca_optimization.sweep_core import (
    _init_sweep_logger,
    _prepare_for_copairs,
    _run_guide_threshold_sweep,
    _save_sweep_outputs,
)


def _lazy_globals():
    """Resolve mutable globals (CHAD path, sweep thresholds, MIN_PCS) at call time."""
    from ops_model.post_process.combination.pca_optimization import (
        CHAD_ANNOTATION_PATH,
        DEFAULT_SWEEP_THRESHOLDS,
        MIN_PCS,
    )

    return CHAD_ANNOTATION_PATH, DEFAULT_SWEEP_THRESHOLDS, MIN_PCS


def aggregate_channels(
    output_dir: str,
    norm_method: str = "ntc",
    per_unit_subdir: str = "per_channel",
    distance: str = "cosine",
    random_seed: int = 42,
    agg_method: str = "mean",
    chromosome_csv: Optional[str] = None,
    umap_type: str = "max",
) -> str:
    """Load per-channel (or per-signal) h5ads, concatenate, normalize, score, save.

    Top-level function (not a method) so submitit can pickle it.

    Args:
        per_unit_subdir: subdirectory containing guide/gene h5ads.
            "per_channel" for standard mode, "per_signal" for downsampled mode.
        agg_method: cells→guides / guides→geneKOs reduction (``mean`` or
            ``median``). Default ``mean``.
        chromosome_csv: optional CSV mapping perturbation → chromosome /
            chromosome_arm; used to color gene-level UMAP + PHATE.
    """
    CHAD_ANNOTATION_PATH, _, MIN_PCS = _lazy_globals()

    _logger = _init_sweep_logger()
    t_start = time.time()
    output_dir = Path(output_dir)
    per_unit_dir = output_dir / per_unit_subdir

    # Step 1: Load per-channel/per-signal blocks
    (guide_blocks, gene_blocks, report_rows, total_cells,
     per_reporter_metric_dfs) = _load_per_unit_blocks(
        per_unit_dir, norm_method, _logger, distance=distance
    )
    if guide_blocks is None:
        return "FAILED: no valid per-channel data loaded"

    # Save the per-reporter mAP matrices for all 4 metrics — atlas and
    # downstream consumers read them from <output_dir>/plots/marker_overlay/
    # to drive per-marker channel selection (CHAD consistency for
    # complex pages, distinctiveness for gene pages, etc.).
    _save_per_reporter_metric_matrices(
        per_reporter_metric_dfs, output_dir, _logger,
    )

    # Step 2: Concat + normalize
    adata_guide, adata_gene = _concat_and_normalize(
        guide_blocks, gene_blocks, norm_method, _logger, agg_method=agg_method
    )
    total_feats = adata_guide.n_vars

    # Step 3: Activity scoring
    metrics_dir = output_dir / "metrics"
    activity_map, r, a = _score_activity_aggregated(
        adata_guide, metrics_dir, _logger, distance=distance
    )

    # Step 4: Save h5ads (before slow metrics) — store X_pca now; UMAP/PHATE added after step 6
    adata_guide.obsm["X_pca"] = np.asarray(adata_guide.X, dtype=np.float32)
    variance_ratio_per_pc = np.array(
        [float(row.get("explained_variance", 0)) for row in report_rows],
        dtype=np.float32,
    )
    adata_guide.uns["pca"] = {
        "params": {"n_components": total_feats, "zero_center": True}
    }
    _save_aggregated_h5ads(
        adata_guide,
        adata_gene,
        report_rows,
        output_dir,
        r,
        a,
        norm_method,
        total_cells,
        _logger,
    )

    # Step 5: Plots
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if activity_map is not None:
        try:
            fig, ax = plt.subplots(figsize=(8, 7))
            plot_map_scatter(ax, activity_map, "Activity", r, show_ntc=False)
            fig.suptitle(
                f"Phenotypic Activity — {total_feats} features",
                fontsize=13,
                fontweight="bold",
            )
            fig.tight_layout()
            fig.savefig(plots_dir / "map_activity.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            _logger.info(f"  Saved plots/map_activity.png")
        except Exception as e:
            _logger.warning(f"  Activity plot failed: {e}")

    plot_sweep_curves_summary(
        per_unit_dir, output_dir, plots_dir, r, a, plt, _logger, min_pcs=MIN_PCS
    )

    # Step 6: Embeddings (UMAP + PHATE) — stored directly into adata_guide.obsm/uns
    # Returns adata_gene_embed (NTC-split gene-level object) with embeddings in obsm
    metric_lookup = build_metric_lookup(activity_map)
    adata_gene_embed = _compute_and_plot_embeddings(
        adata_guide, metric_lookup, plots_dir, plt, _logger,
        random_seed=random_seed,
        chromosome_csv=chromosome_csv,
        umap_type=umap_type,
    )
    # Re-save guide + gene-embed with embeddings now populated
    _atomic_write_h5ad(adata_guide, output_dir / "guide_pca_optimized.h5ad", _logger)
    if adata_gene_embed is not None:
        _atomic_write_h5ad(
            adata_gene_embed, output_dir / "gene_embedding_pca_optimized.h5ad", _logger
        )
    _logger.info(
        "  Re-saved guide_pca_optimized.h5ad + gene_embedding_pca_optimized.h5ad with embeddings"
    )

    # Step 7: Distinctiveness + consistency (slow) — active-only
    dist_map, dist_ratio = _score_distinctiveness(
        adata_guide,
        activity_map,
        r,
        total_feats,
        plots_dir,
        metrics_dir,
        plt,
        _logger,
        distance=distance,
    )
    ebi_plus_map, ebi_plus_ratio = _score_ebi_plus(
        adata_guide,
        activity_map,
        r,
        total_feats,
        plots_dir,
        metrics_dir,
        plt,
        _logger,
        distance=distance,
    )
    corum_map, corum_ratio, chad_map, chad_ratio, ebi_map, ebi_ratio = _score_consistency(
        adata_gene,
        activity_map,
        total_feats,
        plots_dir,
        metrics_dir,
        plt,
        _logger,
        distance=distance,
    )

    # Per-reporter bar chart with all 4 aggregate baselines (active-only)
    plot_channel_peaks_bar(
        report_rows,
        r,
        plots_dir,
        plt,
        _logger,
        dist_ratio=dist_ratio,
        corum_ratio=corum_ratio,
        chad_ratio=chad_ratio,
    )

    # Per-reporter bar chart — unfiltered (all genes, not just active).
    # _score_distinctiveness / _score_consistency above already run with
    # active_only=False (the default), so reuse those ratios verbatim.
    dist_ratio_all = dist_ratio
    corum_ratio_all = corum_ratio
    chad_ratio_all = chad_ratio

    # Build report rows remapped to the _all columns for the unfiltered plot
    unfiltered_rows = []
    for row in report_rows:
        r2 = dict(row)
        r2["distinctiveness"] = r2.get("distinctiveness_all", float("nan"))
        r2["corum"] = r2.get("corum_all", float("nan"))
        r2["chad"] = r2.get("chad_all", float("nan"))
        unfiltered_rows.append(r2)
    plot_channel_peaks_bar(
        unfiltered_rows,
        r,
        plots_dir,
        plt,
        _logger,
        dist_ratio=dist_ratio_all,
        corum_ratio=corum_ratio_all,
        chad_ratio=chad_ratio_all,
        filename="per_channel_peaks_all_genes.png",
    )

    # Bar charts for all 4 metrics (per-perturbation mAP)
    plot_metric_map_bar(
        activity_map, "Activity", "perturbation", r, plots_dir, plt, _logger
    )
    plot_metric_map_bar(
        dist_map, "Distinctiveness", "perturbation", dist_ratio, plots_dir, plt, _logger
    )
    if corum_map is not None:
        corum_entity_col = (
            "complex_id" if "complex_id" in corum_map.columns else corum_map.columns[0]
        )
        plot_metric_map_bar(
            corum_map,
            "Consistency (CORUM)",
            corum_entity_col,
            corum_ratio,
            plots_dir,
            plt,
            _logger,
        )
    if chad_map is not None:
        chad_entity_col = (
            "complex_num" if "complex_num" in chad_map.columns else chad_map.columns[0]
        )
        plot_metric_map_bar(
            chad_map,
            "Consistency (CHAD)",
            chad_entity_col,
            chad_ratio,
            plots_dir,
            plt,
            _logger,
        )

    # Summary violin: per-item mAP distribution per metric (1st-pass).
    try:
        plot_metric_violins(
            metric_maps={
                "Activity": activity_map,
                "Distinctiveness": dist_map,
                "EBI+": ebi_plus_map,
                "EBI": ebi_map,
                "CHAD": chad_map,
                "CORUM": corum_map,
            },
            plots_dir=plots_dir,
            plt=plt,
            _logger=_logger,
            filename="violin_metric_mAPs.png",
            suptitle="1st-pass mAP distributions",
        )
    except Exception as exc:
        _logger.warning(f"  1st-pass metric violin plot failed: {exc}")

    _chad_path = CHAD_ANNOTATION_PATH or "/hpc/projects/icd.ops/configs/gene_clusters/chad_positive_controls_v5_hierarchy.yml"
    if adata_gene_embed is not None and "X_umap" in adata_gene_embed.obsm:
        try:
            import yaml as _yaml
            with open(_chad_path) as f:
                chad_clusters = _yaml.safe_load(f)
            gene_to_cluster = {}
            for cid, cdata in chad_clusters.items():
                name = cdata.get("name", f"cluster_{cid}")
                for gene in cdata.get("genes", []):
                    gene_to_cluster[gene.strip()] = name

            _plot_chad_umap(
                adata_gene_embed.obsm["X_umap"],
                adata_gene_embed.obs["perturbation"].values,
                gene_to_cluster,
                plots_dir / "umap_chad_clusters.png",
                plt, _logger,
            )
        except Exception as e:
            _logger.warning(f"  CHAD UMAP failed: {e}")

    # Extra overlays (super-category, Leiden, interactive HTML)
    try:
        from ops_model.post_process.combination.embedding_overlays import (
            save_extra_overlays,
        )

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
        # Re-save h5ads now that leiden_r* columns + neighbors graph have been
        # added to the in-memory adata objects by save_extra_overlays
        _atomic_write_h5ad(adata_guide, output_dir / "guide_pca_optimized.h5ad", _logger)
        if adata_gene_embed is not None:
            _atomic_write_h5ad(
                adata_gene_embed,
                output_dir / "gene_embedding_pca_optimized.h5ad",
                _logger,
            )
    except Exception as e:
        _logger.warning(f"  Extra overlays failed: {e}")

    elapsed = time.time() - t_start
    _logger.info(f"\nDone in {elapsed/60:.1f} minutes")
    _logger.info(f"  {len(report_rows)} channels, {total_feats} total PCA features")
    _logger.info(f"  Baseline: {r:.1%} active, AUC={a:.4f}")

    return f"SUCCESS: {total_feats} features, {r:.1%} active, AUC={a:.4f}"


# =============================================================================
# Phase 3 (optional): Second-pass PCA on concatenated guide features
# =============================================================================


def _save_pc_marker_contributions(
    components: np.ndarray,
    input_feature_names: List[str],
    n_pcs: int,
    cumvar: np.ndarray,
    out_dir: Path,
    plots_dir: Path,
    plt,
    _logger,
    top_n_pcs_for_bars: int = 50,
    top_k_markers_per_pc: int = 3,
) -> None:
    """For each second-pass PC, attribute variance back to source markers.

    Input feature names follow the ``<signal>_PC<n>`` convention from Phase 1
    (``_save_sweep_outputs``). For each second-pass PC ``sPCi`` we sum the
    squared loadings (``components[i, j] ** 2``) of all input features that
    came from a given signal — that's the share of ``sPCi``'s unit variance
    explained by that marker.

    Outputs:
      ``second_pca_marker_contributions.csv`` — wide table, rows=sPCi,
        columns=marker. Values sum to 1 across markers within each row.
      ``plots/second_pca_marker_contributions.png`` — heatmap (n_pcs x markers).
      ``plots/second_pca_marker_contributions_stacked.png`` — stacked bar of
        the top ``top_n_pcs_for_bars`` PCs.
      ``second_pca_top_markers_per_pc.csv`` — the top ``top_k_markers_per_pc``
        markers for each PC with their fractional contribution.
    """
    import re

    pat = re.compile(r"^(?P<signal>.+)_PC\d+$")
    feat_to_signal: List[str] = []
    for f in input_feature_names:
        m = pat.match(f)
        feat_to_signal.append(m.group("signal") if m else f)

    feat_to_signal_arr = np.asarray(feat_to_signal)
    unique_signals = sorted(set(feat_to_signal))

    # Squared loadings, normalized so each PC's row sums to 1
    sq = np.square(components).astype(np.float64)  # (n_pcs, n_feats_in)
    row_sums = sq.sum(axis=1, keepdims=True)
    sq_norm = np.divide(sq, row_sums, out=np.zeros_like(sq), where=row_sums > 0)

    contrib = np.zeros((n_pcs, len(unique_signals)), dtype=np.float64)
    for s_idx, signal in enumerate(unique_signals):
        cols = np.where(feat_to_signal_arr == signal)[0]
        if cols.size == 0:
            continue
        contrib[:, s_idx] = sq_norm[:, cols].sum(axis=1)

    pc_names = [f"sPC{i}" for i in range(n_pcs)]
    explained_var_per_pc = np.diff(np.concatenate([[0.0], cumvar])).astype(np.float64)
    df = pd.DataFrame(contrib, index=pc_names, columns=unique_signals)
    df.insert(0, "explained_variance", explained_var_per_pc[:n_pcs])
    df.to_csv(out_dir / "second_pca_marker_contributions.csv", index_label="pc")
    _logger.info("  Saved second_pca_marker_contributions.csv")

    # Top-K markers per PC (long-format CSV)
    top_rows = []
    for i in range(n_pcs):
        order = np.argsort(-contrib[i])
        for rank in range(min(top_k_markers_per_pc, len(unique_signals))):
            j = order[rank]
            top_rows.append({
                "pc": pc_names[i],
                "rank": rank + 1,
                "signal": unique_signals[j],
                "contribution": float(contrib[i, j]),
                "explained_variance": float(explained_var_per_pc[i]),
            })
    pd.DataFrame(top_rows).to_csv(out_dir / "second_pca_top_markers_per_pc.csv", index=False)
    _logger.info("  Saved second_pca_top_markers_per_pc.csv")

    # Heatmap (n_pcs x markers)
    try:
        fig_w = max(10, 0.25 * len(unique_signals) + 4)
        fig_h = max(6, 0.18 * n_pcs + 2)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(contrib, aspect="auto", cmap="viridis", vmin=0, vmax=contrib.max() or 1)
        ax.set_xticks(range(len(unique_signals)))
        ax.set_xticklabels(unique_signals, rotation=80, ha="right", fontsize=8)
        step = max(1, n_pcs // 60)
        ax.set_yticks(range(0, n_pcs, step))
        ax.set_yticklabels([pc_names[i] for i in range(0, n_pcs, step)], fontsize=8)
        ax.set_xlabel("Source marker (Phase 1 signal)")
        ax.set_ylabel("Second-pass PC")
        ax.set_title(
            f"Second-pass PC × marker contribution (squared loadings, row-normalized) — "
            f"{n_pcs} PCs × {len(unique_signals)} markers"
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cbar.set_label("share of PC variance")
        fig.tight_layout()
        fig.savefig(plots_dir / "second_pca_marker_contributions.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        _logger.info("  Saved plots/second_pca_marker_contributions.png")
    except Exception as exc:
        _logger.warning(f"  Heatmap failed: {exc}")

    # Stacked bar of top PCs — adjacent signals must look very different in
    # BOTH color and hatch, so any segment in the stack can be matched to its
    # legend entry by either dimension. We walk color and hatch on independent
    # coprime strides near n/phi (~138°), so each step i→i+1 jumps far in hue
    # AND far in the hatch list, and neither dimension repeats until its full
    # cycle is exhausted.
    try:
        n_show = min(top_n_pcs_for_bars, n_pcs)
        import math

        import seaborn as sns

        hatch_patterns = [
            "", "///", "\\\\\\", "xxx", "...", "+++",
            "ooo", "|||", "---", "**", "OO", "//\\\\",
        ]
        n_colors = 40
        n_hatches = len(hatch_patterns)
        husl = sns.color_palette("husl", n_colors)
        phi = (1 + math.sqrt(5)) / 2

        def _coprime_stride(n: int, target: float) -> int:
            s = max(1, int(round(target)))
            for d in range(n):
                for sign in (1, -1):
                    cand = s + sign * d
                    if 1 <= cand < n and math.gcd(cand, n) == 1:
                        return cand
            return 1

        color_stride = _coprime_stride(n_colors, n_colors / phi)
        hatch_stride = _coprime_stride(n_hatches, n_hatches / phi)
        signal_style = {
            sig: (
                husl[(i * color_stride) % n_colors],
                hatch_patterns[(i * hatch_stride) % n_hatches],
            )
            for i, sig in enumerate(unique_signals)
        }

        fig, ax = plt.subplots(figsize=(max(14, 0.50 * n_show + 4), 9))
        bottom = np.zeros(n_show)
        for s_idx, signal in enumerate(unique_signals):
            vals = contrib[:n_show, s_idx]
            if vals.sum() < 1e-6:
                continue
            color, hatch = signal_style[signal]
            ax.bar(
                range(n_show),
                vals,
                bottom=bottom,
                color=color,
                hatch=hatch,
                label=signal if len(signal) <= 35 else signal[:32] + "...",
                width=0.85,
                edgecolor="black",
                linewidth=0.4,
            )
            bottom += vals
        ax.set_xticks(range(n_show))
        ax.set_xticklabels(pc_names[:n_show], rotation=70, fontsize=8)
        ax.set_ylabel("share of PC variance")
        ax.set_xlabel("Second-pass PC (in order)")
        ax.set_title(
            f"Marker contribution to top {n_show} second-pass PCs"
        )
        ax.set_ylim(0, 1.0)
        ncol = 2 if len(unique_signals) > 20 else 1
        ax.legend(
            bbox_to_anchor=(1.02, 1.0), loc="upper left",
            fontsize=8, ncol=ncol, columnspacing=0.6, handletextpad=0.3,
        )
        fig.tight_layout()
        fig.savefig(
            plots_dir / "second_pca_marker_contributions_stacked.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)
        _logger.info("  Saved plots/second_pca_marker_contributions_stacked.png")
    except Exception as exc:
        _logger.warning(f"  Stacked bar failed: {exc}")

    # Top-3 summary log
    summary_lines = []
    for i in range(min(50, n_pcs)):
        order = np.argsort(-contrib[i])[:3]
        bits = ", ".join(
            f"{unique_signals[j]}({contrib[i, j]*100:.0f}%)" for j in order
        )
        summary_lines.append(f"  {pc_names[i]} (var={explained_var_per_pc[i]:.3f}): {bits}")
    if summary_lines:
        _logger.info("  Top contributors per second-pass PC:")
        for line in summary_lines:
            _logger.info(line)


def apply_second_pass_pca(
    output_dir: str,
    threshold: float = 0.0,
    distance: str = "cosine",
    norm_method: str = "ntc",
    subdir: Optional[str] = None,
    run_sweep: bool = True,
    sweep_thresholds: Optional[List[float]] = None,
    random_seed: int = 42,
    agg_method: str = "mean",
    chromosome_csv: Optional[str] = None,
    input_path: Optional[str] = None,
    subdir_suffix: str = "",
    skip_pca: bool = False,
    umap_type: str = "max",
    consensus_metrics=None,
    sweep_metric: str = "mean_map",
) -> str:
    """Second-pass PCA on the already-concatenated NTC-normalized guide features.

    Reads a guide-level h5ad (default ``<output_dir>/guide_pca_optimized.h5ad``;
    override with ``input_path`` to point at a corrected variant), fits a PCA
    on its horizontally-concatenated features, retains the top ``threshold``
    of cumulative variance, re-aggregates to gene level, re-scores all
    phenotypic metrics, and writes results to ``<output_dir>/<subdir>/``.

    Default ``subdir`` is ``second_pca_consensus`` (consensus mode) or
    ``second_pca_<pct>`` (fixed threshold). ``subdir_suffix`` appends a tag
    (e.g. ``_chrom_arm_corr``) so an opt-in correction run never clobbers the
    untouched baseline output sitting next to it.

    ``skip_pca``: when True, skip the 2nd-pass PCA fit entirely — treat the
    input guide ``.X`` as already-final features, aggregate to gene level,
    score the 4 metrics, and write outputs to a ``chrom_arm_corr<suffix>/``
    subdir (NOT ``second_pca_*``). Useful when you want the full plotting +
    scoring pipeline run on a chrom-arm-corrected guide h5ad without the
    additional compression layer.
    """
    CHAD_ANNOTATION_PATH, DEFAULT_SWEEP_THRESHOLDS, _ = _lazy_globals()

    _logger = _init_sweep_logger()
    t_start = time.time()
    output_dir = Path(output_dir)

    # Resolve the consensus_metrics list into a canonical tuple + a subdir tag.
    # Tag = "" for the default {activity, distinctiveness, ebi}; otherwise
    # "_ACT" / "_CHAD" / "_DIST_EBI" / "_ACT_DIST_CHAD" / etc. — based on
    # which subset is in the list. Lets a single second_pca_consensus*/ dir
    # per metric-subset coexist as siblings without clobbering.
    from ops_model.post_process.combination.pca_optimization.sweep_core import (
        _normalize_consensus_metrics,
        consensus_metrics_subdir_tag,
    )
    try:
        consensus_metrics = _normalize_consensus_metrics(consensus_metrics)
    except ValueError as e:
        return f"FAILED: {e}"
    metric_subdir_tag = consensus_metrics_subdir_tag(consensus_metrics)

    # sweep_metric ("mean_map" default, "ratio" alternative) controls whether
    # per-threshold scores in the sweep are continuous means of per-item mAP
    # (the historical pre-May-23 behavior; more stable) or fraction-significant
    # counts. To avoid clobbering existing canonical second_pca_consensus/
    # output that was generated under the ratio picker, mean_map output lands
    # in a sibling "_MEANMAP" subdir. ratio output stays at the canonical
    # path (no suffix).
    sweep_metric = (sweep_metric or "mean_map").lower()
    if sweep_metric not in ("ratio", "mean_map"):
        return f"FAILED: sweep_metric must be 'ratio' or 'mean_map', got {sweep_metric!r}"
    if sweep_metric == "mean_map":
        metric_subdir_tag += "_MEANMAP"

    # threshold <= 0 means "consensus sweep" — pick the threshold that maximizes the
    # normalized sum of activity + distinctiveness + (EBI or CHAD, configurable).
    # Forces the sweep on.
    use_consensus = threshold is None or threshold <= 0
    if skip_pca:
        # No PCA fit → no sweep + a different subdir prefix. The wrapper passes
        # subdir_suffix already containing "_chrom_arm_corr[_method]", so we
        # just strip the leading underscore and use it directly as the subdir.
        run_sweep = False
        use_consensus = False
        if subdir is None:
            subdir = (subdir_suffix.lstrip("_") if subdir_suffix
                      else "chrom_arm_corr")
    elif use_consensus:
        run_sweep = True
        if subdir is None:
            subdir = f"second_pca_consensus{metric_subdir_tag}{subdir_suffix}"
    else:
        if subdir is None:
            subdir = f"second_pca_{int(round(threshold * 100)):02d}{metric_subdir_tag}{subdir_suffix}"
    out_dir = output_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    if input_path is not None:
        guide_path = Path(input_path)
    else:
        guide_path = output_dir / "guide_pca_optimized.h5ad"
    if not guide_path.exists():
        return f"FAILED: {guide_path} not found — run aggregation first"

    _logger.info(f"Loading first-pass concatenated guide h5ad: {guide_path}")
    adata_in = ad.read_h5ad(guide_path)
    n_guides = adata_in.n_obs
    n_feats_in = adata_in.n_vars
    _logger.info(
        f"  Input: {n_guides:,} guides x {n_feats_in:,} concatenated NTC-normalized features"
    )

    if skip_pca:
        # Treat input X as already-final features — no PCA fit, no sweep,
        # no slicing. The downstream scoring + aggregation + plot pipeline
        # runs unchanged on the input AnnData.
        _logger.info(
            f"  skip_pca=True → using input features directly "
            f"({n_feats_in} features, no 2nd-pass compression)"
        )
        adata_guide = adata_in.copy()
        # Keep var_names from the corrected input (e.g. "Phase::PC0"
        # for combined-titration outputs).
        n_pcs = adata_guide.n_vars
        cumvar = np.full(n_pcs, np.nan, dtype=np.float32)
        pca_model = None
        sweep_result = None
        adata_guide.obsm["X_pca"] = np.asarray(adata_guide.X, dtype=np.float32)
        adata_guide.uns["pca"] = {
            "variance_ratio": np.full(n_pcs, np.nan, dtype=np.float32),
            "params": {"n_components": int(n_pcs), "skip_pca": True},
        }
        adata_guide.uns["second_pca"] = {
            "input_features": int(n_feats_in),
            "input_path": str(guide_path),
            "skip_pca": True,
        }
        adata_guide.uns["norm_method"] = norm_method
        input_feature_names = list(adata_in.var_names)
        threshold = float("nan")
    else:
        X_in = np.asarray(adata_in.X, dtype=np.float32)
        X_in = np.nan_to_num(X_in, nan=0.0, posinf=0.0, neginf=0.0)
        _logger.info(f"  Fitting second-pass PCA on {n_guides:,} x {n_feats_in:,} matrix...")
        t_pca = time.time()
        X_pcs, cumvar, pca_model = fit_pca(X_in)
        _logger.info(
            f"  PCA done in {time.time() - t_pca:.0f}s — {X_pcs.shape[1]} components computed"
        )

        # Sweep variance thresholds at guide level (input is already NTC-normalized)
        sweep_result = None
        if run_sweep:
            thresholds = sweep_thresholds or DEFAULT_SWEEP_THRESHOLDS
            _logger.info(
                f"  Sweeping {len(thresholds)} variance thresholds at guide level..."
            )
            t_sweep = time.time()
            sweep_result = _run_guide_threshold_sweep(
                X_pcs,
                cumvar,
                adata_in.obs,
                thresholds,
                _logger=_logger,
                distance=distance,
                consensus_metrics=consensus_metrics,
                sweep_metric=sweep_metric,
            )
            _logger.info(f"  Sweep done in {time.time() - t_sweep:.0f}s")

        if use_consensus:
            if sweep_result is None:
                return f"FAILED: consensus mode requested but sweep produced no valid thresholds"
            threshold = float(sweep_result["consensus_t"])
            n_pcs = int(sweep_result["consensus_n"])
            _logger.info(
                f"  Consensus mode: chose {threshold:.0%} ({n_pcs} PCs) — "
                f"max of normalized act+dist+chad"
            )
        else:
            n_pcs = n_pcs_for_threshold(cumvar, threshold)
            _logger.info(
                f"  Retaining {n_pcs} PCs at {threshold:.0%} "
                f"(explained={float(cumvar[n_pcs - 1]):.3f})"
            )
            if sweep_result is not None:
                _logger.info(
                    f"  (sweep consensus peak: {sweep_result['consensus_t']:.0%} = "
                    f"{sweep_result['consensus_n']} PCs)"
                )

        X_reduced = X_pcs[:, :n_pcs].astype(np.float32)
        pc_names = [f"sPC{i}" for i in range(n_pcs)]

        obs = adata_in.obs.copy()
        adata_guide = ad.AnnData(
            X=X_reduced,
            obs=obs,
            var=pd.DataFrame(index=pc_names),
        )
        variance_ratio_per_pc = np.diff(np.concatenate([[0.0], cumvar])).astype(np.float32)
        input_feature_names = list(adata_in.var_names)
        second_pca_uns = {
            "input_features": int(n_feats_in),
            "input_path": str(guide_path),
            "threshold": float(threshold),
            "n_pcs": int(n_pcs),
            "explained_variance": float(cumvar[n_pcs - 1]),
            "components": pca_model.components_[:n_pcs].astype(np.float32),
            "input_feature_names": input_feature_names,
        }
        adata_guide.obsm["X_pca"] = X_reduced.copy()
        adata_guide.uns["pca"] = {
            "variance_ratio": variance_ratio_per_pc[:n_pcs],
            "params": {
                "n_components": int(n_pcs),
                "threshold": float(threshold),
                "zero_center": True,
            },
        }
        adata_guide.uns["second_pca"] = second_pca_uns
        adata_guide.uns["norm_method"] = norm_method

    adata_guide = _prepare_for_copairs(adata_guide)
    adata_gene = aggregate_to_level(
        adata_guide, "gene", method=agg_method, preserve_batch_info=False, subsample_controls=False
    )
    adata_gene = _prepare_for_copairs(adata_gene)
    adata_gene.obsm["X_pca"] = np.asarray(adata_gene.X, dtype=np.float32)
    adata_gene.uns["pca"] = adata_guide.uns["pca"]
    # In skip_pca mode there's no fitted-PCA metadata to carry across — reuse
    # the same uns block we stamped on adata_guide in the skip_pca branch.
    adata_gene.uns["second_pca"] = adata_guide.uns["second_pca"]
    adata_gene.uns["norm_method"] = norm_method

    total_feats = adata_guide.n_vars
    _logger.info(
        f"  After 2nd PCA: {adata_guide.n_obs:,} guides x {total_feats} PCs, "
        f"{adata_gene.n_obs:,} genes"
    )

    metrics_dir = out_dir / "metrics"
    activity_map, r, a = _score_activity_aggregated(
        adata_guide, metrics_dir, _logger, distance=distance
    )

    adata_guide.uns["pca_optimized"] = True
    adata_guide.uns["second_pca_applied"] = True
    adata_guide.uns["baseline_activity"] = float(r)
    adata_guide.uns["baseline_auc"] = float(a)
    adata_gene.uns["pca_optimized"] = True
    adata_gene.uns["second_pca_applied"] = True
    adata_guide.write_h5ad(out_dir / "guide_pca_optimized.h5ad")
    adata_gene.write_h5ad(out_dir / "gene_pca_optimized.h5ad")
    _logger.info(f"  Saved guide/gene_pca_optimized.h5ad in {subdir}/")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not skip_pca:
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(np.arange(1, len(cumvar) + 1), cumvar, lw=2, color="steelblue")
            ax.axhline(threshold, ls="--", color="red", alpha=0.6, label=f"{threshold:.0%}")
            ax.axvline(n_pcs, ls=":", color="black", alpha=0.5, label=f"{n_pcs} PCs")
            ax.set_xlabel("PC index")
            ax.set_ylabel("Cumulative explained variance")
            ax.set_title(
                f"Second-pass PCA — {n_feats_in} features → {n_pcs} PCs at {threshold:.0%}"
            )
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(plots_dir / "second_pca_cumvar.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            _logger.info("  Saved plots/second_pca_cumvar.png")
        except Exception as exc:
            _logger.warning(f"  Cumvar plot failed: {exc}")

    if sweep_result is not None:
        sweep_df = pd.DataFrame(sweep_result["sweep_rows"])
        sweep_csv = out_dir / "second_pca_sweep.csv"
        sweep_df.to_csv(sweep_csv, index=False)
        _logger.info(f"  Saved {sweep_csv.name}")
        try:
            plot_pca_sweep(
                sweep_df,
                signal="2nd PCA",
                peak_t=threshold,
                peak_n=n_pcs,
                suptitle=(
                    f"2nd-pass PCA sweep — {n_feats_in} concat features → "
                    f"chosen={n_pcs} PCs @ {threshold:.0%} "
                    f"(consensus={sweep_result['consensus_t']:.0%} = "
                    f"{sweep_result['consensus_n']} PCs"
                    + (f", sweep_metric={sweep_metric}" if sweep_metric != 'ratio' else "")
                    + ")"
                ),
                plots_dir=plots_dir,
                file_prefix="second_pca",
                fixed_threshold=threshold,
                sweep_peak_t=sweep_result["consensus_t"],
                metric_peaks={
                    "peak_act_t": sweep_result["peak_act_t"],
                    "peak_dist_t": sweep_result["peak_dist_t"],
                    "peak_ebi_t": sweep_result.get("peak_ebi_t"),
                },
                sweep_metric=sweep_metric,
            )
            _logger.info("  Saved plots/second_pca_sweep.png")
        except Exception as exc:
            _logger.warning(f"  Sweep plot failed: {exc}")

    # Per-PC marker contribution (which signals drive each second-pass PC)
    # Skip in no-PCA mode — there's no fitted PCA basis to attribute back to.
    if not skip_pca:
        try:
            _save_pc_marker_contributions(
                components=pca_model.components_[:n_pcs],
                input_feature_names=input_feature_names,
                n_pcs=n_pcs,
                cumvar=cumvar,
                out_dir=out_dir,
                plots_dir=plots_dir,
                plt=plt,
                _logger=_logger,
            )
        except Exception as exc:
            _logger.warning(f"  Per-PC marker contribution plot failed: {exc}")

    if activity_map is not None:
        try:
            fig, ax = plt.subplots(figsize=(8, 7))
            plot_map_scatter(ax, activity_map, "Activity", r, show_ntc=False)
            fig.suptitle(
                f"Phenotypic Activity (2nd PCA) — {total_feats} features",
                fontsize=13,
                fontweight="bold",
            )
            fig.tight_layout()
            fig.savefig(plots_dir / "map_activity.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            _logger.warning(f"  Activity plot failed: {exc}")

    metric_lookup = build_metric_lookup(activity_map) if activity_map is not None else {}
    adata_gene_embed = _compute_and_plot_embeddings(
        adata_guide, metric_lookup, plots_dir, plt, _logger,
        random_seed=random_seed,
        chromosome_csv=chromosome_csv,
        umap_type=umap_type,
    )
    _atomic_write_h5ad(adata_guide, out_dir / "guide_pca_optimized.h5ad", _logger)
    if adata_gene_embed is not None:
        _atomic_write_h5ad(
            adata_gene_embed, out_dir / "gene_embedding_pca_optimized.h5ad", _logger
        )

    dist_map, dist_ratio = _score_distinctiveness(
        adata_guide,
        activity_map,
        r,
        total_feats,
        plots_dir,
        metrics_dir,
        plt,
        _logger,
        distance=distance,
    )
    ebi_plus_map, ebi_plus_ratio = _score_ebi_plus(
        adata_guide,
        activity_map,
        r,
        total_feats,
        plots_dir,
        metrics_dir,
        plt,
        _logger,
        distance=distance,
    )
    corum_map, corum_ratio, chad_map, chad_ratio, ebi_map, ebi_ratio = _score_consistency(
        adata_gene,
        activity_map,
        total_feats,
        plots_dir,
        metrics_dir,
        plt,
        _logger,
        distance=distance,
    )

    if activity_map is not None:
        plot_metric_map_bar(
            activity_map, "Activity", "perturbation", r, plots_dir, plt, _logger
        )
    if dist_map is not None:
        plot_metric_map_bar(
            dist_map,
            "Distinctiveness",
            "perturbation",
            dist_ratio,
            plots_dir,
            plt,
            _logger,
        )
    if corum_map is not None:
        col = (
            "complex_id" if "complex_id" in corum_map.columns else corum_map.columns[0]
        )
        plot_metric_map_bar(
            corum_map,
            "Consistency (CORUM)",
            col,
            corum_ratio,
            plots_dir,
            plt,
            _logger,
        )
    if chad_map is not None:
        col = (
            "complex_num" if "complex_num" in chad_map.columns else chad_map.columns[0]
        )
        plot_metric_map_bar(
            chad_map,
            "Consistency (CHAD)",
            col,
            chad_ratio,
            plots_dir,
            plt,
            _logger,
        )

    # Summary violin: per-item mAP distribution per metric, mean annotated.
    try:
        plot_metric_violins(
            metric_maps={
                "Activity": activity_map,
                "Distinctiveness": dist_map,
                "EBI+": ebi_plus_map,
                "EBI": ebi_map,
                "CHAD": chad_map,
                "CORUM": corum_map,
            },
            plots_dir=plots_dir,
            plt=plt,
            _logger=_logger,
            filename="violin_metric_mAPs.png",
            suptitle=f"2nd-pass mAP distributions (n_pcs={n_pcs} @ {threshold:.0%})",
        )
    except Exception as exc:
        _logger.warning(f"  Metric violin plot failed: {exc}")

    _chad_path = (
        CHAD_ANNOTATION_PATH
        or "/hpc/projects/icd.ops/configs/gene_clusters/chad_positive_controls_v5_hierarchy.yml"
    )
    if adata_gene_embed is not None and "X_umap" in adata_gene_embed.obsm:
        try:
            import yaml as _yaml

            with open(_chad_path) as f:
                chad_clusters = _yaml.safe_load(f)
            gene_to_cluster = {}
            for cid, cdata in chad_clusters.items():
                name = cdata.get("name", f"cluster_{cid}")
                for gene in cdata.get("genes", []):
                    gene_to_cluster[gene.strip()] = name
            _plot_chad_umap(
                adata_gene_embed.obsm["X_umap"],
                adata_gene_embed.obs["perturbation"].values,
                gene_to_cluster,
                plots_dir / "umap_chad_clusters.png",
                plt,
                _logger,
            )
        except Exception as exc:
            _logger.warning(f"  CHAD UMAP failed: {exc}")

    # Extra overlays (super-category, Leiden, interactive HTML)
    try:
        from ops_model.post_process.combination.embedding_overlays import (
            save_extra_overlays,
        )

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
        # Re-save h5ads with leiden_r* columns + neighbors graph just added
        _atomic_write_h5ad(adata_guide, out_dir / "guide_pca_optimized.h5ad", _logger)
        if adata_gene_embed is not None:
            _atomic_write_h5ad(
                adata_gene_embed,
                out_dir / "gene_embedding_pca_optimized.h5ad",
                _logger,
            )
    except Exception as exc:
        _logger.warning(f"  Extra overlays failed: {exc}")

    summary = {
        "n_guides": int(n_guides),
        "n_features_input": int(n_feats_in),
        "threshold": float(threshold),
        "n_pcs": int(n_pcs),
        "explained_variance": float(cumvar[n_pcs - 1]),
        "activity": float(r),
        "auc": float(a),
        "distinctiveness": float(dist_ratio),
        "corum": float(corum_ratio),
        "chad": float(chad_ratio),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "second_pca_summary.csv", index=False)

    elapsed = time.time() - t_start
    _logger.info(
        f"Done in {elapsed/60:.1f} min — {n_pcs}/{n_feats_in} PCs @ {threshold:.0%}, "
        f"act={r:.1%} AUC={a:.4f} dist={dist_ratio:.1%} corum={corum_ratio:.1%} chad={chad_ratio:.1%}"
    )
    return (
        f"SUCCESS: {n_pcs}/{n_feats_in} PCs @ {threshold:.0%}, "
        f"act={r:.1%} AUC={a:.4f}"
    )

