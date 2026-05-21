"""Modality-distinctiveness analysis at a fixed cells/guide budget.

At one shared cells/guide budget (default = the smallest median across
requested groups, ~199 cells/guide for cp/4i/matched_livecell), score each
modality independently and identify what each captures that the others don't:

- Per-gene distinctiveness (which genes are *significant*)
- Per-CORUM-complex consistency (which complexes are *significant*)
- Per-CHAD-cluster consistency (which clusters are *significant*)

Cross-modality outputs:
- Set comparisons → which genes/complexes/clusters are picked up by which
  subset of {cp, 4i, matched_livecell} (cp_only, 4i_only, livecell_only,
  pairwise intersections, all-three, none).
- Pie charts of gene **super-categories** for each modality-exclusive set
  (chad_boosted, 8 categories).
- CHAD cluster heatmap (rows=clusters, cols=modalities, value=cluster mAP).
- CORUM complex heatmap (rows=complexes, cols=modalities, value=complex mAP).

Output dir is fully isolated from other titration modes:
    .../with_cp/with_4i/all_livecell/<...>/modality_distinctiveness/
        cells_per_guide_<budget>/cp_vs_4i_vs_matched_livecell/

Usage::

    uv run python -m ops_model.post_process.combination.pca_modality_distinctiveness \\
        --cell-dino --paper-v1 --groups cp,4i,matched_livecell --slurm
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd

from ops_model.features.anndata_utils import (
    aggregate_to_level,
    hconcat_by_perturbation,
    normalize_guide_adata,
)
from ops_model.post_process.combination.pca_combined_titration import (
    DEFAULT_MATCHING_CONFIG,
    _build_parser as _combined_parser,
    _per_guide_median,
    _per_guide_pool,
    _resolve_group_paths,
    _resolve_matched_set_membership,
    _resolve_output_dir,
)
from ops_model.post_process.combination.pca_titration import (
    _prepare_for_copairs,
    _subsample_per_guide_and_aggregate,
)
from ops_utils.analysis.map_scores import (
    phenotypic_consistency_corum,
    phenotypic_consistency_manual_annotation,
    phenotypic_distinctivness,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NULL_SIZE = 100_000
DEFAULT_GROUPS = ("cp", "4i", "matched_livecell_best")
DEFAULT_SUPERCATEGORY_CONFIG = Path(
    "/home/gav.sturm/linked_folders/mydata/ops_mono/organelle_profiler/configs/"
    "gene_supercategory_mapping.yaml"
)
GROUP_PALETTE = {
    "cp": "#d97706",                     # amber
    "4i": "#2563eb",                     # blue
    "matched_livecell": "#10b981",       # green
    "matched_livecell_best": "#10b981",  # green (curated best per organelle)
    "livecell": "#0891b2",               # teal (all standard live-cell markers)
    "all": "#6b7280",                    # gray
}
SUPERCAT_FALLBACK_COLOR = "#9ca3af"


# ---------------------------------------------------------------------------
# Output-dir resolution (independent of combined_titration's compare dir)
# ---------------------------------------------------------------------------


def _resolve_modality_dir(
    args: argparse.Namespace, groups: Sequence[str], cells_per_guide: int,
) -> Path:
    """Top-level dir for the modality-distinctiveness analysis. Nests under the
    union of channel-sets needed by the groups (mirrors compare-dir logic)."""
    ns = argparse.Namespace(**vars(args))
    ns.only_4i = False
    ns.only_cp = False
    ns.include_cp = any(g in ("cp", "all") for g in groups)
    ns.include_4i = any(g in ("4i", "all") for g in groups)
    base = _resolve_output_dir(ns)
    return (
        base / "modality_distinctiveness"
        / f"cells_per_guide_{cells_per_guide}"
        / "_vs_".join(groups)
    )


# ---------------------------------------------------------------------------
# Combined-matrix builder (mirrors pca_combined_titration but exposes the score
# map outputs directly instead of just the aggregate stats)
# ---------------------------------------------------------------------------


def _build_combined_at_target(
    cells_blocks: List[ad.AnnData],
    cells_per_guide: int,
    norm_method: str,
    rng: np.random.RandomState,
) -> ad.AnnData:
    """Subsample → guide-aggregate → NTC-normalize each reporter, then h-concat."""
    blocks = []
    for adata in cells_blocks:
        g_sub = _subsample_per_guide_and_aggregate(adata, cells_per_guide, rng)
        g_norm = normalize_guide_adata(g_sub, norm_method)
        sig = str(adata.obs.get("signal", pd.Series(["?"])).iloc[0])
        g_norm.var_names = [f"{sig}::{v}" for v in g_norm.var_names]
        blocks.append(g_norm)
    return hconcat_by_perturbation(blocks, level="guide")


def score_one_group(
    cells_h5ad_paths: List[str],
    output_dir: str,
    cells_per_guide: int,
    norm_method: str = "ntc",
    distance: str = "cosine",
    random_seed: int = 42,
    group_label: str = "combined",
    cache: bool = True,
) -> str:
    """Score one modality at the fixed cells/guide budget. Writes three CSVs:

    - <out>/distinctiveness.csv  (per-perturbation mAP + significance)
    - <out>/corum.csv            (per-CORUM-complex mAP + significance)
    - <out>/chad.csv             (per-CHAD-cluster mAP + significance)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dist = out_dir / "distinctiveness.csv"
    csv_corum = out_dir / "corum.csv"
    csv_chad = out_dir / "chad.csv"

    if cache and all(p.is_file() for p in (csv_dist, csv_corum, csv_chad)):
        logger.info(f"[{group_label}] All 3 score CSVs cached; skipping recompute.")
        return f"CACHED: {out_dir}"

    rng = np.random.RandomState(random_seed)
    cells_blocks: List[ad.AnnData] = []
    for p in cells_h5ad_paths:
        a = ad.read_h5ad(p)
        if "signal" not in a.obs.columns:
            a.obs["signal"] = Path(p).stem.replace("_cells", "")
        cells_blocks.append(a)
    logger.info(
        f"[{group_label}] Loaded {len(cells_blocks)} reporters; "
        f"sampling {cells_per_guide:,} cells/sgRNA..."
    )

    t0 = time.time()
    combined = _build_combined_at_target(cells_blocks, cells_per_guide, norm_method, rng)
    logger.info(
        f"[{group_label}] Combined matrix: {combined.n_obs:,} guides × "
        f"{combined.n_vars:,} features (build: {time.time() - t0:.0f}s)"
    )

    # Distinctiveness — per-perturbation map (gene level)
    g_copairs = _prepare_for_copairs(combined.copy())
    dist_map, dist_ratio = phenotypic_distinctivness(
        g_copairs, plot_results=False, null_size=NULL_SIZE, distance=distance,
    )
    dist_map = dist_map.copy()
    dist_map["group"] = group_label
    dist_map.to_csv(csv_dist, index=False)
    logger.info(
        f"[{group_label}] distinctiveness: {len(dist_map)} perts, "
        f"{int(dist_map['below_corrected_p'].sum())} significant "
        f"(ratio={dist_ratio:.1%})"
    )

    # Aggregate to gene level for consistency scoring
    e_norm = aggregate_to_level(
        g_copairs, "gene", preserve_batch_info=False, subsample_controls=False,
    )
    e_copairs = _prepare_for_copairs(e_norm)

    corum_map, corum_ratio = phenotypic_consistency_corum(
        e_copairs, plot_results=False, null_size=NULL_SIZE,
        cache_similarity=True, distance=distance,
    )
    corum_map = corum_map.copy()
    corum_map["group"] = group_label
    corum_map.to_csv(csv_corum, index=False)
    logger.info(
        f"[{group_label}] CORUM: {len(corum_map)} complexes, "
        f"{int(corum_map['below_corrected_p'].sum())} significant "
        f"(ratio={corum_ratio:.1%})"
    )

    chad_map, chad_ratio = phenotypic_consistency_manual_annotation(
        e_copairs, plot_results=False, null_size=NULL_SIZE,
        cache_similarity=True, distance=distance,
    )
    chad_map = chad_map.copy()
    chad_map["group"] = group_label
    chad_map.to_csv(csv_chad, index=False)
    logger.info(
        f"[{group_label}] CHAD: {len(chad_map)} clusters, "
        f"{int(chad_map['below_corrected_p'].sum())} significant "
        f"(ratio={chad_ratio:.1%})"
    )
    return f"SUCCESS: {out_dir}"


# ---------------------------------------------------------------------------
# Cross-modality set comparisons
# ---------------------------------------------------------------------------


def _build_distinguished_table(
    score_dfs: Dict[str, pd.DataFrame],
    key_col: str,
    map_threshold: Optional[float],
) -> pd.DataFrame:
    """Wide table: rows=key (gene/complex/cluster), cols=group flags.

    A key is "distinguished" by a group if either ``below_corrected_p`` is True
    or its mAP ≥ ``map_threshold`` (when provided). Adds a ``modality_set``
    column with the sorted-tuple name (e.g. "cp_only", "cp+livecell", "all", "none").
    """
    rows: Dict[str, Dict] = {}
    for g, df in score_dfs.items():
        if key_col not in df.columns:
            continue
        for _, r in df.iterrows():
            k = r[key_col]
            if pd.isna(k):
                continue
            entry = rows.setdefault(str(k), {"key": str(k)})
            mAP = float(r.get("mean_average_precision", float("nan")))
            sig = bool(r.get("below_corrected_p", False))
            distinguished = sig or (
                map_threshold is not None and np.isfinite(mAP) and mAP >= map_threshold
            )
            entry[f"{g}_mAP"] = mAP
            entry[f"{g}_sig"] = sig
            entry[f"{g}_distinguished"] = distinguished

    df_out = pd.DataFrame(list(rows.values()))
    if df_out.empty:
        return df_out
    groups = list(score_dfs.keys())
    flag_cols = [f"{g}_distinguished" for g in groups]
    for c in flag_cols:
        if c not in df_out.columns:
            df_out[c] = False

    def _label(row) -> str:
        hits = tuple(g for g in groups if bool(row.get(f"{g}_distinguished", False)))
        if not hits:
            return "none"
        if len(hits) == len(groups):
            return "all"
        if len(hits) == 1:
            return f"{hits[0]}_only"
        return "+".join(hits)

    df_out["modality_set"] = df_out.apply(_label, axis=1)
    return df_out


# ---------------------------------------------------------------------------
# Annotations: gene super-categories, CHAD names, CORUM names
# ---------------------------------------------------------------------------


def _load_supercategory_map(config_path: Path) -> Dict[str, str]:
    """Build gene -> 8-category super-category map (chad_boosted)."""
    if not config_path.is_file():
        logger.warning(f"Supercategory config missing: {config_path}")
        return {}
    try:
        import yaml

        from ops_utils.analysis.gene_supercategories import build_gene_supercategory_map

        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        return build_gene_supercategory_map(cfg, boosted=True)
    except Exception as exc:
        logger.warning(f"Supercategory map load failed: {exc}")
        return {}


def _supercat_palette(categories: Sequence[str]) -> Dict[str, str]:
    """Stable color per supercategory."""
    palette = [
        "#d97706", "#2563eb", "#10b981", "#7c3aed", "#dc2626",
        "#0891b2", "#a16207", "#db2777", "#65a30d", "#9333ea",
    ]
    return {c: palette[i % len(palette)] for i, c in enumerate(sorted(categories))}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_pies(
    table: pd.DataFrame,
    gene_to_supercat: Dict[str, str],
    out_dir: Path,
    *,
    title_prefix: str = "Gene super-categories",
) -> None:
    """One pie per non-empty modality_set: supercategory composition."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    sets_present = [s for s in table["modality_set"].unique() if s != "none"]

    rows = []
    for s in sets_present:
        members = table[table["modality_set"] == s]["key"].tolist()
        cats = pd.Series(
            [gene_to_supercat.get(g, "Other (unannotated)") for g in members]
        )
        counts = cats.value_counts()
        for cat, n in counts.items():
            rows.append({"modality_set": s, "supercategory": cat, "n_genes": int(n)})
    if not rows:
        logger.warning("No modality-exclusive genes to pie-chart.")
        return
    counts_df = pd.DataFrame(rows)
    csv_path = out_dir / "supercategory_counts.csv"
    counts_df.to_csv(csv_path, index=False)
    logger.info(f"  Wrote {csv_path}")

    all_cats = sorted(counts_df["supercategory"].unique())
    palette = _supercat_palette(all_cats)

    # Individual pies + a combined grid
    for s in sets_present:
        sub = counts_df[counts_df["modality_set"] == s].sort_values("n_genes", ascending=False)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(
            sub["n_genes"], labels=sub["supercategory"],
            colors=[palette[c] for c in sub["supercategory"]],
            autopct=lambda p: f"{p:.0f}%" if p >= 5 else "",
            startangle=90, wedgeprops=dict(linewidth=1, edgecolor="white"),
        )
        ax.set_title(f"{title_prefix} — {s} (n={int(sub['n_genes'].sum())})", fontsize=12)
        stem = out_dir / f"pie_{s.replace('+', '_and_')}"
        fig.savefig(stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)

    # Grid: 1 row × N cols, exclusive sets first then intersections
    priority = sorted(sets_present, key=lambda s: (
        0 if s.endswith("_only") else 1 if s == "all" else 2, s,
    ))
    n = len(priority)
    cols = min(n, 4)
    grid_rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(grid_rows, cols, figsize=(5.5 * cols, 5.5 * grid_rows))
    if grid_rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif grid_rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[a] for a in axes])
    for i, s in enumerate(priority):
        ax = axes[i // cols][i % cols]
        sub = counts_df[counts_df["modality_set"] == s].sort_values("n_genes", ascending=False)
        ax.pie(
            sub["n_genes"], labels=sub["supercategory"],
            colors=[palette[c] for c in sub["supercategory"]],
            autopct=lambda p: f"{p:.0f}%" if p >= 5 else "",
            startangle=90, wedgeprops=dict(linewidth=1, edgecolor="white"),
            textprops={"fontsize": 9},
        )
        ax.set_title(f"{s} (n={int(sub['n_genes'].sum())})", fontsize=11)
    # Hide any unused axes
    for j in range(n, grid_rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(title_prefix, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    stem = out_dir / "pie_grid"
    fig.savefig(stem.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _plot_cluster_heatmap(
    table: pd.DataFrame,
    groups: Sequence[str],
    out_dir: Path,
    *,
    cluster_label: str,        # 'CHAD' or 'CORUM'
    n_show: int = 60,
    name_map: Optional[Dict[str, str]] = None,  # cluster id (str) → human name
) -> None:
    """Heatmap of cluster mAP per modality. Sorted by max mAP across modalities;
    significant cells get a star marker. Saves <label>_heatmap.png/svg + .csv.

    ``name_map`` (when provided): maps the integer cluster id (as a string) to
    a human-readable name; used for the row labels so e.g. CHAD numeric IDs
    show up as "Endosomal trafficking" instead of "42".
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if table.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    map_cols = [f"{g}_mAP" for g in groups]
    sig_cols = [f"{g}_sig" for g in groups]
    for c in map_cols + sig_cols:
        if c not in table.columns:
            table[c] = float("nan") if c.endswith("mAP") else False

    df = table.copy()
    df["max_mAP"] = df[map_cols].max(axis=1)
    if name_map:
        df["name"] = df["key"].astype(str).map(lambda k: name_map.get(k, k))
    else:
        df["name"] = df["key"].astype(str)
    df = df.sort_values("max_mAP", ascending=False)
    csv_path = out_dir / f"{cluster_label.lower()}_heatmap.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"  Wrote {csv_path}")

    plot_df = df.head(n_show)
    M = plot_df[map_cols].fillna(0.0).to_numpy(dtype=float)
    S = plot_df[sig_cols].fillna(False).to_numpy(dtype=bool)

    fig, ax = plt.subplots(figsize=(1.0 + 1.4 * len(groups), 0.28 * len(plot_df) + 1.5))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.1, M.max()))
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["name"].tolist(), fontsize=8)
    ax.set_title(
        f"{cluster_label} mAP per modality — top {len(plot_df)}/{len(df)} "
        f"(★ = significant after correction)",
        fontsize=12,
    )
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            txt_color = "white" if M[i, j] < 0.6 * M.max() else "black"
            mark = "★" if S[i, j] else ""
            ax.text(j, i, mark, ha="center", va="center", color=txt_color, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="cluster mAP")
    fig.tight_layout()
    stem = out_dir / f"{cluster_label.lower()}_heatmap"
    fig.savefig(stem.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _plot_delta_rankings(
    table: pd.DataFrame,
    transitions: Sequence[Tuple[str, str]],
    out_dir: Path,
    *,
    level_label: str,           # "genes" or "CHAD"
    label_col: str = "key",     # column with display label
    name_map: Optional[Dict[str, str]] = None,  # optional id → human name
    n_top: int = 20,
) -> None:
    """Horizontal bar chart of top-N delta mAP per source→target transition.

    For each (source, target) in ``transitions``:
      - delta = mAP_target − mAP_source
      - Positive deltas = items enriched in target relative to source
      - Plot top-N by positive delta
    Saves one combined canvas (1 row × len(transitions) cols) plus one CSV with
    full sorted deltas per transition.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if table.empty or not transitions:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save full sorted deltas per transition into a single CSV (long format)
    rows = []
    per_panel: List[Tuple[str, str, pd.DataFrame]] = []
    for src, tgt in transitions:
        m_src, m_tgt = f"{src}_mAP", f"{tgt}_mAP"
        if m_src not in table.columns or m_tgt not in table.columns:
            continue
        sub = table[[label_col, m_src, m_tgt, f"{src}_sig", f"{tgt}_sig"]].copy()
        sub["delta"] = sub[m_tgt] - sub[m_src]
        sub = sub.dropna(subset=["delta"]).sort_values("delta", ascending=False)
        per_panel.append((src, tgt, sub))
        for _, r in sub.iterrows():
            rows.append({
                "source": src, "target": tgt,
                "key": r[label_col],
                f"mAP_{src}": float(r[m_src]),
                f"mAP_{tgt}": float(r[m_tgt]),
                "delta": float(r["delta"]),
                f"sig_{src}": bool(r[f"{src}_sig"]),
                f"sig_{tgt}": bool(r[f"{tgt}_sig"]),
            })
    if not per_panel:
        return
    csv_path = out_dir / f"delta_{level_label.lower()}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info(f"  Wrote {csv_path}")

    n_panels = len(per_panel)
    cols = min(n_panels, 3)
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(7 * cols, max(6, 0.32 * n_top + 1.5) * rows),
        squeeze=False,
    )
    flat_axes = axes.flatten()
    for j in range(n_panels, len(flat_axes)):
        flat_axes[j].axis("off")
    for ax, (src, tgt, sub) in zip(flat_axes, per_panel):
        top = sub.head(n_top).iloc[::-1]   # reverse for nicer top-down ordering
        if top.empty:
            ax.axis("off")
            continue
        labels = [
            (name_map.get(str(k), str(k)) if name_map else str(k))
            for k in top[label_col].tolist()
        ]
        deltas = top["delta"].to_numpy(dtype=float)
        # Color by which modality reaches significance
        colors = []
        for _, r in top.iterrows():
            if bool(r[f"{tgt}_sig"]) and not bool(r[f"{src}_sig"]):
                colors.append(GROUP_PALETTE.get(tgt, "#10b981"))
            elif bool(r[f"{tgt}_sig"]) and bool(r[f"{src}_sig"]):
                colors.append("#6b7280")
            else:
                colors.append("#cbd5e1")
        ax.barh(range(len(top)), deltas, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(f"Δ mAP ({tgt} − {src})", fontsize=11)
        ax.set_title(
            f"Top {min(n_top, len(top))} {level_label} enriched: {src} → {tgt}",
            fontsize=12, fontweight="bold",
        )
        ax.axvline(0, color="black", lw=0.8)
        ax.grid(True, axis="x", alpha=0.3)
        # Annotate sig stars
        for i, (_, r) in enumerate(top.iterrows()):
            if bool(r[f"{tgt}_sig"]):
                ax.text(
                    deltas[i] + (0.005 if deltas[i] >= 0 else -0.005), i, "★",
                    ha="left" if deltas[i] >= 0 else "right",
                    va="center", fontsize=9, color="black",
                )
    fig.suptitle(
        f"Δ mAP rankings — {level_label} (★ = significant in target modality)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    stem = out_dir / f"delta_{level_label.lower()}_top{n_top}"
    fig.savefig(stem.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Wrote {stem}.png/svg")


def _chad_cluster_names(chad_path: Optional[Path] = None) -> Dict[str, str]:
    """Build CHAD cluster_id (str) → human-readable name from the CHAD hierarchy."""
    try:
        from ops_utils.analysis.gene_supercategories import (
            _load_chad_hierarchy, DEFAULT_CHAD_PATH,
        )

        chad = _load_chad_hierarchy(chad_path or DEFAULT_CHAD_PATH)
        return {
            str(cid): cluster["name"]
            for cid, cluster in chad.items()
            if isinstance(cluster, dict) and "name" in cluster
        }
    except Exception as exc:
        logger.warning(f"CHAD name map load failed: {exc}")
        return {}


def _write_summary(
    summary_dir: Path,
    groups: Sequence[str],
    gene_table: pd.DataFrame,
    corum_table: pd.DataFrame,
    chad_table: pd.DataFrame,
) -> None:
    """summary.csv: counts and percentages per modality_set across all 3 levels."""
    rows = []
    for label, t in (("genes", gene_table), ("corum", corum_table), ("chad", chad_table)):
        if t.empty:
            continue
        total = len(t)
        for s, n in t["modality_set"].value_counts().items():
            rows.append({
                "level": label, "modality_set": s, "n": int(n),
                "total": int(total), "pct": float(100 * n / max(total, 1)),
            })
    if rows:
        df = pd.DataFrame(rows).sort_values(["level", "n"], ascending=[True, False])
        path = summary_dir / "summary.csv"
        df.to_csv(path, index=False)
        logger.info(f"Wrote {path}")
        for level in df["level"].unique():
            sub = df[df["level"] == level]
            print(f"\n[{level}] modality-set breakdown:")
            for _, r in sub.iterrows():
                print(f"  {r['modality_set']:>18}  {int(r['n']):>5} / {int(r['total'])}  ({r['pct']:.1f}%)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = _combined_parser()
    p.description = "Modality-distinctiveness analysis at a fixed cells/guide budget."
    # Override the inherited --groups default: we want the curated best-per-
    # organelle live-cell set, not the first-marker-listed-in-YAML set.
    p.set_defaults(groups=",".join(DEFAULT_GROUPS))
    p.add_argument(
        "--cells-per-guide", type=int, default=None,
        help="Cells/sgRNA budget at which to score every group. Default: smallest "
             "non-NTC sgRNA median across the requested groups.",
    )
    p.add_argument(
        "--mAP-threshold", dest="map_threshold", type=float, default=None,
        help="If set, also flag a key as distinguished when mAP ≥ this value "
             "(in addition to below_corrected_p significance).",
    )
    p.add_argument(
        "--supercategory-config", type=str,
        default=str(DEFAULT_SUPERCATEGORY_CONFIG),
        help="YAML for gene super-category mapping (chad_boosted mode).",
    )
    return p


def run_modality_analysis(args_dict: dict) -> str:
    """Top-level (picklable) worker — runs the *entire* modality-distinctiveness
    pipeline (score every group sequentially + cross-modality tables + plots)
    inside one SLURM job. ``args_dict`` is ``vars(parsed_args)``.
    """
    args = argparse.Namespace(**args_dict)
    args.slurm = False  # we're already inside the job; never re-submit

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        )

    groups = [g.strip() for g in args.groups.split(",") if g.strip()] or list(DEFAULT_GROUPS)
    custom_paths = (
        [p.strip() for p in args.custom_paths.split(",") if p.strip()]
        if args.custom_paths else None
    )

    # Resolve per-group cells.h5ad paths and (for matched_livecell_setN) the
    # organelle → marker membership.
    group_paths: Dict[str, List[Path]] = {}
    matched_set_membership: Dict[str, Dict[str, str]] = {}
    for g in groups:
        group_paths[g] = _resolve_group_paths(
            args, g, custom_paths=custom_paths, matching_config=args.matching_config,
        )
        print(f"[{g}] {len(group_paths[g])} reporters")
        m = _resolve_matched_set_membership(args, g, args.matching_config)
        if m:
            matched_set_membership[g] = m

    # Determine the shared cells/guide budget
    if args.cells_per_guide is not None:
        budget = int(args.cells_per_guide)
        print(f"\nUser-specified cells/guide budget: {budget}")
    else:
        medians = {g: _per_guide_median(group_paths[g]) for g in groups}
        budget = min(medians.values())
        print(f"\nMedians: {medians}  → using min median = {budget} cells/guide")

    # Output dirs
    root_dir = _resolve_modality_dir(args, groups, budget)
    print(f"Output: {root_dir}")
    per_group_dir = root_dir / "per_group"
    per_group_dir.mkdir(parents=True, exist_ok=True)

    # Drop the matched-livecell membership manifest up front so the user can
    # inspect which markers each setN picked even before scoring finishes.
    if matched_set_membership:
        rows = []
        for g, mp in matched_set_membership.items():
            for organelle, sig in mp.items():
                rows.append({"group": g, "organelle": organelle, "signal": sig})
        pd.DataFrame(rows).to_csv(root_dir / "matched_set_membership.csv", index=False)
        print("\nMatched-livecell set membership:")
        for g, mp in matched_set_membership.items():
            print(f"  {g}: " + ", ".join(f"{o}={s}" for o, s in mp.items()))

    # Score each group sequentially in this single job
    for g in groups:
        score_one_group(
            cells_h5ad_paths=[str(p) for p in group_paths[g]],
            output_dir=str(per_group_dir / g),
            cells_per_guide=budget,
            norm_method=args.norm_method,
            distance=args.distance,
            random_seed=int(args.seed),
            group_label=g,
            cache=bool(args.cache),
        )

    # Load per-group score CSVs
    dist_dfs: Dict[str, pd.DataFrame] = {}
    corum_dfs: Dict[str, pd.DataFrame] = {}
    chad_dfs: Dict[str, pd.DataFrame] = {}
    for g in groups:
        gd = per_group_dir / g
        try:
            dist_dfs[g] = pd.read_csv(gd / "distinctiveness.csv")
            corum_dfs[g] = pd.read_csv(gd / "corum.csv")
            chad_dfs[g] = pd.read_csv(gd / "chad.csv")
        except FileNotFoundError as exc:
            print(f"  WARNING: missing score CSV for {g}: {exc}")

    if not dist_dfs:
        raise SystemExit("No per-group score CSVs found — scoring step failed.")

    # ---------------------------- Per-gene set analysis ----------------------
    gene_table = _build_distinguished_table(
        dist_dfs, key_col="perturbation", map_threshold=args.map_threshold,
    )
    gene_table.to_csv(root_dir / "gene_modality_assignment.csv", index=False)

    # ------------------------------- CORUM --------------------------------
    corum_key = (
        "complex_id" if "complex_id" in next(iter(corum_dfs.values())).columns
        else "complex_num"
    )
    corum_table = _build_distinguished_table(
        corum_dfs, key_col=corum_key, map_threshold=args.map_threshold,
    )
    corum_table.to_csv(root_dir / "corum_modality_assignment.csv", index=False)

    # ------------------------------- CHAD --------------------------------
    chad_key = (
        "complex_num" if "complex_num" in next(iter(chad_dfs.values())).columns
        else "complex_id"
    )
    chad_table = _build_distinguished_table(
        chad_dfs, key_col=chad_key, map_threshold=args.map_threshold,
    )
    chad_table.to_csv(root_dir / "chad_modality_assignment.csv", index=False)

    # ------------------------------- Plots --------------------------------
    gene_to_super = _load_supercategory_map(Path(args.supercategory_config))
    chad_names = _chad_cluster_names()
    _plot_pies(gene_table, gene_to_super, root_dir / "pies", title_prefix="Gene super-categories")
    _plot_cluster_heatmap(
        chad_table, groups, root_dir / "chad", cluster_label="CHAD",
        name_map=chad_names,
    )
    _plot_cluster_heatmap(
        corum_table, groups, root_dir / "corum", cluster_label="CORUM",
        # CORUM complex_id is already a representative gene symbol from
        # phenotypic_consistency_corum (no numeric IDs to translate).
    )

    # ------------------- Δ mAP rankings (source → target) ------------------
    # Build every pairwise transition (both directions) across the requested
    # groups so cp ↔ 4i, cp ↔ livecell_setN, 4i ↔ livecell_setN, AND
    # livecell_setN ↔ livecell_setM all get their own panel. This way adding
    # matched_livecell_set1 / _set2 automatically gets cross-set comparisons.
    transitions: List[Tuple[str, str]] = []
    for i, src in enumerate(groups):
        for tgt in groups[i + 1:]:
            transitions.append((src, tgt))
            transitions.append((tgt, src))
    if transitions:
        _plot_delta_rankings(
            gene_table, transitions, root_dir / "delta_genes",
            level_label="genes", n_top=20,
        )
        _plot_delta_rankings(
            chad_table, transitions, root_dir / "delta_chad",
            level_label="CHAD", name_map=chad_names, n_top=20,
        )

    # Summary
    _write_summary(root_dir, groups, gene_table, corum_table, chad_table)
    print(f"\nDone. Outputs in {root_dir}")
    return f"SUCCESS: {root_dir}"


# ---------------------------------------------------------------------------
# CLI entry point: submit run_modality_analysis as one tracked SLURM job
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args()

    if not args.slurm:
        run_modality_analysis(vars(args))
        return

    # Submit the full analysis as a single SLURM job with progress tracking.
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

    slurm_params = {
        "mem": args.slurm_memory,
        "cpus_per_task": args.slurm_cpus,
        "slurm_partition": args.slurm_partition,
        "timeout_min": args.slurm_time,
    }
    worker_args = vars(args).copy()
    worker_args["slurm"] = False  # avoid recursion inside the job
    submit_parallel_jobs(
        jobs_to_submit=[{
            "name": "pca_modality_distinctiveness",
            "func": run_modality_analysis,
            "kwargs": {"args_dict": worker_args},
            "metadata": {
                "groups": args.groups,
                "cells_per_guide": args.cells_per_guide,
            },
        }],
        experiment="pca_modality_distinctiveness",
        slurm_params=slurm_params,
        log_dir="pca_optimization",
        manifest_prefix="modality_dist",
        wait_for_completion=True,
    )


if __name__ == "__main__":
    main()
