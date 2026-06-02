"""
Attention atlas with SHAP feature lollipops + per-channel captions.

Same multi-page landscape PDF as `attention_atlas.py` (top-attention KO/NTC
cells per gene × 4 channels), but each gene page also gets a vertical strip
of 4 SHAP panels on the right — one per channel (Phase + 3 fluor) — showing:

  • horizontal lollipop chart of the top-5 SHAP features:
        bar length = |shap_importance|
        sign / color = `direction` column (red ↑ if KO-up, blue ↓ if KO-down)
        feature name on y-axis, AUROC in panel title
  • the per-channel slice of `caption` parsed from the gene's caption row,
    wrapped to fit below the chart.

Implementation reuses everything in `attention_atlas.py` via hooks
(`fig_factory` / `post_render_hook` on `render_gene_page`,
`hook_factory_fqn` on the worker, `extra_per_gene_data` on `_build_tasks`)
so this file is purely the SHAP-specific layer.

CLI mirrors `attention_atlas.py`, plus `--shap-features-csv` and
`--shap-captions-csv`. Default SHAP CSVs are Alex's combined files in
`/hpc/projects/icd.fast.ops/models/alex_lin_attention/`.

Examples:

  # Local 5-gene smoke test:
  python ops_analysis/napari/attention_atlas_shap.py --local \\
    --max-genes 5 \\
    --output /hpc/mydata/gav.sturm/atlas_shap.pdf

  # Full atlas via SLURM against the top20 SHAP CSVs:
  uv run python ops_process/ops_analysis/napari/attention_atlas_shap.py \\
    --shap-features-csv /hpc/projects/icd.fast.ops/models/alex_lin_attention/top20/ko_shap_features.csv \\
    --shap-captions-csv /hpc/projects/icd.fast.ops/models/alex_lin_attention/top20/ko_shap_captions.csv \\
    --shap-cache-phase /hpc/projects/icd.fast.ops/models/alex_lin_attention/shap_caches/phase \\
    --shap-cache-fluor /hpc/projects/icd.fast.ops/models/alex_lin_attention/shap_caches/fluor \\
    --output /hpc/projects/icd.fast.ops/models/alex_lin_attention/top20/atlas_shap_top20.pdf \\
    --skip-cp-4i --no-strict

  # → ~5 min login-node prep (loads CSVs, builds per-gene violin/bg
  #   arrays from caches, picks NTC pool) followed by a 1000-task SLURM
  #   array (1 gene/task, 15-min timeout default, 16 GB / 2 CPUs / `cpu`
  #   partition) and a dependent merge job that stitches per-gene PDFs.
  # → Add `--time-minutes-per-task 20` for extra headroom on slow nodes.
  # → Add `--genes-per-task 5` to bundle into 200 array slots (less
  #   submitit overhead, longer per task).
  # → Logs land in `slurm_logs/attention_atlas/atlas_shap_top20/`.
"""

import os
import re
import sys
import textwrap
import zlib
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
import numpy as np
import pandas as pd

import attention_atlas as aa


DEFAULT_SHAP_FEATURES_CSV = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/top20_v4/ko_shap_features.csv"
)
DEFAULT_SHAP_CAPTIONS_CSV = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/top20_v4/ko_shap_captions.csv"
)
# Complex-grain (CHAD) variants — auto-swapped by main() when
# --aggregation-level complex AND user kept defaults.
_CHAD_SHAP_FEATURES_CSV = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/top20_v4_chad/ko_shap_features.csv"
)
_CHAD_SHAP_CAPTIONS_CSV = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/top20_v4_chad/ko_shap_captions.csv"
)

# NTC + median variant defaults — the same renderer drives all three
# atlases (distinctiveness / NTC / median) selected by --contrast. When
# --contrast in {ntc, global} AND the user kept the default
# --shap-features-csv / --shap-captions-csv, both auto-swap to the
# ntc_v2 paths (see main()).
NTC_BASE_DIR = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/ntc_v2"
)
NTC_DEFAULT_FEATURES_CSV = NTC_BASE_DIR / "ntc_shap_features.csv"
NTC_DEFAULT_CAPTIONS = {
    "ntc":    NTC_BASE_DIR / "ntc_shap_captions_ntc.csv",
    "global": NTC_BASE_DIR / "ntc_shap_captions_global.csv",
}
# CHAD-grain counterparts — populated by run_shap_pipeline.py
# --variant ntc --aggregation-level complex.
_NTC_BASE_DIR_CHAD = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/ntc_v2_chad"
)
_NTC_FEATURES_CHAD_CSV = _NTC_BASE_DIR_CHAD / "ntc_shap_features.csv"
_NTC_CAPTIONS_CHAD = {
    "ntc":    _NTC_BASE_DIR_CHAD / "ntc_shap_captions_ntc.csv",
    "global": _NTC_BASE_DIR_CHAD / "ntc_shap_captions_global.csv",
}
# Per-cell NTC SHAP caches — written by ntc_shap_features.py as
# {ntc_caches/v2/phase, ntc_caches/v2/<viz_channel>/, ...}. Phase is a
# single cache; fluor is a parent dir of per-channel subdirs (each with
# its own feature schema), loaded lazily by `_get_fluor_subpack`.
NTC_CACHE_ROOT = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/ntc_caches/v2"
)
NTC_DEFAULT_CACHE_PHASE = NTC_CACHE_ROOT / "phase"
NTC_DEFAULT_CACHE_FLUOR = NTC_CACHE_ROOT  # multi-channel parent dir
_NTC_CACHE_ROOT_CHAD = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/ntc_caches/v2_chad"
)
_NTC_CACHE_PHASE_CHAD = _NTC_CACHE_ROOT_CHAD / "phase"
_NTC_CACHE_FLUOR_CHAD = _NTC_CACHE_ROOT_CHAD

# Image-row cell sources for NTC variant (written by ntc_pick_cells.py).
# Each row is one cell exemplifying the top SHAP features for its
# (gene, viz_channel) — drop-in replacement for pma_top_*_v3.csv so the
# renderer's existing fluor_csv/phase_csv plumbing works unchanged.
NTC_DEFAULT_PHASE_CELLS_CSV = NTC_BASE_DIR / "ntc_picked_phase.csv"
NTC_DEFAULT_FLUOR_CELLS_CSV = NTC_BASE_DIR / "ntc_picked_fluor.csv"
_NTC_PHASE_CELLS_CHAD = NTC_BASE_DIR / "ntc_picked_phase_chad.csv"
_NTC_FLUOR_CELLS_CHAD = NTC_BASE_DIR / "ntc_picked_fluor_chad.csv"

# Panel-title framing per --contrast. The renderer embeds these into
# `shap_data["title_prefix"]` per gene; falls back to "{gene} geneKO"
# when --contrast distinct (the default).
CONTRAST_FRAMING = {
    "distinct": None,             # → "{gene} geneKO" (legacy)
    "ntc":      "KO vs NTC",
    "global":   "KO vs cohort",
}

# Per-(gene, marker) mAP distinctiveness matrix from the cell_dino
# pca_optimization run. Drives the "top 3 fluor channels by mAP"
# selection on each atlas page. None disables → fall back to
# whatever order the SHAP CSV's viz_channels string supplies (legacy).
# Both default paths live under the FIRST-PCA aggregate output dir
# (.../cosine/plots/marker_overlay/) which is where
# pca_optimization._save_per_reporter_metric_matrices writes them.
# NOT under second_pca_consensus/ — that subdir holds a different,
# legacy distinctiveness matrix produced by
# gene_best_marker_assignment.py.
_PCA_AGG_OVERLAY = (
    Path("/home/gav.sturm/linked_folders/icd.fast.ops/organelle_attribution/")
    / "pca_optimized_v0.3" / "cell_dino" / "zscore_per_exp" / "paper_v1"
    / "with_cp" / "with_4i" / "all_livecell" / "fixed_80%" / "cosine"
    / "plots" / "marker_overlay"
)
DEFAULT_MARKER_MAP_CSV = (
    # GENE-LEVEL: per-gene × per-marker mAP DISTINCTIVENESS.
    # The with_cp/with_4i variant carries all 56 fluor markers +
    # Phase (live-cell + Cell Painting + 4i markers).
    _PCA_AGG_OVERLAY / "gene_reporter_distinctiveness_raw.csv"
)
DEFAULT_CHAD_CONSISTENCY_CSV = (
    # COMPLEX-LEVEL: per-CHAD-complex × per-marker mAP CONSISTENCY.
    # Atlas REQUIRES this file at --aggregation-level complex —
    # distinctiveness is never used as a complex-level fallback
    # per user spec. Rows are keyed by complex_num (integer YAML
    # keys); we map to complex names via the v5 YAML at load time.
    _PCA_AGG_OVERLAY / "complex_reporter_chad_consistency.csv"
)
# CHAD positive-controls config used to roll the gene-level mAP matrix
# up to complex level via mean-across-member-genes (mirrors
# gene_best_marker_assignment.plot_chad_complex_marker_heatmap).
# v5_hierarchy is the canonical default across the atlas pipeline:
# matches Alex's pma_top_*_chad_v1.csv predicted_class names 90/90 and
# is used by consolidate, ntc_shap_features, gene_best_marker_assignment,
# and fe_graphs.
DEFAULT_CHAD_COMPLEX_CONFIG = Path(
    "/hpc/projects/icd.fast.ops/configs/gene_clusters/"
    "chad_positive_controls_v5_hierarchy.yml"
)
# Default # of fluor channels to render per page (Phase always shown
# in addition).
DEFAULT_TOP_FLUOR_CHANNELS = 3

# Marker-name aliases mirrored from gene_best_marker_assignment._norm_channel
# so the mAP CSV's "ER, NCLN" / "Endoplasmic Reticulum, ConA" join cleanly
# against the SHAP CSV's "ER_NCLN" / "Endoplasmic_Reticulum_ConA".
_MARKER_NORM_ALIASES = (
    ("wheat germ agglutinin", "wga"),
    ("concanavalin a", "cona"),
    ("endoplasmic reticulum", "er"),
    ("nucleus", "nuclei"),
)


def _norm_channel_key(name) -> str:
    """Normalize a viz_channel / marker name into a stable lowercase
    key. Mirrors `gene_best_marker_assignment._norm_channel`.

    Bridges two upstream naming conventions for the same biological
    channel: the mAP matrix uses `p21 (4i)` / `ER, ConA (cp)` /
    `Mitochondria_TOMM20 (cp)` (modality suffix tags), while the SHAP
    CSV (sourced from Alex's pma_top CSVs) uses
    `p21_p21 (rabbit-647)` / `Endoplasmic Reticulum_Concanavalin A` /
    `Mitochondria_TOMM20` (probe-name antibody tags, no modality
    suffix). Without this bridge the bar selector silently misses the
    CP/4i channels — they appear high on the bar but get no SHAP
    rendering, leaving blank rows + lollipop fallbacks where violins
    should be.
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    n = str(name).replace(", ", "_").replace(",", "_").strip().lower()
    # Strip every trailing parenthetical: matrix " (cp)" / " (4i)" AND
    # SHAP " (rabbit-647)" / " (mouse-488)" antibody descriptors. Strip
    # in a loop so multiple consecutive parens both go.
    while n.endswith(")"):
        i = n.rfind(" (")
        if i == -1:
            break
        n = n[:i].rstrip()
    # Collapse duplicate-token format: `p21_p21` → `p21`,
    # `b-catenin_b-catenin` → `b-catenin`, etc. SHAP CSV's "p21_p21"
    # mirrors the matrix's plain "p21" once both have lost their
    # parentheticals.
    if "_" in n:
        head, tail = n.split("_", 1)
        if head and head == tail:
            n = head
    for long_form, short_form in _MARKER_NORM_ALIASES:
        n = n.replace(long_form, short_form)
    return n


def _load_marker_map(marker_map_csv):
    """Load the genes × markers mAP matrix; return DataFrame indexed by
    gene with normalized column keys. Returns None on missing/empty
    file (caller falls back to legacy first-N channel order).
    """
    if marker_map_csv is None:
        return None
    p = Path(marker_map_csv)
    if not p.exists():
        print(f"  [marker-map] NOT FOUND: {p} — top-3 selection disabled, "
              f"falling back to viz_channels CSV order", flush=True)
        return None
    df = pd.read_csv(p, index_col=0)
    # The with_cp/with_4i raw matrix may carry an `all_combined`
    # summary column that's not a real marker. Pull it OUT of the
    # per-marker matrix (so it doesn't appear in the bar or top-3
    # selection); we'll prefer the authoritative metrics/ scalar below.
    inline_all_combined = {}
    if "all_combined" in df.columns:
        inline_all_combined = {
            str(g): (float(v) if v == v else None)
            for g, v in df["all_combined"].items()
        }
        df = df.drop(columns=["all_combined"])

    # Track display name (pre-normalization) keyed by normalized name so
    # the heatmap bar can show the original "ER, NCLN" label.
    display_by_norm = {_norm_channel_key(c): str(c) for c in df.columns}
    df.attrs["display_by_norm"] = display_by_norm
    df.columns = [_norm_channel_key(c) for c in df.columns]
    df.index = df.index.astype(str)

    # Per-gene "all-markers combined" mAP DISTINCTIVENESS — three
    # tiers, in order of preference:
    #   1. metrics/phenotypic_distinctiveness.csv from the sibling
    #      `metrics/` dir (authoritative scalar from the FULL
    #      aggregated PCA feature space — what we want for the badge).
    #   2. Inline `all_combined` column on the matrix (legacy
    #      gene_best_marker_assignment output).
    #   3. Mean across markers per gene (last-resort proxy).
    metrics_csv = (
        Path(p).resolve().parent.parent.parent
        / "metrics" / "phenotypic_distinctiveness.csv"
    )
    metrics_all_combined = {}
    if metrics_csv.is_file():
        try:
            mdf = pd.read_csv(metrics_csv)
            if {"perturbation", "mean_average_precision"}.issubset(mdf.columns):
                metrics_all_combined = {
                    str(g): float(v)
                    for g, v in zip(mdf["perturbation"], mdf["mean_average_precision"])
                }
                print(f"  [marker-map] all_combined source: "
                      f"{metrics_csv.name} ({len(metrics_all_combined)} genes)",
                      flush=True)
        except Exception as e:
            print(f"  [marker-map] could not read {metrics_csv}: {e}",
                  flush=True)

    if metrics_all_combined:
        df.attrs["all_combined"] = metrics_all_combined
    elif inline_all_combined:
        df.attrs["all_combined"] = inline_all_combined
    else:
        df.attrs["all_combined"] = {
            g: float(df.loc[g].dropna().mean()) if df.loc[g].notna().any() else None
            for g in df.index
        }
        print(f"  [marker-map] all_combined source: mean across markers "
              f"(metrics CSV not found at {metrics_csv})", flush=True)

    print(f"  [marker-map] loaded {df.shape[0]} genes × {df.shape[1]} markers "
          f"from {p.name}", flush=True)
    return df


DEFAULT_CHAD_DESCRIPTIONS_YAML = (
    Path(__file__).resolve().parent / "chad_complex_descriptions.yml"
)


def _load_chad_descriptions(yaml_path=DEFAULT_CHAD_DESCRIPTIONS_YAML):
    """Load {complex_name: description} from a YAML file. Used to
    surface a one-line functional description on each complex atlas
    page (combined with the member-gene list).
    """
    import yaml
    p = Path(yaml_path) if yaml_path else None
    if p is None or not p.is_file():
        return {}
    try:
        with open(p) as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"  [chad-desc] could not load {p}: {e}", flush=True)
        return {}
    return {str(k): str(v) for k, v in data.items() if v is not None}


def _load_chad_complex_members(chad_config):
    """Load complex_name -> [member_genes] from a CHAD positive-controls
    YAML. Used to surface "Members (N): GENE1, GENE2, ..." on each
    complex atlas page even if the base atlas's `_load_chad_annotations`
    failed (different code path / different YAML loader).
    """
    import yaml
    if chad_config is None:
        return {}
    p = Path(chad_config)
    if not p.is_file():
        return {}
    try:
        with open(p) as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"  [chad-members] could not load {p}: {e}", flush=True)
        return {}
    out = {}
    for entry in data.values():
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        genes = list(entry.get("genes", []) or [])
        if name and genes and name.upper() != "NTCS":
            out[name] = genes
    return out


def _load_chad_consistency_matrix(consistency_csv, chad_config):
    """Load the per-(CHAD complex × marker) mAP CONSISTENCY matrix.

    The CSV (saved by pca_optimization._save_per_reporter_metric_matrices)
    is keyed by `complex_num` (integer YAML keys). This helper:
      1. Reads the CSV.
      2. Maps complex_num → complex name via the CHAD YAML at
         `chad_config` (so downstream lookups can use Alex's
         predicted_class strings like "subu Proteasome 19s").
      3. Drops Phase from the columns if present (renamed to keep
         the same display semantics as the gene-level distinctiveness
         matrix's "Phase" column).
      4. Returns the DataFrame indexed by complex NAME, plus a
         {marker_norm_key: pretty_marker_name} display map.

    Returns (matrix, display_by_norm).
    Raises SystemExit if the file is missing — atlas refuses to fall
    back to distinctiveness at the complex level (per user spec:
    "distinctiveness is never used as the mAP score to select markers
    for the pathway level atlas").
    """
    import yaml
    p = Path(consistency_csv) if consistency_csv else None
    if p is None or not p.is_file():
        raise SystemExit(
            f"[chad-consistency] required matrix not found: {p}\n"
            f"  CHAD complex pages need per-(complex × marker) mAP consistency.\n"
            f"  Generate it by re-running pca_optimization with the new\n"
            f"  --aggregate-only step:\n"
            f"    python -m ops_model.post_process.combination.pca_optimization \\\n"
            f"        --output-dir /hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3 \\\n"
            f"        --cell-dino --zscore-per-experiment \\\n"
            f"        --paper-v1 /hpc/projects/icd.fast.ops/configs/good_experiment_list_v1.yml \\\n"
            f"        --with-cp --with-4i --aggregate-only --slurm\n"
            f"  Or pass --marker-map-csv NONE to disable mAP-driven channel\n"
            f"  selection entirely (legacy first-N order)."
        )
    df = pd.read_csv(p, index_col=0)
    # Build complex_num → name map from the v5 YAML.
    cf = Path(chad_config) if chad_config else None
    if cf is None or not cf.is_file():
        raise SystemExit(
            f"[chad-consistency] CHAD config not found: {cf} — needed to map "
            f"the consistency CSV's complex_num keys to complex names."
        )
    with open(cf) as f:
        data = yaml.safe_load(f) or {}
    num_to_name = {}
    for k, v in data.items():
        if isinstance(v, dict) and v.get("name"):
            num_to_name[int(k)] = str(v["name"]).strip()
    # The CSV's index is complex_num as int. Reindex by name.
    df.index = df.index.astype(int)
    rename = {}
    for cn in df.index:
        nm = num_to_name.get(int(cn))
        if nm is None or nm.upper() == "NTCS":
            continue
        rename[cn] = nm
    df = df.loc[list(rename.keys())].copy()
    df.index = [rename[cn] for cn in df.index]
    df.index.name = "complex"

    # Drop the `all_combined` column if present (saved alongside per-
    # marker columns); we'll prefer the metrics/ scalar below.
    all_combined_col = None
    if "all_combined" in df.columns:
        all_combined_col = df["all_combined"].astype(float).to_dict()
        df = df.drop(columns=["all_combined"])

    # Display map keyed on normalized marker name → pretty name.
    display_by_norm = {_norm_channel_key(c): str(c) for c in df.columns}
    df.columns = [_norm_channel_key(c) for c in df.columns]
    df.attrs["display_by_norm"] = display_by_norm

    # Per-complex "all-markers combined" CHAD consistency mAP — three
    # tiers, in order of preference:
    #   1. The authoritative `phenotypic_consistency_manual.csv` from
    #      the sibling `metrics/` dir (computed on the FULL aggregated
    #      PCA feature space — what we actually want for the badge).
    #   2. An inline `all_combined` column if pca_optimization happens
    #      to have saved one alongside the per-marker matrix.
    #   3. Mean across markers per complex (least preferred — proxy
    #      only).
    metrics_csv = (
        Path(consistency_csv).resolve().parent.parent.parent
        / "metrics" / "phenotypic_consistency_manual.csv"
    )
    metrics_all_combined = {}
    if metrics_csv.is_file():
        try:
            metrics_df = pd.read_csv(metrics_csv)
            if {"complex_num", "mean_average_precision"}.issubset(metrics_df.columns):
                metrics_all_combined = {
                    rename[int(cn)]: float(v)
                    for cn, v in zip(
                        metrics_df["complex_num"], metrics_df["mean_average_precision"]
                    )
                    if int(cn) in rename
                }
                print(f"  [chad-consistency] all_combined source: "
                      f"{metrics_csv.name} ({len(metrics_all_combined)} complexes)",
                      flush=True)
        except Exception as e:
            print(f"  [chad-consistency] could not read {metrics_csv}: {e}",
                  flush=True)

    if metrics_all_combined:
        df.attrs["all_combined"] = metrics_all_combined
    elif all_combined_col:
        df.attrs["all_combined"] = all_combined_col
    else:
        # Mean across markers per complex (proxy fallback).
        df.attrs["all_combined"] = {
            c: float(df.loc[c].dropna().mean()) if df.loc[c].notna().any() else None
            for c in df.index
        }
        print(f"  [chad-consistency] all_combined source: mean across markers "
              f"(metrics CSV not found at {metrics_csv})", flush=True)
    print(f"  [chad-consistency] loaded {df.shape[0]} complexes × "
          f"{df.shape[1]} markers from {p.name}", flush=True)
    return df


def _build_chad_complex_map_matrix(raw_df, chad_config, min_genes=1):
    """Aggregate gene-level mAP matrix to CHAD-complex level (mean
    across member genes per marker). Mirrors
    `gene_best_marker_assignment.plot_chad_complex_marker_heatmap` and
    its `_aggregate_groups_to_marker_matrix` helper.

    Returns DataFrame (complexes × markers). Empty if config missing.
    """
    import yaml
    if raw_df is None:
        return None
    p = Path(chad_config) if chad_config else None
    if p is None or not p.is_file():
        print(f"  [marker-map] CHAD config not found: {p} — top-3 disabled "
              f"at complex level", flush=True)
        return None
    with open(p) as f:
        data = yaml.safe_load(f) or {}
    rows, names = [], []
    for entry in data.values():
        if not isinstance(entry, dict):
            continue
        cname = str(entry.get("name", "")).strip()
        cgenes = entry.get("genes", []) or []
        if not cname or not cgenes or cname.upper() == "NTCS":
            continue
        present = [g for g in cgenes if g in raw_df.index]
        if len(present) < min_genes:
            continue
        rows.append(raw_df.loc[present].mean(axis=0).values)
        names.append(cname)
    if not rows:
        return None
    out = pd.DataFrame(rows, index=names, columns=raw_df.columns)
    print(f"  [marker-map] CHAD complex-level: {out.shape[0]} complexes "
          f"× {out.shape[1]} markers (from {raw_df.shape[0]} genes)", flush=True)
    return out


def select_top_channels(gene, available_channels, map_df, top_k=DEFAULT_TOP_FLUOR_CHANNELS):
    """Pick top-K fluor channels for `gene` by mAP, plus Phase always first.

    Args:
        gene: gene symbol or CHAD complex name (matched against map_df.index).
        available_channels: list of viz_channel names that have SHAP rows
            for this gene (the candidate pool — we never select a channel
            with no SHAP data).
        map_df: genes × markers mAP DataFrame, columns already normalized
            via `_norm_channel_key`. None → fallback (first top_k fluor in
            input order).
        top_k: how many fluor channels to keep.

    Returns:
        (channels_list, map_per_channel) where:
          channels_list = ["Phase", top1_fluor, top2_fluor, top3_fluor]
            in mAP-descending order. May be shorter if too few fluor
            channels in `available_channels` have mAP data.
          map_per_channel = {channel_name: mAP_score | None} — surfaced
            in atlas panel titles.
    """
    available_channels = list(available_channels)
    fluor_avail = [c for c in available_channels if c.lower() != "phase"]
    has_phase = any(c.lower() == "phase" for c in available_channels)

    if map_df is None or gene not in map_df.index:
        head = (["Phase"] if has_phase else []) + fluor_avail[:top_k]
        return head, {c: None for c in available_channels}

    gene_row = map_df.loc[gene]
    map_per_channel = {}
    for ch in available_channels:
        v = gene_row.get(_norm_channel_key(ch))
        map_per_channel[ch] = float(v) if v is not None and not pd.isna(v) else None

    fluor_with_map = [
        (ch, map_per_channel[ch]) for ch in fluor_avail
        if map_per_channel.get(ch) is not None
    ]
    fluor_with_map.sort(key=lambda x: x[1], reverse=True)
    top_fluors = [ch for ch, _ in fluor_with_map[:top_k]]
    # Pad with mAP-less fluor channels if fewer than top_k had mAP scores
    # — preserves visual layout (3 fluor panels) when mAP data is sparse.
    if len(top_fluors) < top_k:
        leftover = [c for c in fluor_avail if c not in top_fluors]
        top_fluors += leftover[: top_k - len(top_fluors)]

    return (["Phase"] if has_phase else []) + top_fluors, map_per_channel
# Optional SHAP caches written by ko_shap_features.py — when provided, each
# feature's z-scored value distribution across the gene's top-attention KO
# cells is embedded in the task and rendered as a horizontal violin instead
# of a lollipop bar. Live alongside the SHAP CSVs under alex_lin_attention/
# so all SHAP artifacts are co-located.
DEFAULT_CACHE_PHASE = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/shap_caches/v4/phase"
)
DEFAULT_CACHE_FLUOR = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/shap_caches/v4/fluor"
)
# Complex-grain (CHAD) variants — auto-swapped by
# `apply_aggregation_level_defaults` when --aggregation-level complex
# AND the user kept the gene-level defaults. Without this swap, atlas
# loads the stale gene-level caches and per-channel violin lookups
# silently miss for CHAD pages → random fallback to lollipops.
_CHAD_CACHE_PHASE = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/shap_caches/v4_chad/phase"
)
_CHAD_CACHE_FLUOR = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/shap_caches/v4_chad/fluor"
)

# Number of top features to show per channel in the lollipop chart.
N_TOP_FEATURES = 5
# Extra SHAP-rank candidates loaded beyond N_TOP_FEATURES so that any of the
# top features with no violin spread (flat distribution → can't be plotted
# meaningfully) can be replaced by the next-highest-ranked feature that
# does have spread. 5 extras = up to 10 candidates per channel.
N_EXTRA_FEATURES = 5
# Minimum IQR (in z-units, since violin values are z-scored) required for a
# feature's combined ko+bg distribution to be considered "with spread". Below
# this, both violins collapse to flat lines and the row carries no signal.
MIN_VIOLIN_IQR = 0.1
# Color by direction: KO-up = blue, KO-down = red. (Counterintuitive vs the
# usual "red = high" volcano-plot mapping, but it's what the user asked for —
# red marks down-features so they pop out as decreases.)
COLOR_UP = "#1F46A6"   # blue
COLOR_DOWN = "#C81E1E" # red
# Total figure: 24.5" wide × 18" tall. Image grid 19" wide; SHAP column
# ~3.8" wide. Caption tiers + wrap widths (below) are calibrated so even
# the longest top20 captions wrap inside this narrower SHAP column.
FIG_WIDTH_IN = 24.5
FIG_HEIGHT_IN = 18.0
# In figure-coord units: where the image grid ends and the SHAP panels begin.
IMG_GRID_RIGHT = 19.0 / FIG_WIDTH_IN          # 19/24.5 ≈ 0.776
# 1.7" gap — moderate whitespace for the lollipop y-tick labels.
SHAP_GRID_LEFT = (19.0 + 1.7) / FIG_WIDTH_IN  # 20.7/24.5 ≈ 0.845
# Right margin: pulled in from the canvas edge so the SHAP charts /
# captions don't get clipped at the page boundary by PDF viewers /
# printers that crop a few px off the edges. 0.040 fig-frac ≈ 1.0".
SHAP_GRID_RIGHT = 1.0 - 0.040
# Right gutter removed: chart fills the full panel width (no extra
# whitespace between the chart and the canvas right edge).
SHAP_WIDTH_RATIOS = None
# Each SHAP panel is caption / chart / bottom gutter. Bumped caption
# allocation 0.9 → 1.5 (now ~25% of panel ≈ 1.07" tall) so it actually
# holds the title (1-2 lines) + caption (up to ~5 lines @ 8.5pt) without
# overflowing into the chart. Chart shrinks 4.5 → 4.0; gutter 0.3 → 0.2.
# Total still sums to ~5.7; chart goes from 79% → 67% — still 1.3× taller
# than the original lollipop layout, plenty of room for the side-by-side
# violins.
SHAP_HEIGHT_RATIOS = (1.5, 4.0, 0.2)   # caption, chart, bottom gutter
# Wrap feature names at this character count per line (preserves underscore
# segments — never mid-word) — keeps names readable without abbreviation.
# Bumped from 17 → 22 so most features land on 1–2 lines instead of 3,
# reducing each lollipop row's vertical height.
FEATURE_WRAP_CHARS = 22

# Caption auto-shrink tiers — n_chars compared = len(per-panel caption)
# + len(title). Calibrated to the observed per-panel distribution from
# the top-20 captions CSV (median ~165 + ~40 title = ~205, p95 ~273,
# max ~325, plus a buffer for future heavier captions).
# Each tier is (max_total_chars, caption_pt, title_pt, caption_wrap_chars).
# At smaller fonts we also widen the wrap — each glyph is narrower so
# the same ~6.2" SHAP column fits more chars per line. The last tier
# is the floor; ~7.5pt stays readable at PDF DPI.
# Calibrated against actual top20 per-section length distribution
# (median 127, p95 245, p99 284, max 327) AND the ~3.8"-wide SHAP panel:
#   - ≤180  → tier 1 (10.5pt italic) — wrap 44 chars (~3.7" max line)
#   - ≤240  → tier 2 ( 9.5pt italic) — wrap 49 chars
#   - ≤290  → tier 3 ( 8.5pt italic) — wrap 55 chars
#   - >290  → tier 4 ( 7.5pt italic) — wrap 62 chars
# Each tier is (max_total_chars, caption_pt, title_pt, caption_wrap_chars).
# Verified programmatically that max single-line render width < 3.8" at
# every tier (matplotlib bbox measurement, italic glyph widths).
# (max_chars, caption_pt, title_pt, wrap_w).
# Each tier shrinks font AND widens wrap so total line count grows slowly
# with caption length. Tail tiers added (was a floor at 7.5pt/62) because
# long auto-generated captions were overflowing into the violin row below.
# 5.8pt floor stays readable in print and survives wider wraps at the
# bottom (92 chars/line ≈ panel width at that point size).
CAPTION_FONT_TIERS = (
    (180, 10.5, 11.0, 44),
    (240,  9.5, 10.0, 49),
    (290,  8.5,  9.5, 55),
    (360,  7.5,  9.0, 62),
    (440,  6.8,  8.5, 72),
    (520,  6.2,  8.0, 82),
    (float("inf"), 5.8, 7.5, 92),
)

def _wrap_feature_name(name, max_chars=FEATURE_WRAP_CHARS):
    """Wrap a long ops/cellprofiler feature name onto multiple lines.

    Splits at underscore boundaries so segments stay intact (never mid-word),
    then greedily packs segments up to `max_chars` per line. Keeps the full
    name readable without inventing cryptic abbreviations.
    """
    if name is None:
        return ""
    n = str(name)
    # Strip 'chN/' channel prefix used in fluor SHAP rows.
    if "/" in n:
        n = n.split("/", 1)[1]
    if n.startswith("op_"):
        n = n[3:]
    parts = n.split("_")
    lines = []
    cur = ""
    for p in parts:
        if not cur:
            cur = p
        elif len(cur) + 1 + len(p) <= max_chars:
            cur = cur + "_" + p
        else:
            lines.append(cur)
            cur = p
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def _channel_order_from_viz(viz_channels_str):
    """Parse `viz_channels` cell ('chA | chB | chC | Phase') into a 4-tuple
    [Phase, ch1, ch2, ch3] in the order the atlas expects (phase first,
    then the 3 fluor channels in Alex's CSV order)."""
    if not isinstance(viz_channels_str, str):
        return ["Phase"]
    parts = [p.strip() for p in viz_channels_str.split("|")]
    fluors = [p for p in parts if p and p.lower() != "phase"]
    return ["Phase"] + fluors


def _parse_caption_per_channel(caption, channels):
    """Split a gene's caption into per-channel descriptions.

    Caption format (one line per gene, post `_make_caption`):
      "{gene} geneKO: Phase — <text>; <ch1> — <text>; <ch2> — <text>."

    Channel names contain spaces, slashes, underscores, and dashes. The
    description text itself can contain "; " as a phrase separator — so
    we can't naively split the body on "; ".

    Strategy: detect EVERY channel boundary in the caption (not just the
    `channels` we want to render) by splitting where "; " is followed by
    a token that looks like "{channel-name} — ". That gives every
    section's text bounded correctly. Then we filter to the requested
    `channels`.

    This is critical for CHAD complex pages where the caption may carry
    sections for all ~56 channels but the page only renders the
    top-3-by-mAP. The previous "look up only requested channels"
    parser would let each requested channel's text run THROUGH all
    intervening non-requested sections — captions panels showed every
    channel's description glued together rather than the per-row slice.
    """
    out = {ch: "" for ch in channels}
    if not isinstance(caption, str) or not caption:
        return out

    # Strip the leading "{gene} {label}: " — the colon-space following
    # the header. (Prior captions had "(... auroc=...): " parens; the
    # current `_make_caption` drops those, so a plain ": " match is fine.)
    m = re.search(r":\s+", caption)
    body = caption[m.end():] if m else caption
    body = body.rstrip(".").strip()

    # Split body on "; " ONLY when the next chunk starts with a
    # "{name} — " channel header (i.e., looks like the start of a new
    # channel section). The lookahead `[^;—]+? — ` matches a channel
    # name (no ";" or em-dash inside) followed by " — ", which can't
    # occur inside a description's phrase list (those use "; " between
    # phrases but never reintroduce a " — " channel marker).
    sections = re.split(r";\s+(?=[^;—]+? — )", body)
    requested = set(channels)
    for s in sections:
        if " — " not in s:
            continue
        ch_name, _, text = s.partition(" — ")
        ch_name = ch_name.strip()
        text = text.strip().rstrip(";").strip()
        if ch_name in requested:
            # If a channel somehow appears multiple times (shouldn't),
            # join with "; " so we don't drop the second slice.
            out[ch_name] = (
                out[ch_name] + "; " + text if out[ch_name] else text
            )
    return out


def _build_gene_idx(obs: pd.DataFrame) -> dict[str, np.ndarray]:
    """One-pass gene → row-index map via groupby.

    The naive `{g: np.where(col == g)[0] for g in col.unique()}` is O(N · G)
    — fine at 100k rows but ~30 min at the NTC phase cache's 8.4M rows.
    `groupby(...).indices` is one O(N) pass.
    """
    if "gene" not in obs.columns:
        return {}
    grouped = obs.groupby("gene", observed=True, sort=False).indices
    return {str(k): np.asarray(v, dtype=np.int64) for k, v in grouped.items()}


def _load_single_cache(cdir: Path) -> dict:
    """Load one {X.npy, obs.parquet, features.txt, median.npy, global_std.npy}
    cache as the standard pack dict.

    CHAD-aware: if a sibling `obs_chad.parquet` exists, it is preferred as
    `obs` — it carries CHAD complex names in the `gene` column (replacing
    raw gene-KO labels). Its `_x_idx` column maps each row back to the
    canonical `X.npy`. The pack stores it as `pack["x_idx"]` and callers
    must translate row indices through it before slicing `pack["X"]`
    (see `_pack_x_indices` below). `X.npy` itself is never duplicated.
    """
    cdir = Path(cdir)
    X = np.load(cdir / "X.npy", mmap_mode="r")
    obs_chad_path = cdir / "obs_chad.parquet"
    if obs_chad_path.exists():
        obs = pd.read_parquet(obs_chad_path)
        if "_x_idx" not in obs.columns:
            raise SystemExit(
                f"{obs_chad_path} missing required `_x_idx` column "
                f"(written by ntc_shap_features.py)."
            )
        x_idx = obs["_x_idx"].to_numpy(dtype=np.int64)
        obs = obs.drop(columns=["_x_idx"]).reset_index(drop=True)
    else:
        obs = pd.read_parquet(cdir / "obs.parquet")
        x_idx = None
    fnames = (cdir / "features.txt").read_text().splitlines()
    fi = {f: i for i, f in enumerate(fnames)}
    median = np.load(cdir / "median.npy")
    std = np.load(cdir / "global_std.npy").clip(1e-6)
    pack = {"X": X, "obs": obs, "fi": fi, "median": median, "std": std}
    if x_idx is not None:
        pack["x_idx"] = x_idx
    return pack


def _pack_x_indices(pack: dict, row_idx) -> np.ndarray:
    """Translate pack-relative row indices to canonical X.npy row indices.

    No-op for regular packs (gene-level caches). For CHAD packs the obs
    has been filtered/relabeled to complexes, so position `i` in obs maps
    to `pack["x_idx"][i]` in the on-disk `X.npy`.
    """
    x_idx = pack.get("x_idx")
    if x_idx is None:
        return np.asarray(row_idx)
    return np.asarray(x_idx[row_idx])


def _is_multi_channel_dir(cdir: Path) -> bool:
    """Detect the NTC per-channel cache layout — a parent dir whose children
    are per-channel subdirs each holding {X.npy, ...}, rather than X.npy
    directly. Used to pick the right loader."""
    cdir = Path(cdir)
    if (cdir / "X.npy").exists():
        return False
    if not cdir.is_dir():
        return False
    for child in cdir.iterdir():
        if child.is_dir() and (child / "X.npy").exists():
            return True
    return False


def _build_multi_fluor_pack(cache_root: Path) -> dict:
    """Index per-channel fluor sub-caches under `cache_root` without loading
    their X.npy yet. Each subdir's obs.parquet is read for viz_channel only
    (1 cell value) so we can map viz_channel → subdir without paying the
    full obs cost up front.

    Sub-packs are loaded lazily by `_get_fluor_subpack` on first hit.
    """
    cache_root = Path(cache_root)
    subdir_by_viz: dict[str, Path] = {}
    skipped: list[str] = []
    for child in sorted(cache_root.iterdir()):
        if not child.is_dir() or not (child / "X.npy").exists():
            continue
        obs_path = child / "obs.parquet"
        if not obs_path.exists():
            skipped.append(child.name)
            continue
        try:
            obs_vc = pd.read_parquet(obs_path, columns=["viz_channel"])
            if not len(obs_vc):
                skipped.append(child.name)
                continue
            viz_channel = str(obs_vc["viz_channel"].iloc[0])
        except Exception as e:
            print(f"  [violin] skipping {child.name}: {e}", flush=True)
            skipped.append(child.name)
            continue
        subdir_by_viz[viz_channel] = child
        # Also key by the sanitized subdir name (lowercase + underscores)
        # so a caller passing either form resolves.
        subdir_by_viz[child.name] = child
    print(
        f"  [violin] multi-channel fluor cache: {len(set(subdir_by_viz.values()))} "
        f"channels at {cache_root}",
        flush=True,
    )
    if skipped:
        print(f"  [violin] skipped (no obs/X): {skipped}", flush=True)
    return {
        "_multi": True,
        "_root": cache_root,
        "_subdir_by_viz": subdir_by_viz,
        "_loaded": {},  # viz_channel/subdir_name → loaded sub-pack
    }


def _viz_lookup_keys(viz_channel: str) -> list[str]:
    """Candidate keys for looking up a multi-channel sub-pack by viz_channel.
    Order: raw, lowercase+spaces→underscores, lowercase+spaces+slashes→underscores.
    """
    raw = str(viz_channel)
    keys = [raw]
    k2 = raw.lower().replace(" ", "_")
    if k2 != raw:
        keys.append(k2)
    k3 = k2.replace("/", "_").replace("-", "-")
    if k3 != k2:
        keys.append(k3)
    return keys


def _get_fluor_subpack(fluor_pack, viz_channel):
    """Resolve the per-channel sub-pack for `viz_channel`.

    For a single-cache fluor_pack (distinctiveness layout), returns it
    unchanged. For a multi-channel fluor_pack (NTC layout), loads the
    matching channel's sub-pack on first hit and caches it.
    """
    if fluor_pack is None:
        return None
    if not fluor_pack.get("_multi"):
        return fluor_pack
    subdir_by_viz = fluor_pack["_subdir_by_viz"]
    loaded = fluor_pack["_loaded"]
    subdir = None
    for k in _viz_lookup_keys(viz_channel):
        if k in subdir_by_viz:
            subdir = subdir_by_viz[k]
            break
    if subdir is None:
        return None
    key = str(subdir)
    if key in loaded:
        return loaded[key]
    print(f"  [violin] lazy-loading fluor sub-pack: {subdir.name}", flush=True)
    sub = _load_single_cache(subdir)
    # gene → row-index for this single-channel sub-pack. No need for
    # (gene, viz_channel) key — the whole sub-pack is one viz_channel.
    sub["gene_idx_by_gene"] = _build_gene_idx(sub["obs"])
    # Expose the (gene, viz_channel) keying that the caller expects.
    vc_uniq = sub["obs"]["viz_channel"].astype(str).iloc[0] if len(sub["obs"]) else viz_channel
    sub["gene_idx_viz"] = {
        (g, vc_uniq): idx for g, idx in sub["gene_idx_by_gene"].items()
    }
    # Legacy (gene, channel_rank) fallback — derive from obs's
    # channel_rank if present (mode/first per gene).
    sub["gene_idx"] = {}
    if "channel_rank" in sub["obs"].columns:
        cr_col = sub["obs"]["channel_rank"]
        for g, idx in sub["gene_idx_by_gene"].items():
            if not len(idx):
                continue
            cr = cr_col.iloc[idx[0]]
            try:
                sub["gene_idx"][(g, int(cr))] = idx
            except (TypeError, ValueError):
                continue
    loaded[key] = sub
    return sub


def _load_violin_caches(cache_phase, cache_fluor):
    """Load mmap-friendly views of the SHAP caches for violin extraction.

    Returns (phase_pack, fluor_pack). Each is a dict with X, obs,
    fi (feature → col), median, std, and gene_idx (gene → cell-index array).

    For the NTC layout the fluor cache is a parent dir of per-channel
    subdirs (each with its own X.npy/obs.parquet/features.txt/...).
    `_is_multi_channel_dir` detects that case and `fluor_pack` is
    returned as a lazy multi-channel index (see `_build_multi_fluor_pack`);
    sub-packs are loaded on first lookup via `_get_fluor_subpack`.

    Returns (None, None) if either cache directory is missing — the caller
    falls back to the no-cache placeholder."""
    if not cache_phase or not cache_fluor:
        return None, None
    if not Path(cache_phase).exists() or not Path(cache_fluor).exists():
        print(
            f"  [violin] cache(s) missing at {cache_phase} / {cache_fluor};"
            f" falling back to no-cache placeholder",
            flush=True,
        )
        return None, None

    print(f"  [violin] loading phase cache mmap from {cache_phase}", flush=True)
    phase_pack = _load_single_cache(Path(cache_phase))
    # gene → row-index (O(N) groupby — scales to NTC's 8.4M phase cells).
    phase_pack["gene_idx"] = _build_gene_idx(phase_pack["obs"])

    if _is_multi_channel_dir(Path(cache_fluor)):
        # NTC layout: lazy-load per-channel sub-packs on demand.
        fluor_pack = _build_multi_fluor_pack(Path(cache_fluor))
    else:
        # Distinctiveness layout: one fluor cache covering all channels.
        print(f"  [violin] loading fluor cache mmap from {cache_fluor}", flush=True)
        fluor_pack = _load_single_cache(Path(cache_fluor))
        fluor_top = (
            fluor_pack["obs"][fluor_pack["obs"]["rank_type"] == "top"]
            if "rank_type" in fluor_pack["obs"].columns
            else fluor_pack["obs"]
        )
        # Primary index: (gene, viz_channel) — unambiguous at both gene and
        # complex aggregation levels. (At CHAD level, `gene` is the complex
        # name and a given channel_rank can map to several different
        # viz_channels across member genes, so the (gene, channel_rank) key
        # mixed cells from different markers and produced wrong KO violins
        # for CP/4i channels.) Legacy (gene, channel_rank) key is kept as
        # a fallback for older caches without a viz_channel column.
        fluor_idx_vc: dict[tuple[str, str], np.ndarray] = {}
        if "viz_channel" in fluor_top.columns:
            for (gene, vc), grp in fluor_top.groupby(["gene", "viz_channel"], observed=True):
                fluor_idx_vc[(str(gene), str(vc))] = grp.index.to_numpy()
        fluor_pack["gene_idx_viz"] = fluor_idx_vc
        fluor_idx: dict[tuple[str, int], np.ndarray] = {}
        for (gene, cr), grp in fluor_top.groupby(["gene", "channel_rank"], observed=True):
            fluor_idx[(str(gene), int(cr))] = grp.index.to_numpy()
        fluor_pack["gene_idx"] = fluor_idx
    return phase_pack, fluor_pack


def _violin_values(pack, feature, idx, max_cells=200):
    """Z-scored value distribution of `feature` across the cells in `idx`.
    Returns None when the feature isn't in the cache or fewer than 5
    finite values exist.

    Samples are drawn from the FINITE subset rather than from `idx`
    blindly: a CP/4i channel's cells have 95-100% NaN for op_* features
    (different imaging modality), so a uniform sample from `idx`
    expects only ~5 finite values per 150 picked → often hits the
    `< 5 finite` floor purely by chance even when 800+ finite cells
    exist in the pool. Pre-filtering eliminates that bias.
    """
    if pack is None:
        return None
    fi = pack["fi"].get(feature)
    if fi is None or len(idx) == 0:
        return None
    # Read all candidate values then keep only the finite ones — the
    # NaN cells contribute nothing and shouldn't eat sample budget.
    vals_all = np.asarray(pack["X"][_pack_x_indices(pack, idx), fi],
                            dtype=np.float64)
    finite_mask = np.isfinite(vals_all)
    n_finite = int(finite_mask.sum())
    if n_finite < 5:
        return None
    finite_vals = vals_all[finite_mask]
    # Subsample so we ship at most `max_cells` floats per (gene, feature).
    if len(finite_vals) > max_cells:
        rng = np.random.default_rng(zlib.crc32(feature.encode("utf-8")) & 0xFFFFFFFF)
        sub_idx = rng.choice(len(finite_vals), size=max_cells, replace=False)
        finite_vals = finite_vals[sub_idx]
    z = (finite_vals - pack["median"][fi]) / pack["std"][fi]
    return [float(v) for v in z]


# Background sample size per (gene, feature). Stored alongside the KO
# violin values so each task pickle stays small (5 feats × 4 channels ×
# 150 floats × 8 bytes = ~24 KB extra per gene).
N_BG_CELLS = 150


def _bg_indices(pack, modality, channel_rank=None, viz_channel=None,
                exclude_gene=None, contrast="distinct"):
    """Return cell indices for the bg violin reference.

    Mirrors `ko_shap_features.py`'s negative class — TOP-attention cells
    from genes OTHER than `exclude_gene`, restricted to the SAME
    viz_channel for fluor. No bottom-attention cells are used anywhere;
    the SHAP classifier itself was retrained against this same
    other-gene-top pool, so the bg violin shows exactly what SHAP
    scored against.

    Filter precedence: `viz_channel` > `channel_rank`. The earlier
    channel_rank-only filter mixed cells across viz_channels (each
    gene's rank-1 channel is a different marker), which silently
    erased the CP/4i bg violin: cp_* / 4i_* features are NaN for
    live-cell rows, and most rank-1 cells across the cohort ARE
    live-cell, so the resulting bg pool had ~no finite cp_* values
    and the bg violin disappeared. Filtering on viz_channel keeps
    the bg pool inside the same imaging modality as the KO, which
    is what the SHAP fit actually used.
    """
    obs = pack["obs"]
    # The bg pool MUST mirror whatever negative class SHAP was trained
    # against — otherwise the violin's color/position contradicts the
    # caption (which is derived from SHAP-aligned effect_size). The
    # contrast determines the pool:
    #   * 'distinct' → other-gene top-attention (what the SHAP
    #                  classifier in `ko_shap_features.py` actually
    #                  sees as the negative class).
    #   * 'ntc'      → cells labeled `gene == "NTC"` in the cache
    #                  (the negative class used by `ntc_shap_features
    #                  --contrast ntc`).
    #   * 'global'   → every cell in the cache (the negative class
    #                  used by `ntc_shap_features --contrast global`,
    #                  which samples 50k cells channel-wide as the
    #                  cohort reference).
    contrast_key = str(contrast or "distinct").lower()
    if contrast_key == "ntc" and "gene" in obs.columns:
        mask = obs["gene"].astype(str) == "NTC"
    elif contrast_key == "global":
        mask = np.ones(len(obs), dtype=bool)
    else:
        # distinct: other-gene top-attention in the same channel
        gene_col = obs["gene"].astype(str)
        mask = gene_col != str(exclude_gene)
        if modality == "fluor":
            if "rank_type" in obs.columns:
                mask = mask & (obs["rank_type"] == "top")
    if modality == "fluor":
        if viz_channel is not None and "viz_channel" in obs.columns:
            mask = mask & (obs["viz_channel"].astype(str) == str(viz_channel))
        elif channel_rank is not None and "channel_rank" in obs.columns:
            mask = mask & (obs["channel_rank"] == channel_rank)
    return np.where(np.asarray(mask))[0]


def _candidate_slugs(channel_key: str):
    """Yield filename slugs for `all_cells_fluor_<slug>.h5ad` lookup,
    most-specific first. Mirrors the helper in ko_shap_features —
    handles both standard channels (full slug) and 4i markers where
    organelle == marker is collapsed to a short name (e.g.
    `p21_p21 (rabbit-647)` → `p21`). Same logic in both files keeps
    SHAP-side neg-pool reads and atlas-side BG reads in lockstep.
    """
    import re as _re
    raw = str(channel_key).lower()
    no_paren = _re.sub(r"\s*\([^)]*\)", "", raw).strip()

    def _sanitize(s):
        for ch in (" ", "/", ",", "(", ")"):
            s = s.replace(ch, "_")
        while "__" in s:
            s = s.replace("__", "_")
        return s.strip("_")

    yield _sanitize(raw)
    if no_paren != raw:
        yield _sanitize(no_paren)
    parts = no_paren.split("_", 1)
    if len(parts) == 2 and parts[0].strip() == parts[1].strip():
        yield _sanitize(parts[0])
    if "_" in no_paren:
        yield _sanitize(no_paren.split("_", 1)[0])


class _ConsolidatedBgLoader:
    """Violin-BG reader for --contrast {ntc, global}.

    Reads BG feature values from the SAME SHAP cache as the positives,
    so they pass through the same extraction pipeline and value scale.
    Two operating modes — auto-detected from the cache obs:

      • TOP-ATTENTION mode (consolidate_top_attention_cells.py has
        ingested gene='NTC' / gene='GLOBAL' rows via
        --ntc-{phase,fluor}-csv / --global-per-channel):
          ntc    → rows where gene == 'NTC'.
          global → rows where gene == 'GLOBAL'.
      • ALL-CELLS mode (per-channel all_cells_*.h5ad caches —
        ntc_shap_features pipeline):
          ntc    → rows where gene == 'NTC'.
          global → random sample across ALL cells in the cache
                   (i.e. the "all non-this-gene" cohort, channel-wide).

    Detection: if gene='GLOBAL' rows exist → top-attention mode for
    global. Otherwise → all-cells mode. NTC is unambiguous (real label
    in both modes). HARD-errors only when the requested cohort is
    genuinely unreachable (e.g. ntc but no NTC cells in cache).
    """

    def __init__(self, phase_pack: dict | None, fluor_pack: dict | None,
                 contrast: str):
        if contrast not in ("ntc", "global"):
            raise ValueError(f"contrast must be 'ntc' or 'global', got {contrast!r}")
        self.contrast = contrast
        self.phase_pack = phase_pack
        self.fluor_pack = fluor_pack
        # Cached per-channel index lookup → (pack, np.ndarray[indices])
        self._idx_by_channel: dict[str, tuple[dict, np.ndarray] | None] = {}

        # Probe the caches to decide source semantics.
        n_ntc, n_global, n_total = 0, 0, 0
        for pack in (phase_pack, fluor_pack):
            if pack is None: continue
            obs = pack.get("obs")
            if obs is None: continue
            g = obs["gene"].astype(str)
            n_ntc    += int((g == "NTC").sum())
            n_global += int((g == "GLOBAL").sum())
            n_total  += int(len(g))

        if contrast == "ntc":
            self.mode = "label_filter"
            self.gene_label = "NTC"
            if n_ntc == 0:
                raise SystemExit(
                    "--contrast ntc requires NTC cells in the SHAP cache. "
                    "Re-run consolidate_top_attention_cells.py with "
                    "--ntc-{phase,fluor}-csv (defaults wired), or for the "
                    "all-cells pipeline rebuild the per-channel caches "
                    "with NTC cells present."
                )
            print(f"  [bg/ntc] {n_ntc:,} gene='NTC' rows in cache "
                  f"(of {n_total:,} total) → label_filter mode", flush=True)
        else:  # global
            if n_global > 0:
                # consolidate_top_attention_cells injected explicit GLOBAL rows.
                self.mode = "label_filter"
                self.gene_label = "GLOBAL"
                print(f"  [bg/global] {n_global:,} gene='GLOBAL' rows in cache "
                      f"→ label_filter mode (top-attention pipeline)",
                      flush=True)
            else:
                # No synthetic GLOBAL label — sample randomly across the
                # whole per-channel cohort (all-cells pipeline).
                self.mode = "random_sample"
                self.gene_label = None
                if n_total == 0:
                    raise SystemExit(
                        "--contrast global: SHAP cache appears empty — "
                        "no rows to sample BG from."
                    )
                print(f"  [bg/global] no gene='GLOBAL' rows; sampling "
                      f"randomly across {n_total:,} cells (all-cells "
                      f"pipeline mode)", flush=True)

    def _pack_for(self, channel_key: str) -> dict | None:
        return (self.phase_pack if channel_key in ("phase", "Phase")
                else self.fluor_pack)

    def _indices(self, channel_key: str) -> tuple[dict, np.ndarray] | None:
        if channel_key in self._idx_by_channel:
            return self._idx_by_channel[channel_key]
        pack = self._pack_for(channel_key)
        if pack is None:
            self._idx_by_channel[channel_key] = None
            return None
        obs = pack.get("obs")
        if obs is None:
            self._idx_by_channel[channel_key] = None
            return None
        # Channel filter is always applied for fluor packs (which mix
        # channels in one pack). Phase packs are single-channel by
        # construction, so the channel filter is a no-op.
        if channel_key not in ("phase", "Phase"):
            chan_mask = obs["viz_channel"].astype(str) == channel_key
        else:
            chan_mask = pd.Series(True, index=obs.index)
        if self.mode == "label_filter":
            mask = (obs["gene"].astype(str) == self.gene_label) & chan_mask
        else:  # random_sample (all-cells pipeline, contrast=global)
            mask = chan_mask
        idx = np.where(np.asarray(mask))[0]
        result = (pack, idx) if len(idx) >= 5 else None
        self._idx_by_channel[channel_key] = result
        return result

    def get_z_values(self, channel_key: str, pack: dict, feature: str,
                     seed_key: str, max_cells: int) -> list[float] | None:
        """Return up to `max_cells` z-scored BG values for `feature` at
        `channel_key`, sampling deterministically by `seed_key`. The
        `pack` arg (positive cache pack) is the z-score reference so
        KO + BG land on the same x-axis."""
        resolved = self._indices(channel_key)
        if resolved is None:
            return None
        bg_pack, idx = resolved
        # Feature lookup is in the BG pack (= same cache as pos for unified mode).
        fi_bg = bg_pack.get("fi", {}).get(feature)
        if fi_bg is None:
            return None
        if len(idx) > max_cells:
            rng = np.random.default_rng(zlib.crc32(seed_key.encode()) & 0xFFFFFFFF)
            idx = idx[rng.choice(len(idx), size=max_cells, replace=False)]
        vals = bg_pack["X"][_pack_x_indices(bg_pack, idx), fi_bg]
        finite = vals[np.isfinite(vals)]
        if len(finite) < 5:
            return None
        # Z-score with the POSITIVE pack's median/std so BG + KO violins
        # are aligned on the same x-axis (when pos / BG packs are the
        # same — i.e. unified mode — this is a no-op equivalence).
        fi_pos = pack["fi"].get(feature)
        if fi_pos is None:
            return None
        z = (finite.astype(np.float64) - pack["median"][fi_pos]) / pack["std"][fi_pos]
        return [float(v) for v in z]

def _bg_values(pack, feature, other_idx, seed_key, max_cells=N_BG_CELLS,
               bg_loader=None, channel_key=None):
    """Dispatch:
      • When `bg_loader` is provided (--contrast in {ntc, global}), read
        the violin BG sample from the shared reference cell pool the
        loader manages. The bg-pool indices arg (`other_idx`) is ignored
        in this path — the loader has its own pool keyed by channel.
      • Otherwise (legacy / --contrast distinct): sample from
        `pack["X"][other_idx, fi]` as before.
    """
    if bg_loader is not None and channel_key is not None:
        return bg_loader.get_z_values(
            channel_key, pack, feature, seed_key, max_cells,
        )
    return _bg_values_from_pack(pack, feature, other_idx, seed_key, max_cells)


def _bg_values_from_pack(pack, feature, other_idx, seed_key, max_cells=N_BG_CELLS):
    """Z-scored value distribution of `feature` across other-gene top cells.

    Sample is drawn deterministically (CRC32-seeded) from the FINITE
    subset of `other_idx` — same finite-first filter as
    `_violin_values`. Without this, a uniform sample from a 30k cell
    pool with 97% NaNs (e.g. p53 channel × `op_*` feature) expects only
    ~5 finite values per 150 picked and trips the `< 5 finite` floor by
    chance, even though 800+ finite cells exist.
    """
    if pack is None or len(other_idx) < 5:
        return None
    fi = pack["fi"].get(feature)
    if fi is None:
        return None

    vals_all = np.asarray(pack["X"][_pack_x_indices(pack, other_idx), fi],
                            dtype=np.float64)
    finite_mask = np.isfinite(vals_all)
    if int(finite_mask.sum()) < 5:
        return None
    finite_vals = vals_all[finite_mask]
    if len(finite_vals) > max_cells:
        rng = np.random.default_rng(zlib.crc32(seed_key.encode("utf-8")) & 0xFFFFFFFF)
        sub_idx = rng.choice(len(finite_vals), size=max_cells, replace=False)
        finite_vals = finite_vals[sub_idx]
    z = (finite_vals - pack["median"][fi]) / pack["std"][fi]
    return [float(v) for v in z]


# Features whose semantics are opaque to a human reader — invariant
# moments, eigenvalues, raw central moments. Captioning these is hard
# (they show up as "more complex shape" / "more dispersed signal" etc.,
# none of which carry crisp biological meaning) so we deprioritize
# them when ≥1 more interpretable candidate is available in the SHAP
# top window.
_UNINTERPRETABLE_PATTERNS = (
    # OP snake_case
    "hu_moment",
    "moments_weighted_hu",
    "weighted_hu",
    "central_moment",
    "inertia_eigval",
    "moments_normalized",
    # CP CamelCase (post-lower()): cp_cell_HuMoment_0 →
    # cp_cell_humoment_0, cp_cell_CentralMoment_0_1 →
    # cp_cell_centralmoment_0_1, etc. Without these patterns the SHAP
    # top-5 silently included a dozen+ moment / eigenvalue features
    # because the snake_case patterns above only matched OP names.
    "humoment",
    "centralmoment",
    "normalizedmoment",
    "spatialmoment",
    "inertiatensoreigenvalues",
)


def _interpretability_rank(feature_name: str) -> int:
    """Lower = more interpretable. Used as a SECONDARY sort key after
    SHAP rank so we promote human-friendly features (network branches,
    counts, area, intensity, etc.) over opaque moments / eigenvalues
    when SHAP ranks them within a few slots of each other.
    """
    name = str(feature_name).lower()
    for p in _UNINTERPRETABLE_PATTERNS:
        if p in name:
            return 1   # demote
    return 0           # keep at SHAP-rank order


def _has_spread(feat: dict, min_iqr: float = MIN_VIOLIN_IQR) -> bool:
    """True if the feature has enough ko+bg violin data to plot a meaningful
    distribution. Requires ≥5 values per side AND that EITHER side's IQR
    ≥ min_iqr (in z-units). The OR (rather than combined-IQR) rule keeps
    features where one violin is flat but the other has shape — the
    contrast itself is informative — and only drops the truly-flat
    "both sides collapsed to a line" case.
    """
    ko = feat.get("violin_values")
    bg = feat.get("bg_values")
    if not ko or not bg or len(ko) < 5 or len(bg) < 5:
        return False
    ko_a = np.asarray(ko, dtype=float)
    bg_a = np.asarray(bg, dtype=float)
    ko_iqr = float(np.percentile(ko_a, 75) - np.percentile(ko_a, 25))
    bg_iqr = float(np.percentile(bg_a, 75) - np.percentile(bg_a, 25))
    return ko_iqr >= min_iqr or bg_iqr >= min_iqr


def load_shap_data(features_csv, captions_csv,
                   cache_phase=None, cache_fluor=None,
                   gene_filter=None,
                   marker_map_csv=None,
                   aggregation_level="gene",
                   chad_config=None,
                   top_fluor_channels=DEFAULT_TOP_FLUOR_CHANNELS,
                   mAP_channel_threshold=0.0,
                   max_fluor_channels=None,
                   pma_per_gene_channels=None,
                   contrast="distinct",
                   bg_all_cells_dir=None,
                   bg_refs_dir=None,
                   ntc_pma_phase_csv=None,
                   ntc_pma_fluor_csv=None):
    """Build {gene: shap_payload}.

    shap_payload is a dict (json/pickle-friendly) with:
      'channels'        : ['Phase', ch1, ch2, ch3]  (mAP-descending fluor)
      'features'        : {channel: [{feature, importance, direction, ...,
                                      violin_values?}, ...]}
      'caption'         : {channel: text}
      'auroc'           : {channel: float}
      'n_cells'         : {channel: int}
      'mAP_per_channel' : {channel: float | None} — mAP for each rendered
                           channel. Surfaced in panel titles.
      'mAP_full_row'    : {marker_norm_key: float} — mAP for ALL markers
                           the matrix carries. Drives the top-of-page
                           heatmap bar (consistent across every page).
      'mAP_marker_display' : {marker_norm_key: original_name} — pretty
                           labels for the heatmap bar tick text.
      'mAP_marker_order'   : list[norm_key] — global, page-invariant
                           column order for the bar (Phase first, then
                           alphabetical).

    When `cache_phase` and `cache_fluor` are valid, each feature dict also
    carries `violin_values`: a z-scored list of feature values across the
    gene's top-attention KO cells (subsampled to <=200 cells / feature).
    The renderer prefers violins when this is present and falls back to
    lollipop bars otherwise.

    Channel selection: When `marker_map_csv` is provided (default in
    main()), per-page channels are picked by mAP distinctiveness from
    that matrix — gene-level matches by gene symbol; complex-level
    aggregates the gene matrix to (complex, marker) means using
    `chad_config`. Fluor channels are sorted descending by mAP, then:
      * If `mAP_channel_threshold > 0`: keep all channels with mAP >=
        threshold, up to `max_fluor_channels` (dynamic per-page count).
      * Else: keep the top `top_fluor_channels` (legacy fixed-3 mode).
    Genes/complexes absent from the mAP matrix fall back to the legacy
    first-N order from the SHAP CSV's `viz_channels` field.
    """
    # Defensive defaults — any of these can land as None when callers
    # forward args from outside the argparse path (notebooks, tests,
    # legacy wrappers). Coerce to safe numerics so the `>=` / `<`
    # checks below never trip a `NoneType` TypeError.
    if mAP_channel_threshold is None:
        mAP_channel_threshold = 0.0
    if top_fluor_channels is None:
        top_fluor_channels = DEFAULT_TOP_FLUOR_CHANNELS
    if max_fluor_channels is None:
        max_fluor_channels = (
            10 if mAP_channel_threshold > 0 else top_fluor_channels
        )
    print(f"  reading {features_csv}", flush=True)
    feat_df = pd.read_csv(features_csv)
    if gene_filter is not None:
        gene_filter = {str(g) for g in gene_filter}
        before = len(feat_df)
        feat_df = feat_df[feat_df["gene"].astype(str).isin(gene_filter)].copy()
        print(
            f"  filtered to {feat_df['gene'].nunique()} genes "
            f"({len(feat_df):,}/{before:,} rows)",
            flush=True,
        )
    print(f"  reading {captions_csv}", flush=True)
    cap_df = pd.read_csv(captions_csv)

    cap_lookup = {}
    if "gene" in cap_df.columns and "caption" in cap_df.columns:
        cap_lookup = dict(zip(cap_df["gene"].astype(str), cap_df["caption"].astype(str)))

    phase_pack, fluor_pack = _load_violin_caches(cache_phase, cache_fluor)

    # Shared NTC / global BG reader. NTC and GLOBAL cells live in the
    # SAME SHAP cache as the positives (consolidate_top_attention_cells.py
    # ingests them via --ntc-{phase,fluor}-csv / --global-per-channel).
    # No all_cells_v2 detour, no fallback — hard-errors at init if the
    # cohort's rows aren't in the cache.
    bg_loader = None
    if contrast in ("ntc", "global"):
        bg_loader = _ConsolidatedBgLoader(phase_pack, fluor_pack, contrast)

    # Marker selection metric depends on aggregation level — STRICT
    # split per spec:
    #   gene-level    → per-gene mAP DISTINCTIVENESS
    #                   (gene_reporter_distinctiveness_raw.csv)
    #   complex-level → per-complex mAP CONSISTENCY
    #                   (complex_reporter_chad_consistency.csv)
    # No silent fallback: if the consistency matrix is missing for a
    # complex-level run, _load_chad_consistency_matrix raises with a
    # "regenerate via pca_optimization --aggregate-only" message.
    map_df_raw = _load_marker_map(marker_map_csv)
    if aggregation_level == "complex":
        map_df = _load_chad_consistency_matrix(
            DEFAULT_CHAD_CONSISTENCY_CSV, chad_config,
        )
        # Surface the per-complex `all_combined` (mean consistency
        # across markers) as the title-bar mAP badge.
        gene_all_combined = {}  # not used at complex level
    else:
        map_df = map_df_raw
        gene_all_combined = (
            map_df_raw.attrs.get("all_combined", {}) if map_df_raw is not None else {}
        )

    # Per-entity "all-markers combined" mAP — one number per
    # gene/complex summarizing the overall mAP for the full marker
    # panel. Surfaced in the page-title quality bar.
    #   gene-level    → distinctiveness `all_combined` (per-gene global score)
    #   complex-level → consistency mean across markers
    #                   (already cached at map_df.attrs["all_combined"]
    #                   by `_load_chad_consistency_matrix`)
    if aggregation_level == "complex":
        complex_all_combined = (
            map_df.attrs.get("all_combined", {}) if map_df is not None else {}
        )
    else:
        complex_all_combined = {}

    # CHAD complex member-gene lookup, for the page-header "Members (N):"
    # line at complex-aggregation level. Loaded from the SAME chad_config
    # used for the mAP aggregation so the rendered names always match
    # the bar's complex labels.
    cluster_members = (
        _load_chad_complex_members(chad_config)
        if aggregation_level == "complex" else {}
    )
    # Hand-written one-line functional description per CHAD complex
    # (e.g. "Arp2/3 actin nucleator — generates branched actin networks").
    # Combined with the member-gene list to form the page-header italic
    # ontology line. Empty for gene-level pages — uniprot_function fills
    # the slot there.
    cluster_descriptions = (
        _load_chad_descriptions()
        if aggregation_level == "complex" else {}
    )

    # Per-page marker order is computed inside the gene loop below so
    # each page's bar reads as mAP-descending (Phase always first).
    # `marker_display` is page-invariant — a {norm_key: pretty_name} map
    # used by the bar to render the gene-symbol labels.
    if map_df is not None and len(map_df.columns):
        # Read display_by_norm from the active map_df (consistency at
        # complex level, distinctiveness at gene level) so the bar
        # labels match the matrix's own column set.
        marker_display = map_df.attrs.get("display_by_norm", {})
        all_keys = list(map_df.columns)
    else:
        marker_display = {}
        all_keys = []

    out = {}
    for gene, gdf in feat_df.groupby("gene"):
        gene = str(gene)
        # Available channels for this gene = ones with SHAP rows.
        available_fluor = sorted(
            gdf[gdf["modality"] == "fluor"]["viz_channel"].astype(str).unique()
        )
        has_phase = (gdf["modality"] == "phase").any()
        available = (["Phase"] if has_phase else []) + available_fluor

        # Build map_per_channel for this gene first (before deciding
        # which channels to render), so we can both populate the bar
        # AND drive selection from the SAME ordering — guarantees the
        # selected fluor channels are the LEFTMOST K on the bar.
        if map_df is not None and gene in map_df.index:
            row = map_df.loc[gene]
            map_per_channel = {
                ch: (float(row.get(_norm_channel_key(ch)))
                     if pd.notna(row.get(_norm_channel_key(ch))) else None)
                for ch in available
            }
        else:
            map_per_channel = {ch: None for ch in available}

        # Full-row mAP (for the top-of-page heatmap bar). Falls back
        # to {} when this gene/complex isn't in the matrix → bar
        # renders all-grey for that page (still a useful visual cue
        # that we have no mAP data for that target).
        if map_df is not None and gene in map_df.index:
            mAP_full_row = {
                k: (float(v) if v is not None and not pd.isna(v) else None)
                for k, v in map_df.loc[gene].items()
            }
        else:
            mAP_full_row = {}

        # Per-page marker order: Phase first (always, regardless of mAP),
        # then fluor markers sorted by (mAP DESCENDING, marker_name
        # ASCENDING) so the bar's brightest cells cluster at the left,
        # with ALPHABETICAL tiebreak for ties at e.g. mAP=1.0. Same
        # tuple-sort is used to drive channel selection below — the
        # selected fluor channels are guaranteed to be the LEFTMOST K
        # fluor on the bar.
        phase_key = _norm_channel_key("Phase")
        if mAP_full_row:
            fluor_with_map = [
                (k, v) for k, v in mAP_full_row.items()
                if k != phase_key and v is not None
            ]
            # Tuple sort: primary -mAP (desc), secondary marker_key (asc).
            fluor_with_map.sort(key=lambda kv: (-kv[1], kv[0]))
            page_marker_order = (
                ([phase_key] if phase_key in mAP_full_row else [])
                + [k for k, _ in fluor_with_map]
            )
            # Tail: any markers that exist in the global key set but
            # have no mAP score for this page (rare — keeps the bar
            # complete so columns line up across pages where possible).
            seen = set(page_marker_order)
            page_marker_order += sorted(k for k in all_keys if k not in seen)
        else:
            # No mAP for this gene/complex — fall back to alphabetical
            # so the bar still renders something stable.
            page_marker_order = (
                ([phase_key] if phase_key in all_keys else [])
                + sorted(k for k in all_keys if k != phase_key)
            )

        # Walk page_marker_order LEFT TO RIGHT and pick all non-Phase
        # keys whose mAP >= `mAP_channel_threshold`, capped at
        # `max_fluor_channels`. The selected channels are therefore the
        # leftmost K fluor on the bar, by construction. Pages get
        # dynamic K: a complex with 9 high-mAP markers renders all 9
        # rows; a low-signal complex with 1 above-threshold marker
        # renders only that 1 row. Set `mAP_channel_threshold = 0` to
        # restore the legacy "always pick top max_fluor_channels" mode.
        #
        # For NTC/global contrasts, `pma_per_gene_channels` restricts
        # available_fluor to channels that pma actually has for this
        # gene (channel_rank 1..3 in pma_top_fluorescent_cells_*.csv).
        # Without this, NTC's full 56-channel SHAP set would pick mAP-
        # top markers outside pma's per-gene set, producing blank image
        # rows because the renderer has no pma cells to crop.
        if pma_per_gene_channels is not None:
            allowed = pma_per_gene_channels.get(str(gene), set())
            if allowed:
                available_fluor = [c for c in available_fluor if c in allowed]
            # If `allowed` is empty (gene not in pma) we keep
            # available_fluor as-is so the selector still picks
            # something rather than rendering an empty page.
        available_fluor_norm = {_norm_channel_key(c): c for c in available_fluor}
        top_fluors = []
        for k in page_marker_order:
            if k == phase_key:
                continue
            disp = available_fluor_norm.get(k)
            if disp is None:
                continue  # marker exists in mAP matrix but no SHAP rows for this gene
            ch_mAP = mAP_full_row.get(k)
            if (mAP_channel_threshold > 0
                    and (ch_mAP is None or ch_mAP < mAP_channel_threshold)):
                # Below threshold: stop walking — page_marker_order is
                # mAP-descending, so nothing further qualifies either.
                break
            top_fluors.append(disp)
            if max_fluor_channels is not None and len(top_fluors) >= max_fluor_channels:
                break
        channels = (["Phase"] if has_phase else []) + top_fluors

        features_per_channel = {}
        auroc_per_channel = {}
        n_cells_per_channel = {}
        for ch in channels:
            if ch == "Phase":
                ch_df = gdf[gdf["modality"] == "phase"]
            else:
                ch_df = gdf[(gdf["modality"] == "fluor") & (gdf["viz_channel"] == ch)]
            if ch_df.empty:
                features_per_channel[ch] = []
                continue

            # Load N_TOP_FEATURES + N_EXTRA_FEATURES candidates so any of
            # the top features with no violin spread (flat distribution) can
            # be replaced by the next-highest-ranked feature that does.
            top = ch_df.sort_values("shap_rank").head(
                N_TOP_FEATURES + N_EXTRA_FEATURES
            )

            # Resolve which cache + cell-index slice this channel pulls
            # from. Phase = phase_pack, indexed by gene only. Fluor = the
            # fluor_pack, indexed by (gene, channel_rank).
            if ch == "Phase":
                pack = phase_pack
                idx = pack["gene_idx"].get(gene, np.array([], dtype=int)) if pack else np.array([], dtype=int)
                bg_idx = (
                    _bg_indices(pack, "phase", exclude_gene=gene,
                                contrast=contrast)
                    if pack else np.array([], dtype=int)
                )
            else:
                # For multi-channel fluor caches (NTC layout), resolve the
                # per-channel sub-pack by viz_channel — sub-pack is loaded
                # lazily on first hit, then reused for the rest of this run.
                pack = _get_fluor_subpack(fluor_pack, ch)
                cr = int(top["channel_rank"].iloc[0]) if "channel_rank" in top.columns else 0
                # Prefer the viz_channel-keyed index (unambiguous at
                # CHAD level); fall back to (gene, channel_rank) for
                # legacy caches built before the viz_channel index.
                gene_idx_vc = (pack.get("gene_idx_viz") or {}) if pack else {}
                if (gene, ch) in gene_idx_vc:
                    idx = gene_idx_vc[(gene, ch)]
                else:
                    idx = (
                        pack["gene_idx"].get((gene, cr), np.array([], dtype=int))
                        if pack else np.array([], dtype=int)
                    )
                bg_idx = (
                    _bg_indices(
                        pack, "fluor",
                        channel_rank=cr, viz_channel=ch,
                        exclude_gene=gene,
                        contrast=contrast,
                    )
                    if pack else np.array([], dtype=int)
                )

            candidate_features = [
                {
                    "feature": str(r["feature"]),
                    "importance": float(r["shap_importance"]) if pd.notna(r["shap_importance"]) else 0.0,
                    "direction": float(r["direction"]) if pd.notna(r["direction"]) else 1.0,
                    "shap_mean": float(r["shap_mean"]) if pd.notna(r["shap_mean"]) else 0.0,
                    "shap_cv": float(r["shap_cv"]) if pd.notna(r["shap_cv"]) else 0.0,
                    "effect_size": float(r["effect_size"]) if pd.notna(r["effect_size"]) else 0.0,
                    # `pct_cells` is the new column added by
                    # `OrganelleProfiler/scripts/ko_shap/add_pct_cells.py`
                    # (PR #5). It's the fraction of the gene's top-attention
                    # cells where the feature value is on the expected side
                    # of the global median — a real "% cells with the
                    # feature" rather than a SHAP-derived proxy. NaN if the
                    # CSV pre-dates the pipeline change; the renderer falls
                    # back to a consistency proxy when this is None.
                    "pct_cells": (
                        float(r["pct_cells"])
                        if "pct_cells" in r and pd.notna(r["pct_cells"])
                        else None
                    ),
                    # Z-scored feature values across the gene's KO cells —
                    # used to draw a horizontal violin per feature in the
                    # SHAP panel. None when caches aren't available; the
                    # renderer falls back to a lollipop bar in that case.
                    "violin_values": _violin_values(pack, str(r["feature"]), idx),
                    # Pale background reference: top-attention cells from
                    # OTHER genes (same channel for fluor) — the SHAP
                    # classifier's negative class.
                    "bg_values": _bg_values(
                        pack, str(r["feature"]), bg_idx,
                        seed_key=f"{gene}|{ch}|{r['feature']}",
                        bg_loader=bg_loader,
                        channel_key=("phase" if ch == "Phase" else ch),
                    ),
                    "organelle": str(r["organelle"]) if pd.notna(r.get("organelle")) else "",
                    "category": str(r["category"]) if pd.notna(r.get("category")) else "",
                }
                for _, r in top.iterrows()
            ]
            # Pick features that have real KO+BG data (both sides ≥5 cells
            # AND at least one side has IQR ≥ MIN_VIOLIN_IQR). For
            # low-signal channels where ALL candidates fail (AUROC≈0.5,
            # shap_importance=0, or features in the wrong imaging
            # modality with all-NaN values) we fall back to the original
            # SHAP-ranked candidates so the panel shows the lollipop
            # view — readers see 5 near-zero bars and immediately read
            # "this channel has no discriminating signal" instead of
            # a misleadingly empty "no SHAP features" placeholder. The
            # padded features that DO have spread are pulled forward
            # so the violin path still gets the best rows when at
            # least one feature qualifies.
            with_spread = [f for f in candidate_features if _has_spread(f)]

            # Two-tier selection:
            #   tier 1 — interpretable features (interpretability_rank == 0),
            #   tier 2 — opaque (hu_moments, eigenvalues, etc.).
            # Tier 2 is used ONLY as backfill when tier 1 has fewer than
            # N_TOP_FEATURES candidates. The previous logic merely
            # reordered all candidates, so opaque features still
            # appeared whenever the SHAP top-20 included them — which
            # is most of the time. Now they're a hard fallback, not a
            # silent slot-stealer.
            def _split_pool(pool):
                interp, opaque = [], []
                for f in pool:
                    (interp if _interpretability_rank(f.get("feature", "")) == 0
                     else opaque).append(f)
                return interp, opaque

            if with_spread:
                kept_ids = {id(f) for f in with_spread}
                padding = [f for f in candidate_features if id(f) not in kept_ids]
                combined = with_spread + padding
            else:
                # No feature has plottable violin spread; surface the
                # SHAP-ranked candidates anyway (panel will show the
                # "no spread" placeholder per row).
                combined = list(candidate_features)
            interp, opaque = _split_pool(combined)
            picked = interp[:N_TOP_FEATURES]
            if len(picked) < N_TOP_FEATURES:
                picked.extend(opaque[: N_TOP_FEATURES - len(picked)])
            features_per_channel[ch] = picked
            if pd.notna(ch_df["auroc"].iloc[0]):
                auroc_per_channel[ch] = float(ch_df["auroc"].iloc[0])
            if "n_pos_cells" in ch_df.columns and pd.notna(ch_df["n_pos_cells"].iloc[0]):
                n_cells_per_channel[ch] = int(ch_df["n_pos_cells"].iloc[0])

        captions_per_channel = _parse_caption_per_channel(
            cap_lookup.get(gene, ""), channels
        )

        # Channel-rank colors so the bar labels and the row labels on
        # the left of the image grid match. Phase always black; the 3
        # selected fluor channels (in mAP-descending order, which is
        # how `channels` is built by `select_top_channels`) get the
        # CHANNEL_RANK_COLORS palette.
        channel_colors_by_name = {}
        selected_colors_by_key = {}
        fluor_rank = 0
        for ch in channels:
            if ch == "Phase":
                channel_colors_by_name[ch] = PHASE_LABEL_COLOR
                selected_colors_by_key[_norm_channel_key(ch)] = PHASE_LABEL_COLOR
            else:
                color = CHANNEL_RANK_COLORS[
                    min(fluor_rank, len(CHANNEL_RANK_COLORS) - 1)
                ]
                channel_colors_by_name[ch] = color
                selected_colors_by_key[_norm_channel_key(ch)] = color
                fluor_rank += 1

        out[gene] = {
            "gene": gene,
            "channels": channels,
            "features": features_per_channel,
            "caption": captions_per_channel,
            "auroc": auroc_per_channel,
            "n_cells": n_cells_per_channel,
            # mAP-driven channel context (NEW). Empty when no marker
            # map is loaded — renderer falls back to legacy display.
            "mAP_per_channel": map_per_channel,
            "mAP_full_row": mAP_full_row,
            "mAP_marker_order": page_marker_order,
            "mAP_marker_display": marker_display,
            "mAP_selected_keys": [_norm_channel_key(c) for c in channels],
            "mAP_selected_colors": selected_colors_by_key,
            # Read by attention_atlas.render_gene_page to color the
            # left-side ylabel of each (KO + NTC) row pair.
            "channel_colors": channel_colors_by_name,
            # Member-gene list + one-line functional description for
            # the page-header italic ontology line on complex pages
            # (None at gene level — uniprot_function path runs).
            "chad_members": cluster_members.get(gene),
            "chad_description": cluster_descriptions.get(gene),
            # Quality-score badges surfaced in the page suptitle
            # alongside the gene/complex name. `mAP_all_combined` is:
            #   gene-level   → DISTINCTIVENESS (per-gene `all_combined`)
            #   complex-level → CONSISTENCY (mean across markers, from
            #                   complex_reporter_chad_consistency.csv)
            # `mAP_metric_label` lets the renderer prefix the badge
            # with the metric type ("distinct" / "consist") so the
            # reader can tell which signal is being reported.
            "mAP_all_combined": (
                complex_all_combined.get(gene)
                if aggregation_level == "complex"
                else gene_all_combined.get(gene)
            ),
            "mAP_metric_label": (
                "consist" if aggregation_level == "complex" else "distinct"
            ),
            # Override the atlas's default channel ordering for the
            # image rows so they MATCH the SHAP bar's mAP-driven
            # selection. Without this, image rows pick channels by
            # Alex's attention rank (channel_rank 1/2/3) while the
            # SHAP bar picks by mAP — they often disagree, leaving
            # the row labels uncolored (no entry in channel_colors).
            "viz_channel_override": [c for c in channels if c != "Phase"],
        }
    return out


# ──────────────────────────────────────────────────────────────────────────
# SLURM prep stage (parallelize the per-gene `load_shap_data` loop)
# ──────────────────────────────────────────────────────────────────────────
def _prep_shap_data_shard_worker(
    features_csv: str,
    captions_csv: str,
    cache_phase: str,
    cache_fluor: str,
    marker_map_csv: str,
    aggregation_level: str,
    chad_config: str,
    top_fluor_channels: int,
    map_channel_threshold: float,
    max_fluor_channels: int,
    gene_shard: list[str],
    out_pkl: str,
    contrast: str = "distinct",
    bg_all_cells_dir: str = "",
    bg_refs_dir: str = "",
    ntc_pma_phase_csv: str = "",
    ntc_pma_fluor_csv: str = "",
) -> dict:
    """Compute `load_shap_data` for one shard of genes; pickle the result.

    Top-level (module-scoped) so cloudpickle can ship it to SLURM workers
    without dragging the entire script's frame. Each worker only reloads
    the SHAP CSVs and the (mmap-backed) cache headers for its own shard
    — the actual hot loop over `gene_shard` then runs single-host but
    the WALL clock is now parallelised N-way across the array.
    """
    import pickle as _pickle
    payload = load_shap_data(
        features_csv=Path(features_csv),
        captions_csv=Path(captions_csv),
        cache_phase=Path(cache_phase) if cache_phase else None,
        cache_fluor=Path(cache_fluor) if cache_fluor else None,
        gene_filter=set(gene_shard),
        marker_map_csv=Path(marker_map_csv) if marker_map_csv else None,
        aggregation_level=aggregation_level,
        chad_config=Path(chad_config) if chad_config else None,
        top_fluor_channels=top_fluor_channels,
        mAP_channel_threshold=map_channel_threshold,
        max_fluor_channels=max_fluor_channels,
        contrast=contrast,
        bg_all_cells_dir=Path(bg_all_cells_dir) if bg_all_cells_dir else None,
        bg_refs_dir=Path(bg_refs_dir) if bg_refs_dir else None,
        ntc_pma_phase_csv=Path(ntc_pma_phase_csv) if ntc_pma_phase_csv else None,
        ntc_pma_fluor_csv=Path(ntc_pma_fluor_csv) if ntc_pma_fluor_csv else None,
    )
    Path(out_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        _pickle.dump(payload, f, protocol=_pickle.HIGHEST_PROTOCOL)
    return {"out_pkl": out_pkl,
            "n_genes": len(payload),
            "shard_head": gene_shard[:3]}


def _prep_shap_data_slurm(args, gene_list: list[str]) -> dict:
    """Fan out `load_shap_data` across a SLURM array, keyed by gene shard,
    then merge the per-shard pickles back into a single {gene: payload}
    dict. Mirrors `_submit_via_slurm`'s submit_parallel_jobs pattern so
    log dirs / manifests live alongside the existing atlas_shap logs.
    """
    import pickle as _pickle
    import tempfile
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

    n = max(1, args.shap_prep_genes_per_shard)
    shards = [gene_list[i:i + n] for i in range(0, len(gene_list), n)]
    # Pin work dir next to --output so pickles survive on the shared
    # filesystem (SLURM workers can't read login-node /tmp).
    work_dir = (Path(args.output).parent
                / f"{Path(args.output).stem}_shap_prep")
    work_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for i, shard in enumerate(shards):
        out_pkl = work_dir / f"shard_{i:05d}.pkl"
        jobs.append({
            "name": f"shap_prep_{Path(args.output).stem}_{i:05d}",
            "func": _prep_shap_data_shard_worker,
            "kwargs": {
                "features_csv": str(args.shap_features_csv),
                "captions_csv": str(args.shap_captions_csv),
                "cache_phase": str(args.shap_cache_phase) if args.shap_cache_phase else "",
                "cache_fluor": str(args.shap_cache_fluor) if args.shap_cache_fluor else "",
                "marker_map_csv": str(args.marker_map_csv) if args.marker_map_csv else "",
                "aggregation_level": getattr(args, "aggregation_level", "gene"),
                "chad_config": str(args.chad_complex_config),
                "top_fluor_channels": args.top_fluor_channels,
                "map_channel_threshold": args.map_channel_threshold,
                "max_fluor_channels": args.max_fluor_channels,
                "gene_shard": shard,
                "out_pkl": str(out_pkl),
                "contrast": args.contrast,
                "bg_all_cells_dir": (str(args.bg_all_cells_dir)
                                      if args.bg_all_cells_dir else ""),
                "bg_refs_dir": (str(args.bg_refs_dir)
                                 if args.bg_refs_dir else ""),
                "ntc_pma_phase_csv": (str(args.ntc_pma_phase_csv)
                                       if args.ntc_pma_phase_csv else ""),
                "ntc_pma_fluor_csv": (str(args.ntc_pma_fluor_csv)
                                       if args.ntc_pma_fluor_csv else ""),
            },
            "metadata": {"shard": i, "n_genes": len(shard),
                          "out_pkl": str(out_pkl)},
        })

    slurm_params = {
        "timeout_min": args.shap_prep_timeout_min,
        "slurm_partition": getattr(args, "partition", "cpu"),
        "cpus_per_task": args.shap_prep_cpus,
        "mem": args.shap_prep_mem,
    }
    print(f"\n[shap-prep] submitting {len(shards)} shard jobs "
          f"({n} genes/shard × {args.shap_prep_cpus} CPUs / "
          f"{args.shap_prep_mem} / {args.shap_prep_timeout_min} min)\n",
          flush=True)
    res = submit_parallel_jobs(
        jobs_to_submit=jobs,
        experiment=f"atlas_shap_prep_{Path(args.output).stem}",
        slurm_params=slurm_params,
        log_dir=f"attention_atlas/{Path(args.output).stem}_shap_prep",
        manifest_prefix=f"shap_prep_{Path(args.output).stem}",
        wait_for_completion=True,
        verbose=True,
    )
    if not res.get("all_completed"):
        print(f"[shap-prep] WARNING: "
              f"{len(res.get('failed', []))} shard jobs failed — merging "
              f"surviving shards", flush=True)

    merged: dict = {}
    for j in jobs:
        p = Path(j["kwargs"]["out_pkl"])
        if not p.exists():
            print(f"  [shap-prep] missing {p}", flush=True)
            continue
        with open(p, "rb") as f:
            merged.update(_pickle.load(f))
    print(f"[shap-prep] merged {len(merged):,} gene payloads "
          f"from {len(shards)} shards", flush=True)
    return merged


# ──────────────────────────────────────────────────────────────────────────
# Rendering
# ──────────────────────────────────────────────────────────────────────────
def _render_violin(ax, features, title, auroc=None):
    """Sideways (horizontal) violin chart of top SHAP features.

    For each top feature: a horizontal violin showing the z-scored
    distribution of that feature's values across the gene's top-attention
    KO cells. Color encodes effect direction (blue = ↑ in KO, red = ↓ in
    KO). Signed pct_cells is overlaid at the right end of each violin.

    No lollipop fallback — when features lack plottable KO+BG data
    (e.g. NTC variant runs without a SHAP cache, or a low-signal
    gene/channel where the SHAP top-5 are essentially noise), the
    panel renders a small placeholder text identifying the reason
    instead of falling back to lollipop.
    """
    # Filter to features with both KO ≥5 finite values AND BG ≥5
    # finite values — anything less can't draw a meaningful violin
    # pair and the empty rows interspersed with real violins read as
    # broken output.
    plottable = [
        f for f in features
        if isinstance(f.get("violin_values"), list)
        and len(f["violin_values"]) >= 5
        and isinstance(f.get("bg_values"), list)
        and len(f["bg_values"]) >= 5
    ]
    if not plottable:
        # Distinguish the two no-violin cases for the reader:
        #   - "no SHAP cache" — NTC variant intentionally disables
        #     the cache, so violin_values is None for every feature.
        #     Caption text still renders elsewhere in the panel.
        #   - "no plottable" — cache loaded but features all fail
        #     the spread / finite-count filters (low-signal channel).
        #   - "no SHAP signal" — additionally low AUROC ≈ 0.5.
        no_cache = (
            features is not None
            and len(features) > 0
            and all(f.get("violin_values") is None for f in features)
        )
        if no_cache:
            msg = "no SHAP cache for this variant"
        elif auroc is not None and auroc < 0.55:
            msg = "no SHAP signal\n(AUROC ≈ 0.5)"
        else:
            msg = "no plottable SHAP features"
        ax.text(
            0.5, 0.5, msg,
            ha="center", va="center", transform=ax.transAxes,
            fontsize=10, color="gray", style="italic",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        return
    features = plottable

    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Title rendered by `_render_caption` (single text block: title +
    # caption stacked) so a long caption can never collide with the
    # title. Pad=0 keeps the chart axes from reserving empty space above.
    ax.set_title("", pad=0)

    if not features:
        ax.text(
            0.5, 0.5, "no SHAP features",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=10, color="gray", style="italic",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        return

    # Single source of truth for color + tick direction + pct_cells:
    # the BG sample actually drawn on this panel. The SHAP CSV's
    # `pct_cells` is computed against the top-attention COHORT mean
    # (every gene's cells pooled) — a different reference than the
    # NTC / all-cells BG the violin shows, so they previously
    # disagreed in sign for features where cohort_mean and bg_mean
    # straddled the KO mean. Recomputing both sign AND pct_cells
    # against the visible BG keeps everything in lockstep.
    directions = np.empty(len(features), dtype=float)
    pct_cells_render = np.empty(len(features), dtype=float)
    for i, feat in enumerate(features):
        ko_v = np.asarray(feat.get("violin_values") or [], dtype=float)
        bg_v = np.asarray(feat.get("bg_values") or [], dtype=float)
        if len(ko_v) >= 5 and len(bg_v) >= 5:
            ko_v_f = ko_v[np.isfinite(ko_v)]
            bg_v_f = bg_v[np.isfinite(bg_v)]
            if len(ko_v_f) >= 5 and len(bg_v_f) >= 5:
                bg_ref = float(np.mean(bg_v_f))
                ko_mean = float(np.mean(ko_v_f))
                if ko_mean > bg_ref:
                    directions[i] = 1.0
                    pct_cells_render[i] = float((ko_v_f > bg_ref).mean())
                elif ko_mean < bg_ref:
                    directions[i] = -1.0
                    pct_cells_render[i] = float((ko_v_f < bg_ref).mean())
                else:
                    directions[i] = 0.0
                    pct_cells_render[i] = 0.5
                continue
        directions[i] = 0.0
        pct_cells_render[i] = float("nan")
    colors = [COLOR_UP if d > 0 else COLOR_DOWN for d in directions]

    y_pos = np.arange(len(features))
    all_vals: list[float] = []

    # Side-by-side layout per feature row: bg violin sits ABOVE the row's
    # y-tick (offset -0.22), KO violin sits BELOW it (offset +0.22). Both
    # at width 0.40 so they don't overlap. Y-tick label sits between them.
    KO_OFFSET = 0.22
    BG_OFFSET = -0.22
    HALF_H = 0.20   # half-height of each violin's median tick mark
    VIOLIN_W = 0.40

    # Per-feature BG-anchored robust z-score, hard-clipped at ±VIOLIN_CLIP
    # BG-IQR units. Each feature is normalized to its OWN bg
    # (median=0, IQR=1) so every row uses the same x-range and rows
    # are directly comparable regardless of raw-feature spread.
    # Clipping prevents a single outlier from stretching the panel —
    # was the root cause of "some features look tight, others wide".
    # 4 IQRs is already a huge effect (~99.7% of BG within ±4 IQR for
    # any reasonable distribution); anything beyond is rendered at
    # the edge rather than dragging the axis.
    VIOLIN_CLIP = 4.0
    def _rescale(feat):
        ko = np.asarray(feat.get("violin_values") or [], dtype=float)
        bg = np.asarray(feat.get("bg_values") or [], dtype=float)
        if len(bg) >= 5:
            bg_q1, bg_q3 = np.percentile(bg, [25, 75])
            bg_med = float(np.median(bg))
            bg_iqr = max(float(bg_q3 - bg_q1), 1e-6)
        else:
            bg_med, bg_iqr = 0.0, 1.0
        ko_z = (np.clip((ko - bg_med) / bg_iqr, -VIOLIN_CLIP, VIOLIN_CLIP)
                if len(ko) else np.array([]))
        bg_z = (np.clip((bg - bg_med) / bg_iqr, -VIOLIN_CLIP, VIOLIN_CLIP)
                if len(bg) else np.array([]))
        return ko_z.tolist(), bg_z.tolist()

    rescaled = [_rescale(f) for f in features]

    for yi, feat, col, (ko_scaled, bg_scaled) in zip(y_pos, features, colors, rescaled):
        vals = ko_scaled
        if not vals or len(vals) < 5:
            continue
        # Pale "background" reference violin ABOVE the y-tick.
        bg_vals = bg_scaled
        if bg_vals and len(bg_vals) >= 5:
            vp_bg = ax.violinplot(
                [bg_vals], positions=[yi + BG_OFFSET], widths=VIOLIN_W,
                vert=False, showmedians=True, showextrema=False,
            )
            for body in vp_bg["bodies"]:
                body.set_facecolor("#CCCCCC")
                body.set_edgecolor("#777777")
                body.set_linewidth(0.4)
                body.set_alpha(0.65)
            if "cmedians" in vp_bg:
                vp_bg["cmedians"].set_visible(False)
            all_vals.extend(bg_vals)

            # Bg-mean tick: short dashed gray line spanning only the bg
            # violin's y-range. Mean (not median) so it aligns with the
            # cohort-mean reference `pct_cells` and `effect_size` are
            # computed against in `ko_shap_features.py`.
            bg_mean = float(np.mean(bg_vals))
            ax.plot(
                [bg_mean, bg_mean],
                [yi + BG_OFFSET - HALF_H, yi + BG_OFFSET + HALF_H],
                color="#666666", linewidth=1.4, linestyle=(0, (3, 2)),
                zorder=5, solid_capstyle="round",
            )

        # KO violin BELOW the y-tick. Higher alpha (0.7) since there's no
        # underlying bg shape to show through.
        vp = ax.violinplot(
            [vals], positions=[yi + KO_OFFSET], widths=VIOLIN_W,
            vert=False, showmedians=True, showextrema=False,
        )
        for body in vp["bodies"]:
            body.set_facecolor(col)
            body.set_edgecolor("black")
            body.set_linewidth(0.5)
            body.set_alpha(0.70)
        if "cmedians" in vp:
            vp["cmedians"].set_visible(False)

        # KO mean tick: saturated solid line spanning only the KO
        # violin's y-range, mirroring the bg tick's footprint. Mean
        # (not median) so the gene-vs-cohort comparison shown in the
        # panel matches what pct_cells reports.
        ko_mean = float(np.mean(vals))
        ax.plot(
            [ko_mean, ko_mean],
            [yi + KO_OFFSET - HALF_H, yi + KO_OFFSET + HALF_H],
            color=col, linewidth=2.6, zorder=7, solid_capstyle="round",
        )
        all_vals.extend(vals)

    wrapped_names = [_wrap_feature_name(f["feature"]) for f in features]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(wrapped_names)
    ax.invert_yaxis()
    ax.set_ylim(len(features) - 0.05, -0.95)

    # Vertical zero line (z-score = 0) for visual reference.
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.5, zorder=1)

    # X-axis: fixed window in BG-IQR units, matching the value clip
    # above. ±4 IQR covers ~all of any plausible distribution; right
    # edge extended by 0.5 leaves room for the pct_cells label.
    x_left, x_right = -VIOLIN_CLIP, VIOLIN_CLIP + 0.5
    ax.set_xlim(x_left, x_right)

    # Pct_cells overlay at the right edge — recomputed above against
    # the visible BG so the % always agrees with the tick direction
    # and color. Falls back to the CSV's pct_cells (cohort-anchored)
    # only when the violin lacks data to recompute (no BG values).
    pct_from_csv = [f.get("pct_cells") for f in features]
    pct_values = np.where(
        np.isfinite(pct_cells_render),
        pct_cells_render,
        np.array(
            [float(v) if v is not None else float("nan") for v in pct_from_csv],
            dtype=float,
        ),
    )
    pct_values = np.clip(pct_values, 0.0, 1.0)
    label_x = x_right - 0.02 * (x_right - x_left)
    for yi, p in zip(y_pos, pct_values):
        if not np.isfinite(p):
            continue
        ax.text(
            label_x, yi, f"{p:.0%}",
            ha="right", va="center", fontsize=9, color="#444",
        )
    # Each feature is BG-anchored (median=0, IQR=1) and clipped to
    # ±4 IQR. 0 = BG median, ±1 = ±1 BG-IQR, edge = ≥4 IQR (rare).
    ax.set_xlabel(
        "BG-IQR units (median=0, IQR=1; clipped at ±4)",
        fontsize=9, labelpad=4,
    )


def _render_caption(ax, text, title=None, auroc=None):
    """Single stacked text block carrying BOTH the chart title (channel
    name + AUROC, regular weight) and the italic caption below it.

    The chart axes calls `set_title("", pad=0)` so this is the only place
    title or caption text appears in the panel — eliminates any chance
    of the caption overlapping the title regardless of how long the
    caption gets.

    Implementation: a `VPacker` stacks one `TextArea` per visual line
    (title rows on top, blank spacer, caption rows below). The whole
    block is anchored to the TOP-CENTER of the caption axes via
    `AnchoredOffsetbox`, so it grows downward as it gets taller. With
    no title-pad on the chart axes below, an over-tall block at most
    overflows into the inter-panel gap.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # Pick caption font size + wrap based on total text length so long
    # captions don't overflow the panel. Floors at the smallest tier.
    n_chars = len(text or "") + len(title or "")
    for max_chars, cap_pt, title_pt, wrap_w in CAPTION_FONT_TIERS:
        if n_chars <= max_chars:
            break

    children = []

    if title:
        # Split on \n so explicit line breaks (e.g. "cohort\nmAP=X")
        # survive — textwrap.wrap collapses whitespace and ignores
        # existing newlines, so we wrap each segment separately. AUROC
        # appends to the LAST segment so the metrics sit visually
        # adjacent on the mAP line.
        title_lines: list[str] = []
        segs = str(title).split("\n")
        for i, seg in enumerate(segs):
            seg = seg.strip()
            if not seg:
                continue
            if i == len(segs) - 1 and auroc is not None:
                seg = f"{seg}  ·  AUROC={auroc:.2f}"
            for ln in textwrap.wrap(seg, width=wrap_w, break_long_words=False):
                title_lines.append(ln)
        for ln in title_lines:
            children.append(TextArea(
                ln, textprops=dict(fontsize=title_pt, color="black"),
            ))

    if title and text:
        # Small visual gap between title row(s) and caption row(s).
        children.append(TextArea(
            "", textprops=dict(fontsize=4),
        ))

    if text:
        wrapped = textwrap.wrap(text, width=wrap_w) or [""]
        # Wrap the whole text in matching curly quotes — only on the
        # first/last lines, not every line.
        wrapped[0] = f'"{wrapped[0]}'
        wrapped[-1] = f'{wrapped[-1]}"'
        for ln in wrapped:
            children.append(TextArea(
                ln,
                textprops=dict(fontsize=cap_pt, color="#111", style="italic"),
            ))
    elif not title:
        children.append(TextArea(
            "(no caption for this channel)",
            textprops=dict(fontsize=9, color="gray", style="italic"),
        ))

    if not children:
        return

    packed = VPacker(children=children, pad=0, sep=2, align="center")
    anchored = AnchoredOffsetbox(
        loc="upper center", child=packed,
        bbox_to_anchor=(0.5, 1.0), bbox_transform=ax.transAxes,
        frameon=False, pad=0, borderpad=0,
    )
    ax.add_artist(anchored)


# mAP heatmap bar — thin horizontal strip just below the suptitle.
# Page-invariant marker order (Phase first, then alphabetical) so the
# same column sits at the same x position across pages. Selected
# channel labels float ABOVE the bar (horizontal, bold) so the strip
# itself reads cleanly without rotated text below.
# Page header layout (figure-fraction y, top → bottom):
#   suptitle           y = 0.995
#   ontology line      y_top = 0.978  (italic; uniprot for genes,
#                                      description + members for
#                                      complexes — can wrap up to
#                                      2-3 lines, span ~0.945-0.978)
#   mAP heatmap bar    y = 0.948-0.956  (very close to the ontology
#                                        line — minimal dead space,
#                                        touches the 2nd line of a
#                                        2-line ontology but doesn't
#                                        overlap visibly)
#   gene-symbol labels (45° rotated, wrap@10) below the bar — span
#                       ~0.946 down to ~0.918
#   image grid top     y = 0.896        (~15.9" tall, ~0.4" gap above)
MAP_BAR_TOP = 0.956
MAP_BAR_BOTTOM = 0.948         # height ≈ 0.008 ≈ 0.144"
MAP_BAR_LEFT = 0.025 * 19.0 / FIG_WIDTH_IN     # align with image-grid left
MAP_BAR_RIGHT = IMG_GRID_RIGHT - 0.005 * 19.0 / FIG_WIDTH_IN
MAP_BAR_CMAP = "magma"
# Font sizes for the gene-symbol labels under the bar (horizontal).
MAP_BAR_LABEL_FONT_UNSELECTED = 6.5   # mid-grey for context — slightly
                                       # bigger + darker than 5/#9A9A9A
                                       # so the unselected markers stay
                                       # legible without competing with
                                       # the bold selected labels.
MAP_BAR_LABEL_FONT_SELECTED = 7.5     # bold + channel color for selected 3 fluor
MAP_BAR_LABEL_FONT_PHASE = 7.5        # bold black for Phase
MAP_BAR_LABEL_COLOR_UNSELECTED = "#555555"   # darker than the prior #9A9A9A

# Per-rank colors for the top-3 fluor channels (mAP-descending order).
# Same palette is used for the row labels on the left side of the
# image grid so the eye can follow channel identity from bar → row.
CHANNEL_RANK_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
PHASE_LABEL_COLOR = "#000000"

# Organelle → color palette for the stripe row above each cell. Keyed
# on lowercased organelle prefix from the marker name; falls back to
# grey for anything not listed. Hand-picked so co-located organelles
# read as similar hues (ER cluster = warm reds; nuclear cluster =
# purples; endosomes = blues; etc.).
ORGANELLE_PALETTE = {
    "phase":                 "#444444",
    "5xupre":                "#F08C2E",
    "chromalive":            "#888888",
    # ER / Golgi cluster — warm reds → oranges
    "er":                    "#E07B7B",
    "endoplasmic reticulum": "#E07B7B",
    "er/golgi":              "#E89E50",
    "er/golgi cop-ii":       "#EE8A30",
    "er/golgi bridge":       "#EFB280",
    "trans-golgi":           "#F0C480",
    # Mitochondria
    "mitochondria":          "#B0413E",
    # Nuclear cluster — purples
    "nuclei":                "#9B6BB8",
    "nucleus":               "#9B6BB8",
    "nucleolus":             "#7A4DAD",
    "nucleolus-dfc":         "#7A4DAD",
    "nucleolus-gc":          "#9F60C2",
    "nuclear speckles":      "#B388CB",
    "chromatin":             "#3A4A8C",
    "laminin":               "#5C5C99",
    # Vesicular / endo-membrane — blues / teals
    "autophagosome":         "#6BAA52",
    "lysosome":              "#4FA59B",
    "endosome":              "#5BAEDB",
    "early endosome":        "#7BC0E0",
    "late endosome":         "#5390BD",
    "recycling endosome":    "#85C5DB",
    "endocytic vesicle ph":  "#9DD0E5",
    "clathrin vesicles":     "#A8D5DE",
    "lipid droplet":         "#E8C547",
    # Proteostasis
    "proteasome":            "#7A5230",
    "chaperones":            "#A0826A",
    # Other organelles
    "peroxisome":            "#D58CB1",
    "actin filament":        "#C04ABA",
    "f-actin":               "#A03A99",
    "microtubules":          "#9D3D9F",
    "plasma membrane":       "#A2D87C",
    # Live-cell stains / functional readouts
    "fe2+":                  "#7B4F2A",
    "caspase activity":      "#5C5C5C",
    "oxidative stress":      "#5C5C5C",
    "cell proliferation marker": "#7B7B7B",
    # 4i + CP fallbacks (gene-name-only marker labels)
    "4i":                    "#2D5F3F",
    "cp":                    "#5C7BA8",
}
DEFAULT_ORGANELLE_COLOR = "#BBBBBB"


def _organelle_key(marker_display_name):
    """Extract organelle prefix from a marker name for palette lookup.

    Examples:
      "ER, NCLN"                 -> "er"
      "ER/Golgi COP-II_SEC23A"   -> "er/golgi cop-ii"
      "autophagosome, ATG101"    -> "autophagosome"
      "p53 (4i)"                 -> "4i"
      "f-actin, Phalloidin (cp)" -> "cp"   (CP/4i suffix wins over organelle)
      "Phase"                    -> "phase"
      "5xUPRE"                   -> "5xupre"
      "ChromaLIVE 488 excitation"-> "chromalive"
    """
    import re
    name = str(marker_display_name).strip()
    low = name.lower()
    if low == "phase":
        return "phase"
    if "(4i)" in low:
        return "4i"
    if "(cp)" in low:
        return "cp"
    if "5xupre" in low:
        return "5xupre"
    if "chromalive" in low:
        return "chromalive"
    # "{organelle}, {gene}" or "{organelle}_{gene}" — split on first
    # comma OR underscore, take the prefix.
    parts = re.split(r"[,_]", name, maxsplit=1)
    return parts[0].strip().lower() if parts else low


def _gene_symbol(marker_display_name):
    """Extract a compact gene/probe symbol for horizontal labels.

    Strips well-known dye/probe trailers ("live cell dye", "live-cell
    dye", "excitation", etc.) so labels show only the meaningful
    symbol (e.g. "BODIPY", "FeRhoNox", "LysoTracker") rather than the
    generic "dye" tail.

    Examples:
      "ER, NCLN"                                -> "NCLN"
      "Mitochondria_TOMM20"                     -> "TOMM20"
      "autophagosome, ATG101"                   -> "ATG101"
      "lipid droplet, BODIPY live cell dye"     -> "BODIPY"
      "Fe2+, FeRhoNox live-cell dye"            -> "FeRhoNox"
      "lysosome, LysoTracker live-cell dye"     -> "LysoTracker"
      "actin filament, FastAct_SPY555 Live Cell Dye" -> "FastAct_SPY555"
      "mitochondria, ChromaLIVE 561 excitation" -> "ChromaLIVE 561"
      "p53 (4i)"                                -> "p53"
      "Phase"                                   -> "Phase"
      "5xUPRE"                                  -> "5xUPRE"
      "ChromaLIVE 488 excitation"               -> "ChromaLIVE 488"
    """
    name = str(marker_display_name).strip()
    low = name.lower()
    if low == "phase":
        return "Phase"
    if "(4i)" in name:
        return name.split("(")[0].strip()
    if "(cp)" in low:
        return name.split("(")[0].strip()

    # Strip common dye/probe trailers (case-insensitive) BEFORE picking
    # the last comma/underscore segment, so the symbol isn't "dye" or
    # "excitation". Order matters: longest first so " live-cell dye"
    # beats " live cell" alone.
    SUFFIXES = (
        " Live Cell Dye", " Live-Cell Dye",
        " live cell dye", " live-cell dye",
        " Live Cell dye", " live Cell dye",
        " live cell dyes", " Live Cell Dyes",
        " live cell stain", " Live Cell Stain",
        " excitation",
    )
    for suf in SUFFIXES:
        if name.lower().endswith(suf.lower()):
            name = name[: -len(suf)].strip()
            break

    # Prefer the comma separator (organelle, probe form) since that's
    # explicitly the "{organelle}, {probe}" delimiter Alex's CSV uses.
    if "," in name:
        return name.split(",")[-1].strip()
    # No comma — fall back to splitting on underscore.
    if "_" in name:
        return name.split("_")[-1].strip()
    return name


def _render_map_bar(bar_ax, *, marker_order, marker_display,
                    full_row, selected_keys, selected_colors=None):
    """Draw the page-top mAP heatmap bar with rotated gene-symbol
    labels below.

    Layout (top → bottom):
      • mAP heatmap — magma colormap of the mAP score for this
        gene/complex across all 57 markers.
      • Red border around the 4 selected markers (Phase + top-3 fluor).
      • Rotated 45° gene-symbol labels below — every marker named
        with just the gene/probe symbol (e.g. "TOMM70A" instead of
        "mitochondria, TOMM70A"); long symbols wrap after 10 chars.
        Selected fluor (3): bold + channel-rank color, larger font.
        Phase: bold black. Unselected: light grey, tiny.
    """
    import textwrap as _tw
    from matplotlib.patches import Rectangle

    if not marker_order:
        bar_ax.set_axis_off()
        return

    full_row = full_row or {}
    selected_set = set(selected_keys or [])
    selected_colors = dict(selected_colors or {})
    marker_display = marker_display or {}
    n = len(marker_order)

    # Defensive fallback: if the caller didn't supply selected_colors
    # (e.g., legacy task pickle predates the change), assign the
    # CHANNEL_RANK_COLORS palette in selected-key encounter order so
    # the bar/row label color-coding still works visually.
    if not selected_colors and selected_set:
        is_phase_key = _norm_channel_key("Phase")
        fluor_idx = 0
        for k in marker_order:
            if k not in selected_set:
                continue
            if k == is_phase_key:
                selected_colors[k] = PHASE_LABEL_COLOR
            else:
                selected_colors[k] = CHANNEL_RANK_COLORS[
                    min(fluor_idx, len(CHANNEL_RANK_COLORS) - 1)
                ]
                fluor_idx += 1

    # mAP bar — main cell row.
    values = np.array(
        [full_row.get(k, np.nan) if full_row else np.nan for k in marker_order],
        dtype=float,
    )
    if np.all(np.isnan(values)):
        bar_ax.imshow(
            np.zeros((1, n), dtype=float),
            aspect="auto", cmap="Greys", vmin=0, vmax=1,
            extent=(-0.5, n - 0.5, -0.5, 0.5),
        )
    else:
        # Lock vmin=0, vmax=1 across all pages so the colormap is
        # comparable page-to-page — a "bright" cell on one gene means
        # the same mAP magnitude as a "bright" cell on another.
        # Per-page autoscaling (previous behavior) made low-mAP pages
        # look as colorful as high-mAP ones, defeating the comparison.
        plot_vals = np.where(np.isnan(values), 0.0, values)
        bar_ax.imshow(
            plot_vals.reshape(1, n),
            aspect="auto", cmap=MAP_BAR_CMAP, vmin=0.0, vmax=1.0,
            extent=(-0.5, n - 0.5, -0.5, 0.5),
        )

    # Red borders around the selected (Phase + top-3 fluor) cells.
    for i, key in enumerate(marker_order):
        if key in selected_set:
            bar_ax.add_patch(Rectangle(
                (i - 0.5, -0.5), 1.0, 1.0,
                fill=False, edgecolor="#FF2D2D", linewidth=1.4,
            ))

    # Gene-symbol labels below the bar — rotated 45° (less steep than
    # the previous 75° so they read more naturally) with the symbol
    # only ("TOMM70A", not "mitochondria, TOMM70A"). Long symbols wrap
    # after 10 chars onto a 2nd line — `break_long_words=False` +
    # `break_on_hyphens=False` so multi-token symbols like
    # "Live Cell Dye" or "TOM-20" stay on whole-word boundaries
    # rather than splitting mid-word.
    is_phase_key = _norm_channel_key("Phase")
    for i, key in enumerate(marker_order):
        disp_full = marker_display.get(key, key)
        symbol = _gene_symbol(disp_full)
        wrapped = "\n".join(_tw.wrap(
            symbol, width=10,
            break_long_words=False, break_on_hyphens=False,
        )) or symbol
        if key == is_phase_key:
            color = PHASE_LABEL_COLOR
            fs = MAP_BAR_LABEL_FONT_PHASE
            weight = "bold"
        elif key in selected_set:
            color = selected_colors.get(key, "#111")
            fs = MAP_BAR_LABEL_FONT_SELECTED
            weight = "bold"
        else:
            color = MAP_BAR_LABEL_COLOR_UNSELECTED
            fs = MAP_BAR_LABEL_FONT_UNSELECTED
            weight = "normal"
        bar_ax.text(
            i, -0.7, wrapped,
            ha="right", va="top", rotation=45, rotation_mode="anchor",
            fontsize=fs, fontweight=weight, color=color,
        )

    bar_ax.set_xticks([])
    bar_ax.set_yticks([])
    for s in bar_ax.spines.values():
        s.set_visible(False)
    # Bold "mAP" anchor label at the bar's left edge.
    bar_ax.text(
        -0.7, 0.0, "mAP",
        ha="right", va="center",
        fontsize=MAP_BAR_LABEL_FONT_SELECTED, fontweight="bold", color="#222",
    )


def build_shap_factories(task):
    """Worker entry point referenced by `hook_factory_fqn`.

    Builds (fig_factory, post_render_hook) closures so the existing
    `render_gene_page` can render its image grid into the LEFT half of
    a wider figure, while we render the SHAP lollipops + captions into the
    RIGHT half AFTER the cells are drawn. Also draws the mAP heatmap bar
    at the very top of the page.

    Page height scales with the per-gene fluor-channel count K (from
    `task["n_fluor_channels"]`): the image grid has 2 + 2K rows, and
    the figure height scales linearly from K=1 (~6") to K=10 (~50").
    The SHAP grid has 1 + K panels (Phase + each fluor) mirroring the
    image rows.
    """
    shap_data = task.get("shap_data") or {}
    panels_holder = []  # captured by both closures
    map_bar_holder = []

    # Resolve per-page fluor-channel count. Override > task field > default.
    n_fluor = task.get("n_fluor_channels")
    if n_fluor is None:
        # Derive from shap_data if available (`channels` includes Phase).
        chans = shap_data.get("channels") or []
        n_fluor = max(1, sum(1 for c in chans if str(c).lower() != "phase")) if chans else aa.N_FLUOR_CHANNELS_PER_GENE
    n_total_rows_page = 2 + 2 * n_fluor
    # Page height: 2.25" per image row (matches atlas.py's standalone
    # render). 3-row baseline: 2 phase + 2*3 fluor = 8 rows × 2.25 = 18".
    # K=1 → 9", K=10 → 49.5".
    fig_height_page = n_total_rows_page * 2.25

    def fig_factory():
        fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_height_page))

        # IMAGE GRID — top edge sits well below the rotated marker
        # labels under the bar so the labels have room without
        # crowding the first row of cell tiles.
        #
        # AT BASELINE (FH=18, 3 fluor channels):
        #   suptitle           0.995
        #   ontology line      0.978 top (italic, wraps to 2-3 lines)
        #   mAP bar            0.948-0.956   (= 0.792"-0.936" from top)
        #   gene-symbol labels (45° rotated, wrap@10)  ~0.918-0.943
        #   ←── ~0.86" gap to image-grid top ──→
        #   image grid top     0.900           (= 1.80" from top)
        #
        # For taller variable pages we anchor the bar + img_top in
        # ABSOLUTE INCHES from the top of the figure (rather than
        # constant fig-fraction). Without this the 0.048 fig-frac gap
        # balloons proportionally with page height — a 50" cohesin
        # page would put the bar 2.4" above the Phase row. The
        # inch-offset rebase keeps that gap visually identical to the
        # 3-channel baseline regardless of n_fluor.
        BAR_BOTTOM_FROM_TOP_IN = 0.936   # = (1 - 0.948) * 18
        BAR_TOP_FROM_TOP_IN    = 0.792   # = (1 - 0.956) * 18
        IMG_TOP_FROM_TOP_IN    = 1.80    # = (1 - 0.900) * 18
        bar_bottom_frac = 1.0 - BAR_BOTTOM_FROM_TOP_IN / fig_height_page
        bar_top_frac    = 1.0 - BAR_TOP_FROM_TOP_IN    / fig_height_page
        img_top         = 1.0 - IMG_TOP_FROM_TOP_IN    / fig_height_page

        img_gs = fig.add_gridspec(
            n_total_rows_page, aa.N_COLS,
            left=0.025 * 19.0 / FIG_WIDTH_IN,
            right=IMG_GRID_RIGHT - 0.005 * 19.0 / FIG_WIDTH_IN,
            top=img_top, bottom=0.01,
            hspace=0.18, wspace=0.06,
        )
        axes = np.empty((n_total_rows_page, aa.N_COLS), dtype=object)
        for r in range(n_total_rows_page):
            for c in range(aa.N_COLS):
                axes[r, c] = fig.add_subplot(img_gs[r, c])

        # mAP HEATMAP BAR — single horizontal strip above the image grid.
        # Spans the same x range as the image grid so the visual columns
        # line up. Uses the height-aware fractions computed above so
        # the bar↔Phase-row gap stays constant in inches across page
        # heights instead of growing with fig_height_page.
        map_bar_ax = fig.add_axes([
            MAP_BAR_LEFT, bar_bottom_frac,
            MAP_BAR_RIGHT - MAP_BAR_LEFT, bar_top_frac - bar_bottom_frac,
        ])
        map_bar_holder.append(map_bar_ax)

        # SHAP GRID — separate, independent gridspec on the right side. 4
        # panels stacked vertically, each spanning 2 image rows for KO/NTC
        # alignment. Within each panel: caption (top), lollipop chart
        # (middle), bottom whitespace gutter (no axes). Top edge tracks
        # the image grid so SHAP panels stay aligned with image rows.
        shap_gs = fig.add_gridspec(
            n_total_rows_page, 1,
            left=SHAP_GRID_LEFT,
            right=SHAP_GRID_RIGHT,
            top=img_top, bottom=0.030,
            hspace=0.18,
        )
        panels = []
        # Phase panel + 1 panel per fluor channel — each panel spans 2
        # image rows (KO + NTC). Total panels = (n_total_rows_page // 2).
        n_shap_panels = n_total_rows_page // 2
        for i in range(n_shap_panels):
            panel_gs = shap_gs[2 * i : 2 * i + 2, 0].subgridspec(
                3, 1,
                height_ratios=list(SHAP_HEIGHT_RATIOS),
                hspace=0.18,
            )
            cap_ax = fig.add_subplot(panel_gs[0])
            bar_ax = fig.add_subplot(panel_gs[1])
            panels.append({"bar": bar_ax, "cap": cap_ax})
        panels_holder.append(panels)
        return fig, axes

    def post_render_hook(fig, axes):
        if not panels_holder:
            return
        panels = panels_holder[0]
        gene = str(shap_data.get("gene") or "")
        channels = shap_data.get("channels") or []
        features = shap_data.get("features") or {}
        captions = shap_data.get("caption") or {}
        auroc = shap_data.get("auroc") or {}
        n_cells = shap_data.get("n_cells") or {}
        map_per_ch = shap_data.get("mAP_per_channel") or {}

        # mAP heatmap bar at the top of the page.
        if map_bar_holder:
            _render_map_bar(
                map_bar_holder[0],
                marker_order=shap_data.get("mAP_marker_order") or [],
                marker_display=shap_data.get("mAP_marker_display") or {},
                full_row=shap_data.get("mAP_full_row") or {},
                selected_keys=shap_data.get("mAP_selected_keys") or [],
                selected_colors=shap_data.get("mAP_selected_colors") or {},
            )

        # Lead each panel title with "{gene} geneKO" so every panel
        # caption is anchored by the gene the page is about. The
        # per-channel slice from the captions CSV strips the "{gene}
        # geneKO:" prefix during parsing, so panels would otherwise
        # have no gene-name anchor of their own. NTC/median atlases
        # override this via shap_data["title_prefix"] (e.g.
        # "{gene} KO vs NTC  ").
        gene_prefix = shap_data.get("title_prefix") or (
            f"{gene} geneKO  " if gene else ""
        )

        for i, panel in enumerate(panels):
            ch = channels[i] if i < len(channels) else None
            if ch is None:
                _render_violin(panel["bar"], [], "(channel n/a)")
                _render_caption(panel["cap"], "",
                                title=f"{gene_prefix}(channel n/a)")
                continue
            # Append "mAP=X.XX" on a NEW LINE below the channel name —
            # keeps the long marker names readable on their own row and
            # makes the mAP / AUROC numbers visually attached but
            # separated from the cohort identity above.
            display_title = f"{gene_prefix}{ch}"
            m = map_per_ch.get(ch)
            if m is not None and not (isinstance(m, float) and m != m):
                display_title = f"{display_title}\nmAP={float(m):.2f}"
            _render_violin(
                panel["bar"],
                features.get(ch, []),
                display_title,
                auroc=auroc.get(ch),
            )
            _render_caption(
                panel["cap"], captions.get(ch, ""),
                title=display_title, auroc=auroc.get(ch),
            )

    return fig_factory, post_render_hook


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
HOOK_FACTORY_FQN = "attention_atlas_shap.build_shap_factories"


def main():
    parser = aa._make_arg_parser()
    parser.add_argument(
        "--contrast", choices=("distinct", "ntc", "global"), default="distinct",
        help="Which atlas variant to render. `distinct` (default) = the "
             "original SHAP-on-attention atlas (`ko_shap_features.csv`); "
             "`ntc` = KO-vs-NTC contrast from `ntc_shap_features.csv` "
             "(rows where contrast=='ntc'); `global` = KO-vs-cohort "
             "(rows where contrast=='global'). When --contrast in {ntc, "
             "global} AND --shap-features-csv / --shap-captions-csv are "
             "left at their defaults, both auto-swap to the ntc_v2 paths.",
    )
    parser.add_argument(
        "--bg-all-cells-dir", type=Path,
        default=Path("/hpc/projects/icd.fast.ops/models/alex_lin_attention/all_cells_v2"),
        help="(--contrast ntc/global) Directory of all_cells_<channel>.h5ad "
             "files. Used to sample the SHARED NTC / global cell reference "
             "for the violin BG. Cells are sampled ONCE per (contrast, "
             "channel) into `bg_refs/` and re-used across every atlas run.",
    )
    parser.add_argument(
        "--bg-refs-dir", type=Path, default=None,
        help="(--contrast ntc/global) Override the shared BG reference "
             "cache dir. Defaults to <bg-all-cells-dir>/../bg_refs/.",
    )
    # Default points at gene-level paths; auto-swaps to the `_chad_`
    # variants in `apply_aggregation_level_defaults` when
    # --aggregation-level complex.
    parser.add_argument(
        "--ntc-pma-phase-csv", type=Path,
        default=Path("/hpc/projects/icd.fast.ops/models/alex_lin_attention/"
                     "v3/attention_v3/pma_top_phase_cells_ntc_v3.csv"),
        help="(--contrast ntc) PMA top-attention NTC phase cells CSV. "
             "When present, the violin's BG sample is drawn from these "
             "high-attention NTC cells (joined to all_cells_phase.h5ad "
             "by (experiment, segmentation), top-N by attention rank) "
             "instead of a uniform random NTC sample. Pass an empty "
             "string to disable. Default auto-swaps to the chad_ntc "
             "variant at --aggregation-level complex.",
    )
    parser.add_argument(
        "--ntc-pma-fluor-csv", type=Path,
        default=Path("/hpc/projects/icd.fast.ops/models/alex_lin_attention/"
                     "v3/attention_v3/pma_top_fluorescent_cells_ntc_v3.csv"),
        help="(--contrast ntc) PMA top-attention NTC fluor cells CSV "
             "(same role as --ntc-pma-phase-csv but for fluor channels; "
             "filtered per row's `channel` column). Default auto-swaps "
             "to the chad_ntc variant at --aggregation-level complex.",
    )
    parser.add_argument(
        "--shap-features-csv", type=Path,
        default=DEFAULT_SHAP_FEATURES_CSV,
        help=f"SHAP features CSV (default for --contrast distinct: "
             f"{DEFAULT_SHAP_FEATURES_CSV}; for ntc/global: "
             f"{NTC_DEFAULT_FEATURES_CSV}).",
    )
    parser.add_argument(
        "--shap-captions-csv", type=Path,
        default=DEFAULT_SHAP_CAPTIONS_CSV,
        help=f"SHAP captions CSV (default for --contrast distinct: "
             f"{DEFAULT_SHAP_CAPTIONS_CSV}; for ntc/global: "
             f"ntc_shap_captions_<contrast>.csv under {NTC_BASE_DIR}).",
    )
    parser.add_argument(
        "--shap-cache-phase", type=Path,
        default=DEFAULT_CACHE_PHASE,
        help="Phase SHAP cache dir (X.npy/obs.parquet/...). When present, "
             "each top feature gets a horizontal violin instead of a lollipop "
             f"bar. Default: {DEFAULT_CACHE_PHASE}",
    )
    parser.add_argument(
        "--shap-cache-fluor", type=Path,
        default=DEFAULT_CACHE_FLUOR,
        help=f"Fluor SHAP cache dir. Default: {DEFAULT_CACHE_FLUOR}",
    )
    parser.add_argument(
        "--marker-map-csv", type=Path,
        default=DEFAULT_MARKER_MAP_CSV,
        help="Genes × markers mAP distinctiveness matrix from "
             "gene_best_marker_assignment.py. Drives top-3 fluor channel "
             "selection (mAP-descending) and the page-top mAP heatmap "
             "bar. Pass `--marker-map-csv NONE` to disable both. "
             f"Default: {DEFAULT_MARKER_MAP_CSV.name}",
    )
    parser.add_argument(
        "--chad-complex-config", type=Path,
        default=DEFAULT_CHAD_COMPLEX_CONFIG,
        help="CHAD positive-controls YAML used to roll the gene-level "
             "mAP matrix up to complex level (mean across member genes) "
             "when --aggregation-level complex. Mismatches with the "
             "consolidator's CHAD config (which writes obs.gene = "
             "complex name into the SHAP h5ad) silently fall back to "
             "the legacy first-N channel order for any complex absent "
             f"from this map. Default: {DEFAULT_CHAD_COMPLEX_CONFIG.name}",
    )
    parser.add_argument(
        "--top-fluor-channels", type=int,
        default=DEFAULT_TOP_FLUOR_CHANNELS,
        help=f"Legacy fixed-K mode: render this many top-mAP fluor "
             f"channels per page (Phase always shown additionally). "
             f"Used only when --map-channel-threshold = 0. Default: "
             f"{DEFAULT_TOP_FLUOR_CHANNELS}.",
    )
    parser.add_argument(
        "--map-channel-threshold", type=float, default=0.0,
        help="Per-channel mAP threshold for dynamic page sizing (opt-in). "
             "When > 0, pages render ALL fluor channels with mAP >= "
             "threshold up to --max-fluor-channels, and page height "
             "scales with K per page. Default 0 (off) — fall back to "
             "the fixed --top-fluor-channels=3 mode that aligns with "
             "each gene's pma top-attention channels. Pass e.g. 0.2 to "
             "enable dynamic mode.",
    )
    parser.add_argument(
        "--max-fluor-channels", type=int, default=10,
        help="Upper cap on fluor channels rendered per page when "
             "--map-channel-threshold > 0. A high-signal complex with "
             "30+ above-threshold markers still caps here to keep the "
             "page from growing unboundedly. Default 10.",
    )
    parser.add_argument(
        "--threshold-map", type=float, default=0.2,
        help="Drop genes/complexes whose all-markers-combined mAP "
             "(distinctiveness for gene-level, consistency for complex-"
             "level) is below this threshold. Default 0.2. Set to 0 to "
             "disable. Used together with --threshold-acc — a page must "
             "pass BOTH filters to be rendered.",
    )
    parser.add_argument(
        "--threshold-acc", type=float, default=0.8,
        help="Drop genes/complexes whose attention-model top1_acc "
             "(max of phase + fluor classifiers, from --eval-csv at "
             "--eval-n-cells) is below this threshold. Default 0.8 "
             "(= model picks the right perturbation 80%+ of the time "
             "with EITHER modality). Set to 0 to disable. Supersedes "
             "--threshold for atlas_shap usage.",
    )
    # ── SHAP-prep parallelization (fan `load_shap_data` over SLURM) ──
    parser.add_argument(
        "--shap-prep-mode", choices=("auto", "inline", "slurm"), default="auto",
        help="How to compute per-gene SHAP payloads. `inline` (legacy) "
             "runs the per-gene loop on the login node (slow: ~1001 genes "
             "× 4 channels × 5 features × an O(N_cells) pandas mask per "
             "(gene,channel) on an 8.4M-row obs for NTC caches). `slurm` "
             "shards genes across an array job. `auto` (default) picks "
             "inline when --local OR --genes is a small (<=10) explicit "
             "list; otherwise slurm.",
    )
    parser.add_argument(
        "--shap-prep-genes-per-shard", type=int, default=50,
        help="Genes per SHAP-prep SLURM task. Smaller = more array tasks "
             "but better load balancing. Default 50.",
    )
    parser.add_argument(
        "--shap-prep-cpus", type=int, default=4,
        help="cpus_per_task for SHAP-prep SLURM jobs (default 4).",
    )
    parser.add_argument(
        "--shap-prep-mem", default="32G",
        help="Memory per SHAP-prep SLURM task. NTC fluor caches mmap an "
             "8.4M-row X array; defaults to 32G to leave headroom. "
             "Default 32G.",
    )
    parser.add_argument(
        "--shap-prep-timeout-min", type=int, default=45,
        help="SHAP-prep SLURM task timeout (minutes). Default 45.",
    )
    args = parser.parse_args()
    # Wire --threshold-acc into the legacy --threshold path the renderer
    # already consumes (`attention_atlas.run_atlas` filters genes by
    # `top1_acc >= threshold`). User-set --threshold takes precedence;
    # otherwise --threshold-acc default (0.66) becomes effective.
    #
    # Exception: at --aggregation-level complex, the accuracy signal is
    # per-gene (cdino_eval_phase_50.csv keys = gene symbols), so it has
    # no entries for CHAD complex names. Forcing the threshold here
    # would drop every complex (top1_acc lookup returns 0). Auto-zero
    # the threshold for complex-level runs so only --threshold-map
    # gates which complexes render. Users who genuinely want a gene-
    # level accuracy gate at complex level can still pass an explicit
    # `--threshold X` to override.
    if args.threshold == 0.0:
        if getattr(args, "aggregation_level", "gene") == "complex":
            args.threshold = 0.0
            print("  [threshold-acc] aggregation-level=complex — "
                  "accuracy filter disabled (eval CSV is per-gene; "
                  "complexes have no top1_acc lookup).", flush=True)
        else:
            args.threshold = args.threshold_acc

    # For --contrast {ntc, global}, the SHAP CSV carries rows for ALL
    # ~56 channels per gene but the atlas's image tiles come from
    # pma's per-gene top-3 attention channels. If the user opts into
    # dynamic --map-channel-threshold here, the selector may pick
    # mAP-top markers that don't align with pma's per-gene channels →
    # blank fluor image rows. Auto-disable to keep image rows
    # populated; users who explicitly want a >0 threshold for NTC
    # will see this warning.
    if (args.contrast in ("ntc", "global")
            and args.map_channel_threshold > 0):
        print(f"  [channel-threshold] contrast={args.contrast} — "
              f"disabling --map-channel-threshold "
              f"({args.map_channel_threshold} → 0); pma's per-gene "
              f"top-attention channels are limited to 3, so any mAP-"
              f"picked markers outside that set produce blank image "
              f"rows.", flush=True)
        args.map_channel_threshold = 0.0

    # When dynamic threshold is off, the cap should match the legacy
    # fixed-K behavior (--top-fluor-channels=3) — NOT the argparse
    # default of --max-fluor-channels=10. Without this, threshold=0
    # but cap=10 would still pick the gene's mAP-top-10 globally,
    # which for NTC has the same pma-misalignment issue as dynamic
    # mode. Force the cap down to top_fluor_channels here.
    if args.map_channel_threshold == 0:
        args.max_fluor_channels = args.top_fluor_channels

    # `--marker-map-csv NONE` (any case-insensitive variant) disables
    # mAP-driven channel selection AND the page-top heatmap bar — falls
    # back to the legacy first-N viz_channels order with no mAP context.
    if (args.marker_map_csv is not None
            and str(args.marker_map_csv).strip().upper() == "NONE"):
        args.marker_map_csv = None

    # Mirror attention_atlas.main()'s CHAD path auto-swap — without
    # this, --aggregation-level complex on attention_atlas_shap reads
    # the gene-level pma CSVs and renders blank cells.
    aa.apply_aggregation_level_defaults(args)

    # SHAP-specific CHAD swap: cache dirs and SHAP feature/captions
    # CSVs default to gene-level paths (top20_v4/, shap_caches/v4/),
    # but at --aggregation-level complex they need to point at the
    # _chad variants. Without this the atlas mixes a CHAD attention
    # CSV with gene-level SHAP rows + stale violin caches → random
    # lollipop fallbacks for whichever (gene, channel) tuples don't
    # exist in the gene-level cache.
    if getattr(args, "aggregation_level", "gene") == "complex":
        if args.shap_features_csv == DEFAULT_SHAP_FEATURES_CSV:
            args.shap_features_csv = _CHAD_SHAP_FEATURES_CSV
            print(f"[CHAD] swapped --shap-features-csv to: "
                  f"{args.shap_features_csv.name}", flush=True)
        if args.shap_captions_csv == DEFAULT_SHAP_CAPTIONS_CSV:
            args.shap_captions_csv = _CHAD_SHAP_CAPTIONS_CSV
            print(f"[CHAD] swapped --shap-captions-csv to: "
                  f"{args.shap_captions_csv.name}", flush=True)
        if args.shap_cache_phase == DEFAULT_CACHE_PHASE:
            args.shap_cache_phase = _CHAD_CACHE_PHASE
            print(f"[CHAD] swapped --shap-cache-phase to: "
                  f"{args.shap_cache_phase}", flush=True)
        if args.shap_cache_fluor == DEFAULT_CACHE_FLUOR:
            args.shap_cache_fluor = _CHAD_CACHE_FLUOR
            print(f"[CHAD] swapped --shap-cache-fluor to: "
                  f"{args.shap_cache_fluor}", flush=True)

    # Auto-swap default CSVs and SHAP caches for the NTC/median variants
    # when the user didn't pass explicit paths. The NTC fluor cache is a
    # parent dir of per-channel subdirs (each with its own feature schema);
    # `_load_violin_caches` detects this layout and uses lazy per-channel
    # sub-pack loading instead of single-cache mmap.
    framing = CONTRAST_FRAMING[args.contrast]
    if args.contrast in ("ntc", "global"):
        # Two CSV layouts feed this branch:
        #   (a) LEGACY ntc_shap_features.csv (ntc_shap_features.py): one
        #       file with a `contrast` column; positives are ALL cells
        #       for the gene; lives under ntc_v2{,_chad}/. Cache must
        #       point at ntc_caches/ because that's where these
        #       positives' z-stats and X.npy live.
        #   (b) NEW ko_shap_features.csv (--contrast {ntc,global}): per-
        #       contrast file, NO `contrast` column; positives are
        #       top-100 attention cells (identical to distinct), lives
        #       under top20_v4{_chad}_{ntc,global}/. Cache MUST stay at
        #       the top-attention SHAP cache (shap_caches/v4{,_chad}/)
        #       because the positives are the top-attention cells the
        #       distinct atlas reads from.
        is_new_layout = True
        try:
            head = pd.read_csv(args.shap_features_csv, nrows=0)
            is_new_layout = "contrast" not in head.columns
        except Exception:
            pass
        is_chad = getattr(args, "aggregation_level", "gene") == "complex"

        if is_new_layout:
            print(f"[--contrast {args.contrast}] new per-contrast CSV layout "
                  f"detected — keeping top-attention SHAP cache "
                  f"({args.shap_cache_phase}).", flush=True)
        else:
            # Legacy: swap to the per-cell NTC cache as before.
            ntc_feat = _NTC_FEATURES_CHAD_CSV if is_chad else NTC_DEFAULT_FEATURES_CSV
            ntc_caps = (_NTC_CAPTIONS_CHAD if is_chad else NTC_DEFAULT_CAPTIONS)[args.contrast]
            if args.shap_features_csv in (DEFAULT_SHAP_FEATURES_CSV,
                                           _CHAD_SHAP_FEATURES_CSV):
                args.shap_features_csv = ntc_feat
                print(f"[--contrast {args.contrast}] features-csv → "
                      f"{args.shap_features_csv}", flush=True)
            if args.shap_captions_csv in (DEFAULT_SHAP_CAPTIONS_CSV,
                                           _CHAD_SHAP_CAPTIONS_CSV):
                args.shap_captions_csv = ntc_caps
                print(f"[--contrast {args.contrast}] captions-csv → "
                      f"{args.shap_captions_csv}", flush=True)
            ntc_phase = _NTC_CACHE_PHASE_CHAD if is_chad else NTC_DEFAULT_CACHE_PHASE
            ntc_fluor = _NTC_CACHE_FLUOR_CHAD if is_chad else NTC_DEFAULT_CACHE_FLUOR
            if args.shap_cache_phase in (DEFAULT_CACHE_PHASE, _CHAD_CACHE_PHASE):
                args.shap_cache_phase = ntc_phase
                print(f"[--contrast {args.contrast}] cache phase → "
                      f"{args.shap_cache_phase}", flush=True)
            if args.shap_cache_fluor in (DEFAULT_CACHE_FLUOR, _CHAD_CACHE_FLUOR):
                args.shap_cache_fluor = ntc_fluor
                print(f"[--contrast {args.contrast}] cache fluor (multi-channel) → "
                      f"{args.shap_cache_fluor}", flush=True)
        # Swap pma image-cell CSVs to the ntc_pick_cells outputs. The atlas
        # eats these unchanged (same schema), but the cells are
        # SHAP-feature exemplars per (gene, viz_channel) instead of pma's
        # global top-attention rank=1/2/3. Without this swap, NTC pages
        # render with empty fluor image rows because pma's per-gene
        # channel set (typically ChromaLIVE only) doesn't overlap with
        # the mAP-driven channel selection used by the renderer.
        # No image-cell CSV swap: all three contrasts render the SAME
        # pma top-attention FOVs. The contrast only changes WHICH SHAP
        # CSV (and therefore which features / captions / bg pool) is
        # used; the rendered cells are constant so a reader can compare
        # the three views side-by-side without the image grid shifting
        # underneath them. (The deprecated ntc_picked_* swap caused
        # empty image rows when those CSVs weren't generated; dropped
        # entirely.)
        print(
            f"[--contrast {args.contrast}] reading {args.shap_features_csv.name} "
            f"+ {args.shap_captions_csv.name}",
            flush=True,
        )

    print("Loading SHAP data...", flush=True)
    # When --genes is passed (smoke tests), filter the SHAP data load to
    # those genes only — avoids iterating all 1000 genes' violin+bg
    # extraction when we'll render just one. For full atlas runs (no
    # --genes filter) we still load everything; that's fine since the
    # login-node prep happens once before the SLURM submit.
    gene_filter = set(args.genes) if args.genes else None
    features_csv_for_load = args.shap_features_csv
    if args.contrast in ("ntc", "global"):
        # Two CSV layouts feed this path:
        #   (a) ntc_shap_features.csv (legacy, from ntc_shap_features.py)
        #       — one combined CSV with both contrasts; row-filter on the
        #       `contrast` column.
        #   (b) ko_shap_features.csv from
        #       `ko_shap_features.py --contrast {ntc,global}` (new) — a
        #       per-contrast file in its own out-dir, NO `contrast` column.
        #       Used as-is.
        import tempfile
        feat_df = pd.read_csv(args.shap_features_csv)
        if "contrast" in feat_df.columns:
            sub = feat_df[feat_df["contrast"].astype(str) == args.contrast]
            if not len(sub):
                raise SystemExit(
                    f"{args.shap_features_csv}: no rows with "
                    f"contrast='{args.contrast}'."
                )
            tmp_dir = Path(tempfile.mkdtemp(prefix=f"atlas_{args.contrast}_"))
            features_csv_for_load = tmp_dir / f"features_{args.contrast}.csv"
            sub.to_csv(features_csv_for_load, index=False)
        else:
            # Per-contrast CSV from the new ko_shap_features path — the
            # whole file is for this contrast already.
            print(
                f"  [--contrast {args.contrast}] no 'contrast' column — "
                f"treating CSV as per-contrast (from "
                f"ko_shap_features --contrast {args.contrast}).",
                flush=True,
            )
        if "contrast" in feat_df.columns:
            print(
                f"  filtered on contrast='{args.contrast}': "
                f"{len(sub):,}/{len(feat_df):,} rows -> {features_csv_for_load}",
                flush=True,
            )

    # Decide whether to run the per-gene loop here (inline, single-thread)
    # or fan it out as a SLURM array. The features CSV already lists all
    # genes; for the auto path we only need its `gene` column to size
    # the shard plan.
    prep_mode = args.shap_prep_mode
    if prep_mode == "auto":
        # Inline when local rendering OR when the user pinned a small
        # explicit gene list (round-tripping through SLURM is dead
        # weight for <=10 genes — the prep finishes in a few seconds).
        small_gene_list = (gene_filter is not None
                           and len(gene_filter) <= 10)
        prep_mode = "inline" if (args.local or small_gene_list) else "slurm"
        print(f"  [shap-prep] mode=auto resolved to '{prep_mode}'"
              f" (local={args.local}, "
              f"|--genes|={len(gene_filter) if gene_filter else 'all'})",
              flush=True)

    if prep_mode == "slurm":
        # Peek at the features CSV's `gene` column to size the shard
        # plan. Apply gene_filter when set so we don't ship empty
        # shards through SLURM.
        gene_list = (
            pd.read_csv(features_csv_for_load, usecols=["gene"])
              ["gene"].astype(str).unique().tolist()
        )
        if gene_filter is not None:
            gene_list = [g for g in gene_list if g in gene_filter]
        n_shards = (len(gene_list) + args.shap_prep_genes_per_shard - 1) \
                   // args.shap_prep_genes_per_shard
        print(f"  [shap-prep] sharding {len(gene_list):,} genes "
              f"into {n_shards} SLURM tasks", flush=True)
        # SHAP-prep workers live in this module (__main__ when run as a
        # script), so cloudpickle inlines them automatically — no
        # register_pickle_by_value needed for this stage.
        shap_data_per_gene = _prep_shap_data_slurm(args, gene_list)
    else:
        shap_data_per_gene = load_shap_data(
            features_csv_for_load, args.shap_captions_csv,
            cache_phase=args.shap_cache_phase, cache_fluor=args.shap_cache_fluor,
            gene_filter=gene_filter,
            marker_map_csv=args.marker_map_csv,
            mAP_channel_threshold=args.map_channel_threshold,
            max_fluor_channels=args.max_fluor_channels,
            aggregation_level=getattr(args, "aggregation_level", "gene"),
            chad_config=args.chad_complex_config,
            top_fluor_channels=args.top_fluor_channels,
            contrast=args.contrast,
            bg_all_cells_dir=args.bg_all_cells_dir,
            bg_refs_dir=args.bg_refs_dir,
            ntc_pma_phase_csv=(args.ntc_pma_phase_csv
                                if str(args.ntc_pma_phase_csv) else None),
            ntc_pma_fluor_csv=(args.ntc_pma_fluor_csv
                                if str(args.ntc_pma_fluor_csv) else None),
        )
    print(f"  Loaded SHAP data for {len(shap_data_per_gene):,} genes", flush=True)

    # --threshold-map filter: drop genes/complexes whose all-combined mAP
    # is below the threshold. Pages with mAP below the threshold are
    # essentially noise (no consistent perturbation signal vs the
    # cohort), so rendering them just dilutes the atlas. Accuracy
    # filtering happens downstream via the legacy --threshold path.
    if args.threshold_map and args.threshold_map > 0:
        before = len(shap_data_per_gene)
        kept = {
            g: p for g, p in shap_data_per_gene.items()
            if isinstance(p.get("mAP_all_combined"), (int, float))
            and not (p["mAP_all_combined"] != p["mAP_all_combined"])  # NaN check
            and p["mAP_all_combined"] >= args.threshold_map
        }
        dropped = before - len(kept)
        metric_label = (
            "consist" if getattr(args, "aggregation_level", "gene") == "complex"
            else "distinct"
        )
        print(f"  --threshold-map {args.threshold_map}: kept "
              f"{len(kept):,}/{before:,} pages with mAP_{metric_label} "
              f">= {args.threshold_map} (dropped {dropped:,})", flush=True)
        shap_data_per_gene = kept

    # Plant the per-gene panel-title prefix when the variant overrides
    # the default "{gene} geneKO" framing. The renderer's
    # `build_shap_factories` reads shap_data["title_prefix"] when set.
    extra_per_gene = {}
    for gene, payload in shap_data_per_gene.items():
        if framing is not None:
            payload["title_prefix"] = f"{gene} {framing}  "
        extra_per_gene[gene] = {"shap_data": payload}

    # Constrain the renderer's gene set to what survived the mAP filter.
    # When the user passed `--genes` explicitly, intersect; otherwise
    # take the post-filter set.
    surviving = list(extra_per_gene.keys())
    if args.genes:
        args.genes = [g for g in args.genes if g in extra_per_gene]
        if not args.genes:
            print("  [filter] no genes survived --threshold-map AND --genes "
                  "intersection — nothing to render. Exiting.", flush=True)
            return
    else:
        args.genes = surviving

    if not args.local:
        # SLURM workers don't have `attention_atlas` on their PYTHONPATH —
        # without this they fail at pickle-load time (before our in-worker
        # `sys.path.insert` can run) because cloudpickle serialized the
        # `attention_atlas._slurm_render_chunk` reference by module name.
        # `register_pickle_by_value` flips it to inline bytecode, which is
        # how the bare attention_atlas.py SLURM mode works (functions there
        # live in `__main__`, which cloudpickle always inlines).
        import cloudpickle
        cloudpickle.register_pickle_by_value(aa)

    if args.local:
        aa._run_atlas_local(
            phase_csv=args.phase_csv,
            fluor_csv=args.fluor_csv,
            output=args.output,
            crop_size=args.crop_size,
            genes_subset=args.genes,
            max_genes=args.max_genes,
            skip=args.skip,
            dpi=args.dpi,
            workers=args.workers,
            keep_pages=args.keep_pages,
            ntc_max_experiments=args.ntc_experiments,
            ntc_max_per_experiment=args.ntc_per_experiment,
            skip_ntc=args.no_ntc,
            strict=args.strict,
            skip_cp_4i=args.skip_cp_4i,
            extra_per_gene_data=extra_per_gene,
            hook_factory_fqn=HOOK_FACTORY_FQN,
            eval_csv=getattr(args, "eval_csv", None),
            eval_n_cells=getattr(args, "eval_n_cells", 100),
            threshold=getattr(args, "threshold", 0.0),
            nuclear_overlay=getattr(args, "nuclear_overlay", "seg"),
            aggregation_level=getattr(args, "aggregation_level", "gene"),
            ntc_pma_phase_csv=(args.ntc_pma_phase_csv
                                if str(args.ntc_pma_phase_csv) else None),
            ntc_pma_fluor_csv=(args.ntc_pma_fluor_csv
                                if str(args.ntc_pma_fluor_csv) else None),
        )
    else:
        aa._submit_via_slurm(
            args,
            extra_per_gene_data=extra_per_gene,
            hook_factory_fqn=HOOK_FACTORY_FQN,
        )


if __name__ == "__main__":
    main()
