"""Pick image-row cells for the NTC atlas variant — exemplars matching the
SHAP top-feature means.

For each (gene, viz_channel) pair in the NTC SHAP features CSV, this script
scores every candidate cell by its L2 distance (in z-units) to the gene's
KO group mean across the gene's top-N SHAP features, then emits the K
nearest as image-row cells. Two cell pools are scored:

  * **KO row** — cells with `obs.gene == gene`. Smallest distance → most
    representative KO examples of the SHAP-identified phenotype.
  * **NTC row** — cells with `obs.gene == "NTC"`. Smallest distance to the
    SAME `mu_KO` → "KO-typical NTC" cells (NTCs that visually resemble the
    KO phenotype). Mirrors the original `_build_ntc_pool` strip design.

Output schema is a drop-in replacement for pma_top_*_v3.csv so
`attention_atlas.py` can consume it via `--phase-csv` / `--fluor-csv`.
The two roles (KO / NTC) get separate `rank_type` values:

    gene, viz_channel, channel_rank, rank, rank_type, experiment, well,
    segmentation, x_pheno, y_pheno, model_confidence, predicted_class,
    pma_attention

For NTC strip rows we set `gene == predicted_class == "NTC"` but write the
TARGET gene into `target_gene` so the renderer can group by intended KO.

CLI:
    python ntc_pick_cells.py \\
        --shap-features-csv .../ntc_v2/ntc_shap_features.csv \\
        --cache-dir         .../ntc_caches/v2 \\
        --out-phase-csv     .../ntc_v2/ntc_picked_phase.csv \\
        --out-fluor-csv     .../ntc_v2/ntc_picked_fluor.csv \\
        --contrast ntc --n-cells 30 --top-n-features 5
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Only load the obs columns we actually need. The full schema has ~25
# columns and per-channel obs.parquet is 100-360 MB — loading everything
# pushes the picker past 5 min/channel. Subset cuts to ~10s/channel.
_OBS_COLS_NEEDED = [
    "gene", "viz_channel", "experiment", "well", "segmentation",
    "x_pheno", "y_pheno", "channel_rank", "predicted_class",
    "pma_attention", "model_confidence", "_pos",
]


def _load_cache(cdir: Path):
    """Load one per-channel cache (mmap-friendly views + feature index)."""
    X = np.load(cdir / "X.npy", mmap_mode="r")
    # Read only the obs columns the picker actually consumes — saves
    # ~100 MB and ~25s per channel.
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(cdir / "obs.parquet")
    available = set(pf.schema.names)
    cols = [c for c in _OBS_COLS_NEEDED if c in available]
    obs = pf.read(columns=cols).to_pandas()
    feats = (cdir / "features.txt").read_text().splitlines()
    fi = {f: i for i, f in enumerate(feats)}
    median = np.load(cdir / "median.npy")
    std = np.load(cdir / "global_std.npy").clip(1e-6)
    return X, obs, fi, median, std


def _read_feature_columns(X, feat_cols):
    """Read selected feature columns from X.npy as ONE contiguous slice per
    column. Column-major reads on a (N_rows, N_features) mmap are
    1000x cheaper than `X[idx, :][:, feat_cols]` for fancy row indexing
    — the row pattern hits non-contiguous mmap pages and on the NTC
    phase cache (8.4M × 2318 × 4 bytes = ~73 GB) drives RSS past 40 GB
    paging the entire array. With column reads we touch only N_features
    × (N_rows × 4 bytes) bytes total, sequential per column.

    Returns ndarray shape (N_rows, len(feat_cols)) — read once,
    re-used across all genes in this channel.
    """
    cols = np.empty((X.shape[0], len(feat_cols)), dtype=np.float32)
    for i, fc in enumerate(feat_cols):
        cols[:, i] = X[:, fc]
    return cols


def _score_cells(X_cached, idx, median_vec, std_vec, mu_ref_z):
    """L2 distance in z-units from `mu_ref_z` for cells `idx` in `X_cached`.

    `X_cached` is the pre-read (N_rows, len(feat_cols)) slice — already
    in RAM, so indexing is fast.
    """
    if len(idx) == 0 or X_cached.shape[1] == 0:
        return np.array([], dtype=np.float64)
    sub = np.asarray(X_cached[idx, :], dtype=np.float64)
    z = (sub - median_vec) / std_vec
    diff = z - mu_ref_z
    diff = np.where(np.isfinite(diff), diff, 0.0)
    return np.sqrt(np.sum(diff * diff, axis=1))


def _viz_to_subdir(cache_root: Path, viz_channel: str) -> Optional[Path]:
    """Resolve a viz_channel value to its per-channel cache subdir.
    NTC features CSV uses values like 'autophagosome_map1lc3b' or
    'er/golgi cop-ii_sec23a'; subdirs use lowercase + underscores +
    no slashes ('er_golgi_cop-ii_sec23a'). Returns None if not found."""
    candidates = [
        viz_channel,
        viz_channel.lower().replace(" ", "_"),
        viz_channel.lower().replace(" ", "_").replace("/", "_"),
    ]
    for c in candidates:
        p = cache_root / c
        if p.is_dir() and (p / "X.npy").exists():
            return p
    return None


def _pick_exemplars_for_channel(
    channel_subdir: Path,
    gene_to_top_feats: dict[str, list[str]],
    n_cells: int,
    chad_map: Optional[dict[str, str]] = None,
):
    """Score and pick exemplars for every gene in this channel.

    Returns list of pandas DataFrame rows (dicts), pma-compatible schema.
    """
    print(f"  loading {channel_subdir.name}...", flush=True)
    X, obs, fi, median_vec, std_vec = _load_cache(channel_subdir)

    # CHAD relabel at iteration time: same logic as ntc_shap_features.
    if chad_map is not None:
        gene_str = obs["gene"].astype(str)
        is_ntc = gene_str == "NTC"
        mapped = gene_str[~is_ntc].map(chad_map)
        new_gene = gene_str.copy()
        new_gene.loc[~is_ntc] = mapped.fillna("__DROP__").values
        obs = obs.assign(gene=new_gene.values)
        keep_mask = obs["gene"].astype(str) != "__DROP__"
        obs = obs[keep_mask].reset_index(drop=True)
        # X indices follow obs indices, but after filter we lose the mapping.
        # Build a positional remap: original row → new row index.
        if "_pos" in obs.columns:
            x_pos = obs["_pos"].to_numpy()
        else:
            # Fallback: assume row order matches X — caller should make sure.
            x_pos = np.arange(len(obs))
    else:
        x_pos = obs["_pos"].to_numpy() if "_pos" in obs.columns else np.arange(len(obs))

    viz_channel = str(obs["viz_channel"].iloc[0]) if len(obs) else channel_subdir.name
    gene_col = obs["gene"].astype(str).to_numpy()

    rows: list[dict] = []
    # NTC pool index (in obs space) — same for every gene in this channel.
    ntc_obs_idx = np.where(gene_col == "NTC")[0]
    ntc_X_idx = x_pos[ntc_obs_idx]

    # Build the union of top-feature columns across all genes in this
    # channel. Reading these columns ONCE upfront (sequentially, one
    # column at a time) avoids the random-row-fancy-index paging
    # explosion on the 73 GB phase mmap.
    union_feats = set()
    for feat_names in gene_to_top_feats.values():
        for f in feat_names:
            if f in fi:
                union_feats.add(f)
    union_cols = sorted(fi[f] for f in union_feats)
    col_lookup = {c: i for i, c in enumerate(union_cols)}
    if not union_cols:
        print(f"    no top features mapped to cache columns; skipping",
              flush=True)
        return rows
    print(f"    reading {len(union_cols)} feature column(s) from "
          f"{X.shape[0]:,}×{X.shape[1]:,} cache...", flush=True)
    X_cached = _read_feature_columns(X, union_cols)
    median_sub = median_vec[union_cols]
    std_sub = std_vec[union_cols]

    for gene, feat_names in gene_to_top_feats.items():
        # Per-gene subset within the union: indices into X_cached's
        # columns axis.
        feat_cols_global = [fi[f] for f in feat_names if f in fi]
        if not feat_cols_global:
            continue
        local_cols = [col_lookup[c] for c in feat_cols_global]
        ko_obs_idx = np.where(gene_col == gene)[0]
        if len(ko_obs_idx) < 5:
            continue
        ko_X_idx = x_pos[ko_obs_idx]

        # KO reference mean in z-units.
        ko_sub = np.asarray(X_cached[ko_X_idx, :][:, local_cols],
                            dtype=np.float64)
        ko_z = (ko_sub - median_sub[local_cols]) / std_sub[local_cols]
        ko_z = np.where(np.isfinite(ko_z), ko_z, np.nan)
        with np.errstate(invalid="ignore"):
            mu_ko = np.nanmean(ko_z, axis=0)
        mu_ko = np.where(np.isfinite(mu_ko), mu_ko, 0.0)

        # KO row: cells closest to KO mean.
        ko_scores = _score_cells(
            X_cached[:, local_cols], ko_X_idx,
            median_sub[local_cols], std_sub[local_cols], mu_ko,
        )
        ko_pick = np.argsort(ko_scores)[: n_cells]
        for r, sel in enumerate(ko_pick, start=1):
            obs_row = obs.iloc[ko_obs_idx[sel]]
            rows.append(_pma_row(obs_row, gene, viz_channel, role="top", rank=r,
                                 target_gene=gene))

        # NTC row: NTC cells closest to KO mean.
        if len(ntc_obs_idx) >= 5:
            ntc_scores = _score_cells(
                X_cached[:, local_cols], ntc_X_idx,
                median_sub[local_cols], std_sub[local_cols], mu_ko,
            )
            ntc_pick = np.argsort(ntc_scores)[: n_cells]
            for r, sel in enumerate(ntc_pick, start=1):
                obs_row = obs.iloc[ntc_obs_idx[sel]]
                rows.append(_pma_row(obs_row, "NTC", viz_channel, role="ntc_ko_typical",
                                     rank=r, target_gene=gene))
    print(f"    {len(rows):,} rows for {channel_subdir.name}", flush=True)
    return rows


def _pma_row(obs_row, gene_value, viz_channel, role, rank, target_gene):
    """Build one pma-schema dict from an obs row.

    `role` is written into `rank_type` so the atlas can route to the
    KO image row (rank_type='ko_top') vs the NTC strip row
    (rank_type='ntc_ko_typical'). `target_gene` is preserved so
    NTC strip cells stay grouped with their target KO page.
    """
    # Normalize well: NTC obs uses "A/1/0", pma format is "A1". The atlas
    # downstream joins on well so we must match pma's flat form. Skip
    # NaN/empty values — leaving them as "" so the atlas drops the row
    # rather than producing the surprising "NAN0".
    well_raw = obs_row.get("well")
    if pd.isna(well_raw) or not str(well_raw).strip() or str(well_raw).lower() == "nan":
        well_flat = ""
    else:
        well_str = str(well_raw)
        parts = well_str.split("/")
        well_flat = parts[0] + parts[1] if len(parts) >= 2 else well_str
    out = {
        "gene": gene_value,
        "viz_channel": viz_channel,
        "channel_rank": int(obs_row.get("channel_rank", 1) or 1),
        "rank": int(rank),
        "rank_type": role,
        "experiment": str(obs_row["experiment"]),
        "well": well_flat,
        "segmentation": int(obs_row["segmentation"]) if pd.notna(obs_row.get("segmentation")) else 0,
        "x_pheno": float(obs_row["x_pheno"]) if pd.notna(obs_row.get("x_pheno")) else 0.0,
        "y_pheno": float(obs_row["y_pheno"]) if pd.notna(obs_row.get("y_pheno")) else 0.0,
        "model_confidence": (
            float(obs_row["model_confidence"])
            if pd.notna(obs_row.get("model_confidence")) else float("nan")
        ),
        "predicted_class": str(obs_row.get("predicted_class", "") or gene_value),
        "pma_attention": (
            float(obs_row["pma_attention"])
            if pd.notna(obs_row.get("pma_attention")) else float("nan")
        ),
        "target_gene": target_gene,
    }
    return out


def _load_chad_map(chad_config_path: Path) -> dict[str, str]:
    import yaml
    with open(chad_config_path) as f:
        config = yaml.safe_load(f)
    gene_to_complex = {}
    for complex_name, info in config.get("complexes", {}).items():
        for g in info.get("genes", []):
            gene_to_complex[str(g)] = str(complex_name)
    return gene_to_complex


def _enrich_with_fluor_seg(rows: pd.DataFrame) -> pd.DataFrame:
    """For every (experiment, well) referenced by `rows`, read the linked-
    pheno CSV (live / CP / 4i variant — picked by the experiment's
    channel_map) and look up `cp_cell_seg_id` / `4i_segmentation_id`,
    writing them into a new `fluor_segmentation` column. Drops rows
    whose mask can't be resolved at all (no valid live OR fluor seg) —
    same filter `_build_ntc_pool` applies for NTC pool cells, so the
    picker's KO+NTC image cells are validated the same way.
    """
    from ops_utils.data.experiment import OpsDataset
    from ops_utils.data.filesystem import resolve_experiment_name
    import sys, os
    # attention_atlas lives a few dirs up the ops_mono tree from this
    # script — climb to the repo root and append the napari path so
    # `import attention_atlas` resolves the right module.
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = here.split("/organelle_profiler/")[0]
    atlas_dir = os.path.join(repo_root, "ops_process", "ops_analysis", "napari")
    if atlas_dir not in sys.path:
        sys.path.insert(0, atlas_dir)
    import attention_atlas as aa

    fluor_seg = pd.Series([pd.NA] * len(rows), index=rows.index, dtype="Int64")
    fluor_y = pd.Series([float("nan")] * len(rows), index=rows.index, dtype="float64")
    fluor_x = pd.Series([float("nan")] * len(rows), index=rows.index, dtype="float64")
    # Per-(experiment, well) cache so we read each linked CSV once.
    seg_cache: dict[tuple[str, str], pd.DataFrame] = {}
    well_norm_cache: dict[str, str] = {}

    def _norm_well(w: str) -> str:
        # Picker's wells are "A1"; the linked CSV expects "A/1" for ds.append_well.
        if w in well_norm_cache:
            return well_norm_cache[w]
        out = f"{w[0]}/{w[1:]}" if len(w) >= 2 and w[1:].isdigit() else w
        well_norm_cache[w] = out
        return out

    grouped = rows.groupby(["experiment", "well"], sort=False)
    print(f"  enriching {grouped.ngroups} unique (experiment, well) pairs "
          f"with CP/4i seg lookups...", flush=True)
    for (exp, well_flat), grp in grouped:
        try:
            ds = OpsDataset(resolve_experiment_name(str(exp)))
        except Exception as e:
            print(f"    [skip] {exp}: resolve failed: {e}", flush=True)
            continue
        cmap = aa._channel_map_for_experiment(str(exp))
        try:
            path, seg_col, fluor_seg_col, y_col, x_col = aa._ntc_linked_paths(
                ds, _norm_well(str(well_flat)), cmap,
            )
        except Exception as e:
            print(f"    [skip] {exp} {well_flat}: linked-paths failed: {e}",
                  flush=True)
            continue
        if fluor_seg_col is None:
            # Live-cell experiment: live-cell segmentation IS the fluor seg.
            # Leave fluor_segmentation as NA; atlas falls back to cell_seg.
            continue
        if not path.exists():
            print(f"    [skip] {exp} {well_flat}: no {path.name}", flush=True)
            continue
        key = (str(exp), str(well_flat))
        if key not in seg_cache:
            try:
                cols = [seg_col, fluor_seg_col, y_col, x_col]
                seg_cache[key] = pd.read_csv(path, usecols=cols)
            except Exception as e:
                print(f"    [skip] {exp} {well_flat}: read failed: {e}",
                      flush=True)
                continue
        link_df = seg_cache[key]
        # Join: row.segmentation == linked.segmentation_id → linked.fluor_seg
        seg_to_fluor = dict(zip(
            pd.to_numeric(link_df[seg_col], errors="coerce").astype("Int64"),
            pd.to_numeric(link_df[fluor_seg_col], errors="coerce").astype("Int64"),
        ))
        # Optional: x/y_pheno_centroid for CP/4i — use those when present.
        seg_to_yx = dict(zip(
            pd.to_numeric(link_df[seg_col], errors="coerce").astype("Int64"),
            zip(link_df[y_col], link_df[x_col]),
        ))
        for idx, row in grp.iterrows():
            seg = row["segmentation"]
            try:
                seg_int = int(seg)
            except (TypeError, ValueError):
                continue
            v = seg_to_fluor.get(seg_int)
            if v is not None and pd.notna(v) and int(v) > 0:
                fluor_seg.loc[idx] = int(v)
                yx = seg_to_yx.get(seg_int)
                if yx is not None and pd.notna(yx[0]) and pd.notna(yx[1]):
                    fluor_y.loc[idx] = float(yx[0])
                    fluor_x.loc[idx] = float(yx[1])

    out = rows.copy()
    out["fluor_segmentation"] = fluor_seg
    # Override x/y with CP/4i centroids when we have them — these match the
    # CP/4i imaging coordinate system, which can drift from the live-cell
    # x/y in 4i wells.
    has_fluor_xy = fluor_y.notna() & fluor_x.notna()
    out.loc[has_fluor_xy, "y_pheno"] = fluor_y.loc[has_fluor_xy].astype(float)
    out.loc[has_fluor_xy, "x_pheno"] = fluor_x.loc[has_fluor_xy].astype(float)

    # Promote CP/4i seg into `segmentation` so the atlas's fluor mask
    # preference (4i_cell_seg → cp_cell_seg → cell_seg) finds the cell
    # in the CP/4i mask label space — same trick as the atlas's NTC
    # pool does (line 2391: `ntc_rows_sub["segmentation"] = ntc_rows_sub
    # ["fluor_segmentation"]`). Without this the live-cell seg gets
    # tried against the CP/4i mask, falls through to cell_seg, and the
    # renderer overlays a Phase-shape mask onto the CP/4i image.
    has_fluor_seg = pd.to_numeric(out["fluor_segmentation"],
                                  errors="coerce") > 0
    out.loc[has_fluor_seg, "segmentation"] = (
        out.loc[has_fluor_seg, "fluor_segmentation"].astype("Int64").astype(int)
    )

    # Validity filter: same as `_build_ntc_pool`. Live-cell rows pass on
    # the (unchanged) live segmentation; CP/4i rows pass on the now-
    # promoted CP/4i seg. Anything still NaN/zero gets dropped so the
    # renderer never tries to crop a cell that has no resolvable mask.
    seg_ok = pd.to_numeric(out["segmentation"], errors="coerce") > 0
    dropped = int((~seg_ok).sum())
    if dropped:
        print(f"  dropped {dropped:,}/{len(out):,} rows with no valid mask "
              f"(no live OR CP/4i seg)", flush=True)
    return out[seg_ok].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--shap-features-csv", type=Path, required=True,
                    help="ntc_shap_features.csv (has gene, viz_channel, contrast, "
                         "shap_rank, feature columns).")
    ap.add_argument("--cache-dir", type=Path, required=True,
                    help="Per-channel cache root (ntc_caches/v2/). Contains "
                         "phase/ and <viz_channel>/ subdirs each with X.npy, "
                         "obs.parquet, features.txt, median.npy, global_std.npy.")
    ap.add_argument("--out-phase-csv", type=Path, required=True)
    ap.add_argument("--out-fluor-csv", type=Path, required=True)
    ap.add_argument("--contrast", choices=("ntc", "global"), default="ntc")
    ap.add_argument("--n-cells", type=int, default=30,
                    help="Cells to emit per (gene, viz_channel, role). "
                         "Default 30 = 3x N_COLS (atlas image row width).")
    ap.add_argument("--top-n-features", type=int, default=5,
                    help="Top-N SHAP features per (gene, viz_channel) to "
                         "compute the KO group mean over.")
    ap.add_argument("--genes", type=str, default=None,
                    help="Comma-separated gene filter (smoke testing).")
    ap.add_argument("--aggregation-level", choices=("gene", "complex"),
                    default="gene",
                    help="`complex` relabels source-gene -> CHAD complex name "
                         "(matching ntc_shap_features.py's CHAD logic).")
    ap.add_argument("--chad-config", type=Path, default=None,
                    help="CHAD positive-controls YAML (required when "
                         "--aggregation-level complex).")
    args = ap.parse_args()

    print(f"reading {args.shap_features_csv}", flush=True)
    feats_df = pd.read_csv(args.shap_features_csv)
    if "contrast" in feats_df.columns:
        feats_df = feats_df[feats_df["contrast"].astype(str) == args.contrast]
    if not len(feats_df):
        raise SystemExit(f"no rows with contrast='{args.contrast}'")

    if args.genes:
        wanted = set(g.strip() for g in args.genes.split(","))
        feats_df = feats_df[feats_df["gene"].astype(str).isin(wanted)]
        print(f"  filtered to {feats_df['gene'].nunique()} genes "
              f"({len(feats_df):,} rows)", flush=True)

    chad_map = None
    if args.aggregation_level == "complex":
        if args.chad_config is None:
            raise SystemExit("--aggregation-level complex requires --chad-config")
        chad_map = _load_chad_map(args.chad_config)
        print(f"  CHAD map: {len(chad_map)} gene->complex assignments", flush=True)

    # Build {viz_channel: {gene: [feat1, feat2, ...]}} index from features CSV.
    feats_df = feats_df.sort_values(["viz_channel", "gene", "shap_rank"])
    per_channel: dict[str, dict[str, list[str]]] = {}
    for (vc, gene), grp in feats_df.groupby(["viz_channel", "gene"], observed=True):
        top_feats = grp["feature"].astype(str).head(args.top_n_features).tolist()
        per_channel.setdefault(str(vc), {})[str(gene)] = top_feats

    phase_rows: list[dict] = []
    fluor_rows: list[dict] = []
    for vc in sorted(per_channel.keys()):
        sub = _viz_to_subdir(args.cache_dir, vc)
        if sub is None:
            print(f"  [skip] no cache subdir for viz_channel={vc!r}", flush=True)
            continue
        rows = _pick_exemplars_for_channel(
            sub,
            per_channel[vc],
            n_cells=args.n_cells,
            chad_map=chad_map,
        )
        target = phase_rows if vc == "Phase" else fluor_rows
        target.extend(rows)

    print(f"writing {args.out_phase_csv} ({len(phase_rows):,} rows)", flush=True)
    args.out_phase_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(phase_rows).to_csv(args.out_phase_csv, index=False)

    # Fluor rows need CP/4i seg lookup so the mask preference fluor_pref
    # (`4i_cell_seg`/`cp_cell_seg`/`cell_seg`) actually finds the cell on
    # CP/4i experiments. Without this, the renderer falls through to
    # the live-cell `cell_seg` mask and draws live-cell outlines over
    # the CP/4i fluor image — visible as the thin/wrong masks in
    # POLR2B's smoke output.
    fluor_df = pd.DataFrame(fluor_rows) if fluor_rows else pd.DataFrame()
    if not fluor_df.empty:
        fluor_df = _enrich_with_fluor_seg(fluor_df)
    print(f"writing {args.out_fluor_csv} ({len(fluor_df):,} rows)", flush=True)
    args.out_fluor_csv.parent.mkdir(parents=True, exist_ok=True)
    fluor_df.to_csv(args.out_fluor_csv, index=False)


if __name__ == "__main__":
    main()
