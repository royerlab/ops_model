"""KO-vs-NTC and KO-vs-global SHAP feature ranking on per-channel all_cells h5ads.

Sibling of `ko_shap_features.py`. Two key differences:

  * **Inputs**: per-channel `all_cells_<viz_channel>.h5ad` files produced by
    `consolidate_all_cells.py`. Each h5ad has a single viz_channel and
    features pre-z-scored per (experiment) batch. Cells live in `obs`
    columns: `gene` (KO gene symbol or "NTC"), `sgRNA`, `experiment`,
    `viz_channel`, etc.
  * **Two contrasts**, run in the same loop over genes:
        - `ntc`:    positives = gene KO cells, negatives = NTC cells
        - `global`: positives = gene KO cells, negatives = ALL non-{this gene}
                    cells (other-gene KOs + NTCs)
    Both produce the same CSV row schema, distinguished by a new `contrast`
    column. Atlas downstream picks one contrast at render time.

Output CSV mirrors `ko_shap_features.csv`'s columns plus `contrast`:
    gene, modality, channel_rank, viz_channel, contrast, shap_rank, feature,
    organelle, category, shap_importance, shap_mean, shap_cv, effect_size,
    direction, pct_cells, auroc, f1, prec, rec, n_pos_cells, n_neg_cells,
    viz_channels

CLI:
    # Smoke test on a few genes, both contrasts:
    python ntc_shap_features.py \\
        --all-cells-dir /hpc/.../alex_lin_attention/all_cells_v2 \\
        --cache-dir    /hpc/.../alex_lin_attention/ntc_caches/v2 \\
        --out-dir      /hpc/.../alex_lin_attention/ntc_v2 \\
        --genes POLR2B,PSMB7,TOMM20 \\
        --contrast both

    # Sharded SLURM (driven by run_ntc_shap_pipeline.py):
    python ntc_shap_features.py \\
        --all-cells-dir ... --cache-dir ... --out-dir ... \\
        --shard 0 --n-shards 50

Reuses `_build_cache`, `_classify`, `_pct_cells`, `LGBM_PHASE`, `LGBM_FLUOR`,
`TOP_N_*`, `MIN_*` from the sibling `ko_shap_features` module.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ko_shap_features as ks  # noqa: E402

LGBM_PHASE = ks.LGBM_PHASE
LGBM_FLUOR = ks.LGBM_FLUOR
TOP_N_PHASE = ks.TOP_N_PHASE
TOP_N_FLUOR = ks.TOP_N_FLUOR
MIN_SHAP_IMPORTANCE = ks.MIN_SHAP_IMPORTANCE
MIN_KEEP_FEATURES = ks.MIN_KEEP_FEATURES

# Per-classifier caps: skip thin (gene, viz_channel) pairs where SHAP would
# be unreliable. With paper_v1's per-sgRNA caps these floors rarely bind for
# any biologically meaningful gene.
MIN_POS_CELLS = 50
MIN_NEG_CELLS = 50

# Per-fit cap on BOTH positive and negative cohorts. Each gene/complex ×
# channel × contrast classifier trains on min(NEG_SIZE, n_pos) vs the same
# count from the negative pool, so cohorts are always balanced 1:1. 2000
# keeps each LightGBM fit fast (<30s) while preserving statistical power
# — empirically SHAP rankings stabilize well below 5k cells per class,
# and dropping from 5000 to 2000 cuts per-shard wall time and cache I/O
# pressure without changing the top-rank features.
NEG_SIZE = 2000

CONTRAST_DISTINCT = "distinct"
CONTRAST_NTC      = "ntc"
CONTRAST_GLOBAL   = "global"
CONTRAST_CHOICES  = (CONTRAST_DISTINCT, CONTRAST_NTC, CONTRAST_GLOBAL)


# Canonical PMA fluor CSV used to recover original-case viz_channel
# strings for the SHAP CSV (the all_cells h5ads store them lowercased
# after _normalize_viz_channel; the atlas's image-row matcher does
# exact-string lookup against PMA-case channel names, so we have to
# remap before writing the SHAP CSV).
_PMA_FLUOR_GENE = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v3/attention_v3/"
    "pma_top_fluorescent_cells_v3.csv"
)


def _build_viz_channel_case_map(pma_csv: Path = _PMA_FLUOR_GENE) -> dict[str, str]:
    """Build {lowercased_normalized → original_case} mapping for viz_channel.

    The all_cells cache stores `viz_channel` post `_normalize_viz_channel`
    (lowercased, parenthetical-stripped, Foo_Foo deduped). The atlas's
    image-row resolver matches against PMA-style channel names which
    keep the original case + parenthetical + repeat. We rebuild that
    mapping here so the SHAP CSV writes the atlas-compatible form.
    """
    try:
        from organelle_profiler.feature_extraction.consolidate_top_attention_cells import (
            _normalize_viz_channel,
        )
    except Exception:
        # Same canonicalization scheme — minimal copy to avoid an import
        # cycle if downstream consumers ever shift module layout.
        def _normalize_viz_channel(s: str) -> str:
            raw = str(s or "").strip()
            paren = raw.find("(")
            if paren > 0:
                raw = raw[:paren].strip()
            parts = raw.split("_")
            if len(parts) == 2 and parts[0] and parts[0].lower() == parts[1].lower():
                raw = parts[0]
            return raw.lower().strip()

    if not pma_csv.exists():
        print(f"  [viz-case] PMA fluor CSV not found at {pma_csv}; "
              f"viz_channel will keep lowercased form (atlas matching may fail).",
              flush=True)
        return {}
    chans = pd.read_csv(pma_csv, usecols=["channel"])["channel"].astype(str).unique()
    return {_normalize_viz_channel(c): c for c in chans}


# Populated once in main() from `_build_viz_channel_case_map`; consulted by
# `_process_channel` to rewrite lowercased viz_channel back to PMA case.
# Module-level dict (mutated via `.update`) so the worker function picks it
# up without threading an extra arg through every callsite.
_VIZ_CASE_MAP: Dict[str, str] = {}


def _write_chad_obs(
    chad_obs_path: Path,
    chad_lock_path: Path,
    obs_to_write: pd.DataFrame,
) -> None:
    """Flock-guarded write of `obs_chad.parquet`.

    Many shards converge on the same cache_dir; only one needs to write,
    and writes must be atomic. We hold an exclusive flock on
    `.chad.lock`, re-check existence inside the critical section, then
    do a tempfile-then-rename swap. The tmp path is PID-unique so that
    if NFS flock fails to actually serialize across nodes (a known
    NFSv3/v4 weakness), shards don't trample each other's tmp files —
    the worst case becomes a redundant write rather than a
    FileNotFoundError on `os.replace`.
    """
    chad_lock_path.touch(exist_ok=True)
    tmp_path = chad_obs_path.with_suffix(f".parquet.tmp.{os.getpid()}")
    try:
        with open(chad_lock_path, "r+") as lock_fp:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
            try:
                if chad_obs_path.exists():
                    try:
                        existing_n = len(pd.read_parquet(chad_obs_path, columns=["_x_idx"]))
                    except (OSError, ValueError, pd.errors.ParserError):
                        existing_n = -1
                    if existing_n == len(obs_to_write):
                        print(f"  [CHAD] obs_chad.parquet written by peer "
                              f"({existing_n:,} rows); skipping", flush=True)
                        return
                obs_to_write.to_parquet(tmp_path, index=False)
                os.replace(tmp_path, chad_obs_path)
                print(f"  [CHAD] persisted obs_chad.parquet: "
                      f"{len(obs_to_write):,} rows → {chad_obs_path}", flush=True)
            finally:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
    finally:
        # Belt-and-braces: clean up our PID-specific tmp file if the
        # rename never happened (exception inside the critical section).
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _viz_channel_from_filename(path: Path) -> str:
    """Recover the canonical viz_channel from `all_cells_*.h5ad` filename.

    The consolidator sanitizes viz_channel for the filename (lowercase + spaces
    -> underscores). Here we just need a stable key — we read the actual
    viz_channel from the h5ad's obs, this is just for filtering/logging.
    """
    name = path.stem
    if name == "all_cells_phase":
        return "phase"
    return name.replace("all_cells_fluor_", "")


def _classify_with_neg_cap(
    X_pos: np.ndarray,
    X_neg_full: np.ndarray,
    lgbm_params: dict,
    rng: np.random.Generator,
    neg_cap: int = NEG_SIZE,
) -> tuple:
    """Wrapper around `ks._classify` that caps the negative pool size."""
    if len(X_neg_full) > neg_cap:
        idx = rng.choice(len(X_neg_full), size=neg_cap, replace=False)
        X_neg = X_neg_full[idx]
    else:
        X_neg = X_neg_full
    return ks._classify(X_pos, X_neg, lgbm_params, rng)


def _emit_records(
    *,
    gene: str,
    modality: str,
    channel_rank: int,
    viz_channel: str,
    contrast: str,
    viz_channels_str: str,
    metrics: dict,
    shap_abs: np.ndarray,
    shap_signed: np.ndarray,
    shap_cv: np.ndarray,
    feat_names: List[str],
    feat_organelle: List[str],
    feat_category: List[str],
    cohort_mean: np.ndarray,
    cohort_std: np.ndarray,
    pos_vals: np.ndarray,
    n_pos: int,
    n_neg: int,
    top_n: int,
) -> List[dict]:
    """Build per-feature CSV rows for one classifier (gene, channel, contrast).

    Schema matches `ko_shap_features.csv` plus `contrast`. `cohort_mean` /
    `cohort_std` are the negative-class statistics (used for `effect_size`
    and `pct_cells` — same convention as ko_shap_features).
    """
    ranked = list(np.argsort(shap_abs)[::-1][:top_n])
    kept = ranked[:MIN_KEEP_FEATURES] + [
        fi for fi in ranked[MIN_KEEP_FEATURES:]
        if float(shap_abs[fi]) >= MIN_SHAP_IMPORTANCE
    ]
    out: List[dict] = []
    for rank_i, fi in enumerate(kept):
        direction = float(np.sign(shap_signed[fi])) or 1.0
        vals = pos_vals[:, fi]
        es_denom = float(cohort_std[fi]) + 1e-8
        effect_size = (float(np.nanmean(vals)) - float(cohort_mean[fi])) / es_denom
        pct = ks._pct_cells(vals, float(cohort_mean[fi]))
        out.append(dict(
            gene=gene,
            modality=modality,
            channel_rank=channel_rank,
            viz_channel=viz_channel,
            contrast=contrast,
            shap_rank=rank_i + 1,
            feature=feat_names[fi],
            organelle=feat_organelle[fi],
            category=feat_category[fi],
            shap_importance=float(shap_abs[fi]),
            shap_mean=float(shap_signed[fi]),
            shap_cv=float(shap_cv[fi]),
            effect_size=effect_size,
            direction=direction,
            pct_cells=pct,
            auroc=metrics["auroc"],
            f1=metrics["f1"],
            prec=metrics["prec"],
            rec=metrics["rec"],
            n_pos_cells=n_pos,
            n_neg_cells=n_neg,
            viz_channels=viz_channels_str,
        ))
    return out


def _process_channel(
    h5ad_path: Path,
    cache_dir: Path,
    genes: List[str],
    contrasts: List[str],
    rng: np.random.Generator,
    chad_map: Optional[Dict[str, str]] = None,
) -> List[dict]:
    """Build/load cache for one per-channel h5ad, train classifiers for each
    (gene, contrast) and return CSV records.

    When `chad_map` is provided, `obs.gene` is relabeled in-place: source
    gene → CHAD complex name, NTCs preserved as "NTC". Cells with no CHAD
    assignment are dropped from the iteration. Effective unit is then
    "complex" instead of "gene"; downstream code is unchanged.
    """
    print(f"\n=== {h5ad_path.name} ===", flush=True)
    X, obs, feat_names, median_, std_, organelle, category = (
        ks._build_cache(h5ad_path, cache_dir, rank_type_filter=None)
    )
    if "_pos" not in obs.columns:
        obs["_pos"] = np.arange(len(obs))

    # CHAD relabel at iteration time. Maps source gene -> complex name;
    # NTCs stay "NTC". Cells with no CHAD assignment are dropped from
    # the iteration. The relabeled obs is ALSO persisted back to the
    # cache_dir's obs.parquet (only at complex-level — gene-level uses
    # a separate cache dir) so the atlas can read CHAD-labelled cells
    # directly from disk when it loads the cache for violin/positive
    # lookup.
    if chad_map is not None:
        gene_str = obs["gene"].astype(str)
        is_ntc = gene_str == "NTC"
        mapped = gene_str[~is_ntc].map(chad_map)
        n_dropped = int(mapped.isna().sum())
        if n_dropped:
            print(f"  [CHAD] dropping {n_dropped:,} cells with no CHAD assignment "
                  f"({mapped.notna().sum():,} cells in {mapped.dropna().nunique()} "
                  f"complexes kept)", flush=True)
        new_gene = gene_str.copy()
        new_gene.loc[~is_ntc] = mapped.fillna("__DROP__").values
        obs = obs.assign(gene=new_gene.values)
        # Filter out dropped rows
        keep = obs["gene"].astype(str) != "__DROP__"
        obs = obs[keep].reset_index(drop=True)

        # Persist chad-relabeled obs as a SIBLING parquet
        # (`obs_chad.parquet`) so the atlas can find complex-labelled
        # rows on disk.  We do NOT touch `obs.parquet` or `X.npy` —
        # 200 concurrent shards rewriting the canonical cache files
        # caused stale-file-handle errors and bus-error core dumps
        # via truncate-during-mmap. The sibling carries an `_x_idx`
        # column mapping each row to its position in the canonical
        # `X.npy` so the atlas can slice features without a duplicate
        # X copy.
        chad_obs_path = cache_dir / "obs_chad.parquet"
        chad_lock_path = cache_dir / ".chad.lock"
        obs_to_write = obs.reset_index(drop=True).copy()
        obs_to_write["_x_idx"] = obs_to_write["_pos"].astype(np.int64)
        obs_to_write = obs_to_write.drop(columns=["_pos"], errors="ignore")

        # Skip if a previous shard already wrote a complete sibling.
        if chad_obs_path.exists():
            try:
                existing_n = len(pd.read_parquet(chad_obs_path, columns=["_x_idx"]))
            except (OSError, ValueError, pd.errors.ParserError):
                existing_n = -1
            if existing_n == len(obs_to_write):
                print(f"  [CHAD] obs_chad.parquet already present "
                      f"({existing_n:,} rows); skipping write", flush=True)
            else:
                _write_chad_obs(chad_obs_path, chad_lock_path, obs_to_write)
        else:
            _write_chad_obs(chad_obs_path, chad_lock_path, obs_to_write)

    viz_uniques = obs["viz_channel"].astype(str).unique()
    if len(viz_uniques) != 1:
        raise SystemExit(
            f"{h5ad_path.name}: expected single viz_channel, got {list(viz_uniques)}"
        )
    viz_channel = str(viz_uniques[0])
    # Remap lowercased viz_channel (from `_normalize_viz_channel`) back
    # to PMA-case so the SHAP CSV's viz_channel matches the atlas's
    # image-row matcher. `_viz_case_map` is set in main() once per run.
    canonical = _VIZ_CASE_MAP.get(viz_channel, viz_channel)
    if canonical != viz_channel:
        print(f"  [viz-case] remapped {viz_channel!r} → {canonical!r}",
              flush=True)
        viz_channel = canonical
    is_phase = viz_channel == "Phase"
    modality = "phase" if is_phase else "fluor"
    lgbm_params = LGBM_PHASE if is_phase else LGBM_FLUOR
    top_n = TOP_N_PHASE if is_phase else TOP_N_FLUOR
    viz_str = "Phase" if is_phase else f"{viz_channel} | Phase"

    gene_col = obs["gene"].astype(str).to_numpy()
    pos_lookup = obs["_pos"].to_numpy()

    ntc_pos = pos_lookup[gene_col == "NTC"]

    # Cohort references for effect_size / pct_cells computed once per contrast.
    # NTC contrast uses NTC cells as the cohort baseline; global contrast
    # uses all-cells (effectively channel-wide median, post-z-score).
    if len(ntc_pos) >= MIN_NEG_CELLS:
        _Xn = np.asarray(X[ntc_pos, :], dtype=np.float64)
        ntc_cohort_mean = np.nanmean(_Xn, axis=0)
        ntc_cohort_std = np.nanstd(_Xn, axis=0).clip(1e-6)
        del _Xn
    else:
        print(f"  [{viz_channel}] NTC cohort too small ({len(ntc_pos)}); "
              f"skipping ntc contrast for this channel", flush=True)
        ntc_cohort_mean = ntc_cohort_std = None

    if CONTRAST_GLOBAL in contrasts:
        # Global cohort mean/std on a sample (full computation is expensive).
        # Post-z-score these are ~(0, 1) globally; use the channel-wide stats
        # so per-feature effect sizes are channel-relative.
        # 5K samples — sufficient for the channel-wide mean/std used only
        # to compute per-feature effect_size + pct_cells in the CSV.
        # Statistical SE at N=5K is ~1.4× larger than N=50K (1/√N),
        # well below feature-level variability. Cuts mmap I/O 10× per
        # channel (was the dominant per-shard cost after the neg-sample
        # patch). Also drops the float64 staging buffer 10× since the
        # nan* reductions are the only consumer.
        sample_n = min(5_000, len(obs))
        sample_idx = rng.choice(len(obs), size=sample_n, replace=False)
        _Xs = np.asarray(X[np.sort(sample_idx), :], dtype=np.float64)
        global_cohort_mean = np.nanmean(_Xs, axis=0)
        global_cohort_std = np.nanstd(_Xs, axis=0).clip(1e-6)
        del _Xs
    else:
        global_cohort_mean = global_cohort_std = None

    if CONTRAST_DISTINCT in contrasts:
        # Distinct cohort = all KO cells EXCLUDING NTCs (other-gene KOs only,
        # sampled across all complexes). Mean/std on a sample matches the
        # `global` cohort treatment.
        ko_mask = gene_col != "NTC"
        ko_pool = pos_lookup[ko_mask]
        if len(ko_pool) >= MIN_NEG_CELLS:
            # 5K samples — same reasoning as the global-cohort sample
            # above. Used only for effect_size / pct_cells statistics.
            sample_n = min(5_000, len(ko_pool))
            sample_idx = rng.choice(len(ko_pool), size=sample_n, replace=False)
            _Xs = np.asarray(X[np.sort(ko_pool[sample_idx]), :], dtype=np.float64)
            distinct_cohort_mean = np.nanmean(_Xs, axis=0)
            distinct_cohort_std  = np.nanstd(_Xs, axis=0).clip(1e-6)
            del _Xs
        else:
            print(f"  [{viz_channel}] KO cohort too small ({len(ko_pool)}); "
                  f"skipping distinct contrast for this channel", flush=True)
            distinct_cohort_mean = distinct_cohort_std = None
    else:
        distinct_cohort_mean = distinct_cohort_std = None

    print(f"  Channel: {viz_channel} ({modality}); cells={len(obs):,} "
          f"(NTC={len(ntc_pos):,})", flush=True)

    records: List[dict] = []
    for i, gene in enumerate(genes, start=1):
        gene_pos = pos_lookup[gene_col == gene]
        if len(gene_pos) < MIN_POS_CELLS:
            continue

        # Cap positives at NEG_SIZE so a fat gene with 50k cells doesn't
        # train against an undersampled 5k negative pool — matching pos
        # to neg keeps the 1:1 balance the negative cap below also
        # enforces. Sampled without replacement; deterministic per gene.
        if len(gene_pos) > NEG_SIZE:
            sub_idx = rng.choice(len(gene_pos), size=NEG_SIZE, replace=False)
            gene_pos = gene_pos[sub_idx]

        X_pos = np.asarray(X[gene_pos, :], dtype=np.float32)
        n_pos = len(gene_pos)
        gene_summary: List[str] = []

        for contrast in contrasts:
            if contrast == CONTRAST_NTC:
                if ntc_cohort_mean is None or len(ntc_pos) < MIN_NEG_CELLS:
                    continue
                neg_pos = ntc_pos
                cohort_mean = ntc_cohort_mean
                cohort_std = ntc_cohort_std
            elif contrast == CONTRAST_DISTINCT:
                # All KO cells from OTHER genes (excludes NTCs, excludes this
                # gene). The all-cells analog of ko_shap_features' distinct
                # contrast — symmetric "this KO's population vs other KOs'
                # populations" at the all-cells grain.
                if distinct_cohort_mean is None:
                    continue
                neg_pos = pos_lookup[(gene_col != gene) & (gene_col != "NTC")]
                if len(neg_pos) < MIN_NEG_CELLS:
                    continue
                cohort_mean = distinct_cohort_mean
                cohort_std  = distinct_cohort_std
            else:  # CONTRAST_GLOBAL
                # All non-{this gene} cells: other-gene KOs + NTCs
                neg_pos = pos_lookup[gene_col != gene]
                if len(neg_pos) < MIN_NEG_CELLS:
                    continue
                cohort_mean = global_cohort_mean
                cohort_std = global_cohort_std

            n_neg = len(neg_pos)
            # SUBSAMPLE neg indices BEFORE reading the mmap. Without this
            # we'd materialize the full neg pool (up to ~8M rows × 2476
            # features × 4 bytes ≈ 80 GB on phase) just to subsample down
            # to 2000 inside _classify_with_neg_cap. With pre-subsample
            # the actual mmap read is 2-5 MB regardless of pool size —
            # the dominant cost on the phase channel disappears.
            neg_cap = min(NEG_SIZE, n_pos)
            if len(neg_pos) > neg_cap:
                sub_idx = rng.choice(len(neg_pos), size=neg_cap, replace=False)
                neg_pos = neg_pos[sub_idx]
            X_neg = np.asarray(X[neg_pos, :], dtype=np.float32)

            # `_classify_with_neg_cap` would also cap, but `len(X_neg)`
            # is already at or below `neg_cap` so its inner sample is a
            # no-op. Kept for the case where MIN_NEG_CELLS ≤ |pool| < neg_cap
            # (small NTC pool) — there the wrapper just passes through.
            m, sh_a, sh_s, sh_cv = _classify_with_neg_cap(
                X_pos, X_neg, lgbm_params, rng, neg_cap,
            )
            records.extend(_emit_records(
                gene=gene, modality=modality, channel_rank=1,
                viz_channel=viz_channel, contrast=contrast,
                viz_channels_str=viz_str,
                metrics=m, shap_abs=sh_a, shap_signed=sh_s, shap_cv=sh_cv,
                feat_names=feat_names, feat_organelle=organelle, feat_category=category,
                cohort_mean=cohort_mean, cohort_std=cohort_std,
                pos_vals=X_pos,
                n_pos=n_pos, n_neg=min(n_neg, neg_cap),
                top_n=top_n,
            ))
            gene_summary.append(f"{contrast}={m['auroc']:.2f}")

        if (i % 25 == 0 or i == len(genes)) and gene_summary:
            print(f"  {i:>4d}/{len(genes)}  {gene:<20s}  {'  '.join(gene_summary)}",
                  flush=True)

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-cells-dir", required=True,
        help="Directory containing all_cells_*.h5ad files from consolidate_all_cells.",
    )
    parser.add_argument(
        "--cache-dir", required=True,
        help="Directory for per-channel feature caches (one subdir per h5ad).",
    )
    parser.add_argument(
        "--out-dir", required=True,
        help="Output directory for ntc_shap_features.csv (or shard CSVs).",
    )
    parser.add_argument(
        "--contrast", choices=("distinct", "ntc", "global", "all"),
        default="all",
        help="Negative-class definition. `distinct`=other-gene KO cells "
             "(excludes NTCs); `ntc`=NTC cells; `global`=all non-{this gene} "
             "cells (other KOs + NTCs); `all` runs all three per (gene, "
             "channel) and emits a `contrast` column (default: all).",
    )
    parser.add_argument(
        "--aggregation-level", choices=("gene", "complex"), default="gene",
        help="Aggregate at gene level (default; iterate over individual gene KO "
             "labels in `obs.gene`) or at CHAD protein-complex level (relabel "
             "obs.gene = complex name from --chad-config; cells with no CHAD "
             "assignment are dropped). Same all_cells h5ads serve both — relabel "
             "happens in-memory at SHAP iteration time. Output CSV's `gene` "
             "column carries the complex name in 'complex' mode; downstream "
             "atlas code is unchanged.",
    )
    from organelle_profiler.feature_extraction.consolidate_top_attention_cells import (
        DEFAULT_CHAD_CONFIG, _load_chad_complexes,
    )
    parser.add_argument(
        "--chad-config", type=Path, default=DEFAULT_CHAD_CONFIG,
        help=f"Path to CHAD positive_controls YAML (default: {DEFAULT_CHAD_CONFIG}).",
    )
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore any existing shard CSV at the output path and "
             "re-process every (gene, viz_channel, contrast) triple "
             "assigned to this shard. Default resume mode silently "
             "masks bug fixes that should add new rows.",
    )
    parser.add_argument(
        "--genes", default="",
        help="Comma-separated gene allowlist (overrides shard when set). "
             "At --aggregation-level complex, pass complex names instead "
             "(e.g. 'RNA Polymerase II Complex,Proteasome').",
    )
    parser.add_argument(
        "--channels", default="",
        help="Comma-separated viz_channel filter on the all_cells_<x>.h5ad "
             "filename suffix (e.g. 'phase,5xupre,gh2ax'). Empty = all channels.",
    )
    args = parser.parse_args()

    all_cells_dir = Path(args.all_cells_dir)
    cache_root = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    # Build the lowercased→PMA-case viz_channel map once. `_process_channel`
    # reads it via the module-level `_VIZ_CASE_MAP`.
    _VIZ_CASE_MAP.update(_build_viz_channel_case_map())
    if _VIZ_CASE_MAP:
        print(f"  [viz-case] built {len(_VIZ_CASE_MAP)} channel-case mappings "
              f"from PMA fluor CSV")

    # Discover per-channel h5ads.
    h5ad_paths = sorted(all_cells_dir.glob("all_cells_*.h5ad"))
    if args.channels:
        wanted = {c.strip().lower() for c in args.channels.split(",") if c.strip()}
        h5ad_paths = [p for p in h5ad_paths
                      if _viz_channel_from_filename(p).lower() in wanted]
    if not h5ad_paths:
        raise SystemExit(f"No all_cells_*.h5ad files matched in {all_cells_dir}")
    print(f"Found {len(h5ad_paths)} per-channel h5ads")

    # CHAD relabel map (None at gene level). Loaded once; passed to each
    # `_process_channel` worker to relabel `obs.gene` in-memory at iteration
    # time so the same all_cells h5ads serve both grain levels.
    chad_map: Optional[Dict[str, str]] = None
    if args.aggregation_level == "complex":
        chad_map = _load_chad_complexes(Path(args.chad_config))
        print(f"CHAD: loaded {len(chad_map)} gene -> complex mappings "
              f"({len(set(chad_map.values()))} complexes) from "
              f"{Path(args.chad_config).name}")

    # Discover the gene/complex universe by reading any one h5ad's obs in
    # backed mode. At complex level, we'd compute gene-level uniques and
    # then map through chad_map to get complex names.
    import anndata as ad
    print(f"Discovering {'complexes' if chad_map else 'genes'} from "
          f"{h5ad_paths[0].name}...", flush=True)
    a0 = ad.read_h5ad(h5ad_paths[0], backed="r")
    raw_genes = (
        a0.obs.loc[a0.obs["gene"].astype(str) != "NTC", "gene"]
        .astype(str).unique().tolist()
    )
    a0.file.close()
    if chad_map is not None:
        all_genes = sorted({chad_map[g] for g in raw_genes if g in chad_map})
        n_unmapped = sum(1 for g in raw_genes if g not in chad_map)
        print(f"  {len(raw_genes):,} unique gene labels in h5ad → "
              f"{len(all_genes):,} CHAD complexes "
              f"({n_unmapped:,} genes have no CHAD assignment)")
    else:
        all_genes = sorted(raw_genes)
        print(f"  {len(all_genes):,} unique KO genes")

    # Apply gene filter / shard.
    gene_filter: set = (set(g.strip() for g in args.genes.split(",") if g.strip())
                        if args.genes else set())
    if gene_filter:
        my_genes = [g for g in all_genes if g in gene_filter]
        if not my_genes:
            raise SystemExit(f"No genes in --genes matched the cohort.")
        print(f"  filter --genes -> {len(my_genes)} genes")
    else:
        my_genes = all_genes[args.shard::args.n_shards]
        print(f"  shard {args.shard}/{args.n_shards} -> {len(my_genes)} genes")

    contrasts = (list(CONTRAST_CHOICES) if args.contrast == "all"
                 else [args.contrast])
    print(f"  contrasts: {contrasts}")

    # Output CSV path. Renamed from `ntc_shap_features.csv` — the
    # script handles ALL 3 contrasts (distinct/ntc/global) for the
    # all-cells variant, so the `ntc_` script-name prefix on output
    # files was misleading. Directory already encodes variant+contrast
    # (e.g. all_cells_distinct_geneKO/) so the filename just needs
    # the pipeline tag.
    if gene_filter:
        out_csv = out_dir / "all_cells_shap_features_targeted.csv"
    elif args.n_shards > 1:
        out_csv = out_dir / f"all_cells_shap_features_shard{args.shard:02d}.csv"
    else:
        out_csv = out_dir / "all_cells_shap_features.csv"

    # Resume support: skip channels for which all (gene, contrast) pairs are
    # already in the existing CSV.
    #
    # Treat 0-byte / empty / corrupt CSVs as "no prior progress" rather than
    # crashing. A previous shard that got killed mid-flush (SLURM timeout,
    # OOM) can leave an empty output file behind. The plain
    # `pd.read_csv(empty)` raised `EmptyDataError`, which then propagated up
    # to submitit and failed the whole shard despite there being no actual
    # work problem — just stale empty state. 13/200 shards in the most
    # recent CHAD run hit this exact path.
    done_pairs: set = set()
    records: List[dict] = []
    if args.no_resume and out_csv.exists():
        print(f"--no-resume: overwriting {out_csv.name} (no triple-skip)")
    elif out_csv.exists() and out_csv.stat().st_size > 0:
        try:
            existing = pd.read_csv(out_csv)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            print(f"  [resume] {out_csv.name} unreadable ({type(e).__name__}: {e}); "
                  f"starting fresh.", flush=True)
            existing = None
        if existing is not None and {"gene", "viz_channel", "contrast"}.issubset(existing.columns):
            done_pairs = set(map(tuple, existing[["gene", "viz_channel", "contrast"]].values))
            records = existing.to_dict("records")
            print(f"Resuming: {len(done_pairs):,} (gene, viz, contrast) "
                  f"triples already in {out_csv.name}")

    rng = np.random.default_rng(args.shard)

    # Iterate channels. Cache is per-channel.
    for h5ad_path in h5ad_paths:
        ch_key = _viz_channel_from_filename(h5ad_path)
        ch_cache = cache_root / ch_key
        # Filter to genes still pending for this channel.
        # (We process at the channel level so reads are amortized.)
        # Determine viz_channel name for the done_pairs check: read from cache's
        # obs.parquet if present, else from h5ad obs.
        viz_label = None
        if (ch_cache / "obs.parquet").exists():
            try:
                _viz = pd.read_parquet(
                    ch_cache / "obs.parquet", columns=["viz_channel"],
                )["viz_channel"].astype(str).iloc[0]
                viz_label = _viz
            except Exception:
                pass
        if viz_label is None:
            _a = ad.read_h5ad(h5ad_path, backed="r")
            viz_label = str(_a.obs["viz_channel"].astype(str).iloc[0])
            _a.file.close()
        # Canonicalize for the done_pairs check — the SHAP CSV stores the
        # PMA-case viz_channel (post-remap), so the lowercased cache label
        # would never match an existing resume row otherwise.
        viz_label = _VIZ_CASE_MAP.get(viz_label, viz_label)

        pending = [g for g in my_genes
                   if any((g, viz_label, c) not in done_pairs for c in contrasts)]
        if not pending:
            print(f"\n[{viz_label}] all (gene × contrast) already done; skipping")
            continue

        ch_records = _process_channel(
            h5ad_path, ch_cache, pending, contrasts, rng,
            chad_map=chad_map,
        )
        records.extend(ch_records)
        # Stream-write so we can resume on preemption.
        pd.DataFrame(records).to_csv(out_csv, index=False)

    # Final summary.
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    if not df.empty:
        print(f"\n=== Summary ===")
        for contrast in contrasts:
            sub = df[(df["contrast"] == contrast) & (df["shap_rank"] == 1)]
            if not len(sub):
                continue
            ph = sub[sub["modality"] == "phase"]
            fl = sub[sub["modality"] == "fluor"]
            print(f"\n[{contrast}]")
            if len(ph):
                print(f"  Phase: {len(ph):>5d} (gene, channel) classifiers; "
                      f"median AUROC={float(np.median(ph['auroc'])):.3f}; "
                      f">0.65 = {(ph['auroc'] > 0.65).sum()}")
            if len(fl):
                print(f"  Fluor: {len(fl):>5d} (gene, channel) classifiers; "
                      f"median AUROC={float(np.median(fl['auroc'])):.3f}; "
                      f">0.65 = {(fl['auroc'] > 0.65).sum()}")
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
