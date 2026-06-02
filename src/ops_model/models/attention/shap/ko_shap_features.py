"""Four-classifier SHAP per KO gene: each modality / channel vs other-gene top-attention.

For each KO gene, runs 4 independent LightGBM classifiers. The negative
class is uniformly "top-attention cells from OTHER gene KOs" across all
four classifiers — matching what the upstream attention model itself
classifies (KO-vs-KO discrimination on attention-selected cells).
Bottom-attention cells are no longer used as negatives anywhere in the
pipeline; the within-gene top-vs-bottom contrast was found to inject
attention-score artifacts that aren't actually KO biology.

  Phase (top_attention_cells_phase.h5ad)
    Positive: 100 top-attention phase cells for this gene
    Negative: 100 top-attention phase cells from OTHER genes
    Question: How does this KO's overall cell morphology differ from the
              cohort of all OTHER KOs?

  Channel 1 / 2 / 3  (top_attention_cells_fluor.h5ad)
    Positive: 100 top-attention cells for this gene in this channel
    Negative: 100 top-attention cells from OTHER genes (same channel_rank)
    Question: How does this KO's most-phenotypic cells differ from the
              most-phenotypic cells of all OTHER KOs in this channel?

Output columns
--------------
  gene, modality, channel_rank, viz_channel,
  shap_rank, feature, organelle, category,
  shap_importance, shap_mean, shap_cv, effect_size, direction,
  pct_cells, auroc, f1, prec, rec, n_pos_cells, viz_channels

  shap_importance : mean |SHAP| over positive cells (feature ranking)
  shap_mean       : mean signed SHAP over positive cells (direction signal)
  shap_cv         : std(|SHAP|) / mean(|SHAP|) — low = consistent effect across cells
  effect_size     : (mean(X_pos) - cohort_mean) / cohort_std — magnitude of
                    raw feature shift; cohort = top-attention cells across genes
                    (top-only for both modalities).
  direction       : sign of shap_mean (+1 = elevated, -1 = reduced)
  pct_cells       : fraction of this gene's top cells past the cohort top-only
                    MEAN in the gene's dominant direction (same cohort as
                    effect_size; aligned with the atlas violin's mean-tick line).
                    Not bounded ≥ 0.5 — for skewed feature distributions a gene
                    can have mean > cohort while a minority of cells exceed it.

Usage:
    python scripts/ko_shap/ko_shap_features.py \\
        --phase-h5ad /path/to/top_attention_cells_phase.h5ad \\
        --fluor-h5ad /path/to/top_attention_cells_fluor.h5ad \\
        --cache-phase /path/to/phase_cache \\
        --cache-fluor /path/to/fluor_cache \\
        --out-dir /path/to/output

    # Sharded (run N_SHARDS jobs in parallel, each with a different --shard index):
    python scripts/ko_shap/ko_shap_features.py --shard 0 --n-shards 10 ...
"""

import argparse
from pathlib import Path

import anndata as ad
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
MIN_NAN_FRAC   = 0.5
HOLDOUT_FRAC   = 0.20
TOP_N_PHASE    = 20
TOP_N_FLUOR    = 20
# Confidence floor + minimum-keep guarantee. Each (gene, modality,
# channel) keeps the top MIN_KEEP_FEATURES rows unconditionally (so the
# violin atlas's 5-row panel always has something to render), then any
# of ranks (MIN_KEEP_FEATURES+1)..TOP_N whose SHAP importance clears
# MIN_SHAP_IMPORTANCE. Net effect: at least 5, at most 20, with the
# additional rows gated by signal strength.
MIN_SHAP_IMPORTANCE = 0.10
MIN_KEEP_FEATURES   = 5
SAMPLE_FOR_NAN = 5_000
NEG_SIZE       = 100

LGBM_PHASE: dict = dict(
    n_estimators=150, learning_rate=0.05, num_leaves=15,
    colsample_bytree=0.2, min_child_samples=5,
    n_jobs=-1, random_state=42, verbose=-1,
)
LGBM_FLUOR: dict = dict(
    n_estimators=150, learning_rate=0.05, num_leaves=31,
    colsample_bytree=0.4, min_child_samples=10,
    n_jobs=-1, random_state=42, verbose=-1,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_cp_metadata(feat_name: str) -> tuple[str | None, str | None]:
    """Infer (organelle, category) for CellProfiler features whose
    h5ad-side `var.organelle` / `var.category` are NaN.

    The v3 h5ad (and prior caches) stores all 222 cp_* features with
    empty organelle / category metadata. Without backfill the SHAP CSV
    rows are uninterpretable — `_resolve_organelle` in the captioning
    script catches them via a feature-name fallback path, but the CSV
    column value is "nan" and downstream `ORGANELLE_DISPLAY` lookups
    miss. Filling these in at cache-build keeps the column
    self-describing and lets the primary dict lookup work.

    Routing:
      cp_cell_*       → ("cp_cell",      "cp_morphology")  (whole-cell)
      cp_cytoplasm_*  → ("cp_cytoplasm", "cp_morphology")  (whole-cell)
      cp_nucleus_*    → ("cp_nucleus",   "cp_morphology")  (nuclear)
      others          → (None, None) — caller keeps the original value
    """
    if feat_name.startswith("cp_cell_"):
        return ("cp_cell", "cp_morphology")
    if feat_name.startswith("cp_cytoplasm_"):
        return ("cp_cytoplasm", "cp_morphology")
    if feat_name.startswith("cp_nucleus_"):
        return ("cp_nucleus", "cp_morphology")
    return (None, None)


def _pct_cells(vals: np.ndarray, cohort_mean: float) -> float:
    """Fraction of finite cells on the dominant side of the cohort mean.

    Direction = sign(gene_mean − cohort_mean); pct_cells = fraction of
    gene values past `cohort_mean` in that direction. Unlike the prior
    median-based version, this is NOT guaranteed to be ≥ 0.5 — for
    skewed feature distributions a gene can have mean > cohort_mean
    (because of a few extreme cells) while fewer than half its cells
    actually exceed the cohort mean. That's intentional: the metric
    matches what the eye reads off the violin's mean tick. NaN when
    the window is empty.
    """
    finite = vals[np.isfinite(vals)]
    if len(finite) == 0:
        return float("nan")
    gene_mean = float(np.mean(finite))
    if gene_mean > cohort_mean:
        return float((finite > cohort_mean).mean())
    if gene_mean < cohort_mean:
        return float((finite < cohort_mean).mean())
    return 0.5


def _metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    y_pred = (y_pred_proba >= 0.5).astype(int)
    return dict(
        auroc=float(roc_auc_score(y_true, y_pred_proba)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),        # type: ignore[arg-type]
        prec=float(precision_score(y_true, y_pred, zero_division=0)),  # type: ignore[arg-type]
        rec=float(recall_score(y_true, y_pred, zero_division=0)),    # type: ignore[arg-type]
    )


def _classify(
    X_pos: np.ndarray,
    X_neg: np.ndarray,
    lgbm_params: dict,
    rng: np.random.Generator,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Train binary classifier; return held-out metrics + SHAP arrays over positives.

    Two fits: clf_eval (80% train) for unbiased metrics; clf_full (100%) for SHAP.
    """
    X = np.concatenate([X_pos, X_neg])
    y = np.array([1] * len(X_pos) + [0] * len(X_neg), dtype=np.int32)

    idx_tr, idx_te = train_test_split(
        np.arange(len(X)), test_size=HOLDOUT_FRAC, stratify=y,
        random_state=int(rng.integers(1 << 31)),
    )
    clf_eval = lgb.LGBMClassifier(**lgbm_params)
    clf_eval.fit(X[idx_tr], y[idx_tr])
    m = _metrics(y[idx_te], clf_eval.predict_proba(X[idx_te])[:, 1])  # type: ignore[index]

    clf_full = lgb.LGBMClassifier(**lgbm_params)
    clf_full.fit(X, y)
    explainer = shap.TreeExplainer(clf_full, feature_perturbation="tree_path_dependent")
    sv = explainer.shap_values(X_pos)
    shap_pos = sv[1] if isinstance(sv, list) else sv  # (n_pos, n_features)

    abs_shap         = np.abs(shap_pos)
    mean_abs_shap    = np.mean(abs_shap, axis=0)
    mean_signed_shap = np.mean(shap_pos, axis=0)
    shap_cv          = np.std(abs_shap, axis=0) / (mean_abs_shap + 1e-8)
    return m, mean_abs_shap, mean_signed_shap, shap_cv



def _build_cache(
    h5ad_path: Path,
    cache_dir: Path,
    rank_type_filter: str | None,
    rebuild: bool = False,
) -> tuple[np.ndarray, pd.DataFrame, list[str], np.ndarray, np.ndarray, list[str], list[str]]:
    """Load feature matrix from cache, building it from h5ad if not yet cached.

    rank_type_filter: None = all rows; "" = phase top-attention rows. Phase
    h5ads tag top-attention rows as `rank_type == ""` in the v2 schema and
    as `rank_type == "top"` in the v3 schema (which added `rank_type` ∈
    {top, bottom} on phase too); we accept either so the same call site
    works against both eras.

    rebuild: when True, deletes any existing cache files and rebuilds from
    the source h5ad. Use after the h5ad has been regenerated upstream
    (e.g. consolidate_top_attention_cells.py re-ran with new NTC/GLOBAL
    rows) so the cache reflects the current data. Also auto-triggers if
    cache obs row count is smaller than the source h5ad's row count —
    same staleness signal, applied silently.

    SLURM-array safety: ~200 shards racing on the same `cache_dir` would
    have one write a partial `X.npy` while another reads it → truncated
    load → `cannot reshape array of size ... into shape (...)`. We
    serialize on a per-cache-dir lock file (`fcntl.flock`) so only one
    shard builds the cache at a time; the rest block until the build
    finishes and then load the validated artifact. Atomic temp-rename
    on every write guards against partial-file reads if a shard is
    killed mid-build (kept files only become visible at rename time).
    """
    import fcntl

    cache_dir.mkdir(parents=True, exist_ok=True)
    cx, co = cache_dir / "X.npy",        cache_dir / "obs.parquet"
    cf, cm = cache_dir / "features.txt", cache_dir / "median.npy"
    cg, cc = cache_dir / "organelle.txt", cache_dir / "category.txt"
    cs     = cache_dir / "global_std.npy"
    ready  = cache_dir / ".ready"
    lockp  = cache_dir / ".build.lock"

    # Detect stale cache: source h5ad has more rows than cache obs.
    # Triggers a forced rebuild even without an explicit flag, since
    # silently loading a stale cache would skip newly-added rows (e.g.
    # NTC/GLOBAL after consolidate_top_attention_cells re-run). Narrow
    # the except so a TypeError or bad-implementation error surfaces
    # instead of being swallowed and causing a stale-cache load
    # (the original `except Exception` masked a real
    # "AnnData has no context manager" TypeError → users hit the
    # downstream NTC/GLOBAL hard-error and saw it as a SHAP failure).
    if ready.exists() and co.exists() and not rebuild:
        try:
            import anndata as _ad
            _src = _ad.read_h5ad(h5ad_path, backed="r")
            try:
                n_src = len(_src.obs)
            finally:
                _src.file.close()
            n_cache = len(pd.read_parquet(co, columns=["gene"]))
            if n_src > n_cache:
                print(f"  [cache stale] {cache_dir.name}: cache has {n_cache:,} "
                      f"rows vs source h5ad {n_src:,} ({n_src - n_cache:,} new). "
                      f"Forcing rebuild.", flush=True)
                rebuild = True
            else:
                print(f"  [cache fresh] {cache_dir.name}: cache={n_cache:,} "
                      f"h5ad={n_src:,}", flush=True)
        except (FileNotFoundError, OSError, ValueError) as e:
            # Specific file / parquet read errors are recoverable —
            # fall through with rebuild=False and let _try_load decide.
            print(f"  [cache stale-check skipped] {cache_dir.name}: "
                  f"{type(e).__name__}: {e}", flush=True)

    if rebuild:
        # Clear sentinel + data files so the build path below runs fresh.
        # Keep the lock file — it serializes the rebuild.
        for f in (ready, cx, co, cf, cm, cg, cc, cs):
            try:
                f.unlink()
            except FileNotFoundError:
                pass
        print(f"  [cache] {cache_dir.name}: rebuild requested → "
              f"sentinels cleared, will rebuild under lock.", flush=True)

    import os as _os, time as _time
    pid = _os.getpid()

    def _replace_with_retry(src: Path, dst: Path, attempts: int = 5) -> None:
        """`os.replace` with retry on EBUSY/ETXTBSY.

        On GPFS / NFS, rename-over-existing-file can fail with errno 16
        ("Device or resource busy") when another node holds the dest
        file open for read (e.g. another shard's `np.load` mmap from
        before this shard cleared the cache). The destination IS being
        properly torn down — the kernel just needs a beat to release
        the inode after the last reader closes its handle. A short
        backoff loop is enough; the original report had a single shard
        race past during a multi-minute rebuild.
        """
        for i in range(attempts):
            try:
                _os.replace(src, dst)
                return
            except OSError as e:
                if e.errno not in (16, 26) or i == attempts - 1:
                    raise
                _time.sleep(0.5 * (1 << i))  # 0.5, 1, 2, 4, 8 sec

    def _atomic_write_npy(path: Path, arr: np.ndarray) -> None:
        # np.save auto-appends .npy if the target path doesn't already
        # end in .npy, so build a temp name that DOES end in .npy to
        # avoid surprises. Hidden-file prefix (.) keeps the temp from
        # showing up in `ls cache_dir/` if a shard dies mid-build.
        tmp = path.parent / f".{path.stem}.tmp.{pid}.npy"
        np.save(tmp, arr)
        _replace_with_retry(tmp, path)

    def _atomic_write_text(path: Path, text: str) -> None:
        tmp = path.parent / f".{path.name}.tmp.{pid}"
        tmp.write_text(text)
        _replace_with_retry(tmp, path)

    def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
        tmp = path.parent / f".{path.name}.tmp.{pid}"
        df.to_parquet(tmp)
        _replace_with_retry(tmp, path)

    def _try_load() -> tuple | None:
        """Best-effort load of a fully-written cache. Returns None if the
        cache is missing, partial, empty, or corrupted (e.g. truncated
        X.npy from a killed builder). Caller must rebuild on None.

        X is mmap'd ('r') so 200 SLURM shards reading the same 78 GB
        phase cache share the OS page cache instead of each
        materializing 78 GB in RAM (which OOM-killed shards under
        --mem 96GB).
        """
        if not ready.exists():
            return None
        if not (cx.exists() and co.exists()):
            return None
        try:
            X = np.load(cx, mmap_mode="r")
            obs = pd.read_parquet(co)
            feature_names = cf.read_text().splitlines()
            global_median = np.load(cm)
            feat_organelle = cg.read_text().splitlines()
            feat_category = cc.read_text().splitlines()
            if cs.exists():
                global_std = np.load(cs)
            else:
                global_std = np.std(X.astype(np.float64), axis=0).astype(np.float32)
                _atomic_write_npy(cs, global_std)
        except (ValueError, OSError, EOFError, FileNotFoundError) as e:
            print(f"    Cache load failed at {cache_dir.name} ({type(e).__name__}: {e}); "
                  f"will rebuild.", flush=True)
            return None
        if X.shape[0] == 0 or X.shape[1] == 0:
            return None
        return X, obs, feature_names, global_median, global_std, feat_organelle, feat_category

    # Fast path: try without taking the lock. If `.ready` is present and
    # the load succeeds, no need to serialize 200 shards behind a single
    # acquire.
    cached = _try_load()
    if cached is not None:
        print(f"  Loaded {cache_dir.name} ({cached[0].shape[0]:,} cells × "
              f"{cached[0].shape[1]} features)", flush=True)
        return cached

    # Slow path: take the per-cache exclusive lock and re-check. If
    # another shard already finished the build while we were waiting,
    # we'll see `.ready` on the second check and load.
    print(f"  [{cache_dir.name}] acquiring build lock...", flush=True)
    with open(lockp, "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        cached = _try_load()
        if cached is not None:
            print(f"  [{cache_dir.name}] cache built by another shard while waiting "
                  f"({cached[0].shape[0]:,} × {cached[0].shape[1]})", flush=True)
            return cached

        # Stale partial files from a previous killed build — clear before
        # rebuilding so the atomic-rename writes don't fight existing
        # truncated artifacts. `missing_ok=True` keeps the cleanup
        # idempotent and race-free: under NFS the test-and-unlink
        # pattern (`if p.exists(): p.unlink()`) can still hit
        # FileNotFound when stat metadata is cached or another shard
        # is mid-cleanup (we've also seen partial pre-lock caches with
        # only X.npy written, no obs.parquet — the loop would crash
        # on the missing file before reaching the next).
        for p in (ready, cx, co, cf, cm, cg, cc, cs):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                # File showed up between this shard's check and unlink,
                # or NFS hiccup — best-effort cleanup, not fatal.
                pass

    print(f"  Building {cache_dir.name} from {h5ad_path.name} ...")
    adata = ad.read_h5ad(h5ad_path, backed="r")

    if rank_type_filter is None:
        pos = np.arange(len(adata))
        obs = pd.DataFrame(adata.obs).reset_index(drop=True)          # type: ignore[arg-type]
    else:
        # Accept either v2 ('') or v3 ('top') tagging for phase top-attention
        # rows. v2 phase h5ads had rank_type='' (or no column); v3 phase
        # h5ads tag rows 'top'. Without this OR, a v3 h5ad fed an "" filter
        # silently produces a 0-cell cache and every shard crashes at
        # `df["modality"]` on the empty result.
        if rank_type_filter == "":
            mask = adata.obs["rank_type"].isin(["", "top"])
        else:
            mask = adata.obs["rank_type"] == rank_type_filter
        pos  = np.where(mask)[0]
        obs  = pd.DataFrame(adata.obs[mask]).reset_index(drop=True)   # type: ignore[index]

    print(f"    {len(pos):,} cells, {adata.n_vars} raw features")

    rng0       = np.random.default_rng(42)
    sample_idx = rng0.choice(len(pos), size=min(SAMPLE_FOR_NAN, len(pos)), replace=False)
    X_sample   = np.array(adata.X[sorted(pos[sample_idx].tolist())])  # type: ignore[index]
    nan_frac   = np.isnan(X_sample).mean(axis=0)
    keep       = nan_frac < MIN_NAN_FRAC
    keep_idx   = np.where(keep)[0]
    del X_sample

    feature_names  = list(adata.var_names[keep])
    feat_meta      = adata.var.loc[feature_names, ["organelle", "category"]].copy()  # type: ignore[attr-defined]
    feat_organelle = [str(v) for v in feat_meta["organelle"]]
    feat_category  = [str(v) for v in feat_meta["category"]]

    # Backfill organelle/category for CellProfiler features — the v3
    # h5ad stores them with NaN var metadata, so without this the
    # SHAP CSV rows show organelle="nan" and downstream lookups have
    # to feature-name-infer instead of using ORGANELLE_DISPLAY.
    n_backfilled = 0
    for i, fname in enumerate(feature_names):
        if feat_organelle[i] not in ("nan", "None", ""):
            continue
        org, cat = _infer_cp_metadata(fname)
        if org is None:
            continue
        feat_organelle[i] = org
        if feat_category[i] in ("nan", "None", ""):
            feat_category[i] = cat or feat_category[i]
        n_backfilled += 1
    if n_backfilled:
        print(f"    backfilled organelle metadata for {n_backfilled} CP features")

    print(f"    {keep.sum()} / {adata.n_vars} features pass NaN filter")
    print(f"    Loading matrix ({len(pos):,} × {keep.sum()}) ...")

    # Fast path: when rank_type_filter is None we're loading EVERY cell,
    # i.e. `pos == np.arange(N)`. Passing 8.4M ints to fancy indexing on
    # a backed h5ad triggers cell-by-cell HDF5 reads — ~30 min on phase.
    # Chunked contiguous slicing reads MB-sized slabs instead, ~3-5×
    # faster, and keeps memory bounded (one chunk + accumulator instead
    # of two full copies for the X_full → X[:, keep_idx] sub-select).
    if rank_type_filter is None:
        N = len(pos)
        X = np.empty((N, len(keep_idx)), dtype=np.float32)
        CHUNK = 500_000  # ~5 GB per chunk at 2476 features × 4 bytes
        for start in range(0, N, CHUNK):
            end = min(start + CHUNK, N)
            slab = np.asarray(adata.X[start:end], dtype=np.float32)  # type: ignore[index]
            X[start:end] = slab[:, keep_idx]
            print(f"      read {end:,}/{N:,} rows ({100*end/N:.0f}%)", flush=True)
    else:
        X_full = np.array(adata.X[pos.tolist()], dtype=np.float32)  # type: ignore[index]
        X      = X_full[:, keep_idx]
        del X_full
    adata.file.close()

    # `global_median` / `global_std` use nan-aware reductions so they
    # reflect the cohort's true center/scale on real data.
    global_median = np.nanmedian(X, axis=0).astype(np.float32)
    global_std    = np.nanstd(X, axis=0).astype(np.float32)

    # NaNs are NOT imputed. LightGBM handles NaN natively (`use_missing=True`,
    # default) by routing missing samples to whichever child branch
    # maximizes split gain at each tree node — so missingness can carry
    # signal when it correlates with the label, and stays neutral when
    # random. Verified empirically that SHAP `TreeExplainer` returns
    # finite values for NaN-containing X. The prior median-imputation
    # step (which fabricated values for ~5% of fluor cells, including
    # 233 cells that were 100% NaN in the kept-feature space) is gone.
    # Downstream consumers in this file already use `np.nanmean` /
    # `np.nanmedian` / `np.isfinite` filters where needed.

    obs["_pos"] = np.arange(len(obs))
    # Atomic temp-rename for every artifact so a killed shard never
    # leaves a half-written X.npy that the next shard would try to load.
    # `.ready` written last is the all-clear signal for `_try_load`.
    _atomic_write_npy(cx, X)
    _atomic_write_npy(cm, global_median)
    _atomic_write_npy(cs, global_std)
    _atomic_write_parquet(obs, co)
    _atomic_write_text(cf, "\n".join(feature_names))
    _atomic_write_text(cg, "\n".join(feat_organelle))
    _atomic_write_text(cc, "\n".join(feat_category))
    ready.touch()
    print(f"    Saved to {cache_dir}", flush=True)
    return X, obs, feature_names, global_median, global_std, feat_organelle, feat_category


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-h5ad",  required=True, help="top_attention_cells_phase.h5ad")
    parser.add_argument("--fluor-h5ad",  required=True, help="top_attention_cells_fluor.h5ad")
    parser.add_argument("--cache-phase", required=True, help="Directory for phase feature cache")
    parser.add_argument("--cache-fluor", required=True, help="Directory for fluor feature cache")
    parser.add_argument("--out-dir",     required=True, help="Output directory for CSV(s)")
    parser.add_argument("--shard",       type=int, default=0)
    parser.add_argument("--n-shards",    type=int, default=1)
    parser.add_argument("--genes",       default="", help="Comma-separated gene filter")
    parser.add_argument(
        "--contrast", choices=("distinct", "ntc", "global"), default="distinct",
        help=(
            "Negative-class source for every classifier. Positives are "
            "ALWAYS this gene's top-attention cells (unchanged across "
            "contrasts). The negative pool swaps:\n"
            "  • distinct (default): other-gene top-attention cells in "
            "the same channel. Answers 'how does this KO's most-attended "
            "cells differ from OTHER KOs' most-attended cells?'.\n"
            "  • ntc: NTC cells from the per-channel all_cells_*.h5ad. "
            "Answers 'how do these KO cells differ from random NTCs?'.\n"
            "  • global: random sample of all cells from the per-channel "
            "all_cells_*.h5ad (any gene). Answers 'how do these KO cells "
            "differ from the global cell population?'."
        ),
    )
    # --all-cells-dir and --ntc-pma-*-csv removed — all negatives now
    # come from the same consolidated h5ad as the positives (gene
    # column distinguishes KO / NTC / GLOBAL). consolidate_top_attention
    # _cells.py ingests NTC + GLOBAL into the h5ad upstream.
    parser.add_argument(
        "--no-resume", action="store_true",
        help=(
            "Ignore any existing shard CSV at the output path and "
            "re-process every gene/complex assigned to this shard. "
            "Default behavior reads the existing CSV (if present), "
            "treats already-listed genes as done, and only fills in "
            "missing ones — which silently masks bug fixes that "
            "should add NEW channels/rows to the same genes (e.g. "
            "the 4i-marker slug fix)."
        ),
    )
    # No explicit --rebuild-cache flag: the auto-stale-detect in
    # `_build_cache` (cache obs row count vs h5ad row count) handles
    # the only scenario where it would have mattered — the source
    # h5ad gained rows since the cache was last built (e.g. after
    # consolidate_top_attention_cells re-ran). If you ever need to
    # force-rebuild without a row-count delta (e.g. h5ad VALUES
    # changed but row count is the same), just `rm` the cache dir.
    args = parser.parse_args()
    # All 3 contrasts source negatives from the same consolidated h5ad
    # as positives (gene column = KO complex / NTC / GLOBAL). Re-run
    # consolidate_top_attention_cells.py first if NTC/GLOBAL rows are
    # missing from the h5ad.

    phase_h5ad  = Path(args.phase_h5ad)
    fluor_h5ad  = Path(args.fluor_h5ad)
    cache_phase = Path(args.cache_phase)
    cache_fluor = Path(args.cache_fluor)
    out_dir     = Path(args.out_dir)
    gene_filter: set[str] = set(args.genes.split(",")) - {""} if args.genes else set()

    out_dir.mkdir(parents=True, exist_ok=True)
    if gene_filter:
        out_csv = out_dir / "ko_shap_features_targeted.csv"
    elif args.n_shards > 1:
        out_csv = out_dir / f"ko_shap_features_shard{args.shard:02d}.csv"
    else:
        out_csv = out_dir / "ko_shap_features.csv"

    # Load / build caches
    print("=== Phase ===")
    X_phase, obs_phase, phase_feat_names, phase_median, phase_std, phase_organelle, phase_category = (
        _build_cache(phase_h5ad, cache_phase, rank_type_filter="")
    )

    print("\n=== Fluor (top + bottom) ===")
    X_fluor, obs_fluor, fluor_feat_names, fluor_median, fluor_std, fluor_organelle, fluor_category = (
        _build_cache(fluor_h5ad, cache_fluor, rank_type_filter=None)
    )

    if "_pos" not in obs_phase.columns:
        obs_phase["_pos"] = np.arange(len(obs_phase))
    if "_pos" not in obs_fluor.columns:
        obs_fluor["_pos"] = np.arange(len(obs_fluor))

    # NTC and GLOBAL cells live in the same SHAP cache as KO positives
    # (after consolidate_top_attention_cells.py ingests them via
    # --ntc-{phase,fluor}-csv and --global-per-channel). We exclude
    # them from the per-gene positive iteration AND from the distinct
    # negative pool (other-complex KO cells) so they only serve as
    # negatives when --contrast {ntc, global}. Slice both modalities
    # first so all downstream references see consistent labels.
    _NEG_LABELS = ("NTC", "GLOBAL")
    obs_phase_ko   = obs_phase[~obs_phase["gene"].astype(str).isin(_NEG_LABELS)]
    obs_phase_ntc  = obs_phase[obs_phase["gene"].astype(str) == "NTC"]
    obs_phase_glob = obs_phase[obs_phase["gene"].astype(str) == "GLOBAL"]
    # KO positives + NTC PMA cells carry rank_type="top". GLOBAL random
    # cells (from `_build_global_cells`) don't have an attention-rank
    # concept, so their rank_type is empty — include them here so the
    # global-contrast classifier can see them as a negative pool.
    obs_fluor_top: pd.DataFrame = obs_fluor[
        (obs_fluor["rank_type"].astype(str) == "top")
        | (obs_fluor["gene"].astype(str) == "GLOBAL")
    ].copy()  # type: ignore[assignment]
    obs_fluor_ko_top   = obs_fluor_top[~obs_fluor_top["gene"].astype(str).isin(_NEG_LABELS)]
    obs_fluor_ntc_top  = obs_fluor_top[obs_fluor_top["gene"].astype(str) == "NTC"]
    obs_fluor_glob_top = obs_fluor_top[obs_fluor_top["gene"].astype(str) == "GLOBAL"]
    print(f"  obs_phase breakdown: {len(obs_phase_ko):,} KO + "
          f"{len(obs_phase_ntc):,} NTC + {len(obs_phase_glob):,} GLOBAL")
    print(f"  obs_fluor_top breakdown: {len(obs_fluor_ko_top):,} KO + "
          f"{len(obs_fluor_ntc_top):,} NTC + {len(obs_fluor_glob_top):,} GLOBAL "
          f"(of {len(obs_fluor_top):,} total)")

    # Build index maps. KO genes only; NTC + GLOBAL are negatives, not positives.
    all_phase_pos = obs_phase_ko["_pos"].to_numpy()
    gene_phase_pos: dict[str, np.ndarray] = {
        str(gene): grp["_pos"].to_numpy()
        for gene, grp in obs_phase_ko.groupby("gene", observed=True)
    }
    phase_ntc_pos  = obs_phase_ntc["_pos"].to_numpy() if len(obs_phase_ntc) else np.array([], dtype=np.int64)
    phase_glob_pos = obs_phase_glob["_pos"].to_numpy() if len(obs_phase_glob) else np.array([], dtype=np.int64)

    gene_ch_top_pos: dict[str, dict[int, np.ndarray]] = {}
    gene_ch_name:    dict[str, dict[int, str]]         = {}
    for gene, grp in obs_fluor_ko_top.groupby("gene", observed=True):
        gene = str(gene)
        gene_ch_top_pos[gene] = {}
        gene_ch_name[gene]    = {}
        for ch_rank, ch_grp in grp.groupby("channel_rank", observed=True):
            gene_ch_top_pos[gene][int(ch_rank)] = ch_grp["_pos"].to_numpy()         # type: ignore[arg-type]
            gene_ch_name[gene][int(ch_rank)]    = str(ch_grp["viz_channel"].iloc[0])  # type: ignore[union-attr]

    # Per-channel TOP-attention pool — keyed by VIZ_CHANNEL (not channel_rank)
    # since NTC and GLOBAL rows have their own per-source channel_rank scheme
    # that doesn't align with KO's. Distinct contrast negatives = other-KO
    # cells with the same viz_channel; ntc/global negatives = NTC/GLOBAL cells
    # with the same viz_channel. All pulled from the same SHAP cache, so
    # they pass through the same feature extraction → no pipeline-fingerprint
    # AUROC=1.0 saturation.
    ch_top_pool_ko:    dict[str, np.ndarray] = {}
    ch_top_pool_ntc:   dict[str, np.ndarray] = {}
    ch_top_pool_glob:  dict[str, np.ndarray] = {}
    for _vc, _grp in obs_fluor_ko_top.groupby("viz_channel", observed=True):
        ch_top_pool_ko[str(_vc)] = _grp["_pos"].to_numpy()  # type: ignore[arg-type]
    for _vc, _grp in obs_fluor_ntc_top.groupby("viz_channel", observed=True):
        ch_top_pool_ntc[str(_vc)] = _grp["_pos"].to_numpy()  # type: ignore[arg-type]
    for _vc, _grp in obs_fluor_glob_top.groupby("viz_channel", observed=True):
        ch_top_pool_glob[str(_vc)] = _grp["_pos"].to_numpy()  # type: ignore[arg-type]
    # Back-compat: legacy channel_rank-keyed KO pool used by the
    # old distinct path. Kept until that path is removed.
    ch_top_pool: dict[int, np.ndarray] = {}
    for _ch_rank, _grp in obs_fluor_ko_top.groupby("channel_rank", observed=True):
        ch_top_pool[int(_ch_rank)] = _grp["_pos"].to_numpy()  # type: ignore[arg-type]

    # Top-attention-only MEAN + std for both modalities — used as the
    # cohort reference for `effect_size` and `pct_cells`. Switched from
    # median to mean so the metric matches the atlas's violin "mean
    # tick" — a 100% pct_cells now means "all gene cells exceed the bg
    # violin's tick" instead of "all gene cells exceed the cohort
    # median" (which can leave a chunk of the gene violin visually
    # past the tick when distributions are skewed).
    # Phase cache is top-only by build, so phase_top_mean is just the
    # column mean of X_phase. Fluor cache spans top+bottom, so we
    # restrict to the top-attention slice first.
    print("\nComputing top-only cohort reference (mean + std)...", flush=True)
    fluor_top_pos_arr = obs_fluor_top["_pos"].to_numpy()

    # Chunked Welford-style streaming compute to keep peak memory bounded
    # under the 5.6M-cell × 595-feature gene-level workload — materializing
    # the full slice as float64 here was OOM-killing 64GB shards. Per chunk
    # we cast to float64 only for accumulation, then drop. Mean/std land
    # in float64 (cheap: 595 doubles) but the bulk arrays stay float32.
    def _streaming_nan_mean_std(arr, idx, chunk=200_000):
        n_feat = arr.shape[1]
        cnt = np.zeros(n_feat, dtype=np.int64)
        s   = np.zeros(n_feat, dtype=np.float64)
        s2  = np.zeros(n_feat, dtype=np.float64)
        for start in range(0, len(idx), chunk):
            block = np.asarray(arr[idx[start:start + chunk], :], dtype=np.float64)
            valid = ~np.isnan(block)
            cnt += valid.sum(axis=0)
            block_zero = np.where(valid, block, 0.0)
            s  += block_zero.sum(axis=0)
            s2 += (block_zero * block_zero).sum(axis=0)
            del block, valid, block_zero
        cnt_safe = np.maximum(cnt, 1)
        mean = s / cnt_safe
        var = (s2 / cnt_safe) - mean * mean
        std = np.sqrt(np.maximum(var, 0.0))
        return mean.astype(np.float32), std.astype(np.float32)

    fluor_top_mean, fluor_top_std = _streaming_nan_mean_std(X_fluor, fluor_top_pos_arr)
    fluor_top_std = fluor_top_std.clip(1e-6)
    phase_idx = np.arange(X_phase.shape[0], dtype=np.int64)
    phase_top_mean, _ = _streaming_nan_mean_std(X_phase, phase_idx)

    print(
        f"  fluor: {len(fluor_top_pos_arr):,} top cells × {X_fluor.shape[1]} features\n"
        f"  phase: {X_phase.shape[0]:,} top cells × {X_phase.shape[1]} features",
        flush=True,
    )

    gene_viz_channels: dict[str, str] = {}
    for gene, grp in obs_fluor_top.groupby("gene", observed=True):
        by_rank = (
            grp[["viz_channel", "channel_rank"]]
            .drop_duplicates(subset=["channel_rank"])  # type: ignore[call-overload]
            .sort_values("channel_rank")
        )
        gene_viz_channels[str(gene)] = " | ".join(by_rank["viz_channel"].astype(str).tolist() + ["Phase"])

    genes_sorted = sorted(set(gene_phase_pos) & set(gene_ch_top_pos))
    if gene_filter:
        genes_sorted = [g for g in genes_sorted if g in gene_filter]
    else:
        genes_sorted = genes_sorted[args.shard::args.n_shards]

    print(f"\nShard {args.shard}/{args.n_shards}: {len(genes_sorted)} genes")
    print(f"  Phase: {len(obs_phase):,} cells × {X_phase.shape[1]} features")
    print(f"  Fluor top: {len(obs_fluor_top):,} cells  |  {X_fluor.shape[1]} features")

    # Resume from partial output. A 0-byte CSV (left behind by an earlier
    # shard that crashed before writing any rows — e.g., the v3 rank_type
    # mismatch run) crashes pd.read_csv with EmptyDataError; treat that
    # the same as "no prior output" and start fresh rather than abort.
    done_genes: set[str] = set()
    records: list[dict]  = []
    if args.no_resume and out_csv.exists():
        print(f"--no-resume: overwriting {out_csv.name} (no gene-skip)")
    elif out_csv.exists() and out_csv.stat().st_size > 0:
        try:
            existing = pd.read_csv(out_csv)
        except pd.errors.EmptyDataError:
            print(f"Empty CSV at {out_csv} — starting fresh")
            existing = None
        if existing is not None and "channel_rank" in existing.columns:
            done_genes = set(existing["gene"].unique())
            records    = existing.to_dict("records")
            print(f"Resuming: {len(done_genes)} / {len(genes_sorted)} genes done")
        elif existing is not None:
            print("Old format CSV detected — starting fresh")

    # Per-gene loop
    rng       = np.random.default_rng(0)
    remaining = [g for g in genes_sorted if g not in done_genes]
    print(f"Processing {len(remaining)} genes ...\n")

    # Negatives for ntc/global contrasts come from gene=NTC / gene=GLOBAL
    # rows in the SAME consolidated h5ad as the positives (post the
    # consolidate_top_attention_cells.py unification). Hard-error when
    # those rows are missing — no fallback to all_cells_v2 (the legacy
    # path produced AUROC=1.0 pipeline-fingerprint saturation, a wrong
    # answer, not a degraded one).
    if args.contrast == "ntc":
        if len(phase_ntc_pos) == 0 and len(obs_fluor_ntc_top) == 0:
            raise SystemExit(
                "--contrast ntc requires gene='NTC' rows in the consolidated "
                "h5ad. Re-run consolidate_top_attention_cells with "
                "--ntc-phase-csv / --ntc-fluor-csv (defaults wired)."
            )
        print(f"  [ntc] negatives: gene='NTC' rows in consolidated h5ad "
              f"(phase={len(phase_ntc_pos):,}, fluor={len(obs_fluor_ntc_top):,})")
    elif args.contrast == "global":
        if len(phase_glob_pos) == 0 and len(obs_fluor_glob_top) == 0:
            raise SystemExit(
                "--contrast global requires gene='GLOBAL' rows in the "
                "consolidated h5ad. Re-run consolidate_top_attention_cells "
                "with --global-per-channel >= 1 (default 100)."
            )
        print(f"  [global] negatives: gene='GLOBAL' rows in consolidated h5ad "
              f"(phase={len(phase_glob_pos):,}, fluor={len(obs_fluor_glob_top):,})")

    for i, gene in enumerate(remaining):
        viz_ch    = gene_viz_channels.get(gene, "Phase")
        phase_pos = gene_phase_pos[gene]

        # 1. Phase classifier — negative pool swaps per --contrast. All
        # three sources (other-KO / NTC / GLOBAL) now live in the same
        # SHAP cache (post consolidate_top_attention_cells unification),
        # so feature scales match the positives exactly — no pipeline-
        # fingerprint AUROC=1.0 saturation.
        if args.contrast == "distinct":
            other_phase = np.array([p for p in all_phase_pos if p not in set(phase_pos.tolist())])
            neg_source = other_phase
        elif args.contrast == "ntc":
            neg_source = phase_ntc_pos
        elif args.contrast == "global":
            neg_source = phase_glob_pos
        else:
            neg_source = np.array([], dtype=np.int64)
        if len(neg_source) == 0:
            print(f"  [{gene}] phase: empty neg pool for contrast={args.contrast!r} "
                  f"— skipping classifier", flush=True)
            continue
        neg_p_idx   = rng.choice(len(neg_source), size=min(NEG_SIZE, len(neg_source)), replace=False)
        X_neg_phase = X_phase[neg_source[neg_p_idx]]
        m_p, shap_p, shap_p_signed, shap_p_cv = _classify(
            X_phase[phase_pos], X_neg_phase, LGBM_PHASE, rng
        )
        ranked_p = list(np.argsort(shap_p)[::-1][:TOP_N_PHASE])
        kept_p = ranked_p[:MIN_KEEP_FEATURES] + [
            fi for fi in ranked_p[MIN_KEEP_FEATURES:]
            if float(shap_p[fi]) >= MIN_SHAP_IMPORTANCE
        ]
        for rank_i, fi in enumerate(kept_p):
            direction   = float(np.sign(shap_p_signed[fi])) or 1.0
            vals        = np.asarray(X_phase[phase_pos, fi], dtype=np.float64)
            # nanmean/nanmedian: cache no longer median-imputes NaNs
            # (LightGBM handles them natively at training time); the
            # gene-vs-cohort effect_size needs to skip NaN cells, not
            # treat them as zeros.
            effect_size = (float(np.nanmean(vals)) - float(phase_top_mean[fi])) / (float(phase_std[fi]) + 1e-8)
            pct_cells   = _pct_cells(vals, float(phase_top_mean[fi]))
            records.append(dict(
                gene=gene, modality="phase", channel_rank=0, viz_channel="Phase",
                shap_rank=rank_i + 1, feature=phase_feat_names[fi],
                organelle=phase_organelle[fi], category=phase_category[fi],
                shap_importance=float(shap_p[fi]),
                shap_mean=float(shap_p_signed[fi]),
                shap_cv=float(shap_p_cv[fi]),
                effect_size=effect_size, direction=direction,
                pct_cells=pct_cells,
                auroc=m_p["auroc"], f1=m_p["f1"], prec=m_p["prec"], rec=m_p["rec"],
                n_pos_cells=len(phase_pos), viz_channels=viz_ch,
            ))

        # 2–4. Per-channel fluor classifiers. All three negative cohorts
        # (other-KO / NTC / GLOBAL) live in the same SHAP cache, keyed
        # by viz_channel. Same feature scale as positives by
        # construction → no pipeline-fingerprint AUROC saturation.
        ch_aurocs: list[float] = []
        for ch_rank, top_pos in sorted(gene_ch_top_pos[gene].items()):
            ch_name  = gene_ch_name[gene][ch_rank]
            if args.contrast == "distinct":
                own_top_set = set(top_pos.tolist())
                pool        = ch_top_pool_ko.get(ch_name, np.array([], dtype=np.int64))
                other_pool  = np.array([p for p in pool if p not in own_top_set])
                if len(other_pool) == 0:
                    continue
            elif args.contrast == "ntc":
                other_pool = ch_top_pool_ntc.get(ch_name, np.array([], dtype=np.int64))
                if len(other_pool) == 0:
                    print(f"  [{gene}/{ch_name}] no NTC cells in consolidated h5ad "
                          f"for this channel — skipping", flush=True)
                    continue
            elif args.contrast == "global":
                other_pool = ch_top_pool_glob.get(ch_name, np.array([], dtype=np.int64))
                if len(other_pool) == 0:
                    print(f"  [{gene}/{ch_name}] no GLOBAL cells in consolidated h5ad "
                          f"for this channel — skipping", flush=True)
                    continue
            else:
                raise SystemExit(f"Unknown --contrast={args.contrast!r}")
            neg_idx = rng.choice(
                len(other_pool), size=min(NEG_SIZE, len(other_pool)), replace=False,
            )
            X_neg_fluor = X_fluor[other_pool[neg_idx]]

            m_f, shap_f, shap_f_signed, shap_f_cv = _classify(
                X_fluor[top_pos], X_neg_fluor, LGBM_FLUOR, rng
            )
            ch_aurocs.append(m_f["auroc"])

            ranked_f = list(np.argsort(shap_f)[::-1][:TOP_N_FLUOR])
            kept_f = ranked_f[:MIN_KEEP_FEATURES] + [
                fi for fi in ranked_f[MIN_KEEP_FEATURES:]
                if float(shap_f[fi]) >= MIN_SHAP_IMPORTANCE
            ]
            for rank_i, fi in enumerate(kept_f):
                direction   = float(np.sign(shap_f_signed[fi])) or 1.0
                vals        = np.asarray(X_fluor[top_pos, fi], dtype=np.float64)
                # Use the TOP-only cohort MEAN as reference (matches
                # atlas violin's mean-tick line) for both effect_size
                # and pct_cells. nanmean: cache no longer imputes NaNs.
                effect_size = (float(np.nanmean(vals)) - float(fluor_top_mean[fi])) / (float(fluor_top_std[fi]) + 1e-8)
                pct_cells   = _pct_cells(vals, float(fluor_top_mean[fi]))
                records.append(dict(
                    gene=gene, modality="fluor", channel_rank=ch_rank, viz_channel=ch_name,
                    shap_rank=rank_i + 1, feature=fluor_feat_names[fi],
                    organelle=fluor_organelle[fi], category=fluor_category[fi],
                    shap_importance=float(shap_f[fi]),
                    shap_mean=float(shap_f_signed[fi]),
                    shap_cv=float(shap_f_cv[fi]),
                    effect_size=effect_size, direction=direction,
                    pct_cells=pct_cells,
                    auroc=m_f["auroc"], f1=m_f["f1"], prec=m_f["prec"], rec=m_f["rec"],
                    n_pos_cells=len(top_pos), viz_channels=viz_ch,
                ))

        if (i + 1) % 50 == 0 or len(remaining) <= 10:
            chs = "  ".join(f"ch{r}={a:.2f}" for r, a in zip(sorted(gene_ch_top_pos[gene]), ch_aurocs))
            print(f"  {i+1:4d}/{len(remaining)}  phase={m_p['auroc']:.2f}  {chs}  ({gene})")

        pd.DataFrame(records).to_csv(out_csv, index=False)

    # Final summary. Empty-shard early exit: when args.n_shards exceeds
    # the number of genes (e.g., the SHAP runner submitted 200 shards
    # against the 90-complex CHAD workload), shards beyond gene_count
    # have 0 genes assigned, write an empty CSV, and would crash at
    # df["modality"] lookup below. Treat that as a successful no-op so
    # the SLURM array doesn't go partially red.
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    if df.empty or "modality" not in df.columns:
        print(f"\n=== Summary ===")
        print(f"No genes processed in this shard (assigned 0 genes "
              f"from genes_sorted[{args.shard}::{args.n_shards}]).")
        print(f"Saved: {out_csv} (empty)")
        return
    phase_r1 = df[(df["modality"] == "phase") & (df["shap_rank"] == 1)]
    fluor_r1  = df[(df["modality"] == "fluor") & (df["shap_rank"] == 1) & (df["channel_rank"] == 1)]
    print(f"\n=== Summary ===")
    print(f"Genes:         {df['gene'].nunique()} / {len(genes_sorted)}")
    if len(phase_r1):
        print(f"Phase   median AUROC: {float(np.median(phase_r1['auroc'])):.3f}  F1: {float(np.median(phase_r1['f1'])):.3f}")
        print(f"Phase auroc > 0.65:  {(phase_r1['auroc'] > 0.65).sum()}")
    if len(fluor_r1):
        print(f"Fluor   median AUROC: {float(np.median(fluor_r1['auroc'])):.3f}  F1: {float(np.median(fluor_r1['f1'])):.3f}")
        print(f"Fluor  auroc > 0.65: {(fluor_r1['auroc'] > 0.65).sum()}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
