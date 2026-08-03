"""Combined-reporter cell-count titration.

Like ``titration`` but instead of titrating each reporter independently, this
samples cells across N reporters at each cell budget, h-concatenates their
NTC-normalized guide-level features into one combined matrix, and scores that.
Produces one curve per metric for the *group*, so you can ask how a panel of
markers collectively degrades with cell count — rather than how each marker
degrades on its own.

Groups are named explicitly with repeated ``--group NAME=path1[,path2,...]``
flags pointing at ``*_cells.h5ad`` files. NAME becomes the output directory
leaf. With two or more groups their mean-mAP curves are overlaid in one figure
per metric; ``--no-compare`` skips that step and ``--compare-only`` re-plots it
from existing per-group CSVs without re-running the titration.

The ``-o`` root plus the usual path-resolution flags (``--cell-dino``,
``--paper-v1``, ``--with-cp``/``--with-4i``, ``--fixed-threshold``,
``--distance``, …) select where output is *written*; the ``--group`` paths say
what is *read*.

Usage::

    # One marker alone, another alone, and the two combined — overlaid.
    python -m ops_model.post_process.combination.titration.combined_titration \\
        -o <pca_optimization_root> --cell-dino --paper-v1 --with-cp --with-4i \\
        --per-guide-median-titration --slurm \\
        --group pRb=<per_signal>/pRb_4i_cells.h5ad \\
        --group p21=<per_signal>/p21_4i_cells.h5ad \\
        --group pRb_p21=<per_signal>/pRb_4i_cells.h5ad,<per_signal>/p21_4i_cells.h5ad
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
    hconcat_by_perturbation,
    normalize_guide_adata,
)
from ops_utils.analysis.pca import fit_pca, n_pcs_for_threshold
from ops_model.post_process.combination.titration.titration import (
    METRIC_COLUMNS,
    METRICS,
    MIN_CELLS,
    SCALES,
    SCALE_LABEL_SHORT,
    TITRATION_MAP_LABELS,
    _UNIT_BY_MODE,
    _aggregate_draws,
    _apply_x_scale,
    _build_parser as _titr_parser,  # reuse common nesting flags
    _build_per_ko_schedule,
    _cache_split,
    _guide_count_pools,
    _merge_and_write,
    _non_ntc,
    _pert_col,
    _plt,
    _resolve_output_dir,
    _score_all_metrics,
    _subsample_one,
    _target_col,
    titration_x_axis_base_label,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------


def _parse_group_specs(specs: List[str]) -> Dict[str, List[Path]]:
    """Parse repeated ``--group NAME=path1,path2`` into {name: [paths]}.

    Every path must be an existing ``*_cells.h5ad``. Names must be unique and
    filesystem-safe, since each one becomes an output directory leaf.
    """
    groups: Dict[str, List[Path]] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(
                f"--group expects NAME=path1[,path2,...], got {spec!r}"
            )
        name, _, path_str = spec.partition("=")
        name = name.strip()
        if not name:
            raise SystemExit(f"--group has an empty name: {spec!r}")
        if any(ch in name for ch in "/\\"):
            raise SystemExit(
                f"--group name {name!r} can't contain a path separator "
                f"(it becomes an output directory name)"
            )
        if name in groups:
            raise SystemExit(f"--group {name!r} given more than once")
        paths = [Path(p.strip()) for p in path_str.split(",") if p.strip()]
        if not paths:
            raise SystemExit(f"--group {name!r} lists no paths")
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise SystemExit(
                f"--group {name!r}: missing cells h5ad(s):\n  "
                + "\n  ".join(missing)
            )
        groups[name] = sorted(paths)
    return groups


def _mode_subdir(sampling_mode: str) -> str:
    """Filesystem-safe subdir name per titration method (keeps modes from clobbering)."""
    return {
        "per_guide": "per_guide_max",
        "per_guide_median": "per_guide_median",
        "per_ko": "per_ko",
        "total": "total_cells",
    }.get(sampling_mode, sampling_mode)


def _second_pca_suffix(args: argparse.Namespace) -> str:
    """Disk-name tag for the active 2nd-pass PCA threshold.

    Returns ``""`` when 2nd-pass is off (so existing no-2nd-pca output dirs
    stay where they were), and ``_sec<XX>`` (e.g. ``_sec40`` for 0.40) when
    on. This lets the same group be titrated at multiple thresholds without
    clobbering each other or invalidating each other's caches.
    """
    thr = float(getattr(args, "second_pca_threshold", 0.0) or 0.0)
    if thr <= 0:
        return ""
    return f"_sec{int(round(thr * 100)):02d}"


def _resolve_group_output_dir(
    args: argparse.Namespace, group: str, sampling_mode: str = "per_guide",
) -> Path:
    """Output dir for one group's combined titration, scoped by titration method.

    The variant directory comes from the same flags titration/pca_optimization
    use (``-o`` plus the channel-set / threshold / distance flags); the group
    name is the leaf, so two groups in one run never collide.
    """
    return (
        _resolve_output_dir(args)
        / "combined_titration"
        / _mode_subdir(sampling_mode)
        / f"{group}{_second_pca_suffix(args)}"
    )


def _groups_tag(groups: Sequence[str], max_len: int = 200) -> str:
    """Filesystem-safe tag joining group names with ``_vs_``.

    Truncated with a deterministic hash suffix if the joined name would push the
    leaf directory past the 255-byte filename limit.
    """
    tag = "_vs_".join(groups)
    if len(tag) > max_len:
        import hashlib
        h = hashlib.sha1("|".join(groups).encode()).hexdigest()[:8]
        tag = tag[: max_len - 12] + f"__{h}"
    return tag


def _resolve_compare_dir(
    args: argparse.Namespace, groups: Sequence[str], sampling_mode: str = "per_guide",
) -> Path:
    """Top-level dir for cross-group comparison plots, scoped by titration method."""
    return (
        _resolve_output_dir(args)
        / "combined_titration_compare"
        / _mode_subdir(sampling_mode)
        / f"{_groups_tag(groups)}{_second_pca_suffix(args)}"
    )


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


def _per_reporter_guide_counts(paths: List[Path]) -> List[np.ndarray]:
    """For each reporter h5ad, return the per-guide cell-count array (non-NTC).

    Falls back to all guides for a reporter with no identifiable NTC.
    """
    per_reporter: List[np.ndarray] = []
    for p in paths:
        sg_counts, non_ntc_counts = _guide_count_pools(ad.read_h5ad(p, backed="r"))
        pool = non_ntc_counts if len(non_ntc_counts) else sg_counts
        per_reporter.append(np.asarray(pool.values, dtype=int))
    return per_reporter


# Allowed values for the median-schedule start policy.
_MEDIAN_START_POLICIES = ("pool", "max_reporter")


def _per_guide_median_start(
    paths: List[Path],
    policy: str = "pool",
) -> int:
    """Start point (cells/guide) for the per-guide-median schedule.

    ``policy='pool'``: median of all per-sgRNA cell counts pooled across
        every reporter (treats every reporter × guide as one observation).
        Conservative — when one reporter has many more cells/guide than the
        others, the median sits in the smaller-reporter cloud.

    ``policy='max_reporter'``: max of per-reporter medians. Lets the schedule
        climb up to the *biggest* reporter's natural median, with smaller
        reporters saturating at their own max along the way (each step caps
        per-reporter at ``min(target, available)``). Use when one reporter
        in a multi-marker group has substantially more cells per guide than
        the rest (e.g. Phase in the ``livecell`` group).
    """
    if policy not in _MEDIAN_START_POLICIES:
        raise ValueError(
            f"policy must be one of {_MEDIAN_START_POLICIES}, got {policy!r}"
        )
    per_reporter = _per_reporter_guide_counts(paths)
    if not per_reporter:
        return 1
    if policy == "pool":
        return int(np.median(np.concatenate(per_reporter)))
    return int(max(int(np.median(arr)) for arr in per_reporter if arr.size))


def _build_per_guide_max_schedule(paths: List[Path]) -> List[int]:
    """cells/guide schedule starting at p90 of pooled non-NTC guide cell counts."""
    per_reporter = _per_reporter_guide_counts(paths)
    pool = np.concatenate(per_reporter) if per_reporter else np.asarray([], dtype=int)
    return _build_per_ko_schedule(int(np.percentile(pool, 90)))


def _build_per_guide_median_schedule(
    paths: List[Path],
    start_override: Optional[int] = None,
    policy: str = "pool",
) -> List[int]:
    """cells/guide schedule from the MEDIAN of pooled non-NTC sgRNA counts
    down to 1 (mirrors max-mode but caps the high end at median instead of
    p90). ``start_override`` clamps the starting cells/guide value (used to
    align starts across groups in cross-group comparisons); when provided,
    we skip the pool computation entirely (saves opening every h5ad).

    ``policy`` selects the start when no override is given — see
    :func:`_per_guide_median_start`.
    """
    if start_override is not None:
        return _build_per_ko_schedule(int(start_override))
    return _build_per_ko_schedule(_per_guide_median_start(paths, policy=policy))


# ─── SLURM-prep worker: compute medians for one group on a compute node ────
# Top-level so cloudpickle can pickle it for submit_parallel_jobs. Mirrors
# `_per_guide_median_start` but emits BOTH start policies (pool + max_reporter)
# plus per-reporter medians so the login node can compute shared/per-group
# starts later without reopening any h5ad.
def _prep_schedule_worker(
    group: str,
    cells_h5ad_paths: List[str],
    sampling_mode: str,
    median_start_policy: str,
    cache_path: str,
) -> dict:
    """Open this group's h5ads in backed mode, compute per-reporter sgRNA
    cell-count arrays, then cache the medians (both ``pool`` and
    ``max_reporter`` policies) to ``cache_path`` as JSON.

    The big-Phase group dominates median-computation wall time because
    Phase's 60M-row obs frame takes ~minutes to groupby. Running 1
    prep job per group in parallel converts that into max(group_times)
    instead of sum, while writing a tiny cache file so re-runs at the
    same (sampling_mode, median_start_policy) skip the prep entirely.
    """
    import json
    from pathlib import Path as _Path

    cache_p = _Path(cache_path)
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    paths = [_Path(p) for p in cells_h5ad_paths]

    counts_per_reporter = _per_reporter_guide_counts(paths)
    per_reporter_medians = [
        int(np.median(arr)) if arr.size else 1 for arr in counts_per_reporter
    ]
    if counts_per_reporter:
        pooled = np.concatenate(counts_per_reporter)
        median_pool = int(np.median(pooled)) if pooled.size else 1
    else:
        median_pool = 1
    median_max_reporter = (
        max(per_reporter_medians) if per_reporter_medians else 1
    )

    payload = {
        "group": group,
        "sampling_mode": sampling_mode,
        "median_start_policy": median_start_policy,
        "n_reporters": len(paths),
        "per_reporter_medians": per_reporter_medians,
        "median_pool": median_pool,
        "median_max_reporter": median_max_reporter,
    }
    cache_p.write_text(json.dumps(payload, indent=2))
    print(f"[prep] {group}: medians cached → {cache_p}", flush=True)
    return payload


def _build_per_ko_max_schedule(paths: List[Path]) -> List[int]:
    """cells/KO schedule from largest reporter's max non-NTC perturbation count."""
    starts = []
    for p in paths:
        a = ad.read_h5ad(p, backed="r")
        counts = a.obs.groupby(_pert_col(a), observed=True).size()
        non_ntc = _non_ntc(counts)
        starts.append(int(non_ntc.max() if len(non_ntc) else counts.max()))
    return _build_per_ko_schedule(int(np.max(starts)))


def _build_total_schedule(paths: List[Path]) -> List[int]:
    """Plain n_cells schedule starting at the SMALLEST reporter's total cell count."""
    totals = [ad.read_h5ad(p, backed="r").n_obs for p in paths]
    return _build_per_ko_schedule(int(min(totals)), MIN_CELLS)


# ---------------------------------------------------------------------------
# Core: subsample → aggregate → NTC-normalize → h-concat → score
# ---------------------------------------------------------------------------


def _apply_fixed_second_pca(
    adata_guide: ad.AnnData,
    threshold: float,
    _logger: Optional[logging.Logger] = None,
) -> ad.AnnData:
    """Fit a fixed-threshold 2nd-pass PCA on the hconcat'd guide feature matrix.

    Returns a new guide-level AnnData whose features are the top PCs covering
    ``threshold`` cumulative variance (var_names ``sPC0``, ``sPC1``, …). This
    is the same operation ``pca_optimization.apply_second_pass_pca`` performs
    in fixed-pct mode, but in-memory and per-titration-step.

    No-op when the matrix is too small to PCA (n_obs < 2 or n_features < 2).
    """
    X = np.asarray(adata_guide.X, dtype=np.float32)
    n_obs, n_feat = X.shape
    if n_obs < 2 or n_feat < 2:
        if _logger is not None:
            _logger.warning(
                f"  2nd-pass PCA skipped: matrix shape {n_obs}x{n_feat} too small"
            )
        return adata_guide
    X_pcs, cumvar, _model = fit_pca(X)
    n_keep = max(min(n_pcs_for_threshold(cumvar, threshold), X_pcs.shape[1]), 1)
    if _logger is not None:
        _logger.info(
            f"  2nd-pass PCA: {n_feat} → {n_keep} PCs "
            f"(cumvar ≥ {threshold:.2f}; covered = {cumvar[n_keep-1]:.3f})"
        )
    X_keep = X_pcs[:, :n_keep].astype(np.float32)
    new_var = pd.DataFrame(index=[f"sPC{i}" for i in range(n_keep)])
    return ad.AnnData(X=X_keep, obs=adata_guide.obs.copy(), var=new_var)


def _build_combined_at_target(
    cells_blocks: List[ad.AnnData],
    target: int,
    sampling_mode: str,
    norm_method: str,
    rng: np.random.RandomState,
    *,
    second_pca_threshold: float = 0.0,
    n_workers: int = 1,
    replace: bool = False,
    _logger: Optional[logging.Logger] = None,
) -> ad.AnnData:
    """For each reporter: subsample → aggregate → NTC-normalize, then h-concat.

    The per-reporter prep loop (subsample + aggregate + z-score) is the
    dominant cost of a titration step (~70% of wall time at large reporter
    counts) and embarrassingly parallel — each reporter is its own file,
    its own NTC pool, its own feature space. ``n_workers`` > 1 parallelizes
    that loop with a ThreadPoolExecutor. Numpy/pandas/anndata release the
    GIL on the heavy ops, so threads beat processes here (no pickling of
    AnnData blocks). Per-reporter RNGs are derived from the parent ``rng``
    *before* the parallel section so reproducibility holds regardless of
    completion order.

    When ``second_pca_threshold > 0`` AND the group has > 1 reporter, the
    concatenated guide matrix is passed through ``_apply_fixed_second_pca``
    before being returned — same as the standard pca_optimization pipeline
    consensus step, but with a fixed variance threshold per titration step.
    """
    # Pre-draw per-reporter seeds so parallel completion order can't change
    # the bootstrap result. Done sequentially on the parent rng.
    reporter_seeds = [
        int(rng.randint(0, 2**31 - 1)) for _ in range(len(cells_blocks))
    ]

    def _prep_one(idx_adata: Tuple[int, ad.AnnData]) -> ad.AnnData:
        idx, adata = idx_adata
        local_rng = np.random.RandomState(reporter_seeds[idx])
        g_sub = _subsample_one(adata, target, sampling_mode, local_rng, replace=replace)
        g_norm = normalize_guide_adata(g_sub, norm_method)
        sig = str(adata.obs.get("signal", pd.Series(["?"])).iloc[0])
        g_norm.var_names = [f"{sig}::{v}" for v in g_norm.var_names]
        return g_norm

    n_workers = max(1, min(int(n_workers), len(cells_blocks)))
    if n_workers == 1 or len(cells_blocks) == 1:
        blocks = [_prep_one((i, a)) for i, a in enumerate(cells_blocks)]
    else:
        from concurrent.futures import ThreadPoolExecutor  # noqa: WPS433
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            # map preserves input order — important for deterministic concat.
            blocks = list(pool.map(_prep_one, list(enumerate(cells_blocks))))

    combined = hconcat_by_perturbation(blocks, level="guide")
    # 2nd-pass PCA only makes sense for MULTI-reporter groups — it reduces
    # the dimensionality of the cross-reporter concat to balance variance
    # contributions. For a single-reporter group (e.g. phase_only) the
    # input is already the per-reporter PC space; running PCA on it again
    # just drops components and adds noise without doing the cross-reporter
    # rebalancing the step was designed for. Skip when there's only one
    # block, regardless of threshold.
    if second_pca_threshold > 0 and len(cells_blocks) > 1:
        combined = _apply_fixed_second_pca(
            combined, second_pca_threshold, _logger=_logger,
        )
    return combined


def run_combined_titration(
    cells_h5ad_paths: List[str],
    output_dir: str,
    sampling_mode: str = "per_guide",
    norm_method: str = "ntc",
    distance: str = "cosine",
    n_bootstraps: int = 1,
    random_seed: int = 42,
    schedule: Optional[List[int]] = None,
    group_label: str = "combined",
    cache: bool = True,
    second_pca_threshold: float = 0.0,
    schedule_start_override: Optional[int] = None,
    median_start_policy: str = "pool",
    n_workers: int = 1,
    replace: bool = False,
) -> str:
    """Run the combined-titration loop for one group and write CSV + plots.

    sampling_mode: 'per_guide' | 'per_ko' | 'total' — interprets ``schedule``
        targets as cells/sgRNA, cells/perturbation, or absolute n_cells.
    cache: when True (default), reuse already-scored rows from any existing
        combined_titration_<group>.csv and only score the missing schedule
        targets. Pass --no-cache (CLI) to force a full recompute. Cached rows
        whose ``second_pca_threshold`` column disagrees with the requested
        value are dropped so the cache stays consistent across threshold
        changes without renaming the CSV.
    second_pca_threshold: when > 0 AND the group has > 1 reporter, fit a
        2nd-pass PCA on the hconcat'd guide matrix at each titration step and
        keep components up to this cumulative-variance threshold. Set to 0
        (default) to disable. Single-reporter groups always skip this step,
        regardless of the threshold value.
    """
    _logger = logging.getLogger(f"combined_titration.{group_label}")
    if not _logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        _logger.addHandler(h)
        _logger.setLevel(logging.INFO)

    paths = [Path(p) for p in cells_h5ad_paths]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(random_seed)

    # 2nd-pass PCA is a multi-reporter dim-reduction step; single-reporter
    # groups (phase_only) skip it inside `_build_combined_at_target`. We
    # still pass the threshold through for cache-key consistency, but the
    # actual transform only runs when len(cells_blocks) > 1.
    effective_second_pca = float(second_pca_threshold) if second_pca_threshold > 0 else 0.0
    if effective_second_pca > 0:
        _logger.info(
            f"[{group_label}] 2nd-pass PCA on at every titration step "
            f"(fixed threshold = {effective_second_pca:.2f})."
        )

    eff_workers = max(1, min(int(n_workers), len(paths)))
    _logger.info(
        f"[{group_label}] Loading {len(paths)} reporter cells.h5ads "
        f"(mode={sampling_mode}, bootstrap={n_bootstraps}, "
        f"prep_threads={eff_workers})..."
    )
    cells_blocks: List[ad.AnnData] = []
    for p in paths:
        a = ad.read_h5ad(p)
        if "signal" not in a.obs.columns:
            sig_guess = p.stem.replace("_cells", "")
            a.obs["signal"] = sig_guess
        cells_blocks.append(a)
        _logger.info(f"  {p.name}: {a.n_obs:,} cells x {a.n_vars} PCs")

    if schedule is None:
        if sampling_mode == "per_guide":
            schedule = _build_per_guide_max_schedule(paths)
        elif sampling_mode == "per_guide_median":
            schedule = _build_per_guide_median_schedule(
                paths,
                start_override=schedule_start_override,
                policy=median_start_policy,
            )
        elif sampling_mode == "per_ko":
            schedule = _build_per_ko_max_schedule(paths)
        else:
            schedule = _build_total_schedule(paths)
    _logger.info(f"[{group_label}] Schedule ({len(schedule)} pts): {schedule}")

    csv_path = out_dir / f"combined_titration_{group_label}.csv"
    target_col = _target_col(sampling_mode)
    # Cache-by-threshold: a cached row only counts when it was scored at the
    # same second_pca_threshold (missing column == 0.0, i.e. a legacy
    # no-second-pca run). That filter is the one combined-specific piece; the
    # rest of the cache split is shared with the per-reporter titration.
    def _same_threshold(df_old: pd.DataFrame) -> pd.DataFrame:
        stored = (
            df_old.get("second_pca_threshold", pd.Series([0.0] * len(df_old)))
                  .fillna(0.0).astype(float)
        )
        keep = np.isclose(stored.to_numpy(), float(effective_second_pca), atol=1e-6)
        n_dropped = int((~keep).sum())
        if n_dropped:
            _logger.info(
                f"[{group_label}] Dropping {n_dropped} cached rows whose "
                f"second_pca_threshold ≠ {effective_second_pca:.2f}"
            )
        return df_old.loc[keep].reset_index(drop=True)

    targets_to_run, cached_rows = (
        _cache_split(
            csv_path, list(schedule), target_col, _logger,
            row_filter=_same_threshold,
        )
        if cache else (list(schedule), [])
    )

    metric_cols = list(METRIC_COLUMNS)
    base_seed = int(rng.randint(0, 2**31 - 1))
    rows = []

    unit = _UNIT_BY_MODE[sampling_mode]

    for target in targets_to_run:
        _logger.info(
            f"[{group_label}] Scoring at {target:,} {unit} "
            f"({n_bootstraps} draw{'s' if n_bootstraps > 1 else ''})..."
        )
        t_step = time.time()

        draws: List[Dict[str, float]] = []
        last_combined = None
        for b in range(n_bootstraps):
            draw_rng = np.random.RandomState(base_seed + b * 9973 + target)
            combined = _build_combined_at_target(
                cells_blocks, target, sampling_mode, norm_method, draw_rng,
                second_pca_threshold=effective_second_pca,
                n_workers=int(n_workers),
                replace=replace,
                _logger=_logger,
            )
            scores_b = _score_all_metrics(combined, _logger)
            draws.append(scores_b)
            last_combined = combined

        scores: Dict[str, float] = _aggregate_draws(draws, metric_cols, n_bootstraps)

        # x-axis bookkeeping
        n_guides = last_combined.n_obs if last_combined is not None else 0
        pert_col = _pert_col(last_combined)
        n_perts = last_combined.obs[pert_col].nunique()
        n_reporters = len(cells_blocks)
        if sampling_mode in ("per_guide", "per_guide_median"):
            scores["cells_per_guide"] = target
            scores["n_cells"] = target * n_guides * n_reporters
            scores["cells_per_perturbation"] = (
                scores["n_cells"] / max(n_perts * n_reporters, 1)
            )
        elif sampling_mode == "per_ko":
            scores["cells_per_perturbation"] = target
            scores["n_cells"] = target * n_perts * n_reporters
            scores["cells_per_guide"] = (
                scores["n_cells"] / max(n_guides * n_reporters, 1)
            )
        else:
            scores["n_cells"] = target * n_reporters
            scores["cells_per_guide"] = target / max(n_guides, 1)
            scores["cells_per_perturbation"] = target / max(n_perts, 1)
        scores["n_guides"] = int(n_guides)
        scores["n_perturbations"] = int(n_perts)
        scores["n_reporters"] = n_reporters
        # n_bootstraps is already set by _aggregate_draws.
        scores["group"] = group_label
        scores["second_pca_threshold"] = float(effective_second_pca)
        rows.append(scores)

        _logger.info(
            f"  act={scores['activity_map_mean']:.3f} "
            f"dist={scores['distinctiveness_map_mean']:.3f} "
            f"corum={scores['corum_map_mean']:.3f} "
            f"chad={scores['chad_map_mean']:.3f} "
            f"ebi={scores['ebi_map_mean']:.3f} "
            f"({time.time() - t_step:.0f}s)"
        )

    # Merge cached + newly-scored, dedupe on the target column, sort descending
    df = _merge_and_write(
        pd.DataFrame(rows), cached_rows, target_col, csv_path, _logger,
    )

    # Per-metric plot
    _plot_group_curves(df, group_label, out_dir, sampling_mode)
    return f"SUCCESS: {csv_path}"


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_group_curves(
    df: pd.DataFrame, group_label: str, out_dir: Path, sampling_mode: str,
) -> None:
    plt = _plt()
    x_col = _target_col(sampling_mode)
    if x_col not in df.columns or df.empty:
        return
    x = df[x_col].values
    out_dir.mkdir(parents=True, exist_ok=True)

    n_metrics = len(METRICS)
    for scale in SCALES:
        fig, axes = plt.subplots(1, n_metrics, figsize=(5.5 * n_metrics, 5), sharex=True)
        for ax, metric in zip(axes, METRICS):
            ycol = f"{metric}_map_mean"
            sem_col = f"{ycol}_sem"
            if ycol not in df.columns:
                continue
            y = df[ycol].values
            ax.plot(x, y, marker="o", lw=2.5, color="darkorange", label=group_label)
            if sem_col in df.columns:
                sem = df[sem_col].values
                ax.fill_between(x, y - sem, y + sem, color="darkorange", alpha=0.25, lw=0)
            ax.set_title(TITRATION_MAP_LABELS[metric], fontsize=12)
            ax.set_xlabel(titration_x_axis_base_label(x_col), fontsize=11)
            ax.set_ylabel("mean mAP", fontsize=11)
            _apply_x_scale(ax, x, scale, tick_fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
        fig.suptitle(
            f"Combined titration — {group_label} "
            f"(n_reporters={int(df['n_reporters'].iloc[0])}, "
            f"x={SCALE_LABEL_SHORT[scale]})",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()
        stem = out_dir / f"combined_titration_{group_label}_{scale}"
        fig.savefig(stem.with_suffix(".png"), dpi=160, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)


def plot_group_comparison(
    csvs_by_group: Dict[str, Path],
    output_dir: Path,
    sampling_mode: str,
    title_prefix: str = "Combined titration",
) -> None:
    """One canvas per x-axis scale, every metric as a subplot, all groups
    overlaid in each subplot."""
    plt = _plt()
    output_dir.mkdir(parents=True, exist_ok=True)
    dfs: Dict[str, pd.DataFrame] = {}
    for g, csv in csvs_by_group.items():
        if not Path(csv).is_file():
            logger.warning("Missing CSV for %s: %s", g, csv)
            continue
        dfs[g] = pd.read_csv(csv)
    if len(dfs) < 2:
        logger.warning("Need >=2 groups for comparison; got %d", len(dfs))
        return

    x_col = _target_col(sampling_mode)

    # Dump the exact (x, y, sem) points plotted, long format, one row per
    # (group, metric, x). Same data for every scale variant of the plot.
    long_rows: List[Dict] = []
    for g, df in dfs.items():
        if x_col not in df.columns:
            continue
        n_rep = int(df["n_reporters"].iloc[0]) if "n_reporters" in df.columns else 0
        for metric in METRICS:
            ycol = f"{metric}_map_mean"
            sem_col = f"{ycol}_sem"
            if ycol not in df.columns:
                continue
            for _, row in df.iterrows():
                long_rows.append({
                    "group": g,
                    "n_reporters": n_rep,
                    "metric": metric,
                    "x_col": x_col,
                    "x": float(row[x_col]),
                    "y": float(row[ycol]),
                    "sem": float(row[sem_col]) if sem_col in df.columns and pd.notna(row.get(sem_col)) else float("nan"),
                })
    if long_rows:
        compare_csv = output_dir / f"compare_all_metrics_{x_col}.csv"
        pd.DataFrame(long_rows).to_csv(compare_csv, index=False)
        logger.info("Wrote %s", compare_csv)
    x_label = titration_x_axis_base_label(x_col)
    # Group names are user-supplied labels now, so colours come from a cycle
    # rather than a name lookup.
    palette = [
        "#d97706", "#2563eb", "#10b981", "#dc2626",
        "#7c3aed", "#0891b2", "#a16207", "#6b7280",
    ]

    group_labels: Dict[str, str] = {}
    for i, (g, df) in enumerate(dfs.items()):
        n_rep = int(df["n_reporters"].iloc[0]) if "n_reporters" in df.columns else 0
        group_labels[g] = f"{g} (n={n_rep})"

    x_all = np.concatenate([
        d[x_col].values for d in dfs.values() if x_col in d.columns
    ])

    for scale in SCALES:
        fig, axes = plt.subplots(1, len(METRICS), figsize=(5.5 * len(METRICS), 6), sharex=True)
        for ax, metric in zip(axes, METRICS):
            ycol = f"{metric}_map_mean"
            sem_col = f"{ycol}_sem"
            for i, (g, df) in enumerate(dfs.items()):
                if x_col not in df.columns or ycol not in df.columns:
                    continue
                x = df[x_col].values
                y = df[ycol].values
                color = palette[i % len(palette)]
                ax.plot(
                    x, y, marker="o", lw=3.0, ms=7, color=color,
                    label=group_labels[g],
                )
                if sem_col in df.columns:
                    sem = df[sem_col].values
                    ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.2, lw=0)
            ax.set_title(TITRATION_MAP_LABELS[metric], fontsize=13)
            ax.set_xlabel(x_label, fontsize=12)
            ax.set_ylabel("mean mAP", fontsize=12)
            _apply_x_scale(ax, x_all, scale, tick_fontsize=10)
            ax.grid(True, alpha=0.3)

        # Single shared legend at the top
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles, labels, loc="upper center", ncol=len(labels),
                fontsize=12, frameon=False, bbox_to_anchor=(0.5, 0.99),
            )
        fig.suptitle(
            f"{title_prefix} — {x_label} ({SCALE_LABEL_SHORT[scale]})",
            fontsize=14, fontweight="bold", y=0.93,
        )

        fig.tight_layout(rect=(0, 0, 1, 0.91))
        stem = output_dir / f"compare_all_metrics_{x_col}_{scale}"
        fig.savefig(stem.with_suffix(".png"), dpi=160, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)
        logger.info("Wrote %s.png/svg", stem)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Reuse titration's parser to share path-resolution flags, then add ours."""
    p = _titr_parser()
    p.description = "Combined-reporter cell-count titration."
    # Combined-titration parallelizes the per-reporter prep loop across CPUs;
    # 32 is the sweet spot on 50+ reporter groups (per Amdahl's analysis on
    # ~35s of serial scoring vs ~225s of parallelizable prep). Override the
    # upstream titration default (8) which was tuned for the 1-reporter
    # per-job case. Users can still pass --slurm-cpus to override.
    p.set_defaults(slurm_cpus=32)
    p.add_argument(
        "--group", dest="group_specs", action="append", metavar="NAME=PATHS",
        required=True,
        help="A named set of reporters to combine, as NAME=path1[,path2,...] "
             "pointing at *_cells.h5ad files. Repeat the flag for several "
             "groups; with two or more, their curves are overlaid in a "
             "comparison plot. NAME becomes the output directory leaf, so it "
             "must be unique and filesystem-safe. Example: "
             "--group phase=/…/Phase_cells.h5ad "
             "--group phase_fe=/…/Phase_cells.h5ad,/…/FeRhoNox_cells.h5ad",
    )
    p.add_argument(
        "--no-compare", action="store_true",
        help="Skip the cross-group comparison plot at the end",
    )
    p.add_argument(
        "--compare-only", action="store_true",
        help="Skip the per-group titration step entirely and just regenerate "
             "the comparison plots from the existing per-group CSVs. "
             "Errors out if any expected combined_titration_<group>.csv is missing.",
    )
    p.add_argument(
        "--no-shared-start", dest="shared_start", action="store_false", default=True,
        help="In --per-guide-median-titration mode, let each group start at its "
             "own median instead of the smallest median across groups (default: "
             "share the start so curves align at the top).",
    )
    p.add_argument(
        "--median-start-policy", type=str, default="pool",
        choices=list(_MEDIAN_START_POLICIES),
        help="Start point for --per-guide-median-titration schedules. "
             "'pool' (default): median of all per-sgRNA cell counts pooled "
             "across reporters in the group. Conservative — dominated by the "
             "more numerous reporters when one reporter has many more "
             "cells/guide. 'max_reporter': max of per-reporter medians. Lets "
             "the schedule climb up to the biggest reporter's natural median, "
             "with smaller reporters saturating at their own max along the "
             "way. Use 'max_reporter' for groups where one reporter (e.g. "
             "Phase) has substantially more cells/guide than the others.",
    )
    p.add_argument(
        "--n-workers", type=int, default=None,
        help="Threads for the per-reporter subsample/aggregate/normalize loop "
             "inside each titration step. Default: --slurm-cpus when --slurm "
             "is set, else os.cpu_count(). Each thread handles one reporter "
             "at a time; numpy releases the GIL on the heavy ops so 8 threads "
             "≈ 5× faster than serial on a 56-reporter group.",
    )
    p.add_argument(
        "--second-pca-threshold", type=float, default=0.0,
        help="Cumulative-variance threshold for a 2nd-pass PCA applied to the "
             "h-concatenated guide matrix at every titration step. Default 0 "
             "(off). Set e.g. 0.40 to match the fixed-pct second-pass that "
             "pca_optimization runs on multi-marker groups — this puts "
             "all_fluor and livecell on the same dimensional footing as the "
             "rest of the paper_v1 pipeline. Single-reporter groups always "
             "skip this step regardless of the threshold value (e.g. "
             "phase_only stays untouched).",
    )
    # --no-cache is inherited from titration's parser (same semantics).
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for cell subsampling",
    )
    # --per-target-slurm is inherited from titration's parser; both
    # scripts use identical (action="store_true") semantics — one task per
    # schedule target.
    p.add_argument(
        "--max-schedule-points", type=int, default=None,
        help="Cap each group's schedule to its top N points (largest "
             "cells/guide first). The downsample ladder spans ~35 targets "
             "from the median down to 1 cell/guide; for a 'does this curve "
             "plateau at X?' diagnostic the bottom of the curve adds little "
             "and the top ~8-12 points already show the asymptote. Combine "
             "with --per-target-slurm for maximum parallel speedup.",
    )
    return p


def _sampling_mode(args: argparse.Namespace) -> str:
    if getattr(args, "per_guide_median_titration", False):
        return "per_guide_median"
    if args.per_guide_max_titration or args.per_guide_min_titration:
        return "per_guide"
    if args.per_ko_max_titration or args.per_ko_min_titration:
        return "per_ko"
    return "total"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args()
    sampling_mode = _sampling_mode(args)
    compare_only = getattr(args, "compare_only", False)

    # --compare-only re-plots from existing per-group CSVs, so the input paths
    # aren't needed (and needn't still exist) — only the names.
    parsed = _parse_group_specs(args.group_specs)
    groups = list(parsed)
    group_paths: Dict[str, List[Path]] = (
        {g: [] for g in groups} if compare_only else parsed
    )
    group_outdirs: Dict[str, Path] = {
        g: _resolve_group_output_dir(args, g, sampling_mode) for g in groups
    }
    for g in groups:
        if compare_only:
            print(f"[{g}] → {group_outdirs[g]}")
        else:
            print(f"[{g}] {len(group_paths[g])} reporters → {group_outdirs[g]}")

    # For median mode with multiple groups, by default cap every group's start
    # (the median) at the smallest median across groups so curves align at the
    # top of the x-axis. Pass --no-shared-start to let each group start at its
    # own median instead. Every group titrates down to 1 cell/guide regardless.
    # In --compare-only mode we never run the schedule, so skip the expensive
    # median computation (which opens every reporter h5ad in backed mode).
    # Schedule construction is deferred to each SLURM worker so the login node
    # doesn't open every h5ad in backed mode for every group. When
    # ``--shared-start`` (default) is on we still need ONE pool computation per
    # group on the login node to find the min median across groups; that
    # number is then passed as ``schedule_start_override`` to each worker, and
    # the worker builds its own schedule with that override (no extra reads).
    # When ``--no-shared-start`` is set, the login node does zero schedule
    # work — every worker independently computes its own median and schedule.
    group_schedules: Dict[str, Optional[List[int]]] = {g: None for g in groups}
    schedule_start_overrides: Dict[str, Optional[int]] = {g: None for g in groups}

    # ── SLURM-prep stage: fan out median computation in parallel ─────────
    # When --per-target-slurm is on we'd otherwise open every reporter
    # h5ad on the login node (sequential per group) just to compute medians
    # for the schedule. That's slow on big-Phase groups (livecell takes
    # ~5-10 min on its own). Submit 1 prep job per group, cache per-group
    # medians as JSON, and let the login node read them back in O(ms).
    # Cache is keyed on (sampling_mode, median_start_policy, n_reporters)
    # so reruns with identical args skip the prep entirely.
    needs_slurm_prep = (
        args.slurm
        and args.per_target_slurm
        and sampling_mode == "per_guide_median"
        and not getattr(args, "compare_only", False)
    )
    prep_payloads: Dict[str, dict] = {}
    if needs_slurm_prep:
        import json as _json
        from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs as _spj
        prep_jobs = []
        for g in groups:
            cache_p = group_outdirs[g] / "schedule_cache.json"
            if args.cache and cache_p.is_file():
                try:
                    p = _json.loads(cache_p.read_text())
                    if (p.get("sampling_mode") == sampling_mode
                            and p.get("median_start_policy") == args.median_start_policy
                            and p.get("n_reporters") == len(group_paths[g])):
                        prep_payloads[g] = p
                        print(f"[prep cache hit] {g}: medians from {cache_p}")
                        continue
                except Exception:
                    pass
            prep_jobs.append({
                "name": f"prep_{g}",
                "func": _prep_schedule_worker,
                "kwargs": {
                    "group": g,
                    "cells_h5ad_paths": [str(p) for p in group_paths[g]],
                    "sampling_mode": sampling_mode,
                    "median_start_policy": args.median_start_policy,
                    "cache_path": str(cache_p),
                },
                "metadata": {"group": g, "cache_path": str(cache_p)},
            })
        if prep_jobs:
            # Prep is I/O-bound (groupby on backed h5ad obs). 8 CPUs is
            # plenty; reuse phase_slurm_time as the budget since the
            # big-Phase group is the long-pole here.
            prep_params = {
                "mem": args.slurm_memory,
                "cpus_per_task": min(int(args.slurm_cpus), 8),
                "slurm_partition": args.slurm_partition,
                "timeout_min": min(int(args.phase_slurm_time), 90),
            }
            print(
                f"\n[prep] {len(prep_jobs)} schedule-prep SLURM job(s) "
                f"({prep_params['timeout_min']}min × {prep_params['cpus_per_task']} CPUs)..."
            )
            _spj(
                jobs_to_submit=prep_jobs,
                experiment="combined_titration",
                slurm_params=prep_params,
                log_dir="pca_optimization",
                manifest_prefix="pca_combtitr_prep",
                wait_for_completion=True,
                verbose=True,
            )
            for j in prep_jobs:
                cache_p = Path(j["metadata"]["cache_path"])
                if cache_p.is_file():
                    prep_payloads[j["metadata"]["group"]] = _json.loads(
                        cache_p.read_text()
                    )
                else:
                    print(f"  [warn] prep job {j['name']} produced no cache file")

        # Resolve schedule_start_overrides from cached medians.
        if prep_payloads:
            policy = args.median_start_policy
            starts = {
                g: (p["median_max_reporter"] if policy == "max_reporter" else p["median_pool"])
                for g, p in prep_payloads.items()
            }
            if args.shared_start and len(groups) > 1:
                shared_start = min(starts.values())
                print(
                    f"[prep] shared start (policy={policy}) = {shared_start} "
                    f"(per-group: {starts})"
                )
                for g in groups:
                    schedule_start_overrides[g] = int(shared_start)
            else:
                print(f"[prep] per-group starts (policy={policy}): {starts}")
                for g, v in starts.items():
                    schedule_start_overrides[g] = int(v)

    if not needs_slurm_prep and (
        sampling_mode == "per_guide_median"
        and len(groups) > 1
        and not getattr(args, "compare_only", False)
        and args.shared_start
    ):
        medians = {
            g: _per_guide_median_start(group_paths[g], policy=args.median_start_policy)
            for g in groups
        }
        shared_start = min(medians.values())
        print(
            f"\nMedian-mode shared start (policy={args.median_start_policy}): "
            f"min start across groups = {shared_start} "
            f"(per-group starts: {medians})"
        )
        for g in groups:
            schedule_start_overrides[g] = int(shared_start)
    elif not needs_slurm_prep and (
        sampling_mode == "per_guide_median"
        and not getattr(args, "compare_only", False)
        and not args.shared_start
    ):
        print(
            f"\nMedian-mode per-group starts (policy={args.median_start_policy}): "
            "each group runs from its own start to 1 (computed inside each "
            "worker — no login-node reads)."
        )

    # ── Pre-build per-group schedules on the login node when needed ──────
    # The per-target SLURM path needs the exact schedule up front (it fans
    # out one task per target), and --max-schedule-points wants the schedule
    # trimmed before workers ever see it. We only pay this cost when either
    # flag is set so the default path stays cheap (worker-side schedule build,
    # no extra login-node h5ad reads).
    if not compare_only and (args.per_target_slurm or args.max_schedule_points):
        print("\nBuilding per-group schedules on login node...")
        for g in groups:
            paths = group_paths[g]
            if sampling_mode == "per_guide_median":
                sched = _build_per_guide_median_schedule(
                    paths,
                    start_override=schedule_start_overrides[g],
                    policy=args.median_start_policy,
                )
            elif sampling_mode == "per_guide":
                sched = _build_per_guide_max_schedule(paths)
            elif sampling_mode == "per_ko":
                sched = _build_per_ko_max_schedule(paths)
            else:
                sched = _build_total_schedule(paths)
            sched = list(sched)
            if args.max_schedule_points and args.max_schedule_points > 0:
                # Schedules are in descending order (largest target first),
                # so the top-N slice = the high-cells-per-guide head of the
                # curve, which is where mAP changes are visible.
                sched = sched[: int(args.max_schedule_points)]
            group_schedules[g] = sched
            print(
                f"  {g}: {len(sched)} target(s) "
                f"(head: {sched[:3]}{', tail: ' + str(sched[-2:]) if len(sched) > 3 else ''})"
            )

    # Run each group (locally or in parallel SLURM)
    csvs_by_group: Dict[str, Path] = {}
    if getattr(args, "compare_only", False):
        # Skip the per-group step entirely; just collect existing CSV paths.
        missing = []
        for g in groups:
            csv = group_outdirs[g] / f"combined_titration_{g}.csv"
            if csv.is_file():
                csvs_by_group[g] = csv
            else:
                missing.append(str(csv))
        if missing:
            raise SystemExit(
                f"--compare-only: {len(missing)} expected per-group CSV(s) missing. "
                f"Run without --compare-only first. Missing:\n  "
                + "\n  ".join(missing[:10])
                + (f"\n  ... (+{len(missing) - 10} more)" if len(missing) > 10 else "")
            )
        print(f"\n--compare-only: skipping per-group step; loading {len(csvs_by_group)} cached CSVs.")
    elif args.slurm and args.per_target_slurm:
        # ── Per-target SLURM fan-out ─────────────────────────────────────
        # One array task per (group, target). Each task reloads its
        # cells.h5ad set (the I/O cost) but computes only ONE schedule
        # point; on a livecell-sized group this turns the inner ~35-target
        # serial loop into N_targets-way parallelism, gated only by
        # cluster slots. We submit ALL (group, target) jobs in one
        # submit_parallel_jobs call so SLURM sees them as a single array.
        from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
        target_col = _target_col(sampling_mode)
        n_workers = (
            int(args.n_workers) if args.n_workers is not None
            else int(args.slurm_cpus)
        )
        thr = float(args.second_pca_threshold)

        all_jobs = []
        skipped_cache = 0
        for g in groups:
            paths = group_paths[g]
            for target in group_schedules[g] or []:
                shard_label = f"{g}_t{int(target)}"
                out_csv = group_outdirs[g] / f"combined_titration_{shard_label}.csv"
                # Cache short-circuit: if this exact (group, target,
                # second_pca_threshold) was already computed, skip the
                # task entirely. Cheaper than letting the worker open
                # h5ads just to no-op via the in-worker cache logic.
                if args.cache and out_csv.is_file():
                    try:
                        d = pd.read_csv(out_csv)
                        stored_thr = (
                            float(d.get("second_pca_threshold", pd.Series([0.0])).iloc[0])
                            if len(d) else None
                        )
                        if stored_thr is not None and np.isclose(stored_thr, thr, atol=1e-6):
                            skipped_cache += 1
                            continue
                    except Exception:
                        pass
                all_jobs.append({
                    "name": f"combtitr_{shard_label}",
                    "func": run_combined_titration,
                    "kwargs": {
                        "cells_h5ad_paths": [str(p) for p in paths],
                        "output_dir": str(group_outdirs[g]),
                        "sampling_mode": sampling_mode,
                        "norm_method": args.norm_method,
                        "distance": args.distance,
                        "n_bootstraps": int(args.bootstrap),
                        "random_seed": int(args.seed),
                        "group_label": shard_label,
                        # We pre-filtered cache hits above, so let the
                        # worker recompute unconditionally for its slice.
                        "cache": False,
                        "schedule": [int(target)],
                        "schedule_start_override": None,
                        "median_start_policy": args.median_start_policy,
                        "second_pca_threshold": thr,
                        "n_workers": int(n_workers),
                        "replace": bool(args.bootstrap_replace),
                    },
                    "metadata": {"group": g, "target": int(target),
                                  "shard_csv": str(out_csv)},
                })
        n_planned = sum(len(group_schedules[g] or []) for g in groups)
        print(
            f"\n[per-target] {len(all_jobs)} task(s) to submit "
            f"({skipped_cache} cached / {n_planned} planned across "
            f"{len(groups)} groups)..."
        )
        if all_jobs:
            slurm_params = {
                "mem": args.slurm_memory,
                "cpus_per_task": args.slurm_cpus,
                "slurm_partition": args.slurm_partition,
                # Each task does (h5ad load) + (one target). For Phase-heavy
                # groups, h5ad load dominates if the user kept --slurm-time
                # at 30; we honor the explicit value, only warning if the
                # group has Phase.
                "timeout_min": args.slurm_time,
            }
            print(
                f"  slurm: {args.slurm_time}min, {args.slurm_memory}, "
                f"{args.slurm_cpus} CPUs per task; partition={args.slurm_partition}"
            )
            submit_parallel_jobs(
                jobs_to_submit=all_jobs,
                experiment="combined_titration",
                slurm_params=slurm_params,
                log_dir="pca_optimization",
                manifest_prefix="pca_combtitr_per_target",
                wait_for_completion=True,
                verbose=True,
            )

        # ── Merge per-target shards into per-group canonical CSVs ────────
        for g in groups:
            parts = []
            for target in group_schedules[g] or []:
                p = group_outdirs[g] / f"combined_titration_{g}_t{int(target)}.csv"
                if p.is_file():
                    try:
                        parts.append(pd.read_csv(p))
                    except pd.errors.EmptyDataError:
                        continue
            if not parts:
                print(f"[merge] {g}: no per-target CSVs produced — skipping group")
                continue
            canonical = group_outdirs[g] / f"combined_titration_{g}.csv"
            # Shared merge helper: dedupes on target_col with keep="last" (this
            # site previously kept the FIRST row, so a re-scored target was
            # silently discarded here while the other three merge sites took it).
            merged = _merge_and_write(
                pd.concat(parts, ignore_index=True), [], target_col, canonical, logger,
            )
            csvs_by_group[g] = canonical
            print(f"[merge] {g}: {len(merged)} rows → {canonical}")
    elif args.slurm:
        from ops_utils.hpc.slurm_batch_utils import (
            submit_parallel_jobs,
            wait_for_multiple_job_arrays,
        )

        base_slurm_params = {
            "mem": args.slurm_memory,
            "cpus_per_task": args.slurm_cpus,
            "slurm_partition": args.slurm_partition,
        }
        # Default to one prep-thread per allocated CPU when on SLURM.
        n_workers = (
            int(args.n_workers) if args.n_workers is not None
            else int(args.slurm_cpus)
        )
        # Auto-bump timeout for any group that includes the Phase reporter
        # (~60M cells / 25GB) to args.phase_slurm_time (default 240min). All
        # reporters in a combined-titration group are h-concatted into one job,
        # so we can't split Phase out the way titration does.
        def _has_phase(paths: List[Path]) -> bool:
            return any("phase" in p.stem.lower() for p in paths)

        # Combined-titration runtime scales with reporter count (each
        # reporter is z-score-normalized + h-concatted at every schedule
        # point), not just Phase presence. Treat any group with ≥10 markers
        # OR Phase as a "big" group and use the larger budget.
        _BIG_GROUP_MIN_MARKERS = 10

        job_arrays = []
        slurm_time_user_set = args.slurm_time != parser.get_default("slurm_time")
        for g in groups:
            n_markers = len(group_paths[g])
            phase_in_group = _has_phase(group_paths[g])
            is_big_group = phase_in_group or n_markers >= _BIG_GROUP_MIN_MARKERS
            if slurm_time_user_set:
                # Explicit --slurm-time always wins.
                timeout_min = args.slurm_time
            elif is_big_group:
                timeout_min = args.phase_slurm_time
            else:
                timeout_min = args.slurm_time
            # Mirror titration's bootstrap autoscaling for the big-group budget
            if (
                is_big_group and args.bootstrap > 1
                and timeout_min == parser.get_default("phase_slurm_time")
            ):
                timeout_min = timeout_min * int(args.bootstrap)
            slurm_params = {**base_slurm_params, "timeout_min": timeout_min}
            job = {
                "name": f"combtitr_{g}",
                "func": run_combined_titration,
                "kwargs": {
                    "cells_h5ad_paths": [str(p) for p in group_paths[g]],
                    "output_dir": str(group_outdirs[g]),
                    "sampling_mode": sampling_mode,
                    "norm_method": args.norm_method,
                    "distance": args.distance,
                    "n_bootstraps": int(args.bootstrap),
                    "random_seed": int(args.seed),
                    "group_label": g,
                    "cache": bool(args.cache),
                    "schedule": group_schedules[g],
                    "schedule_start_override": schedule_start_overrides[g],
                    "median_start_policy": args.median_start_policy,
                    "second_pca_threshold": float(args.second_pca_threshold),
                    "n_workers": int(n_workers),
                    "replace": bool(args.bootstrap_replace),
                },
            }
            if phase_in_group:
                big_reason = " (Phase reporter present — bumped time)"
            elif is_big_group:
                big_reason = f" ({n_markers} markers ≥ {_BIG_GROUP_MIN_MARKERS} — bumped time)"
            else:
                big_reason = ""
            print(
                f"\nSubmitting combined-titration SLURM job for group={g} "
                f"({timeout_min}min, {args.slurm_memory}){big_reason}..."
            )
            result = submit_parallel_jobs(
                jobs_to_submit=[job],
                experiment="combined_titration",
                slurm_params=slurm_params,
                log_dir="pca_optimization",
                manifest_prefix=f"pca_combtitr_{g}",
                wait_for_completion=False,
            )
            if result.get("submitted_jobs"):
                job_arrays.append({
                    "submitted_jobs": result["submitted_jobs"],
                    "base_job_id": result["base_job_id"],
                    "label": g,
                    "slurm_params": slurm_params,
                })
            csvs_by_group[g] = (
                group_outdirs[g] / f"combined_titration_{g}.csv"
            )

        if job_arrays:
            wait_for_multiple_job_arrays(job_arrays, experiment="combined_titration")
    else:
        # Default thread pool size for local mode: --n-workers if set,
        # else os.cpu_count() (capped per group inside _build_combined_at_target).
        import os as _os  # noqa: WPS433
        n_workers_local = (
            int(args.n_workers) if args.n_workers is not None
            else max(1, _os.cpu_count() or 1)
        )
        for g in groups:
            run_combined_titration(
                cells_h5ad_paths=[str(p) for p in group_paths[g]],
                output_dir=str(group_outdirs[g]),
                sampling_mode=sampling_mode,
                norm_method=args.norm_method,
                distance=args.distance,
                n_bootstraps=int(args.bootstrap),
                random_seed=int(args.seed),
                group_label=g,
                cache=bool(args.cache),
                schedule=group_schedules[g],
                schedule_start_override=schedule_start_overrides[g],
                median_start_policy=args.median_start_policy,
                second_pca_threshold=float(args.second_pca_threshold),
                n_workers=int(n_workers_local),
                replace=bool(args.bootstrap_replace),
            )
            csvs_by_group[g] = group_outdirs[g] / f"combined_titration_{g}.csv"

    # Cross-group comparison
    if not args.no_compare and len(csvs_by_group) >= 2:
        compare_dir = _resolve_compare_dir(args, groups, sampling_mode)
        compare_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nComparison plots → {compare_dir}")
        # Manifest of what went into the comparison: the leaf dir name can be
        # truncated, and it doesn't record which reporters each group held.
        (compare_dir / "groups.txt").write_text(
            "".join(
                f"{g}\t" + ",".join(p.name for p in group_paths.get(g, [])) + "\n"
                for g in csvs_by_group
            )
        )
        plot_group_comparison(
            {g: p for g, p in csvs_by_group.items()},
            output_dir=compare_dir,
            sampling_mode=sampling_mode,
            title_prefix=f"Combined titration ({sampling_mode})",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
