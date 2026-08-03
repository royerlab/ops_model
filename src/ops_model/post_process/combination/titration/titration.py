"""Per-reporter cell-count titration analysis.

Reads existing per-signal guide h5ads, repeatedly downsamples cells at each
guide (3/4 ratio per step), scores every metric in ``METRICS`` at each titration
point, and produces two summary plots per reporter plus a combined overview:

  1. **% significant** — fraction of perturbations/complexes passing corrected
     p-value threshold for each metric vs cell count.
  2. **mean mAP** — average mAP score for each metric vs cell count.

Usage::

    # Run locally for a single variant. -o is the same root passed to
    # pca_optimization; the remaining flags select the variant beneath it.
    python -m ops_model.post_process.combination.titration.titration \
        -o <pca_optimization_root> --cell-dino --paper-v1

    # Submit as SLURM jobs
    python -m ops_model.post_process.combination.titration.titration \
        -o <pca_optimization_root> --cell-dino --paper-v1 --slurm
"""

import argparse
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd

from ops_model.features.anndata_utils import (
    _guide_col,
    aggregate_to_level,
    normalize_guide_adata,
)
from ops_utils.analysis.map_scores import phenotypic_activity_assesment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOWNSAMPLE_RATIO = 0.75  # multiply cell count by this each step
MIN_CELLS = 5_000  # stop titrating below this
NULL_SIZE = 10_000  # smaller null for speed (per-reporter)
METRICS = ("activity", "distinctiveness", "corum", "chad", "ebi")
# Canonical per-metric CSV columns. Derived from METRICS so this module and
# combined_titration can't drift apart the way they did when each kept its own
# hand-written list.
METRIC_COLUMNS = tuple(
    f"{metric}_{kind}" for metric in METRICS for kind in ("ratio", "map_mean")
)
SCALES = ("linear", "log2", "log10")  # x-axis scale variants to save

# Shared plot styling / labels (used by compare_titration_versions and below)
TITRATION_METRIC_COLORS = {
    "activity": "steelblue",
    "distinctiveness": "mediumseagreen",
    "corum": "mediumpurple",
    "chad": "darkorange",
    "ebi": "crimson",
}
TITRATION_RATIO_LABELS = {
    "activity": "% Active",
    "distinctiveness": "% Distinctive",
    "corum": "% CORUM consistent",
    "chad": "% CHAD consistent",
    "ebi": "% EBI consistent",
}
TITRATION_MAP_LABELS = {
    "activity": "Activity mAP",
    "distinctiveness": "Distinctiveness mAP",
    "corum": "CORUM mAP",
    "chad": "CHAD mAP",
    "ebi": "EBI mAP",
}
SCALE_LABEL_SHORT = {"linear": "linear", "log2": "log₂", "log10": "log₁₀"}

def _format_cell_count(n: int) -> str:
    """Format cell count as human-readable string: 1.2M, 500K, 50K, etc."""
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{v:.1f}M" if v != int(v) else f"{int(v)}M"
    elif n >= 1_000:
        v = n / 1_000
        return f"{v:.0f}K" if v >= 10 else f"{v:.1f}K"
    return str(n)


def _round_ticks(x_min: float, x_max: float, n: int = 7) -> list:
    """Return ~n round tick values spanning [x_min, x_max].

    Generates geometrically-spaced candidate positions then rounds each to
    1 significant figure, so ticks always land on human-readable values
    (e.g. 50K, 100K, 500K, 1M) regardless of where the raw data falls.
    """
    positions = np.geomspace(x_min, x_max, n)
    ticks = []
    seen = set()
    for v in positions:
        mag = 10 ** math.floor(math.log10(v))
        rounded = int(round(v / mag) * mag)
        if rounded > 0 and rounded not in seen and x_min * 0.5 <= rounded <= x_max * 2:
            ticks.append(rounded)
            seen.add(rounded)
    return sorted(ticks)


def _apply_x_scale(ax, x_values, scale: str, tick_fontsize: int = 12):
    """Apply x-axis scale with round human-readable tick labels.

    Ticks are rounded to 1 significant figure so they always show clean
    values (50K, 100K, 500K, 1M…) regardless of the raw 0.75-ratio data.
    The ``scale`` parameter controls only the visual spacing of the axis.

    linear  — uniform spacing
    log2    — log base-2 spacing
    log10   — log base-10 spacing
    """
    from matplotlib.ticker import FuncFormatter, NullFormatter

    x_min, x_max = min(x_values), max(x_values)
    ticks = _round_ticks(x_min, x_max)

    if scale == "linear":
        ax.set_xscale("linear")
    elif scale == "log2":
        ax.set_xscale("log", base=2)
    elif scale == "log10":
        ax.set_xscale("log", base=10)
    else:
        raise ValueError(f"Unknown scale: {scale!r}")

    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [_format_cell_count(t) for t in ticks],
        rotation=45,
        ha="right",
        fontsize=tick_fontsize,
    )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _format_cell_count(int(v))))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_logger():
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger("copairs").setLevel(logging.WARNING)
    return logging.getLogger(__name__)


def _plt():
    """Return a pyplot bound to the Agg backend (headless SLURM workers)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _pert_col(adata: ad.AnnData) -> str:
    """The .obs column holding perturbation names."""
    return "perturbation" if "perturbation" in adata.obs.columns else "label_str"


def _non_ntc(counts: pd.Series) -> pd.Series:
    """Drop NTC entries from a Series indexed by perturbation name."""
    return counts[~counts.index.astype(str).str.startswith("NTC")]


def _guide_count_pools(adata: ad.AnnData) -> Tuple[pd.Series, pd.Series]:
    """Return (all per-guide cell counts, non-NTC subset) for a cells AnnData.

    One groupby pass computing size and the guide's perturbation together —
    the two-pass version this replaces was the dominant cost on Phase-sized
    obs frames (~60M rows). Works on backed reads (``obs``/``uns`` only).
    """
    guide_col_name = _guide_col(adata)
    if guide_col_name not in adata.obs.columns:
        raise ValueError(
            f"per-guide titration requires {guide_col_name!r} column in obs"
        )
    pert_col = _pert_col(adata)
    sg = adata.obs.groupby(guide_col_name, observed=True).agg(
        n=(pert_col, "size"), pert=(pert_col, "first"))
    return sg["n"], sg.loc[~sg["pert"].astype(str).str.startswith("NTC"), "n"]


# Titration mode -> the schedule/x-axis column its targets are counted in.
_TARGET_COL_BY_MODE = {
    "per_guide": "cells_per_guide",
    "per_guide_median": "cells_per_guide",
    "per_ko": "cells_per_perturbation",
    "total": "n_cells",
}
_UNIT_BY_MODE = {
    "per_guide": "cells/guide",
    "per_guide_median": "cells/guide",
    "per_ko": "cells/KO",
    "total": "cells",
}


def _mode_from_flags(per_guide: bool, per_ko: bool) -> str:
    """Collapse the per_guide/per_ko flag pair to a mode string."""
    if per_guide:
        return "per_guide"
    if per_ko:
        return "per_ko"
    return "total"


def _target_col(mode: str) -> str:
    """Schedule x-axis column for a titration mode. Single source of truth."""
    return _TARGET_COL_BY_MODE[mode]


def _prepare_for_copairs(adata: ad.AnnData) -> ad.AnnData:
    """Strip obs to copairs-required columns and cast X to float64."""
    if "n_cells" not in adata.obs.columns:
        adata.obs["n_cells"] = 1
    keep = [
        c
        for c in [_guide_col(adata), "perturbation", "n_cells"]
        if c in adata.obs.columns
    ]
    adata.obs = adata.obs[keep].copy()
    for col in adata.obs.columns:
        if adata.obs[col].dtype.name == "category":
            adata.obs[col] = adata.obs[col].astype(str)
    adata.X = np.asarray(adata.X, dtype=np.float64)
    return adata


def _bootstrap_indices_per_group(
    group_codes: np.ndarray, budget: int, rng: np.random.RandomState, replace: bool,
) -> np.ndarray:
    """Return positional indices keeping up to ``budget`` cells per group.

    ``group_codes`` is an integer code per cell (e.g. ``pd.Categorical.codes``);
    cells with a negative code (missing / NaN group) are dropped — matching the
    legacy loops where ``sgrnas == nan`` selected nothing. For each group, keeps
    ``min(count, budget)`` cells drawn uniformly:

    - ``replace=False`` — without replacement (subsample). Same semantics as the
      legacy per-group ``rng.choice(..., replace=False)`` loop, but in one pass.
    - ``replace=True`` — with replacement (true nonparametric bootstrap). The
      returned array MAY contain duplicate positions; callers must preserve them
      (do not de-duplicate) so aggregation counts each resampled cell.

    Fully vectorized: cost is O(N log N) instead of the legacy O(n_groups * N)
    ``np.where``-per-group scan — the dominant per-draw cost on large reporters
    like Phase (~60M cells).
    """
    codes = np.asarray(group_codes)
    pos = np.nonzero(codes >= 0)[0]
    if pos.size == 0:
        return pos
    codes = codes[pos]
    if replace:
        # Stable sort just groups equal codes contiguously; within-group order
        # is irrelevant because we draw random offsets below.
        order = np.argsort(codes, kind="stable")
    else:
        # A random key per cell + lexsort groups by code AND randomizes order
        # within each group, so taking the first k of a group == a random k.
        keys = rng.random(codes.size)
        order = np.lexsort((keys, codes))
    pos_sorted = pos[order]
    codes_sorted = codes[order]
    _uniq, sizes = np.unique(codes_sorted, return_counts=True)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    k = np.minimum(sizes, int(budget))
    if replace:
        grp = np.repeat(np.arange(sizes.size), k)
        offs = np.floor(rng.random(int(k.sum())) * sizes[grp]).astype(np.intp)
        # Guard the measure-zero case where random() rounds up to the size.
        offs = np.minimum(offs, sizes[grp] - 1)
        kept = pos_sorted[starts[grp] + offs]
    else:
        rank = np.arange(codes_sorted.size) - np.repeat(starts, sizes)
        kept = pos_sorted[rank < np.repeat(k, sizes)]
    return kept


def _subsample_per_ko_and_aggregate(
    adata_cells: ad.AnnData, cells_per_ko: int, rng: np.random.RandomState,
) -> ad.AnnData:
    """Subsample up to ``cells_per_ko`` cells per perturbation, then aggregate to guide level."""
    perts = adata_cells.obs[_pert_col(adata_cells)].values
    kept_idx = []
    for p in np.unique(perts):
        p_idx = np.where(perts == p)[0]
        if len(p_idx) <= cells_per_ko:
            kept_idx.extend(p_idx)
        else:
            kept_idx.extend(rng.choice(p_idx, cells_per_ko, replace=False))
    kept_idx = np.sort(np.asarray(kept_idx))
    sub = adata_cells[kept_idx].copy()
    return aggregate_to_level(
        sub, level="guide", method="mean",
        preserve_batch_info=False, subsample_controls=False,
    )


def _subsample_per_guide_and_aggregate(
    adata_cells: ad.AnnData, cells_per_guide: int, rng: np.random.RandomState,
    replace: bool = False,
) -> ad.AnnData:
    """Subsample up to ``cells_per_guide`` cells per sgRNA, then aggregate to guide level.

    Unlike ``_subsample_per_ko_and_aggregate`` (which pools cells at the perturbation
    level, so NTC shares one budget across its ~8 sgRNAs while gene KOs split across
    ~4 sgRNAs), this samples directly at sgRNA level so every guide — NTC or KO —
    gets the same cell budget.

    ``replace=True`` switches from subsample-without-replacement to a true
    nonparametric bootstrap (draw ``min(count, budget)`` cells per guide *with*
    replacement). The per-guide cell count is unchanged either way, so
    ``n_cells`` / x-axis bookkeeping stays directly comparable — only the draw
    changes. Both paths share the fully vectorized index helper (see
    :func:`_bootstrap_indices_per_group`).
    """
    guide_col_name = _guide_col(adata_cells)
    if guide_col_name not in adata_cells.obs.columns:
        raise ValueError(
            f"Per-guide titration requires {guide_col_name!r} column in obs"
        )
    codes = pd.Categorical(adata_cells.obs[guide_col_name].values).codes
    kept_idx = _bootstrap_indices_per_group(
        codes, cells_per_guide, rng, replace=replace,
    )
    # Preserve duplicates (do NOT de-dup); sort only to make the row order
    # deterministic for aggregation.
    kept_idx = np.sort(kept_idx)
    sub = adata_cells[kept_idx].copy()
    if replace:
        # With replacement, obs_names repeat — make them unique so downstream
        # AnnData ops don't choke; guide-level grouping is by the guide column,
        # not the index, so this doesn't affect aggregation.
        sub.obs_names_make_unique()
    return aggregate_to_level(
        sub, level="guide", method="mean",
        preserve_batch_info=False, subsample_controls=False,
    )


def _subsample_and_aggregate(
    adata_cells: ad.AnnData, target_n_cells: int, rng: np.random.RandomState,
    min_exp: bool = False,
) -> ad.AnnData:
    """Subsample real cells from the cell-level h5ad, then re-aggregate to guide level.

    If ``min_exp`` is True, selects the fewest experiments (ranked by cell count,
    descending) whose combined cells reach ``target_n_cells``, then randomly
    samples from just those. Otherwise samples uniformly across all cells.
    """
    n_total = adata_cells.n_obs
    if n_total <= target_n_cells:
        sub = adata_cells
    elif min_exp and "experiment" in adata_cells.obs.columns:
        # Pick fewest experiments to reach target, largest first
        exp_counts = adata_cells.obs["experiment"].value_counts()
        kept_exps = []
        running = 0
        for exp_id, count in exp_counts.items():
            kept_exps.append(exp_id)
            running += count
            if running >= target_n_cells:
                break
        mask = adata_cells.obs["experiment"].isin(kept_exps).values
        pool_idx = np.where(mask)[0]
        if len(pool_idx) <= target_n_cells:
            idx = pool_idx
        else:
            idx = rng.choice(pool_idx, target_n_cells, replace=False)
        idx.sort()
        sub = adata_cells[idx].copy()
    else:
        idx = rng.choice(n_total, target_n_cells, replace=False)
        idx.sort()
        sub = adata_cells[idx].copy()

    g = aggregate_to_level(
        sub,
        level="guide",
        method="mean",
        preserve_batch_info=False,
        subsample_controls=False,
    )
    return g


def _subsample_one(
    adata_cells: ad.AnnData, target: int, mode: str, rng: np.random.RandomState,
    replace: bool = False, min_exp: bool = False,
) -> ad.AnnData:
    """Subsample to ``target`` under ``mode``, then aggregate to guide level.

    The single dispatch point over the three samplers, shared by the
    per-reporter loop and combined_titration's per-reporter prep.
    """
    if mode in ("per_guide", "per_guide_median"):
        return _subsample_per_guide_and_aggregate(
            adata_cells, target, rng, replace=replace,
        )
    if mode == "per_ko":
        return _subsample_per_ko_and_aggregate(adata_cells, target, rng)
    return _subsample_and_aggregate(adata_cells, target, rng, min_exp=min_exp)


def _aggregate_draws(
    draw_rows: List[Dict], metric_cols: Sequence[str], n_bootstraps: int,
) -> Dict:
    """Collapse N bootstrap draws to mean + STD + SEM per metric column.

    SEM is std/sqrt(N) over the finite draws. With ``n_bootstraps`` > 1 the raw
    per-draw values are also kept as a pipe-separated ``{col}_draws`` string so
    downstream code can recompute alternative error bars.
    """
    scores: Dict = {}
    for k in metric_cols:
        vals = np.array([r.get(k, float("nan")) for r in draw_rows], dtype=float)
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            scores[k] = float("nan")
            scores[f"{k}_sem"] = float("nan")
            scores[f"{k}_std"] = float("nan")
        else:
            scores[k] = float(np.mean(finite))
            if len(finite) > 1:
                std = float(np.std(finite, ddof=1))
                scores[f"{k}_std"] = std
                scores[f"{k}_sem"] = std / np.sqrt(len(finite))
            else:
                scores[f"{k}_std"] = 0.0
                scores[f"{k}_sem"] = 0.0
        if n_bootstraps > 1:
            scores[f"{k}_draws"] = "|".join(
                "nan" if not np.isfinite(v) else f"{v:.6g}" for v in vals
            )
    scores["n_bootstraps"] = n_bootstraps
    return scores


def _record_metric(result: dict, name: str, map_df, ratio) -> None:
    """Write one metric's ratio + mean mAP into ``result``.

    The scorer's own ratio is authoritative; the mean is read off the map when
    the scorer returned one.
    """
    result[f"{name}_ratio"] = float(ratio)
    if map_df is not None and "mean_average_precision" in map_df.columns:
        result[f"{name}_map_mean"] = float(map_df["mean_average_precision"].mean())


def _score_all_metrics(g_norm: ad.AnnData, _logger, distance="cosine") -> dict:
    """Score every metric in METRICS on a guide-level AnnData.

    Returns a dict of ``{metric}_ratio`` / ``{metric}_map_mean`` (METRIC_COLUMNS),
    NaN for anything that failed to score.
    """
    from ops_utils.analysis.map_scores import (
        phenotypic_distinctivness,
        phenotypic_consistency_corum,
        phenotypic_consistency_ebi,
        phenotypic_consistency_manual_annotation,
    )

    result = {col: math.nan for col in METRIC_COLUMNS}

    # Activity is deliberately not part of the loops below: it builds the
    # copairs view every later metric reuses, and it reads its map unguarded, so
    # a failure here aborts the whole pass instead of falling through.
    try:
        g_copairs = _prepare_for_copairs(g_norm.copy())
        activity_map, active_ratio = phenotypic_activity_assesment(
            g_copairs, plot_results=False, null_size=NULL_SIZE, distance=distance,
        )
        result["activity_ratio"] = float(active_ratio)
        result["activity_map_mean"] = float(
            activity_map["mean_average_precision"].mean()
        )
    except Exception as exc:
        _logger.warning(f"    Activity scoring failed: {exc}")
        return result

    try:
        dist_map, dist_ratio = phenotypic_distinctivness(
            g_copairs, plot_results=False, null_size=NULL_SIZE, distance=distance,
        )
        _record_metric(result, "distinctiveness", dist_map, dist_ratio)
    except Exception as exc:
        _logger.warning(f"    Distinctiveness scoring failed: {exc}")

    # Consistency metrics score the gene-level matrix and share a similarity
    # cache. One try for the group, so a failure in the first skips the rest —
    # matching the previous single-try block.
    gene_scorers = (
        ("corum", phenotypic_consistency_corum),
        ("chad", phenotypic_consistency_manual_annotation),
        ("ebi", phenotypic_consistency_ebi),
    )
    try:
        e_copairs = _prepare_for_copairs(aggregate_to_level(
            g_copairs, "gene", preserve_batch_info=False, subsample_controls=False,
        ))
        for name, scorer in gene_scorers:
            map_df, ratio = scorer(
                e_copairs, plot_results=False, null_size=NULL_SIZE,
                cache_similarity=True, distance=distance,
            )
            _record_metric(result, name, map_df, ratio)
    except Exception as exc:
        _logger.warning(f"    Consistency scoring failed: {exc}")

    return result


# ---------------------------------------------------------------------------
# Core titration function (one per reporter — pickle-friendly for submitit)
# ---------------------------------------------------------------------------


def _run_titration_points(
    adata_cells, cell_targets, norm_method, signal, rng, _logger,
    min_exp=False, per_ko=False, per_guide=False, n_bootstraps=1, replace=False,
):
    """Score all titration points for an adata, returning a DataFrame of results.

    If ``per_ko`` is True, ``cell_targets`` are interpreted as cells-per-perturbation.
    If ``per_guide`` is True, ``cell_targets`` are interpreted as cells-per-sgRNA.
    If ``n_bootstraps`` > 1, runs N independent draws per titration point (different
    seeds) and writes mean + SEM columns for every metric.
    """
    # Drop signal col if present (not needed for scoring, can interfere with aggregation)
    if "signal" in adata_cells.obs.columns:
        adata_cells.obs = adata_cells.obs.drop(columns=["signal"])

    # `replace` (with-replacement bootstrap) is only wired into the per-guide
    # sampler for now. Warn rather than silently ignore for other modes.
    if replace and not per_guide:
        _logger.warning(
            "replace=True (with-replacement bootstrap) is only supported for "
            "per-guide sampling; ignoring for this mode."
        )

    base_seed = int(rng.randint(0, 2**31 - 1))
    metric_cols = list(METRIC_COLUMNS)
    mode = _mode_from_flags(per_guide, per_ko)
    unit = _UNIT_BY_MODE[mode]

    rows = []
    for target in cell_targets:
        _logger.info(f"  Scoring at {target:,} {unit} ({n_bootstraps} draw{'s' if n_bootstraps > 1 else ''})...")
        t_step = time.time()

        draw_rows = []
        g_sub_last = None
        for b in range(n_bootstraps):
            draw_rng = np.random.RandomState(base_seed + b * 9973 + target)
            g_sub = _subsample_one(
                adata_cells, target, mode, draw_rng,
                replace=replace, min_exp=min_exp,
            )
            g_norm = normalize_guide_adata(g_sub, norm_method)
            scores_b = _score_all_metrics(g_norm, _logger)
            draw_rows.append(scores_b)
            g_sub_last = g_sub

        g_sub = g_sub_last
        scores = _aggregate_draws(draw_rows, metric_cols, n_bootstraps)

        pert_col = _pert_col(g_sub)
        n_perts = g_sub.obs[pert_col].nunique()
        if per_guide or per_ko:
            # total cells from guide n_cells sum
            total_cells = int(g_sub.obs.get("n_cells", pd.Series([0])).sum()) if "n_cells" in g_sub.obs.columns else target * g_sub.n_obs
            scores["n_cells"] = total_cells
            if per_guide:
                # cells_per_perturbation varies across perts (NTC has ~8 guides, KOs ~4)
                # Report mean over non-NTC perts so x-axis reflects what gene KOs actually get
                scores["cells_per_guide"] = target
                if "n_cells" in g_sub.obs.columns:
                    cpp = g_sub.obs.groupby(pert_col, observed=True)["n_cells"].sum()
                    non_ntc = ~cpp.index.astype(str).str.startswith("NTC")
                    scores["cells_per_perturbation"] = float(
                        cpp[non_ntc].mean() if non_ntc.any() else cpp.mean()
                    )
                else:
                    scores["cells_per_perturbation"] = total_cells / n_perts if n_perts > 0 else target
            else:
                # per_ko: cells_per_guide = cells_per_pert / mean guides-per-pert
                scores["cells_per_perturbation"] = target
                scores["cells_per_guide"] = total_cells / g_sub.n_obs if g_sub.n_obs > 0 else target
        else:
            scores["n_cells"] = target
            scores["cells_per_perturbation"] = target / n_perts if n_perts > 0 else target
            scores["cells_per_guide"] = target / g_sub.n_obs if g_sub.n_obs > 0 else target
        scores["n_guides"] = g_sub.n_obs
        scores["n_perturbations"] = n_perts
        scores["signal"] = signal
        rows.append(scores)

        _logger.info(
            f"    act={scores['activity_ratio']:.1%}±{scores.get('activity_ratio_sem', 0):.1%} "
            f"dist={scores['distinctiveness_ratio']:.1%}±{scores.get('distinctiveness_ratio_sem', 0):.1%} "
            f"corum={scores['corum_ratio']:.1%}±{scores.get('corum_ratio_sem', 0):.1%} "
            f"chad={scores['chad_ratio']:.1%}±{scores.get('chad_ratio_sem', 0):.1%} "
            f"ebi={scores['ebi_ratio']:.1%}±{scores.get('ebi_ratio_sem', 0):.1%} "
            f"({time.time() - t_step:.0f}s)"
        )
    return pd.DataFrame(rows)


def _build_per_ko_schedule(
    max_cells_per_ko: int, min_cells_per_ko: int = 1, ratio: float = DOWNSAMPLE_RATIO,
) -> list:
    """Build cells-per-KO schedule: max, max*ratio, ... >= min."""
    targets = []
    n = max_cells_per_ko
    while n >= min_cells_per_ko:
        targets.append(int(n))
        nxt = int(n * ratio)
        if nxt == n:  # avoid infinite loop when ratio is too close to 1
            nxt = n - 1
        n = nxt
    if not targets:
        targets = [max_cells_per_ko]
    return targets


def _read_cache(
    csv_path: Path, target_col: str, row_filter=None, _logger=None,
) -> Optional[pd.DataFrame]:
    """Read a titration CSV for cache reuse, or None if it can't be used.

    ``row_filter`` optionally drops rows that don't match the current run
    (combined_titration uses it to discard rows scored at a different
    ``second_pca_threshold``).
    """
    if not csv_path.is_file():
        return None
    try:
        df_old = pd.read_csv(csv_path)
    except Exception as exc:
        if _logger is not None:
            _logger.warning(f"  Cache read failed ({exc}); recomputing all.")
        return None
    if target_col not in df_old.columns:
        return None
    if row_filter is not None:
        df_old = row_filter(df_old)
    return df_old


def _scored_targets(
    csv_path: Path, target_col: str, required_cols: Sequence[str] = (),
    row_filter=None,
) -> set:
    """Targets in ``csv_path`` that are already fully scored.

    A row only counts when every column in ``required_cols`` is non-null — an
    older CSV that has the target but predates a metric must be re-scored, not
    skipped.
    """
    df_old = _read_cache(csv_path, target_col, row_filter=row_filter)
    if df_old is None:
        return set()
    if required_cols:
        if any(c not in df_old.columns for c in required_cols):
            return set()
        df_old = df_old.loc[df_old[list(required_cols)].notna().all(axis=1)]
    return {int(v) for v in df_old[target_col].dropna().astype(int).tolist()}


def _cache_split(
    csv_path: Path, cell_targets: List[int], target_col: str, _logger,
    required_cols: Sequence[str] = (), row_filter=None,
) -> Tuple[List[int], List[Dict]]:
    """Return (targets_to_run, cached_rows) — subset ``cell_targets`` to those
    not yet fully scored in the CSV at ``csv_path`` under ``target_col``.

    On any read error, returns (cell_targets, []) (i.e. recompute everything).
    """
    df_old = _read_cache(csv_path, target_col, row_filter=row_filter, _logger=_logger)
    if df_old is None:
        return list(cell_targets), []
    done = _scored_targets(
        csv_path, target_col, required_cols=required_cols, row_filter=row_filter,
    )
    missing = [t for t in cell_targets if int(t) not in done]
    cached_rows = df_old.to_dict(orient="records")
    if cached_rows:
        _logger.info(
            f"  Cache hit: {len(cached_rows)} existing rows in {csv_path.name}; "
            f"{len(missing)}/{len(cell_targets)} targets need scoring"
        )
    return missing, cached_rows


def _merge_and_write(
    new_df: pd.DataFrame, cached_rows: List[Dict], target_col: str, csv_path: Path,
    _logger,
) -> pd.DataFrame:
    """Concat new rows with cached rows, dedupe on target_col, sort desc, write."""
    all_rows = (cached_rows or []) + new_df.to_dict(orient="records")
    df = pd.DataFrame(all_rows)
    if target_col in df.columns:
        df = (
            df.dropna(subset=[target_col])
              .drop_duplicates(subset=[target_col], keep="last")
              .sort_values(target_col, ascending=False)
              .reset_index(drop=True)
        )
    df.to_csv(csv_path, index=False)
    _logger.info(
        f"  Saved {csv_path} ({len(df)} rows: {len(new_df)} new + "
        f"{len(cached_rows)} cached)"
    )
    return df


def titrate_single_reporter(
    cells_h5ad_path: str,
    output_dir: str,
    norm_method: str = "ntc",
    random_seed: int = 42,
    min_exp: bool = False,
    per_ko: bool = False,
    per_ko_max: bool = False,
    per_guide: bool = False,
    per_guide_max: bool = False,
    per_guide_median: bool = False,
    n_bootstraps: int = 1,
    cache: bool = True,
    replace: bool = False,
) -> str:
    """Run cell-count titration for a single reporter.

    Loads the full cell-level PCA-reduced h5ad, subsamples real cells at each
    titration point, re-aggregates to guide level, and scores every metric in
    ``METRICS``.

    Returns a status string.
    """
    _logger = _init_logger()
    t_start = time.time()
    cells_h5ad_path = Path(cells_h5ad_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata_cells = ad.read_h5ad(cells_h5ad_path)
    signal = adata_cells.obs.get(
        "signal", pd.Series([cells_h5ad_path.stem.replace("_cells", "")])
    ).iloc[0]
    if isinstance(signal, float):
        signal = cells_h5ad_path.stem.replace("_cells", "")
    total_cells = adata_cells.n_obs

    _logger.info(f"Titrating {signal}: {total_cells:,} cells, {adata_cells.n_vars} PCs")

    if per_guide or per_guide_max or per_guide_median:
        sg_counts, non_ntc_counts = _guide_count_pools(adata_cells)
        pool = non_ntc_counts if len(non_ntc_counts) else sg_counts
        if per_guide_median:
            # Start at the MEDIAN of non-NTC cells/guide (more conservative
            # than p90 — only the upper half of the distribution caps at the
            # starting budget). Schedule down to 1 cell/guide just like max mode.
            start_per_guide = int(np.median(pool.values))
            cell_targets = _build_per_ko_schedule(start_per_guide)
            mode_label = f"median (start={start_per_guide:,})"
        elif per_guide_max:
            # Clip to p90 non-NTC so one outlier guide doesn't stretch the schedule
            start_per_guide = int(np.percentile(pool.values, 90))
            cell_targets = _build_per_ko_schedule(start_per_guide)
            mode_label = "max (p90)"
        else:
            start_per_guide = int(non_ntc_counts.min()) if len(non_ntc_counts) else int(sg_counts.min())
            cell_targets = _build_per_ko_schedule(start_per_guide)
            mode_label = "min"
        _logger.info(f"  Per-guide titration points ({mode_label} non-NTC = {start_per_guide:,} cells/guide): {cell_targets}")
        per_guide = True
    elif per_ko or per_ko_max:
        counts = adata_cells.obs[_pert_col(adata_cells)].value_counts()
        non_ntc_counts = _non_ntc(counts)
        if per_ko_max:
            # Clip to p90 non-NTC so one outlier KO doesn't stretch the schedule
            pool = non_ntc_counts if len(non_ntc_counts) else counts
            start_per_ko = int(np.percentile(pool.values, 90))
            mode_label = "max (p90)"
        else:
            # Start at MIN non-NTC count — every KO fully hits target at every point
            start_per_ko = int(non_ntc_counts.min()) if len(non_ntc_counts) else int(counts.min())
            mode_label = "min"
        cell_targets = _build_per_ko_schedule(start_per_ko)
        _logger.info(f"  Per-KO titration points ({mode_label} non-NTC = {start_per_ko:,} cells/KO): {cell_targets}")
        # _run_titration_points uses per_ko flag; per_ko_max uses same sampling code path
        per_ko = True
    else:
        cell_targets = _build_per_ko_schedule(total_cells, MIN_CELLS)
        _logger.info(f"  Titration points: {cell_targets}")

    from ops_utils.data.feature_discovery import sanitize_signal_filename

    sig_safe = sanitize_signal_filename(signal)[:40]

    # Per-reporter subdir keeps CSVs and all scale/format variants together
    reporter_dir = output_dir / sig_safe
    reporter_dir.mkdir(parents=True, exist_ok=True)

    # --- Full titration ---
    csv_path = reporter_dir / f"{sig_safe}_titration.csv"
    full_target_col = _target_col(_mode_from_flags(per_guide, per_ko))
    targets_to_run, cached_rows = (
        _cache_split(csv_path, cell_targets, full_target_col, _logger)
        if cache else (list(cell_targets), [])
    )
    if targets_to_run:
        df_new = _run_titration_points(
            adata_cells.copy(),
            targets_to_run,
            norm_method,
            signal,
            np.random.RandomState(random_seed),
            _logger,
            min_exp=min_exp,
            per_ko=per_ko,
            per_guide=per_guide,
            n_bootstraps=n_bootstraps,
            replace=replace,
        )
    else:
        df_new = pd.DataFrame()
        _logger.info("  All targets cached; skipping recompute.")
    df_full = _merge_and_write(df_new, cached_rows, full_target_col, csv_path, _logger)

    # Plot — PNG + SVG for each scale
    try:
        _plot_titration(df_full, signal, reporter_dir, sig_safe, _plt())
    except Exception as exc:
        _logger.warning(f"  Plotting failed: {exc}")

    elapsed = time.time() - t_start
    return f"SUCCESS: {signal} — {len(cell_targets)} titration points in {elapsed:.0f}s"


_X_AXIS_VARIANTS = [
    ("n_cells", "Total Cells", "totalcells"),
    ("cells_per_perturbation", "Cells / Perturbation", "perpert"),
    ("cells_per_guide", "Cells / Guide", "perguide"),
]


def titration_x_axis_base_label(x_col: str) -> str:
    """Human-readable x-axis title segment for a titration CSV column."""
    for col, label, _ in _X_AXIS_VARIANTS:
        if col == x_col:
            return label
    return x_col


def _plot_titration(df, signal, reporter_dir: Path, sig_safe, plt):
    """Generate titration plots for one reporter across all scales and x-axis types.

    Saves PNG + SVG for each (scale × x-axis) combination into ``reporter_dir``.
    All text is sized at 1.5× the matplotlib default for legibility.
    """
    metrics = METRICS
    reporter_dir = Path(reporter_dir)
    reporter_dir.mkdir(parents=True, exist_ok=True)

    colors = TITRATION_METRIC_COLORS
    ratio_labels = TITRATION_RATIO_LABELS
    map_labels = TITRATION_MAP_LABELS
    _scale_label = SCALE_LABEL_SHORT

    for x_col, x_label_base, x_suffix in _X_AXIS_VARIANTS:
        if x_col not in df.columns:
            continue
        x = df[x_col].values

        for scale in SCALES:
            fig, axes = plt.subplots(1, 2, figsize=(22, 9))
            xlabel = f"{x_label_base} ({_scale_label[scale]})"

            # Panel 1: % significant
            ax = axes[0]
            for metric in metrics:
                _plot_series(
                    ax, x, df, f"{metric}_ratio", scale_y=100,
                    marker="o", color=colors[metric],
                    label=ratio_labels[metric], linewidth=3.5, markersize=8,
                )
            ax.set_xlabel(xlabel, fontsize=22)
            ax.set_ylabel("% Significant", fontsize=22)
            ax.set_title(f"{signal} — % Significant", fontsize=24)
            ax.tick_params(axis="y", labelsize=18)
            _apply_x_scale(ax, x, scale, tick_fontsize=18)

            # Panel 2: mean mAP
            ax = axes[1]
            for metric in metrics:
                _plot_series(
                    ax, x, df, f"{metric}_map_mean",
                    marker="s", color=colors[metric],
                    label=map_labels[metric], linewidth=3.5, markersize=8,
                )
            ax.set_xlabel(xlabel, fontsize=22)
            ax.set_ylabel("Mean mAP", fontsize=22)
            ax.set_title(f"{signal} — Mean mAP", fontsize=24)
            ax.tick_params(axis="y", labelsize=18)
            _apply_x_scale(ax, x, scale, tick_fontsize=18)

            # Single legend below the plots
            handles, labels_list = axes[0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels_list,
                loc="lower center",
                ncol=4,
                fontsize=19,
                bbox_to_anchor=(0.5, -0.02),
            )

            fig.suptitle(
                f"Cell Count Titration — {signal}  [{scale}]",
                fontsize=31,
                fontweight="bold",
            )
            fig.tight_layout(rect=[0, 0.06, 1, 0.97])

            stem = reporter_dir / f"{sig_safe}_titration_{x_suffix}_{scale}"
            fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
            fig.savefig(f"{stem}.svg", bbox_inches="tight")
            plt.close(fig)


def _plot_series(ax, x, df, col, scale_y=1.0, **kwargs):
    """Plot a metric as line or errorbar depending on presence of `{col}_sem`."""
    if col not in df.columns:
        return
    vals = df[col].values * scale_y
    sem_col = f"{col}_sem"
    if sem_col in df.columns and df[sem_col].notna().any():
        ax.errorbar(x, vals, yerr=df[sem_col].values * scale_y,
                    capsize=4, elinewidth=1.5, **kwargs)
    else:
        ax.plot(x, vals, **kwargs)


def _plot_combined_titration(
    output_dir,
    plt,
    csv_glob="**/*_titration.csv",
    title_suffix=None,
    filename_prefix="titration_combined",
):
    """Combine all per-reporter titration CSVs into one summary plot.

    Saves PNG + SVG for each scale (linear, log2, log10).
    CSVs are discovered recursively so reporter subdirs are supported.
    """
    csv_files = sorted(Path(output_dir).glob(csv_glob))
    if not csv_files:
        return

    all_dfs = [pd.read_csv(f) for f in csv_files]
    combined = pd.concat(all_dfs, ignore_index=True)

    # Export the combined DataFrame as CSV next to the plots
    combined.to_csv(Path(output_dir) / f"{filename_prefix}.csv", index=False)

    signals = combined["signal"].unique()
    n_signals = len(signals)
    # Pair a 20-color cycle with a 9-marker cycle. gcd(20, 9) = 1 → 180 unique
    # (color, marker) pairs before any repeat, so every reporter (up to 180)
    # gets a unique combo. Critically, markers cycle every reporter — not every
    # 20 — so adjacent reporters in the legend are easy to distinguish.
    _COLOR_CYCLE = plt.cm.tab20(np.linspace(0, 1, 20))
    _MARKER_CYCLE = ["o", "s", "D", "^", "v", "P", "X", "*", "h"]
    def _style_for(i):
        return _COLOR_CYCLE[i % len(_COLOR_CYCLE)], _MARKER_CYCLE[i % len(_MARKER_CYCLE)]

    # BF-slice titration color scheme (only when BF_z* signals are present, so
    # production biological-signal plots are unaffected): the raw brightfield
    # z-slices are an ordered focal sweep, so map them along a single sequential
    # perceptual colormap (viridis: BF_z0 dark-purple → BF_zN yellow). Phase2D and
    # Focus3D are distinct anchors (black, red) that stand out from viridis.
    _bf_idxs = sorted(
        int(str(s)[len("BF_z"):]) for s in signals
        if str(s).startswith("BF_z") and str(s)[len("BF_z"):].isdigit()
    )
    _is_bf_run = bool(_bf_idxs)
    _bf_mid = 3  # middle z-index (for friendly labels: BF-mid)
    _bf_lo, _bf_hi = (_bf_idxs[0], _bf_idxs[-1]) if _bf_idxs else (0, 1)

    def _bf_color(sig):
        if not _is_bf_run:
            return None
        if sig in ("Phase", "Phase2D"):
            return "#000000"   # black anchor
        if sig == "Focus3D":
            return "#d62728"   # red anchor (stands out from viridis)
        if str(sig).startswith("BF_z"):
            try:
                k = int(str(sig)[len("BF_z"):])
            except ValueError:
                return None
            t = 0.0 if _bf_hi == _bf_lo else (k - _bf_lo) / (_bf_hi - _bf_lo)
            return plt.cm.viridis(t)
        return None

    def _style_for2(i, sig):
        c, mk = _style_for(i)
        bc = _bf_color(sig)
        return (bc if bc is not None else c), mk

    def _bf_label(sig):
        """Friendly legend label for the BF run: Phase2D, Focus3D, BF-mid, BF±n."""
        if not _is_bf_run:
            return str(sig)[:25]
        if sig in ("Phase", "Phase2D"):
            return "Phase2D"
        if sig == "Focus3D":
            return "Focus3D"
        if str(sig).startswith("BF_z"):
            try:
                k = int(str(sig)[len("BF_z"):])
            except ValueError:
                return str(sig)[:25]
            return "BF-mid" if k == _bf_mid else f"BF{k - _bf_mid:+d}"
        return str(sig)[:25]

    # Short labels for the dense multi-reporter grid; the longer
    # TITRATION_RATIO_LABELS wording doesn't fit these subplot titles.
    metric_info = [
        ("activity", "% Active"),
        ("distinctiveness", "% Distinctive"),
        ("corum", "% CORUM"),
        ("chad", "% CHAD"),
        ("ebi", "% EBI"),
    ]
    # BF run: collection plots show only activity, distinctiveness, EBI
    # (CORUM/CHAD mAP dropped).
    if _is_bf_run:
        metric_info = [m for m in metric_info
                       if m[0] in ("activity", "distinctiveness", "ebi")]

    _scale_label = SCALE_LABEL_SHORT

    dest = Path(output_dir)

    for x_col, x_label_base, x_suffix in _X_AXIS_VARIANTS:
        if x_col not in combined.columns:
            continue
        x_all = combined[x_col].values
        x_min, x_max = float(x_all.min()), float(x_all.max())

        n_metrics_plot = len(metric_info)
        for scale in SCALES:
            fig, axes = plt.subplots(2, n_metrics_plot, figsize=(14 * n_metrics_plot, 18))
            xlabel = f"{x_label_base} ({_scale_label[scale]})"

            def _style_combined_axis(ax, _scale=scale, _xmin=x_min, _xmax=x_max):
                _apply_x_scale(ax, [_xmin, _xmax], _scale, tick_fontsize=19)
                ax.set_xlim(_xmin * 0.7, _xmax * 1.3)

            # Row 0: % significant per metric
            for col_idx, (metric, label) in enumerate(metric_info):
                ax = axes[0, col_idx]
                ratio_col = f"{metric}_ratio"
                for i, sig in enumerate(sorted(signals)):
                    sub = combined[combined["signal"] == sig].sort_values(x_col)
                    if ratio_col in sub.columns:
                        sem_col = f"{ratio_col}_sem"
                        c, mk = _style_for2(i, sig)
                        if sem_col in sub.columns and sub[sem_col].notna().any():
                            ax.errorbar(
                                sub[x_col], sub[ratio_col] * 100, yerr=sub[sem_col] * 100,
                                marker=mk, color=c, label=_bf_label(sig),
                                linewidth=3, markersize=9, alpha=0.8,
                                capsize=3, elinewidth=1.2,
                            )
                        else:
                            ax.plot(
                                sub[x_col], sub[ratio_col] * 100,
                                marker=mk, color=c, label=_bf_label(sig),
                                linewidth=3, markersize=9, alpha=0.8,
                            )
                ax.set_xlabel(xlabel, fontsize=24)
                ax.set_ylabel("% Significant", fontsize=24)
                ax.set_title(label, fontsize=26)
                ax.tick_params(axis="y", labelsize=19)
                _style_combined_axis(ax)

            # Row 1: mean mAP per metric
            for col_idx, (metric, label) in enumerate(metric_info):
                ax = axes[1, col_idx]
                map_col = f"{metric}_map_mean"
                for i, sig in enumerate(sorted(signals)):
                    sub = combined[combined["signal"] == sig].sort_values(x_col)
                    if map_col in sub.columns:
                        sem_col = f"{map_col}_sem"
                        c, mk = _style_for2(i, sig)
                        if sem_col in sub.columns and sub[sem_col].notna().any():
                            ax.errorbar(
                                sub[x_col], sub[map_col], yerr=sub[sem_col],
                                marker=mk, color=c, label=_bf_label(sig),
                                linewidth=3, markersize=9, alpha=0.8,
                                capsize=3, elinewidth=1.2,
                            )
                        else:
                            ax.plot(
                                sub[x_col], sub[map_col],
                                marker=mk, color=c, label=_bf_label(sig),
                                linewidth=3, markersize=9, alpha=0.8,
                            )
                ax.set_xlabel(xlabel, fontsize=24)
                ax.set_ylabel("Mean mAP", fontsize=24)
                ax.set_title(f"{label} mAP", fontsize=26)
                ax.tick_params(axis="y", labelsize=19)
                _style_combined_axis(ax)

            # Reserve space at the bottom for the legend so it never overlaps x-axes
            title_tag = f" — {title_suffix}" if title_suffix else ""
            fig.suptitle(
                f"Cell Count Titration — All Reporters{title_tag}  [{scale}]",
                fontsize=34,
                fontweight="bold",
            )
            # Bigger bottom margin so multi-row legend sits below axis labels.
            # n_signals ~44 → ncol=8 → ~6 rows of legend entries.
            n_legend_rows = max(1, int(np.ceil(n_signals / min(8, n_signals))))
            legend_frac = min(0.22, 0.025 * n_legend_rows + 0.04)
            fig.tight_layout(rect=[0, legend_frac, 1, 0.97])

            handles, labels_list = axes[0, 0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels_list,
                loc="lower center",
                ncol=min(8, n_signals),
                fontsize=19,
                bbox_to_anchor=(0.5, 0.005),
            )

            stem = dest / f"{filename_prefix}_{x_suffix}_{scale}"
            fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
            fig.savefig(f"{stem}.svg", bbox_inches="tight")
            plt.close(fig)

            # Delta figure: Δ mean mAP (Phase2D − BF-mid) per metric across bins
            # — how much Phase2D improves over the mid BF slice.
            # Phase2D and BF_z3 schedules can differ, so interpolate Phase2D's
            # mAP onto BF-mid's cell-count bins before differencing.
            _sigset = set(signals)
            _phase_sig = ("Phase2D" if "Phase2D" in _sigset
                          else "Phase" if "Phase" in _sigset else None)
            if (_is_bf_run and "BF_z3" in _sigset
                    and _phase_sig is not None
                    and x_suffix == "perpert" and scale == "log10"):
                _delta_dir = Path(output_dir) / "phase2d_vs_bfmid"
                _delta_dir.mkdir(exist_ok=True)
                bf = combined[combined["signal"] == "BF_z3"].sort_values(x_col)
                ph = combined[combined["signal"] == _phase_sig].sort_values(x_col)
                # Only activity, distinctiveness, EBI shown here (not CORUM/CHAD).
                _dmetrics = [m for m in metric_info
                             if m[0] in ("activity", "distinctiveness", "ebi")]
                _ndm = len(_dmetrics)
                dfig, daxes = plt.subplots(1, _ndm, figsize=(14 * _ndm, 9))
                if _ndm == 1:
                    daxes = [daxes]
                for col_idx, (metric, label) in enumerate(_dmetrics):
                    ax = daxes[col_idx]
                    map_col = f"{metric}_map_mean"
                    if (map_col in bf.columns and map_col in ph.columns
                            and len(bf) and len(ph)):
                        xb = bf[x_col].to_numpy(float); yb = bf[map_col].to_numpy(float)
                        xp = ph[x_col].to_numpy(float); yp = ph[map_col].to_numpy(float)
                        order = np.argsort(xp)
                        yp_i = np.interp(xb, xp[order], yp[order],
                                         left=np.nan, right=np.nan)
                        delta = yp_i - yb
                        finite = np.isfinite(delta)
                        ax.plot(xb[finite], delta[finite], marker="o",
                                color="#222222", linewidth=3, markersize=9)
                    ax.axhline(0, color="#d62728", linestyle="--",
                               linewidth=1.5, alpha=0.8)
                    ax.set_xlabel(xlabel, fontsize=24)
                    ax.set_ylabel("Δ mean mAP (Phase2D − BF-mid)", fontsize=18)
                    ax.set_title(f"{label} mAP", fontsize=26)
                    ax.tick_params(axis="y", labelsize=19)
                    _style_combined_axis(ax)
                dfig.suptitle(
                    f"Phase2D − BF-mid : Δ mean mAP{title_tag}  [{scale}]",
                    fontsize=32, fontweight="bold")
                dfig.tight_layout(rect=[0, 0, 1, 0.95])
                dstem = (_delta_dir
                         / f"{filename_prefix}_delta_Phase2D_vs_BFmid_{x_suffix}_{scale}")
                dfig.savefig(f"{dstem}.png", dpi=150, bbox_inches="tight")
                dfig.savefig(f"{dstem}.svg", bbox_inches="tight")
                plt.close(dfig)

                # Percent improvement: 100 * (Phase2D − BF-mid) / BF-mid.
                pfig, paxes = plt.subplots(1, _ndm, figsize=(14 * _ndm, 9))
                if _ndm == 1:
                    paxes = [paxes]
                for col_idx, (metric, label) in enumerate(_dmetrics):
                    ax = paxes[col_idx]
                    map_col = f"{metric}_map_mean"
                    if (map_col in bf.columns and map_col in ph.columns
                            and len(bf) and len(ph)):
                        xb = bf[x_col].to_numpy(float); yb = bf[map_col].to_numpy(float)
                        xp = ph[x_col].to_numpy(float); yp = ph[map_col].to_numpy(float)
                        order = np.argsort(xp)
                        yp_i = np.interp(xb, xp[order], yp[order],
                                         left=np.nan, right=np.nan)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            pct = 100.0 * (yp_i - yb) / yb
                        finite = np.isfinite(pct)
                        ax.plot(xb[finite], pct[finite], marker="o",
                                color="#222222", linewidth=3, markersize=9)
                    ax.axhline(0, color="#d62728", linestyle="--",
                               linewidth=1.5, alpha=0.8)
                    ax.set_xlabel(xlabel, fontsize=24)
                    ax.set_ylabel("% mean mAP improvement (Phase2D vs BF-mid)",
                                  fontsize=18)
                    ax.set_title(f"{label} mAP", fontsize=26)
                    ax.tick_params(axis="y", labelsize=19)
                    _style_combined_axis(ax)
                pfig.suptitle(
                    f"Phase2D vs BF-mid : % mean mAP improvement{title_tag}  [{scale}]",
                    fontsize=32, fontweight="bold")
                pfig.tight_layout(rect=[0, 0, 1, 0.95])
                pstem = (_delta_dir
                         / f"{filename_prefix}_pctimprove_Phase2D_vs_BFmid_{x_suffix}_{scale}")
                pfig.savefig(f"{pstem}.png", dpi=150, bbox_inches="tight")
                pfig.savefig(f"{pstem}.svg", bbox_inches="tight")
                plt.close(pfig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Per-reporter cell-count titration analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Disable argparse's prefix matching so unknown flags fail loudly
        # instead of silently aliasing to a longer flag (e.g. `--compare`
        # mapping to `--compare-only` and skipping the titration step).
        allow_abbrev=False,
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, required=True,
        help="Root output directory — the same path passed to pca_optimization -o.",
    )
    parser.add_argument("--norm-method", type=str, default="ntc",
                        help="Normalization method (default: ntc)")
    parser.add_argument("--slurm", action="store_true",
                        help="Submit one SLURM job per reporter")
    parser.add_argument("--slurm-memory", type=str, default="200GB")
    parser.add_argument(
        "--slurm-time",
        type=int,
        default=30,
        help="SLURM time limit per job in minutes (default: 30)",
    )
    parser.add_argument(
        "--phase-slurm-time",
        type=int,
        default=240,
        help="SLURM time limit for the Phase reporter job in minutes (default: 240). "
             "Phase has ~60M cells so needs far more time than other reporters.",
    )
    parser.add_argument("--slurm-cpus", type=int, default=8)
    parser.add_argument("--slurm-partition", type=str, default="cpu,gpu")
    parser.add_argument("--replot", action="store_true",
                        help="Regenerate all plots from existing CSVs without recomputing scores")
    parser.add_argument("--downsampled", action="store_true",
                        help="Look in downsampled/ subdir")
    parser.add_argument("--downsample-per-guide", action="store_true",
                        help="Look in downsampled_per_guide/ subdir (matches pca_optimization --downsample-per-guide)")
    parser.add_argument("--cell-profiler", action="store_true",
                        help="Look in cellprofiler/ subdir")
    parser.add_argument("--cell-dino", action="store_true",
                        help="Look in cell_dino/ subdir (matches pca_optimization --cell-dino)")
    parser.add_argument(
        "--include-cellpainting", action="store_true",
        help="Look under with_cellpainting/ (same as pca_optimization --include-cellpainting)",
    )
    parser.add_argument(
        "--with-4i", dest="include_4i", action="store_true",
        help="Look under with_4i/ (same as pca_optimization --with-4i). "
             "Composes with --with-cp and --include-cellpainting.",
    )
    parser.add_argument(
        "--with-cp", dest="include_cp", action="store_true",
        help="Look under with_cp/ (same as pca_optimization --with-cp). Composes with --with-4i.",
    )
    parser.add_argument(
        "--only-4i", action="store_true",
        help="Look under only_4i/ (same as pca_optimization --only-4i). Implies --with-4i.",
    )
    parser.add_argument(
        "--only-cp", action="store_true",
        help="Look under only_cp/ (same as pca_optimization --only-cp). Implies --with-cp.",
    )
    parser.add_argument(
        "--paper-v1", action="store_true",
        help="Look under paper_v1/ (mirrors pca_optimization --paper-v1). This "
             "script only uses it to nest the output path, never to read an "
             "experiment list, so it takes no value.",
    )
    parser.add_argument(
        "--run-tag", type=str, default=None,
        help="Match pca_optimization --run-tag: inserts an extra subfolder "
             "after paper_v1/ and before the channel-set subdir (e.g. "
             "'corrected' resolves to paper_v1/corrected/phase_only/...).",
    )
    parser.add_argument(
        "--fixed-threshold", type=float, default=0.80,
        help="Match pca_optimization --fixed-threshold (uses fixed_<pct>/ not consensus_sweep/). "
             "Default: 0.80. Pass --fixed-threshold 0 to disable and use consensus_sweep/.",
    )
    parser.add_argument(
        "--distance", type=str, default="cosine", choices=["cosine", "euclidean"],
        help="Match pca_optimization --distance (default: cosine)",
    )
    parser.add_argument("--zscore-per-experiment", dest="zscore_per_experiment",
                        action="store_true", default=True,
                        help="Look in zscore_per_exp/ subdir (default: True)")
    parser.add_argument("--no-zscore-per-experiment", dest="zscore_per_experiment",
                        action="store_false",
                        help="Disable zscore_per_exp/ subdir lookup")
    parser.add_argument("--min-exp-titration", action="store_true",
                        help="At each titration level, draw cells from the fewest "
                             "experiments needed (largest first) instead of sampling "
                             "across all experiments. Output → titration_min_exp/")
    parser.add_argument("--per-ko-min-titration", "--per-gene-min-titration",
                        dest="per_ko_min_titration", action="store_true",
                        help="Titrate by cells-per-geneKO (aliased as --per-gene-min-titration). "
                             "Schedule starts at MIN non-NTC cells/KO so every KO hits target fully. "
                             "Output → titration_geneKO_min/")
    parser.add_argument("--per-ko-max-titration", "--per-gene-max-titration",
                        dest="per_ko_max_titration", action="store_true",
                        help="Titrate by cells-per-geneKO (aliased as --per-gene-max-titration). "
                             "Schedule starts at MAX non-NTC cells/KO — large KOs keep gaining cells, "
                             "small KOs cap out. Output → titration_geneKO_max/")
    parser.add_argument("--per-guide-min-titration", action="store_true",
                        help="Titrate by cells-per-sgRNA (every guide, incl. each NTC sgRNA, "
                             "gets the same budget). Schedule starts at MIN non-NTC cells/guide. "
                             "Output → titration_guide_min/")
    parser.add_argument("--bootstrap", type=int, default=1, metavar="N",
                        help="Run N bootstrap draws per titration point (default: 1 = no bootstrap). "
                             "Adds {metric}_sem columns and error bars on plots.")
    parser.add_argument("--bootstrap-replace", dest="bootstrap_replace",
                        action="store_true", default=False,
                        help="Draw each bootstrap sample WITH replacement (true nonparametric "
                             "bootstrap) instead of subsampling without replacement. "
                             "Per-guide sampling only. Default: without replacement.")
    parser.add_argument("--per-guide-max-titration", action="store_true",
                        help="Titrate by cells-per-sgRNA. Schedule starts at MAX non-NTC "
                             "cells/guide — large guides keep gaining, small guides cap out. "
                             "Output → titration_per_guide/")
    parser.add_argument("--per-guide-median-titration", action="store_true",
                        help="Titrate by cells-per-sgRNA. Schedule starts at MAX (p90 non-NTC) "
                             "and STOPS once the median non-NTC cells/guide is reached "
                             "(upper-half of guide cell-count range only). "
                             "Output → titration_guide_median/")
    parser.add_argument("--no-cache", dest="cache", action="store_false", default=True,
                        help="Disable row-level caching (recompute every titration point even "
                             "if an existing <reporter>_titration.csv already has it).")
    parser.add_argument(
        "--per-target-slurm", dest="per_target_slurm",
        action="store_true", default=True,
        help="(Default: ON) Fan out ONE SLURM task per (reporter, target) bin "
             "instead of one per reporter. Mirrors combined_titration "
             "--per-target-slurm: schedules are pre-built on the login node "
             "(backed-mode reads), each task scores its single target and "
             "writes a shard CSV (<reporter>_titration_t<target>.csv), and "
             "shards are merged into the canonical <reporter>_titration.csv "
             "after all tasks complete. Requires --slurm. Trades extra "
             "h5ad-load I/O per task for wall-clock parallelism.",
    )
    parser.add_argument(
        "--no-per-target-slurm", dest="per_target_slurm", action="store_false",
        help="Disable per-(reporter,target) SLURM fan-out and revert to "
             "one job per reporter.",
    )
    phase_group = parser.add_mutually_exclusive_group()
    phase_group.add_argument("--phase-only", action="store_true")
    phase_group.add_argument("--no-phase", action="store_true")
    return parser


def _resolve_output_dir(args) -> Path:
    """Mirror pca_optimization.main() output nesting (non --direct)."""
    only_4i = getattr(args, "only_4i", False)
    only_cp = getattr(args, "only_cp", False)
    if only_4i:
        args.include_4i = True
    if only_cp:
        args.include_cp = True
    include_standard = not (only_4i or only_cp)

    output_dir = Path(args.output_dir)
    if args.cell_profiler:
        output_dir = output_dir / "cellprofiler"
    elif getattr(args, "cell_dino", False):
        output_dir = output_dir / "cell_dino"
    else:
        output_dir = output_dir / "dino"

    if getattr(args, "zscore_per_experiment", False):
        output_dir = output_dir / "zscore_per_exp"

    if getattr(args, "paper_v1", None):
        output_dir = output_dir / "paper_v1"

    if getattr(args, "run_tag", None):
        output_dir = output_dir / args.run_tag

    if getattr(args, "include_cellpainting", False):
        output_dir = output_dir / "with_cellpainting"

    if getattr(args, "include_cp", False):
        output_dir = output_dir / ("only_cp" if only_cp and not include_standard else "with_cp")

    if getattr(args, "include_4i", False):
        output_dir = output_dir / ("only_4i" if only_4i and not include_standard else "with_4i")

    ds_suffix = "_per_guide" if getattr(args, "downsample_per_guide", False) else ""
    ds_on = args.downsampled or getattr(args, "downsample_per_guide", False)
    if getattr(args, "phase_only", False) and ds_on:
        output_dir = output_dir / f"phase_only_downsampled{ds_suffix}"
    elif getattr(args, "no_phase", False) and ds_on:
        output_dir = output_dir / f"no_phase_downsampled{ds_suffix}"
    elif getattr(args, "phase_only", False):
        output_dir = output_dir / "phase_only"
    elif getattr(args, "no_phase", False):
        output_dir = output_dir / "no_phase"
    elif ds_on:
        output_dir = output_dir / f"downsampled{ds_suffix}"
    else:
        output_dir = output_dir / "all_livecell"

    ft = getattr(args, "fixed_threshold", None)
    if ft is not None and ft > 0:
        output_dir = output_dir / f"fixed_{ft:.0%}"
    else:
        output_dir = output_dir / "consensus_sweep"

    output_dir = output_dir / args.distance
    return output_dir


def _emit_combined_plots(titration_dir) -> None:
    """Across-reporter combined plot — the tail step of every run mode
    (local, per-reporter SLURM, per-target SLURM, and --replot)."""
    titration_dir = Path(titration_dir)
    print("Generating combined titration plot...")
    _plot_combined_titration(titration_dir, _plt())
    print(f"Saved {titration_dir}/titration_combined_{{linear,log2,log10}}.{{png,svg}}")


def _replot_one(csv_path: Path) -> str:
    """Plot a single reporter from its CSV; returns sig_safe for progress reporting."""
    plt = _plt()
    df = pd.read_csv(csv_path)
    signal = (
        df["signal"].iloc[0]
        if "signal" in df.columns
        else csv_path.stem.replace("_titration", "")
    )
    reporter_dir = csv_path.parent
    sig_safe = reporter_dir.name
    _plot_titration(df, signal, reporter_dir, sig_safe, plt)

    plt.close("all")
    return sig_safe


def _replot(titration_dir):
    """Regenerate all per-reporter and combined plots from existing CSVs, in parallel."""
    titration_dir = Path(titration_dir)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from ops_utils.hpc.resource_manager import get_optimal_workers
    from tqdm import tqdm

    csv_files = sorted(titration_dir.glob("*/*_titration.csv"))
    if not csv_files:
        print(f"No *_titration.csv files found under {titration_dir}")
        return

    # Plotting is CPU + light RAM bound; leave GPU out of the equation
    n_workers = get_optimal_workers(use_gpu=False, model_ram_gb=0.05, data_ram_gb=0.2)
    print(f"Replotting {len(csv_files)} reporters with {n_workers} workers...")

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_replot_one, csv_path): csv_path
            for csv_path in csv_files
        }
        with tqdm(total=len(futures), unit="reporter") as pbar:
            for fut in as_completed(futures):
                try:
                    pbar.set_postfix_str(fut.result())
                except Exception as exc:
                    pbar.set_postfix_str(f"ERROR {futures[fut].stem}: {exc}")
                pbar.update(1)

    _emit_combined_plots(titration_dir)

# ---------------------------------------------------------------------------
# Per-target SLURM helpers (picklable top-level functions for submitit)
# ---------------------------------------------------------------------------


def _build_schedule_for_cells_path(
    cells_h5ad_path: Path,
    per_guide_min: bool,
    per_guide_max: bool,
    per_guide_median: bool,
    per_ko_min: bool,
    per_ko_max: bool,
) -> List[int]:
    """Build the titration schedule for one reporter using backed-mode read.

    Mirrors the schedule-building branches in ``titrate_single_reporter`` but
    only touches ``obs`` (no X load), so it's cheap enough to call on a login
    node before fanning out per-target SLURM tasks.
    """
    a = ad.read_h5ad(cells_h5ad_path, backed="r")
    if per_guide_min or per_guide_max or per_guide_median:
        sg_counts, non_ntc_counts = _guide_count_pools(a)
        pool = non_ntc_counts if len(non_ntc_counts) else sg_counts
        if per_guide_median:
            start = int(np.median(pool.values))
        elif per_guide_max:
            start = int(np.percentile(pool.values, 90))
        else:  # per_guide_min
            start = (
                int(non_ntc_counts.min())
                if len(non_ntc_counts) else int(sg_counts.min())
            )
        return _build_per_ko_schedule(start)
    if per_ko_min or per_ko_max:
        counts = a.obs[_pert_col(a)].value_counts()
        non_ntc_counts = _non_ntc(counts)
        if per_ko_max:
            pool = non_ntc_counts if len(non_ntc_counts) else counts
            start = int(np.percentile(pool.values, 90))
        else:  # per_ko_min
            start = (
                int(non_ntc_counts.min())
                if len(non_ntc_counts) else int(counts.min())
            )
        return _build_per_ko_schedule(start)
    return _build_per_ko_schedule(a.n_obs, MIN_CELLS)


def titrate_single_target_for_reporter(
    cells_h5ad_path: str,
    output_dir: str,
    target: int,
    norm_method: str = "ntc",
    random_seed: int = 42,
    min_exp: bool = False,
    per_ko: bool = False,
    per_guide: bool = False,
    n_bootstraps: int = 1,
    replace: bool = False,
) -> str:
    """Score one reporter at ONE schedule target; write a shard CSV.

    Top-level + picklable so ``submit_parallel_jobs`` can fan out one task
    per (reporter, target) pair. The shard goes to
    ``<output_dir>/<reporter>/<reporter>_titration_t<target>.csv`` and is
    merged into the canonical CSV by :func:`_merge_per_target_shards` once
    every shard for a reporter has been written.
    """
    _logger = _init_logger()
    t_start = time.time()
    cells_h5ad_path = Path(cells_h5ad_path)
    output_dir = Path(output_dir)
    rng = np.random.RandomState(random_seed + int(target))

    adata_cells = ad.read_h5ad(cells_h5ad_path)
    signal = adata_cells.obs.get(
        "signal", pd.Series([cells_h5ad_path.stem.replace("_cells", "")])
    ).iloc[0]
    if isinstance(signal, float):
        signal = cells_h5ad_path.stem.replace("_cells", "")

    from ops_utils.data.feature_discovery import sanitize_signal_filename
    sig_safe = sanitize_signal_filename(signal)[:40]
    reporter_dir = output_dir / sig_safe
    reporter_dir.mkdir(parents=True, exist_ok=True)

    df = _run_titration_points(
        adata_cells.copy(),
        [int(target)],
        norm_method,
        signal,
        rng,
        _logger,
        min_exp=min_exp,
        per_ko=per_ko,
        per_guide=per_guide,
        n_bootstraps=n_bootstraps,
        replace=replace,
    )
    shard_csv = reporter_dir / f"{sig_safe}_titration_t{int(target)}.csv"
    df.to_csv(shard_csv, index=False)
    return f"SUCCESS: {signal} @ {target:,} → {shard_csv} ({time.time() - t_start:.0f}s)"


def _merge_per_target_shards(
    titration_dir: Path, target_col: str, _logger,
) -> List[Path]:
    """Concat all per-target shards into per-reporter canonical CSVs + plots.

    For each reporter dir under ``titration_dir``, concatenate every
    ``*_titration_t*.csv`` shard, dedupe on ``target_col``, sort by it
    (descending), and write ``<reporter>_titration.csv``. Returns the list
    of canonical CSV paths produced.
    """
    titration_dir = Path(titration_dir)
    written: List[Path] = []
    for reporter_dir in sorted(titration_dir.iterdir()):
        if not reporter_dir.is_dir():
            continue
        sig_safe = reporter_dir.name
        shards = sorted(reporter_dir.glob(f"{sig_safe}_titration_t*.csv"))
        if not shards:
            continue
        dfs = []
        for s in shards:
            try:
                d = pd.read_csv(s)
                if not d.empty:
                    dfs.append(d)
            except pd.errors.EmptyDataError:
                continue
        if not dfs:
            continue
        # Honor any pre-existing canonical rows (e.g. from a prior non-per-target
        # run) by feeding them in as the cached side of the shared merge, so the
        # newly-merged shards win on duplicate targets.
        canonical = reporter_dir / f"{sig_safe}_titration.csv"
        prior = _read_cache(canonical, target_col, _logger=_logger)
        merged = _merge_and_write(
            pd.concat(dfs, ignore_index=True),
            prior.to_dict(orient="records") if prior is not None else [],
            target_col, canonical, _logger,
        )
        written.append(canonical)
        _logger.info(
            f"[merge] {sig_safe}: {len(merged)} rows ({len(shards)} shards) → {canonical}"
        )
        try:
            plt = _plt()
            signal = (
                str(merged["signal"].iloc[0])
                if "signal" in merged.columns and len(merged)
                else sig_safe
            )
            _plot_titration(merged, signal, reporter_dir, sig_safe, plt)
        except Exception as exc:
            _logger.warning(f"  Plotting {sig_safe} failed: {exc}")
    return written


def main():
    parser = _build_parser()
    args = parser.parse_args()
    _logger = _init_logger()

    # Auto-scale SLURM time when the user kept the default and is bootstrapping.
    # Explicit --slurm-time overrides are respected as-is (treated as total).
    if args.bootstrap > 1 and args.slurm_time == parser.get_default("slurm_time"):
        scaled = args.slurm_time * args.bootstrap
        print(
            f"[bootstrap={args.bootstrap}] Auto-scaling --slurm-time from "
            f"{args.slurm_time}min → {scaled}min (pass --slurm-time to override)"
        )
        args.slurm_time = scaled

    variant_dir = _resolve_output_dir(args)
    if args.per_guide_median_titration:
        titration_subdir = "titration_guide_median"
    elif args.per_guide_max_titration:
        titration_subdir = "titration_per_guide"
    elif args.per_guide_min_titration:
        titration_subdir = "titration_guide_min"
    elif args.per_ko_max_titration:
        titration_subdir = "titration_geneKO_max"
    elif args.per_ko_min_titration:
        titration_subdir = "titration_geneKO_min"
    elif args.min_exp_titration:
        titration_subdir = "titration_min_exp"
    else:
        titration_subdir = "titration"
    titration_dir = variant_dir / titration_subdir

    if args.replot:
        titration_dir.mkdir(parents=True, exist_ok=True)
        if args.slurm:
            from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
            print(f"Submitting replot as a single SLURM job ({titration_dir})...")
            jobs = [{
                "name": f"titr_replot_{titration_dir.parent.name}",
                "func": _replot,
                "kwargs": {"titration_dir": str(titration_dir)},
            }]
            slurm_params = {
                "timeout_min": args.slurm_time,
                "mem": args.slurm_memory,
                "cpus_per_task": args.slurm_cpus,
                "slurm_partition": args.slurm_partition,
            }
            submit_parallel_jobs(
                jobs_to_submit=jobs,
                experiment="titration",
                slurm_params=slurm_params,
                log_dir="pca_optimization",
                manifest_prefix="titration_replot",
                wait_for_completion=True,
            )
        else:
            _replot(titration_dir)
        return

    per_signal_dir = variant_dir / "per_signal"
    titration_dir.mkdir(parents=True, exist_ok=True)

    cells_files = sorted(per_signal_dir.glob("*_cells.h5ad"))
    if not cells_files:
        print(f"No cell-level h5ads (*_cells.h5ad) found in {per_signal_dir}")
        print("  Re-run Phase 1 (pca_optimization --slurm --clean) to generate them.")
        return

    print(f"Found {len(cells_files)} reporters in {per_signal_dir}")
    print(f"Titration output: {titration_dir}")

    # --per-target-slurm is the default; silently fall back to
    # one-job-per-reporter for the two cases where it isn't compatible.
    if args.per_target_slurm and not args.slurm:
        print("[info] --per-target-slurm is the default but only takes effect "
              "under --slurm; running locally (one reporter at a time).")
        args.per_target_slurm = False

    if args.per_target_slurm:
        from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

        target_col = _target_col(_mode_from_flags(
            per_guide=(args.per_guide_min_titration or args.per_guide_max_titration
                       or args.per_guide_median_titration),
            per_ko=(args.per_ko_min_titration or args.per_ko_max_titration),
        ))

        # Build per-reporter schedules in parallel — each reads one cells h5ad's
        # obs (~60M rows) + a groupby, which is independent across reporters.
        # Bounded worker count to keep login-node memory in check.
        import os as _os
        from concurrent.futures import ProcessPoolExecutor, as_completed
        _sched_kw = dict(
            per_guide_min=args.per_guide_min_titration,
            per_guide_max=args.per_guide_max_titration,
            per_guide_median=args.per_guide_median_titration,
            per_ko_min=args.per_ko_min_titration,
            per_ko_max=args.per_ko_max_titration,
        )
        _n_workers = min(len(cells_files), (_os.cpu_count() or 4), 8)
        print(f"\nBuilding per-reporter schedules ({len(cells_files)} reporters, "
              f"{_n_workers} parallel workers)...")
        per_reporter_schedule: Dict[Path, List[int]] = {}
        with ProcessPoolExecutor(max_workers=_n_workers) as _pool:
            _futs = {_pool.submit(_build_schedule_for_cells_path, cf, **_sched_kw): cf
                     for cf in cells_files}
            for _fut in as_completed(_futs):
                cf = _futs[_fut]
                sched = _fut.result()
                per_reporter_schedule[cf] = sched
                print(f"  {cf.stem.replace('_cells', '')}: {len(sched)} target(s) "
                      f"(head: {sched[:3]}, tail: {sched[-2:]})" if len(sched) > 3
                      else f"  {cf.stem.replace('_cells', '')}: {sched}")

        # Cache short-circuit: if a canonical CSV already has this target, skip.
        # A row counts as "cached" only when every metric is actually scored —
        # an old CSV that has the target but is missing EBI columns (added
        # after CHAD on 2026-05-19) should NOT be skipped.
        required_metric_cols = [f"{m}_map_mean" for m in METRICS]

        all_jobs = []
        skipped_cache = 0
        n_planned = 0
        from ops_utils.data.feature_discovery import sanitize_signal_filename
        for cf in cells_files:
            sig = cf.stem.replace("_cells", "")
            sig_safe = sanitize_signal_filename(sig)[:40]
            reporter_dir = titration_dir / sig_safe
            canonical = reporter_dir / f"{sig_safe}_titration.csv"
            cached_targets: set = set()
            if args.cache:
                cached_targets = _scored_targets(
                    canonical, target_col, required_cols=required_metric_cols,
                )
            for target in per_reporter_schedule[cf]:
                n_planned += 1
                if int(target) in cached_targets:
                    skipped_cache += 1
                    continue
                # Skip if a complete shard already exists (idempotent re-submits).
                # A shard from a pre-EBI run won't have ebi_map_mean so it
                # falls through and gets re-scored.
                shard = reporter_dir / f"{sig_safe}_titration_t{int(target)}.csv"
                if args.cache and int(target) in _scored_targets(
                    shard, target_col, required_cols=required_metric_cols,
                ):
                    skipped_cache += 1
                    continue
                all_jobs.append({
                    "name": f"titr_{sig_safe}_t{int(target)}",
                    "func": titrate_single_target_for_reporter,
                    "kwargs": {
                        "cells_h5ad_path": str(cf),
                        "output_dir": str(titration_dir),
                        "target": int(target),
                        "norm_method": args.norm_method,
                        "min_exp": args.min_exp_titration,
                        "per_ko": (args.per_ko_min_titration or args.per_ko_max_titration),
                        "per_guide": (
                            args.per_guide_min_titration
                            or args.per_guide_max_titration
                            or args.per_guide_median_titration
                        ),
                        "n_bootstraps": int(args.bootstrap),
                        "replace": bool(args.bootstrap_replace),
                    },
                })

        print(
            f"\n[per-target] {len(all_jobs)} task(s) to submit "
            f"({skipped_cache} cached / {n_planned} planned across "
            f"{len(cells_files)} reporters)..."
        )

        if all_jobs:
            slurm_params = {
                "mem": args.slurm_memory,
                "cpus_per_task": args.slurm_cpus,
                "slurm_partition": args.slurm_partition,
                "timeout_min": args.slurm_time,
            }
            print(
                f"  slurm: {args.slurm_time}min, {args.slurm_memory}, "
                f"{args.slurm_cpus} CPUs per task; partition={args.slurm_partition}"
            )
            submit_parallel_jobs(
                jobs_to_submit=all_jobs,
                experiment="titration",
                slurm_params=slurm_params,
                log_dir="pca_optimization",
                manifest_prefix="titration_per_target",
                wait_for_completion=True,
                verbose=True,
            )

        print("\nMerging per-target shards into canonical reporter CSVs...")
        _merge_per_target_shards(titration_dir, target_col, _logger)

        # Combined plot across reporters
        _emit_combined_plots(titration_dir)
        return

    if args.slurm:
        from ops_utils.hpc.slurm_batch_utils import (
            submit_parallel_jobs,
            wait_for_multiple_job_arrays,
        )

        def _make_job(cf):
            sig_safe = cf.stem.replace("_cells", "")[:40]
            return {
                "name": f"titr_{sig_safe}",
                "func": titrate_single_reporter,
                "kwargs": {
                    "cells_h5ad_path": str(cf),
                    "output_dir": str(titration_dir),
                    "norm_method": args.norm_method,
                    "min_exp": args.min_exp_titration,
                    "per_ko": args.per_ko_min_titration,
                    "per_ko_max": args.per_ko_max_titration,
                    "per_guide": args.per_guide_min_titration,
                    "per_guide_max": args.per_guide_max_titration,
                    "per_guide_median": args.per_guide_median_titration,
                    "n_bootstraps": int(args.bootstrap),
                    "cache": bool(args.cache),
                    "replace": bool(args.bootstrap_replace),
                },
            }

        phase_files = [cf for cf in cells_files if cf.stem.lower().startswith("phase")]
        other_files = [cf for cf in cells_files if cf not in phase_files]
        regular_jobs = [_make_job(cf) for cf in other_files]
        phase_jobs = [_make_job(cf) for cf in phase_files]

        # Auto-scale phase time by bootstrap when user kept the default
        phase_time = args.phase_slurm_time
        if args.bootstrap > 1 and phase_time == parser.get_default("phase_slurm_time"):
            phase_time = phase_time * args.bootstrap
            print(f"[bootstrap={args.bootstrap}] Auto-scaling --phase-slurm-time → {phase_time}min")

        base_slurm_params = {
            "mem": args.slurm_memory,
            "cpus_per_task": args.slurm_cpus,
            "slurm_partition": args.slurm_partition,
        }

        # Submit both arrays without waiting so they run in parallel on SLURM,
        # then watch them together (mirrors pca_optimization _handle_downsampled).
        job_arrays = []
        regular_params = {**base_slurm_params, "timeout_min": args.slurm_time}
        phase_params = {**base_slurm_params, "timeout_min": phase_time}

        if regular_jobs:
            print(f"\nSubmitting {len(regular_jobs)} SLURM titration jobs ({args.slurm_time}min each)...")
            result_reg = submit_parallel_jobs(
                jobs_to_submit=regular_jobs,
                experiment="titration",
                slurm_params=regular_params,
                log_dir="pca_optimization",
                manifest_prefix="titration",
                wait_for_completion=False,
            )
            if result_reg.get("submitted_jobs"):
                job_arrays.append({
                    "submitted_jobs": result_reg["submitted_jobs"],
                    "base_job_id": result_reg["base_job_id"],
                    "label": "reporters",
                    "slurm_params": regular_params,
                })

        if phase_jobs:
            print(f"\nSubmitting {len(phase_jobs)} Phase titration job(s) ({phase_time}min)...")
            result_phase = submit_parallel_jobs(
                jobs_to_submit=phase_jobs,
                experiment="titration_phase",
                slurm_params=phase_params,
                log_dir="pca_optimization",
                manifest_prefix="titration_phase",
                wait_for_completion=False,
            )
            if result_phase.get("submitted_jobs"):
                job_arrays.append({
                    "submitted_jobs": result_phase["submitted_jobs"],
                    "base_job_id": result_phase["base_job_id"],
                    "label": "phase",
                    "slurm_params": phase_params,
                })

        all_failed = []
        if job_arrays:
            wait_result = wait_for_multiple_job_arrays(
                job_arrays,
                experiment="titration",
            )
            all_failed = wait_result.get("failed", []) or []

        total_jobs = len(regular_jobs) + len(phase_jobs)
        if all_failed:
            print(f"\n{len(all_failed)} jobs failed")
        else:
            print(f"\nAll {total_jobs} titration jobs complete")

        # Generate combined plot from all CSVs
        _emit_combined_plots(titration_dir)

    else:
        print("\nRunning locally (sequential)...")
        for cf in cells_files:
            result = titrate_single_reporter(
                cells_h5ad_path=str(cf),
                output_dir=str(titration_dir),
                norm_method=args.norm_method,
                min_exp=args.min_exp_titration,
                per_ko=args.per_ko_min_titration,
                per_ko_max=args.per_ko_max_titration,
                per_guide=args.per_guide_min_titration,
                per_guide_max=args.per_guide_max_titration,
                per_guide_median=args.per_guide_median_titration,
                n_bootstraps=int(args.bootstrap),
                cache=bool(args.cache),
                replace=bool(args.bootstrap_replace),
            )
            print(f"  {result}")

        _emit_combined_plots(titration_dir)


if __name__ == "__main__":
    main()
