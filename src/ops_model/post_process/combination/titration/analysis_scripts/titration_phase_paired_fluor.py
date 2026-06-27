"""Titration sweep that pairs Phase with each fluorescent marker at every bin.

For each bin in the existing Phase per-guide titration, this wrapper:

  1. Subsamples Phase cells per sgRNA at ``K cells/guide`` and aggregates to guide.
  2. Subsamples each fluorescent marker's cells per sgRNA at the same ``K`` and
     aggregates to guide.
  3. Horizontally concatenates the two guide-level matrices (inner join on
     ``sgRNA``) so each guide carries ``[phase_features || marker_features]``.
  4. NTC-normalizes and scores the four mAP metrics (activity, distinctiveness,
     CHAD, EBI consistency) on the paired guide-level features.

After all (marker, bin) cells have been scored we pick, per bin, the marker
that maximises each metric — i.e. *which fluorescent channel paired with Phase
would most boost mAP at this per-guide cell budget?* — and overlay a new line
on the existing Phase titration plot with the winning marker annotated at each
bin.

Reuses ``titration``'s subsample + scoring helpers verbatim. The only new
machinery is the per-marker driver and the post-hoc winners plot. Each marker
runs as one SLURM job (10 bins × 1 marker × 4 metrics fits in <5 min on cpu).

Usage::

    # Submit one SLURM task per fluor marker (~40 jobs, parallel)
    uv run python -m ops_model.post_process.combination.titration.titration_phase_paired_fluor --slurm

    # Aggregate already-scored shards + replot (no SLURM)
    uv run python -m ops_model.post_process.combination.titration.titration_phase_paired_fluor --replot

The default ``--per-signal-dir`` points at the canonical ``all_livecell`` run-tag
where every fluorescent marker has a per-signal cells.h5ad. ``--phase-titration-csv``
defaults to the per-guide-median phase titration we already ran for the
attention-weighting analysis.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd

from ops_model.features.anndata_utils import (
    hconcat_by_perturbation,
    normalize_guide_adata,
)
from ops_model.post_process.combination.titration.titration import (
    _score_all_metrics,
    _subsample_per_guide_and_aggregate,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_PER_SIGNAL_DIR = Path(
    "/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3/"
    "cell_dino/zscore_per_exp/paper_v1/all_livecell/fixed_80%/cosine/per_signal"
)
DEFAULT_PHASE_TITRATION_CSV = Path(
    "/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3/"
    "cell_dino/zscore_per_exp/paper_v1/phase_only/fixed_80%/cosine/"
    "titration_guide_median/Phase/Phase_titration.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3/"
    "cell_dino/zscore_per_exp/paper_v1/phase_only/fixed_80%/cosine/"
    "titration_phase_plus_fluor"
)

# Channel-name → "is this a fluorescent marker" classifier. We exclude Phase
# itself, autofluorescence channels, Cell Painting markers, and 4i markers
# since the user's question is specifically about the live-cell fluor panel.
CELLPAINTING_PREFIXES = (
    "Nucleus_Hoechst", "Mitochondria_TOMM20", "Plasma_Membrane_Wheat_Germ",
    "F-actin_Phalloidin", "Nucleoli_NPM1", "Microtubules_Tubulin",
    "Endoplasmic_Reticulum_Concanavalin_A",
)
FOUR_I_KEYWORDS = ("p21", "p53", "pRb", "pS6", "c-Myc", "b-catenin")


def _is_fluor_signal(name: str) -> bool:
    """Filter to live-cell fluorescent markers, exclude Phase + CP + 4i."""
    if name == "Phase":
        return False
    if name.startswith(CELLPAINTING_PREFIXES):
        return False
    if any(k in name for k in FOUR_I_KEYWORDS):
        return False
    return True


# ---------------------------------------------------------------------------
# Per-marker worker (SLURM-picklable; top level)
# ---------------------------------------------------------------------------

def run_one_marker_titration(
    phase_h5ad_path: str,
    marker_h5ad_path: str,
    marker_label: str,
    target: int,
    norm_method: str,
    output_dir: str,
    random_seed: int = 42,
) -> str:
    """SLURM worker: score Phase + ``marker_label`` at ONE bin ``target``.

    Writes a per-(marker, target) shard CSV at
    ``<output_dir>/Phase+<marker>/Phase+<marker>_titration_t<target>.csv``.
    Top-level + picklable so ``submit_parallel_jobs`` can fan it out per
    (marker, bin) — same pattern as ``titration._run_per_target_shard``.
    The shards are merged by ``_merge_shards`` after all jobs complete.
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    _logger = logging.getLogger(f"pair[{marker_label}@{target}]")
    t_start = time.time()

    marker_dir = Path(output_dir) / f"Phase+{marker_label}"
    marker_dir.mkdir(parents=True, exist_ok=True)
    shard_csv = marker_dir / f"Phase+{marker_label}_titration_t{int(target)}.csv"
    if shard_csv.exists() and not _force_recompute():
        return f"SKIP {marker_label} @ {target}: shard already exists"

    phase = ad.read_h5ad(phase_h5ad_path)
    marker = ad.read_h5ad(marker_h5ad_path)
    _logger.info(
        "Loaded Phase=%d cells / %d feats; %s=%d cells / %d feats",
        phase.n_obs, phase.n_vars, marker_label, marker.n_obs, marker.n_vars,
    )

    # Prefix var_names so column identities stay distinct after hconcat.
    phase.var_names = pd.Index([f"phase__{v}" for v in phase.var_names])
    marker.var_names = pd.Index([f"{marker_label}__{v}" for v in marker.var_names])

    # Seed offset by target so different bins draw different cells but a
    # re-run at the same (marker, target) is reproducible.
    rng = np.random.RandomState(random_seed + int(target))
    K = int(target)
    phase_g = _subsample_per_guide_and_aggregate(phase, K, rng)
    marker_g = _subsample_per_guide_and_aggregate(marker, K, rng)
    combined = hconcat_by_perturbation([phase_g, marker_g], level="guide")
    if combined.n_obs == 0:
        return f"FAIL {marker_label} @ {target}: no shared guides"
    combined = normalize_guide_adata(combined, norm_method)
    scores = _score_all_metrics(combined, _logger)
    scores.update({
        "cells_per_guide": K,
        "marker": marker_label,
        "n_guides_phase": phase_g.n_obs,
        "n_guides_marker": marker_g.n_obs,
        "n_guides_paired": combined.n_obs,
        "n_features_total": int(combined.n_vars),
        "signal": f"Phase+{marker_label}",
    })
    pd.DataFrame([scores]).to_csv(shard_csv, index=False)
    return (
        f"OK {marker_label} @ K={K}: guides={combined.n_obs} "
        f"act={scores.get('activity_map_mean', float('nan')):.3f} "
        f"ebi={scores.get('ebi_map_mean', float('nan')):.3f} "
        f"({int(time.time() - t_start)}s)"
    )


def _merge_shards(output_dir: Path, _logger=logger) -> None:
    """Concat every ``Phase+<marker>_titration_t*.csv`` shard into one canonical
    ``Phase+<marker>_titration.csv`` per marker — mirrors
    ``titration._merge_per_target_shards``.
    """
    for marker_dir in sorted(output_dir.iterdir()):
        if not marker_dir.is_dir() or not marker_dir.name.startswith("Phase+"):
            continue
        shards = sorted(marker_dir.glob(f"{marker_dir.name}_titration_t*.csv"))
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
        merged = pd.concat(dfs, ignore_index=True)
        merged = merged.sort_values("cells_per_guide").reset_index(drop=True)
        merged.to_csv(marker_dir / f"{marker_dir.name}_titration.csv", index=False)
        _logger.info("  merged %d shards → %s", len(dfs),
                     f"{marker_dir.name}/{marker_dir.name}_titration.csv")


_FORCE_FLAG = {"value": False}


def _force_recompute() -> bool:
    return _FORCE_FLAG["value"]


# ---------------------------------------------------------------------------
# Coordinator: discover markers, fan out, plot
# ---------------------------------------------------------------------------

def _discover_fluor_markers(per_signal_dir: Path) -> List[Tuple[str, Path]]:
    """Find every fluor marker's cells.h5ad in ``per_signal_dir``."""
    out = []
    for p in sorted(per_signal_dir.glob("*_cells.h5ad")):
        # Skip the sub-sampled companion files (handled by the canonical pipeline only).
        if p.stem.endswith("_cells_sub") or p.stem == "Phase_cells":
            continue
        marker_label = p.stem[:-len("_cells")]
        if not _is_fluor_signal(marker_label):
            continue
        out.append((marker_label, p))
    return out


def _load_phase_schedule(phase_titration_csv: Path) -> List[int]:
    """Read existing Phase titration's per-guide bin schedule (sorted asc)."""
    df = pd.read_csv(phase_titration_csv)
    if "cells_per_guide" not in df.columns:
        raise ValueError(
            f"{phase_titration_csv} has no 'cells_per_guide' column; was it "
            "produced by --per-guide-median-titration?"
        )
    targets = sorted(set(int(x) for x in df["cells_per_guide"].dropna().unique()))
    return targets


def _load_or_build_marker_median_cpg(
    per_signal_dir: Path, cache_path: Path
) -> Dict[str, int]:
    """Median cells-per-guide for every marker in ``per_signal_dir``. Cached
    next to the titration outputs so we don't re-open 39 h5ads each replot.
    Only reads obs/sgRNA via h5py — cheap.
    """
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        return {r["marker"]: int(r["median_cpg"]) for _, r in df.iterrows()}
    import h5py
    rows = []
    for p in sorted(per_signal_dir.glob("*_cells.h5ad")):
        marker = p.name.replace("_cells.h5ad", "")
        if marker == "Phase":
            continue
        try:
            with h5py.File(p, "r") as f:
                obj = f["obs/sgRNA"]
                if isinstance(obj, h5py.Group):
                    codes = obj["codes"][:]
                    sizes = np.bincount(codes[codes >= 0])
                else:
                    _, sizes = np.unique(obj[:], return_counts=True)
            rows.append((marker, int(np.median(sizes)), int(len(sizes)),
                         int(sizes.sum())))
        except Exception as e:
            logger.warning("median_cpg failed for %s: %s", marker, e)
    df = pd.DataFrame(rows, columns=["marker", "median_cpg", "n_guides", "n_cells"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.info("wrote %s (%d markers)", cache_path, len(df))
    return {r["marker"]: int(r["median_cpg"]) for _, r in df.iterrows()}


def aggregate_shards(output_dir: Path) -> pd.DataFrame:
    """Walk every ``Phase+<marker>/Phase+<marker>_titration.csv`` (already
    merged from per-target shards by ``_merge_shards``) into one long-form
    DataFrame keyed by (marker, cells_per_guide).
    """
    rows = []
    for marker_dir in sorted(output_dir.iterdir()):
        if not marker_dir.is_dir() or not marker_dir.name.startswith("Phase+"):
            continue
        canonical = marker_dir / f"{marker_dir.name}_titration.csv"
        if canonical.exists():
            df = pd.read_csv(canonical)
        else:
            # Fall back to direct shard concat (in case merge hasn't run)
            shards = sorted(marker_dir.glob(f"{marker_dir.name}_titration_t*.csv"))
            if not shards:
                continue
            df = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(output_dir / "phase_plus_fluor_combined_titration.csv", index=False)
    return out


def compute_winners(
    combined: pd.DataFrame, phase_df: pd.DataFrame, min_markers: int = 5,
) -> pd.DataFrame:
    """For each (cells_per_guide, metric) pair, pick the marker with the
    highest ``<metric>_map_mean`` across all paired markers; also record the
    Phase-only baseline at the same K for the delta.

    Skips any (K, metric) where fewer than ``min_markers`` markers have a
    finite score — too few candidates to call one a winner.
    """
    metrics = ("activity", "distinctiveness", "chad", "ebi")
    rows = []
    phase_by_K = phase_df.set_index("cells_per_guide")
    for K, sub in combined.groupby("cells_per_guide"):
        for m in metrics:
            col = f"{m}_map_mean"
            if col not in sub.columns or sub[col].isna().all():
                continue
            n_candidates = sub[col].dropna().shape[0]
            if n_candidates < min_markers:
                continue
            i = sub[col].idxmax()
            winner = sub.loc[i]
            phase_at_K = (
                phase_by_K.loc[K, col]
                if K in phase_by_K.index and col in phase_by_K.columns
                else float("nan")
            )
            rows.append({
                "cells_per_guide": int(K),
                "metric": m,
                "winning_marker": winner["marker"],
                "paired_map_mean": float(winner[col]),
                "phase_only_map_mean": float(phase_at_K),
                "gain_vs_phase": (
                    float(winner[col]) - float(phase_at_K)
                    if np.isfinite(phase_at_K) else float("nan")
                ),
                "n_guides_paired": int(winner.get("n_guides_paired", 0) or 0),
            })
    df = pd.DataFrame(rows).sort_values(["metric", "cells_per_guide"])
    return df


# ---------------------------------------------------------------------------
# Plotting — delegated to the shared ``titration_paired_plots`` module so
# the dual-fluor sibling can share these 5 functions verbatim. The thin wrappers
# below preserve the existing call sites and the ``_pretty_marker`` import that
# the dual-fluor script used.
# ---------------------------------------------------------------------------

from ops_model.post_process.combination.titration import titration_paired_plots as _plots
from ops_model.post_process.combination.titration.titration_paired_plots import (
    pretty_marker as _pretty_marker,  # re-export
    bold_palette as _bold_palette,    # re-export
)


def plot_winners(winners: pd.DataFrame, phase_df: pd.DataFrame, out_dir: Path) -> None:
    _plots.plot_winners(
        winners, phase_df, out_dir,
        winning_col="winning_marker",
        label_fn=_plots.pretty_marker,
        out_prefix="phase_plus_fluor",
        line_legend="Phase + best fluor marker",
        suptitle="Phase + best fluor-marker pairing — per-bin winner annotated",
    )


def plot_rank_bump(combined: pd.DataFrame, out_dir: Path, top_n: int = 5) -> None:
    _plots.plot_rank_bump(
        combined, out_dir,
        entity_col="marker", label_fn=_plots.pretty_marker,
        out_prefix="phase_plus_fluor",
        ylabel="rank (1 = best paired marker)",
        suptitle=f"Marker rank-ordering (top-{top_n} highlighted per metric)",
        top_n=top_n,
    )


def plot_delta_heatmap(combined: pd.DataFrame, phase_df: pd.DataFrame, out_dir: Path) -> None:
    _plots.plot_delta_heatmap(
        combined, phase_df, out_dir,
        entity_col="marker", label_fn=_plots.pretty_marker,
        out_prefix="phase_plus_fluor",
        suptitle="Per-bin mAP gain over Phase baseline (markers sorted by mean gain)",
    )


def plot_topn_curves(combined: pd.DataFrame, phase_df: pd.DataFrame, out_dir: Path, top_n: int = 5) -> None:
    _plots.plot_topn_curves(
        combined, phase_df, out_dir,
        entity_col="marker", label_fn=_plots.pretty_marker,
        out_prefix="phase_plus_fluor",
        ribbon_label="paired 5–95% (39 markers)",
        suptitle=f"Phase + top-{top_n} fluor markers (per metric) with 5-95% paired ribbon",
        top_n=top_n,
    )


def plot_win_share(combined: pd.DataFrame, out_dir: Path, top_k_rank: int = 3) -> None:
    _plots.plot_win_share(
        combined, out_dir,
        entity_col="marker", label_fn=_plots.pretty_marker,
        out_prefix="phase_plus_fluor",
        suptitle=f"Marker win-share — top-{top_k_rank} per (bin, metric)",
        top_k_rank=top_k_rank,
    )




# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-signal-dir", type=Path, default=DEFAULT_PER_SIGNAL_DIR,
                   help=f"Dir with per-signal cells.h5ad files (default: {DEFAULT_PER_SIGNAL_DIR})")
    p.add_argument("--phase-titration-csv", type=Path,
                   default=DEFAULT_PHASE_TITRATION_CSV,
                   help=f"Existing Phase titration CSV to source the bin schedule "
                        f"and the phase-only baseline curve (default: {DEFAULT_PHASE_TITRATION_CSV})")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help=f"Where to write per-marker shards + winners CSV + plot "
                        f"(default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--norm-method", default="ntc",
                   choices=("ntc", "global"))
    p.add_argument("--markers", nargs="*", default=None,
                   help="Optional subset of marker labels to score "
                        "(default: every fluor marker in --per-signal-dir)")
    p.add_argument("--bins", nargs="*", type=int, default=None,
                   help="Optional subset of cells-per-guide bins to score "
                        "(default: every bin in --phase-titration-csv)")
    p.add_argument("--force", action="store_true",
                   help="Recompute even if a marker's shard CSV already exists")

    p.add_argument("--slurm", action="store_true",
                   help="Submit one SLURM job per marker (default: run locally serially)")
    p.add_argument("--slurm-memory", default="48GB")
    p.add_argument("--slurm-time", type=int, default=60)
    p.add_argument("--slurm-cpus", type=int, default=4)
    p.add_argument("--slurm-partition", default="cpu,gpu")

    p.add_argument("--replot", action="store_true",
                   help="Skip scoring; just aggregate existing shards + replot")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    _FORCE_FLAG["value"] = args.force

    per_signal_dir = args.per_signal_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    phase_cells_h5ad = per_signal_dir / "Phase_cells.h5ad"
    if not phase_cells_h5ad.exists():
        raise FileNotFoundError(f"Missing {phase_cells_h5ad}")
    if not args.phase_titration_csv.exists():
        raise FileNotFoundError(f"Missing {args.phase_titration_csv}")

    phase_df = pd.read_csv(args.phase_titration_csv).sort_values("cells_per_guide")
    cell_targets = _load_phase_schedule(args.phase_titration_csv)
    if args.bins:
        wanted_bins = set(args.bins)
        unknown = wanted_bins - set(cell_targets)
        if unknown:
            logger.warning("requested bins not in phase titration: %s", sorted(unknown))
        cell_targets = [b for b in cell_targets if b in wanted_bins]
    logger.info("Phase titration bin schedule: %s", cell_targets)

    available = _discover_fluor_markers(per_signal_dir)
    if args.markers:
        wanted = set(args.markers)
        available = [(m, p) for m, p in available if m in wanted]
        missing = wanted - {m for m, _ in available}
        if missing:
            logger.warning("requested markers not found: %s", sorted(missing))
    logger.info("Found %d fluor markers to pair with Phase", len(available))

    if not args.replot:
        # Build (marker × target) job list — one SLURM task per (marker, bin)
        # so the cluster runs them all in parallel. Same pattern as
        # titration's --per-target-slurm fan-out.
        jobs_spec = [
            (marker, path, target)
            for marker, path in available
            for target in cell_targets
        ]
        logger.info("Fanning out %d (marker × bin) tasks (%d markers × %d bins)",
                    len(jobs_spec), len(available), len(cell_targets))

        if args.slurm:
            from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
            jobs = [{
                "name": f"phase+{marker}@{target}",
                "func": run_one_marker_titration,
                "kwargs": {
                    "phase_h5ad_path": str(phase_cells_h5ad),
                    "marker_h5ad_path": str(path),
                    "marker_label": marker,
                    "target": target,
                    "norm_method": args.norm_method,
                    "output_dir": str(output_dir),
                },
                "metadata": {"marker": marker, "target": target},
            } for marker, path, target in jobs_spec]
            slurm_params = {
                "timeout_min": args.slurm_time,
                "mem": args.slurm_memory,
                "cpus_per_task": args.slurm_cpus,
                "slurm_partition": args.slurm_partition,
            }
            submit_parallel_jobs(
                jobs_to_submit=jobs,
                experiment="phase_plus_fluor_titration",
                slurm_params=slurm_params,
                log_dir="slurm_step_logs/phase_plus_fluor_titration",
                manifest_prefix="phase_plus_fluor_titration",
                wait_for_completion=True,
            )
        else:
            for marker, path, target in jobs_spec:
                msg = run_one_marker_titration(
                    phase_h5ad_path=str(phase_cells_h5ad),
                    marker_h5ad_path=str(path),
                    marker_label=marker,
                    target=target,
                    norm_method=args.norm_method,
                    output_dir=str(output_dir),
                )
                logger.info(msg)

        # Merge shards into per-marker canonical CSVs before plotting.
        _merge_shards(output_dir, logger)

    combined = aggregate_shards(output_dir)
    if combined.empty:
        logger.warning("No shards found in %s — nothing to plot", output_dir)
        return 1
    # Drop bins where K exceeds the marker's median cells-per-guide — beyond
    # that point we're padding with Phase guides only, no fluor uplift.
    median_cpg = _load_or_build_marker_median_cpg(
        per_signal_dir, output_dir / "marker_median_cpg.csv",
    )
    before = len(combined)
    combined["_median_cpg"] = combined["marker"].map(median_cpg)
    combined = combined[combined["cells_per_guide"] <= combined["_median_cpg"]].copy()
    combined = combined.drop(columns=["_median_cpg"])
    logger.info("dropped %d / %d rows where cells_per_guide > marker median_cpg",
                before - len(combined), before)
    combined.to_csv(
        output_dir / "phase_plus_fluor_combined_titration_capped.csv", index=False,
    )
    winners = compute_winners(combined, phase_df)
    winners.to_csv(output_dir / "phase_plus_fluor_winners.csv", index=False)
    logger.info("wrote %s/phase_plus_fluor_winners.csv (%d rows)",
                output_dir, len(winners))
    plot_winners(winners, phase_df, output_dir)
    plot_rank_bump(combined, output_dir, top_n=5)
    plot_delta_heatmap(combined, phase_df, output_dir)
    plot_topn_curves(combined, phase_df, output_dir, top_n=5)
    plot_win_share(combined, output_dir, top_k_rank=3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
