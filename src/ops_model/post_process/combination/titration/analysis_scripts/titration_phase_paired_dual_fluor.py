"""Triple-channel titration: Phase + TWO fluorescent markers per (bin) pair.

For every channel-disjoint pair (A, B) of live-cell fluor markers, this wrapper:

  1. Subsamples Phase, marker A, and marker B each at K cells/guide (capped at
     ``min(median_cpg(A), median_cpg(B))`` — past that point we'd be padding
     with Phase guides only with no fluor uplift on the smaller marker).
  2. Aggregates each to guide-level and inner-joins on ``sgRNA``.
  3. NTC-normalizes and scores the four mAP metrics on
     ``[phase_features || A_features || B_features]``.

After every (pair, bin) cell is scored, we pick — per bin and per metric — the
pair that maximises mAP. The constraint that A and B come from different
channels (GFP / mCherry / Cy5 / farred, parsed from ``ops_channel_maps.yaml``)
mirrors the experimental reality: two fluor channels can be imaged
simultaneously, but two markers on the *same* channel cannot.

Reuses the per-marker subsample + scoring helpers from ``titration`` and
the median-cpg cache from the single-fluor sibling. Same 5 plot types
(winners / rank-bump / delta-heatmap / top-N / win-share) but pair-keyed.
"""
from __future__ import annotations

import argparse
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import yaml

from ops_model.features.anndata_utils import (
    hconcat_by_perturbation,
    normalize_guide_adata,
)
from ops_model.post_process.combination.titration.titration import (
    _score_all_metrics,
    _subsample_per_guide_and_aggregate,
)
from ops_model.post_process.combination.titration.titration_phase_paired_fluor import (
    DEFAULT_PER_SIGNAL_DIR,
    DEFAULT_PHASE_TITRATION_CSV,
    _discover_fluor_markers,
    _load_or_build_marker_median_cpg,
    _load_phase_schedule,
)

logger = logging.getLogger(__name__)


DEFAULT_OUTPUT_DIR = Path(
    "/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3/"
    "cell_dino/zscore_per_exp/paper_v1/phase_only/fixed_80%/cosine/"
    "titration_phase_plus_dual_fluor"
)
DEFAULT_CHANNEL_MAP_YAML = Path(
    "/hpc/mydata/gav.sturm/ops_mono/ops_process/ops_analysis/configs/ops_channel_maps.yaml"
)


# ---------------------------------------------------------------------------
# Marker → channel(s) parsing from ops_channel_maps.yaml
# ---------------------------------------------------------------------------

_EXCLUDE_LABEL_TOKENS = ("no_label", "bleedthrough", "bleedthough", "autofluorescence")


def _slugify_label(label: str) -> str:
    """Convert a yaml label like "ER/Golgi, COPE" → "ER-Golgi_COPE" matching the
    per_signal h5ad filename convention.
    """
    s = label.strip()
    s = re.sub(r"\s*/\s*", "-", s)
    s = re.sub(r"[, ]+", "_", s)
    return s.rstrip("_")


def _build_marker_channel_map(
    channel_map_yaml: Path, available: List[str]
) -> Dict[str, Set[str]]:
    """For each marker in ``available``, the set of physical channels (GFP /
    mCherry / Cy5 / farred / ...) it has been imaged on across all
    experiments in ``channel_map_yaml``.
    """
    data = yaml.safe_load(channel_map_yaml.read_text())
    out: Dict[str, Set[str]] = defaultdict(set)
    avail = set(available)
    for entries in data.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict) or "channel_name" not in entry:
                continue
            ch = entry["channel_name"]
            lab = entry.get("label", "")
            if not lab or ch == "BF" or ch.startswith("4i_") or ch.startswith("CP"):
                continue
            slug = _slugify_label(lab)
            if any(tok in slug for tok in _EXCLUDE_LABEL_TOKENS):
                continue
            if slug in avail:
                out[slug].add(ch)
    return dict(out)


def _generate_channel_disjoint_pairs(
    marker_channels: Dict[str, Set[str]],
) -> List[Tuple[str, str]]:
    mlist = sorted(marker_channels)
    pairs = []
    for i, a in enumerate(mlist):
        for b in mlist[i + 1:]:
            if marker_channels[a].isdisjoint(marker_channels[b]):
                pairs.append((a, b))
    return pairs


def _generate_all_pairs(markers: List[str]) -> List[Tuple[str, str]]:
    """Unconstrained: every unordered pair of distinct markers."""
    mlist = sorted(markers)
    return [(a, b) for i, a in enumerate(mlist) for b in mlist[i + 1:]]


def _top_n_markers_from_single_fluor(
    single_fluor_dir: Path, n: int,
) -> List[str]:
    """Rank markers by mean (activity + distinct + ebi) mAP across bins from the
    single-fluor combined CSV; return top-N marker labels.
    """
    csv = single_fluor_dir / "phase_plus_fluor_combined_titration_capped.csv"
    if not csv.exists():
        csv = single_fluor_dir / "phase_plus_fluor_combined_titration.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"No single-fluor combined CSV in {single_fluor_dir} — "
            f"point --single-fluor-dir at the run that produced "
            f"phase_plus_fluor_combined_titration_capped.csv first."
        )
    df = pd.read_csv(csv)
    cols = [c for c in ("activity_map_mean", "distinctiveness_map_mean",
                        "ebi_map_mean") if c in df.columns]
    df["composite"] = df[cols].mean(axis=1)
    ranked = df.groupby("marker")["composite"].mean().sort_values(ascending=False)
    return list(ranked.head(n).index)


# ---------------------------------------------------------------------------
# Per-(pair, bin) worker (SLURM-picklable; top level)
# ---------------------------------------------------------------------------

def run_one_pair_titration(
    phase_h5ad_path: str,
    marker_a_h5ad_path: str,
    marker_a_label: str,
    marker_b_h5ad_path: str,
    marker_b_label: str,
    target: int,
    norm_method: str,
    output_dir: str,
    random_seed: int = 42,
) -> str:
    """Score Phase + marker_a + marker_b at ONE bin ``target``.

    Writes a per-(pair, target) shard CSV at
    ``<output_dir>/Phase+<A>+<B>/Phase+<A>+<B>_titration_t<target>.csv``.
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    pair_key = f"Phase+{marker_a_label}+{marker_b_label}"
    _logger = logging.getLogger(f"trip[{pair_key}@{target}]")
    t_start = time.time()

    pair_dir = Path(output_dir) / pair_key
    pair_dir.mkdir(parents=True, exist_ok=True)
    shard_csv = pair_dir / f"{pair_key}_titration_t{int(target)}.csv"
    if shard_csv.exists():
        return f"SKIP {pair_key} @ {target}: shard already exists"

    phase = ad.read_h5ad(phase_h5ad_path)
    a = ad.read_h5ad(marker_a_h5ad_path)
    b = ad.read_h5ad(marker_b_h5ad_path)
    _logger.info(
        "Loaded Phase=%d, %s=%d, %s=%d",
        phase.n_obs, marker_a_label, a.n_obs, marker_b_label, b.n_obs,
    )

    phase.var_names = pd.Index([f"phase__{v}" for v in phase.var_names])
    a.var_names = pd.Index([f"{marker_a_label}__{v}" for v in a.var_names])
    b.var_names = pd.Index([f"{marker_b_label}__{v}" for v in b.var_names])

    K = int(target)
    rng = np.random.RandomState(random_seed + K)
    phase_g = _subsample_per_guide_and_aggregate(phase, K, rng)
    a_g = _subsample_per_guide_and_aggregate(a, K, rng)
    b_g = _subsample_per_guide_and_aggregate(b, K, rng)
    merged = hconcat_by_perturbation([phase_g, a_g, b_g], level="guide")
    if merged.n_obs == 0:
        return f"FAIL {pair_key} @ {target}: no shared guides"
    merged = normalize_guide_adata(merged, norm_method)
    scores = _score_all_metrics(merged, _logger)
    scores.update({
        "cells_per_guide": K,
        "marker_a": marker_a_label,
        "marker_b": marker_b_label,
        "pair": f"{marker_a_label}+{marker_b_label}",
        "n_guides_phase": phase_g.n_obs,
        "n_guides_a": a_g.n_obs,
        "n_guides_b": b_g.n_obs,
        "n_guides_paired": merged.n_obs,
        "n_features_total": int(merged.n_vars),
        "signal": pair_key,
    })
    pd.DataFrame([scores]).to_csv(shard_csv, index=False)
    return (
        f"OK {pair_key} @ K={K}: guides={merged.n_obs} "
        f"act={scores.get('activity_map_mean', float('nan')):.3f} "
        f"ebi={scores.get('ebi_map_mean', float('nan')):.3f} "
        f"({int(time.time() - t_start)}s)"
    )


def _merge_shards(output_dir: Path, _logger=logger) -> None:
    for pair_dir in sorted(output_dir.iterdir()):
        if not pair_dir.is_dir() or not pair_dir.name.startswith("Phase+"):
            continue
        shards = sorted(pair_dir.glob(f"{pair_dir.name}_titration_t*.csv"))
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
        merged.to_csv(pair_dir / f"{pair_dir.name}_titration.csv", index=False)
        _logger.info("  merged %d shards → %s", len(dfs),
                     f"{pair_dir.name}/{pair_dir.name}_titration.csv")


def aggregate_shards(output_dir: Path) -> pd.DataFrame:
    rows = []
    for pair_dir in sorted(output_dir.iterdir()):
        if not pair_dir.is_dir() or not pair_dir.name.startswith("Phase+"):
            continue
        canonical = pair_dir / f"{pair_dir.name}_titration.csv"
        if canonical.exists():
            df = pd.read_csv(canonical)
        else:
            shards = sorted(pair_dir.glob(f"{pair_dir.name}_titration_t*.csv"))
            if not shards:
                continue
            df = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(output_dir / "phase_plus_dual_fluor_combined_titration.csv", index=False)
    return out


def compute_winners(
    combined: pd.DataFrame, phase_df: pd.DataFrame, min_pairs: int = 5,
) -> pd.DataFrame:
    """For each (bin, metric), pick the pair that maximises mAP_mean. Skips
    any (K, metric) where fewer than ``min_pairs`` pairs have a finite score —
    too few candidates to call one a winner.
    """
    METRICS = ["activity", "distinctiveness", "chad", "ebi"]
    rows = []
    phase_by_K = phase_df.set_index("cells_per_guide")
    for K, sub in combined.groupby("cells_per_guide"):
        for m in METRICS:
            col = f"{m}_map_mean"
            if col not in sub.columns:
                continue
            ok = sub.dropna(subset=[col])
            if len(ok) < min_pairs:
                continue
            top = ok.loc[ok[col].idxmax()]
            phase_val = float(phase_by_K[col].get(int(K), float("nan"))) \
                if col in phase_by_K.columns else float("nan")
            rows.append({
                "cells_per_guide": int(K),
                "metric": m,
                "winning_pair": top["pair"],
                "winning_marker_a": top["marker_a"],
                "winning_marker_b": top["marker_b"],
                "paired_map_mean": float(top[col]),
                "phase_only_map_mean": phase_val,
                "delta_vs_phase": float(top[col]) - phase_val,
            })
    return pd.DataFrame(rows).sort_values(["metric", "cells_per_guide"])


# ---------------------------------------------------------------------------
# Plotting — delegated to the shared ``titration_paired_plots`` module so
# the single-fluor sibling and this script share the same 5 plot functions.
# ---------------------------------------------------------------------------

from ops_model.post_process.combination.titration import titration_paired_plots as _plots


def plot_winners(winners: pd.DataFrame, phase_df: pd.DataFrame, out_dir: Path) -> None:
    _plots.plot_winners(
        winners, phase_df, out_dir,
        winning_col="winning_pair",
        label_fn=_plots.pretty_pair,
        out_prefix="phase_plus_dual_fluor",
        line_legend="Phase + best fluor pair",
        suptitle="Phase + best fluor-pair — per-bin winner annotated",
        fig_w=7,    # roughly square per-panel
        fig_h=6.5,
        wspace=0.15,
    )


def plot_rank_bump(combined: pd.DataFrame, out_dir: Path, top_n: int = 5) -> None:
    _plots.plot_rank_bump(
        combined, out_dir,
        entity_col="pair", label_fn=_plots.pretty_pair,
        out_prefix="phase_plus_dual_fluor",
        ylabel="rank (1 = best pair)",
        suptitle=f"Pair rank-ordering (top-{top_n} highlighted)",
        top_n=top_n,
    )


def plot_delta_heatmap(
    combined: pd.DataFrame, phase_df: pd.DataFrame, out_dir: Path,
    top_n_rows: int = 60,
) -> None:
    _plots.plot_delta_heatmap(
        combined, phase_df, out_dir,
        entity_col="pair", label_fn=_plots.pretty_pair,
        out_prefix="phase_plus_dual_fluor",
        suptitle=f"Top-{top_n_rows} pairs by per-bin gain over Phase baseline",
        top_n_rows=top_n_rows,
        label_truncate=45,
        figsize_h=13,
    )


def plot_topn_curves(combined: pd.DataFrame, phase_df: pd.DataFrame, out_dir: Path, top_n: int = 5) -> None:
    _plots.plot_topn_curves(
        combined, phase_df, out_dir,
        entity_col="pair", label_fn=_plots.pretty_pair,
        out_prefix="phase_plus_dual_fluor",
        ribbon_label="paired 5–95% (all pairs)",
        suptitle=f"Phase + top-{top_n} fluor pairs with 5-95% paired ribbon",
        top_n=top_n,
    )


def plot_win_share(combined: pd.DataFrame, out_dir: Path, top_k_rank: int = 3) -> None:
    _plots.plot_win_share(
        combined, out_dir,
        entity_col="pair", label_fn=_plots.pretty_pair,
        out_prefix="phase_plus_dual_fluor",
        suptitle=f"Pair win-share (top-{top_k_rank} per bin × metric)",
        drop_zero_wins=True,   # 230 pairs — only show pairs that won at least once
        top_k_rank=top_k_rank,
    )




# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-signal-dir", type=Path, default=DEFAULT_PER_SIGNAL_DIR)
    p.add_argument("--phase-titration-csv", type=Path,
                   default=DEFAULT_PHASE_TITRATION_CSV)
    p.add_argument("--channel-map-yaml", type=Path,
                   default=DEFAULT_CHANNEL_MAP_YAML,
                   help="YAML with per-experiment channel_name → label entries; "
                        "drives the channel-disjoint constraint.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--norm-method", default="ntc", choices=("ntc", "global"))
    p.add_argument("--pairs", nargs="*", default=None,
                   help='Optional subset of pairs as "A+B" strings.')
    p.add_argument("--bins", nargs="*", type=int, default=None)
    p.add_argument("--no-channel-constraint", action="store_true",
                   help="Drop the channel-disjoint filter — pair any two markers "
                        "regardless of whether they share a physical channel.")
    p.add_argument("--top-n-markers", type=int, default=None,
                   help="Filter to the top-N markers from the single-fluor run "
                        "(ranked by mean of activity+distinctiveness+ebi mAP across bins).")
    p.add_argument("--single-fluor-dir", type=Path,
                   default=Path("/hpc/projects/icd.fast.ops/organelle_attribution/"
                                "pca_optimized_v0.3/cell_dino/zscore_per_exp/paper_v1/"
                                "phase_only/fixed_80%/cosine/titration_phase_plus_fluor"),
                   help="Single-fluor output dir whose combined CSV is used for --top-n-markers ranking.")
    p.add_argument("--slurm", action="store_true")
    p.add_argument("--slurm-memory", default="64GB")
    p.add_argument("--slurm-time", type=int, default=90)
    p.add_argument("--slurm-cpus", type=int, default=4)
    p.add_argument("--slurm-partition", default="cpu,gpu")
    p.add_argument("--replot", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the (pair, bin) job list without submitting.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)

    per_signal_dir = args.per_signal_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_cells_h5ad = per_signal_dir / "Phase_cells.h5ad"
    if not phase_cells_h5ad.exists():
        raise FileNotFoundError(phase_cells_h5ad)
    if not args.phase_titration_csv.exists():
        raise FileNotFoundError(args.phase_titration_csv)

    phase_df = pd.read_csv(args.phase_titration_csv).sort_values("cells_per_guide")
    cell_targets = _load_phase_schedule(args.phase_titration_csv)
    if args.bins:
        wanted = set(args.bins)
        cell_targets = [b for b in cell_targets if b in wanted]
    logger.info("bin schedule: %s", cell_targets)

    # Discover markers + their physical channels + median cells/guide.
    available = _discover_fluor_markers(per_signal_dir)
    marker_paths = dict(available)
    marker_channels = _build_marker_channel_map(args.channel_map_yaml,
                                                list(marker_paths))
    median_cpg = _load_or_build_marker_median_cpg(
        per_signal_dir, output_dir / "marker_median_cpg.csv",
    )

    # Optional top-N filter from single-fluor ranking.
    eligible = list(marker_paths)
    if args.top_n_markers:
        top = _top_n_markers_from_single_fluor(args.single_fluor_dir,
                                               args.top_n_markers)
        eligible = [m for m in top if m in marker_paths]
        logger.info("top-%d markers from single-fluor: %s",
                    args.top_n_markers, eligible)

    if args.no_channel_constraint:
        pairs_all = _generate_all_pairs(eligible)
        logger.info("unconstrained pairs (no channel filter): %d", len(pairs_all))
    else:
        subset_channels = {m: marker_channels.get(m, set()) for m in eligible}
        pairs_all = _generate_channel_disjoint_pairs(subset_channels)
        logger.info("channel-disjoint pairs: %d", len(pairs_all))

    if args.pairs:
        wanted_pairs = set(args.pairs)
        pairs_all = [(a, b) for a, b in pairs_all
                     if f"{a}+{b}" in wanted_pairs or f"{b}+{a}" in wanted_pairs]

    # Build (pair, bin) job list with median-cap.
    jobs_spec = []
    for a, b in pairs_all:
        cap = min(median_cpg.get(a, 0), median_cpg.get(b, 0))
        bins_for_pair = [t for t in cell_targets if t <= cap]
        for t in bins_for_pair:
            jobs_spec.append((a, b, t))
    logger.info(
        "total (pair × bin) jobs: %d  (mean %.1f bins/pair)",
        len(jobs_spec), len(jobs_spec) / max(len(pairs_all), 1),
    )

    if args.dry_run:
        df = pd.DataFrame(jobs_spec, columns=["marker_a", "marker_b", "K"])
        df.to_csv(output_dir / "dual_fluor_job_plan.csv", index=False)
        logger.info("wrote %s (dry run, no submission)",
                    output_dir / "dual_fluor_job_plan.csv")
        return 0

    if not args.replot:
        if args.slurm:
            from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
            jobs = [{
                "name": f"trip+{a}+{b}@{t}",
                "func": run_one_pair_titration,
                "kwargs": {
                    "phase_h5ad_path": str(phase_cells_h5ad),
                    "marker_a_h5ad_path": str(marker_paths[a]),
                    "marker_a_label": a,
                    "marker_b_h5ad_path": str(marker_paths[b]),
                    "marker_b_label": b,
                    "target": t,
                    "norm_method": args.norm_method,
                    "output_dir": str(output_dir),
                },
                "metadata": {"marker_a": a, "marker_b": b, "target": t},
            } for a, b, t in jobs_spec]
            slurm_params = {
                "timeout_min": args.slurm_time,
                "mem": args.slurm_memory,
                "cpus_per_task": args.slurm_cpus,
                "slurm_partition": args.slurm_partition,
            }
            submit_parallel_jobs(
                jobs_to_submit=jobs,
                experiment="phase_plus_dual_fluor_titration",
                slurm_params=slurm_params,
                log_dir="slurm_step_logs/phase_plus_dual_fluor_titration",
                manifest_prefix="phase_plus_dual_fluor_titration",
                wait_for_completion=True,
            )
        else:
            for a, b, t in jobs_spec:
                msg = run_one_pair_titration(
                    phase_h5ad_path=str(phase_cells_h5ad),
                    marker_a_h5ad_path=str(marker_paths[a]),
                    marker_a_label=a,
                    marker_b_h5ad_path=str(marker_paths[b]),
                    marker_b_label=b,
                    target=t,
                    norm_method=args.norm_method,
                    output_dir=str(output_dir),
                )
                logger.info(msg)
        _merge_shards(output_dir, logger)

    combined = aggregate_shards(output_dir)
    if combined.empty:
        logger.warning("No shards found in %s — nothing to plot", output_dir)
        return 1
    winners = compute_winners(combined, phase_df)
    winners.to_csv(output_dir / "phase_plus_dual_fluor_winners.csv", index=False)
    logger.info("wrote %s/phase_plus_dual_fluor_winners.csv (%d rows)",
                output_dir, len(winners))
    plot_winners(winners, phase_df, output_dir)
    plot_rank_bump(combined, output_dir, top_n=5)
    plot_delta_heatmap(combined, phase_df, output_dir)
    plot_topn_curves(combined, phase_df, output_dir, top_n=5)
    plot_win_share(combined, output_dir, top_k_rank=3)

    # Decomposition plot: per top winning pair, show Phase / +A / +B / +A+B.
    # Needs the single-fluor combined CSV (capped, with the median-cpg trim).
    single_csv = args.single_fluor_dir / "phase_plus_fluor_combined_titration_capped.csv"
    if not single_csv.exists():
        single_csv = args.single_fluor_dir / "phase_plus_fluor_combined_titration.csv"
    if single_csv.exists():
        single_combined = pd.read_csv(single_csv)
        _plots.plot_winner_decomposition(
            winners, combined, single_combined, phase_df, output_dir,
            out_prefix="phase_plus_dual_fluor",
            suptitle="Top winning pairs: decomposition into Phase / + A / + B / + A and B",
            top_n_pairs=4,
        )
    else:
        logger.warning("single-fluor combined CSV not found at %s — skipping "
                       "winner_decomposition plot", single_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
