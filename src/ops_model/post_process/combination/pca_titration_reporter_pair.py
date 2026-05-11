"""Phase + Fe (or any two reporters) combination titration comparison.

For each titration point (N cells per reporter):
  1. Downsample + aggregate reporter A to guide level
  2. Downsample + aggregate reporter B to guide level
  3. Normalize each independently (NTC or global)
  4. Score each reporter alone across all 4 mAP metrics
  5. Combine features (inner join on guide, horizontal concat), score combined

Output: 4 figures (one per mAP metric: activity, distinctiveness, CORUM, CHAD).
Each figure has 2 panels:
  - Left:  % ratio (significant) vs cell count
  - Right: mean mAP vs cell count
  - 3 lines per panel: reporter A alone | reporter B alone | A+B combined

Usage::

    python -m ops_model.post_process.combination.pca_titration_reporter_pair \\
        --reporter-a /path/to/pca_optimized_v2/dino/all/per_signal/Phase_cells.h5ad \\
        --reporter-b /path/to/pca_optimized_v2/dino/all/per_signal/FeRhoNox_cells.h5ad \\
        -o /hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v2/dino/all/titration/phase_fe_comparison

    # Phase vs Fe+ comparison (correct paths):
    python -m ops_model.post_process.combination.pca_titration_reporter_pair \\
        --reporter-a /hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v2/dino/all/per_signal/Phase_cells.h5ad \\
        --reporter-b "/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v2/dino/all/per_signal/Fe2+_FeRhoNox_live-cell_dye_cells.h5ad" \\
        --name-a "Phase" --name-b "Fe+" \\
        -o /hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v2/dino/all/titration/phase_fe_comparison \\
        --slurm --yes
"""

import argparse
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd

from ops_model.features.anndata_utils import (
    _guide_col,
    aggregate_to_level,
    normalize_guide_adata,
)

# Reuse utilities from the main titration script
from ops_model.post_process.combination.pca_titration import (
    DOWNSAMPLE_RATIO,
    MIN_CELLS,
    METRICS,
    NULL_SIZE,
    _init_logger,
    _subsample_and_aggregate,
    _score_all_metrics,
    _apply_x_scale,
    _format_cell_count,
    _prepare_for_copairs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Combination helper
# ---------------------------------------------------------------------------


def _combine_reporters(
    adata_a: ad.AnnData, adata_b: ad.AnnData
) -> Optional[ad.AnnData]:
    """Inner join two guide-level adatas on sgRNA key and concatenate features.

    Both adatas must already be normalized. Rows are aligned by the per-construct
    identifier (resolved via ``uns["guide_col"]`` — ``"sgRNA"`` for CRISPR,
    ``"minibinder_perturbation"`` for minibinder), giving an unambiguous 1:1
    mapping even when random subsampling causes slightly different guide sets to
    survive in each reporter. Falls back to ``perturbation`` (gene name) if the
    guide column is not available.

    Returns None if fewer than 2 guides are shared.
    """
    import scipy.sparse as sp

    # Choose join key: per-construct identifier if available, else perturbation.
    guide_col_a = _guide_col(adata_a)
    guide_col_b = _guide_col(adata_b)
    key = (
        guide_col_a
        if guide_col_a == guide_col_b
        and guide_col_a in adata_a.obs.columns
        and guide_col_b in adata_b.obs.columns
        else "perturbation"
    )

    keys_a = set(adata_a.obs[key])
    keys_b = set(adata_b.obs[key])
    common = sorted(keys_a & keys_b)

    if len(common) < 2:
        logger.warning(
            f"    Only {len(common)} shared {key}s — skipping combined scoring"
        )
        return None

    logger.info(f"    Combining on {key}: {len(common)} shared guides")

    def _filter_sort(adata: ad.AnnData) -> ad.AnnData:
        mask = adata.obs[key].isin(common)
        sub = adata[mask].copy()
        order = sub.obs[key].map({k: i for i, k in enumerate(common)})
        return sub[order.argsort()].copy()

    sub_a = _filter_sort(adata_a)
    sub_b = _filter_sort(adata_b)

    def _to_dense(x):
        return x.toarray() if sp.issparse(x) else np.asarray(x)

    X_combined = np.hstack([_to_dense(sub_a.X), _to_dense(sub_b.X)]).astype(np.float32)

    keep_cols = [
        c
        for c in ["perturbation", _guide_col(sub_a), "n_cells"]
        if c in sub_a.obs.columns
    ]
    combined = ad.AnnData(X=X_combined, obs=sub_a.obs[keep_cols].copy())
    combined.uns["guide_col"] = _guide_col(sub_a)
    return combined


# ---------------------------------------------------------------------------
# Core titration function
# ---------------------------------------------------------------------------


def titrate_pair(
    cells_a_path: Path,
    cells_b_path: Path,
    name_a: str,
    name_b: str,
    output_dir: Path,
    norm_method: str = "ntc",
    random_seed: int = 42,
) -> pd.DataFrame:
    """Run titration for reporter A, reporter B, and their combination.

    At each titration point N, both reporters are independently downsampled
    to N cells, aggregated to guide level, normalized, then scored alone and
    combined. The titration schedule is determined by the smaller reporter's
    cell count so both reporters always have matched N.

    Returns a long-form DataFrame with columns:
        n_cells, reporter, {metric}_ratio, {metric}_map_mean  (× 4 metrics)
    """
    _logger = _init_logger()
    rng = np.random.RandomState(random_seed)

    _logger.info(f"Loading {name_a} from {cells_a_path}")
    adata_a = ad.read_h5ad(cells_a_path)
    _logger.info(f"  {name_a}: {adata_a.n_obs:,} cells, {adata_a.n_vars} PCs")

    _logger.info(f"Loading {name_b} from {cells_b_path}")
    adata_b = ad.read_h5ad(cells_b_path)
    _logger.info(f"  {name_b}: {adata_b.n_obs:,} cells, {adata_b.n_vars} PCs")

    # Drop signal column if present — can interfere with aggregation
    for adata in (adata_a, adata_b):
        if "signal" in adata.obs.columns:
            adata.obs = adata.obs.drop(columns=["signal"])

    # Titration schedule: based on the smaller reporter so N is matched
    min_total = min(adata_a.n_obs, adata_b.n_obs)
    cell_targets: List[int] = []
    n = min_total
    while n >= MIN_CELLS:
        cell_targets.append(int(n))
        n = int(n * DOWNSAMPLE_RATIO)
    if not cell_targets:
        cell_targets = [min_total]
    _logger.info(f"Titration points ({len(cell_targets)}): {cell_targets}")

    rows = []
    for target in cell_targets:
        _logger.info(f"\n--- {target:,} cells ---")
        t = time.time()

        # Downsample + aggregate each reporter
        g_a_raw = _subsample_and_aggregate(adata_a, target, rng)
        g_b_raw = _subsample_and_aggregate(adata_b, target, rng)

        # Normalize each independently
        g_a = normalize_guide_adata(g_a_raw, norm_method)
        g_b = normalize_guide_adata(g_b_raw, norm_method)

        # Score reporter A alone
        scores_a = _score_all_metrics(_prepare_for_copairs(g_a.copy()), _logger)
        _logger.info(
            f"  {name_a}: act={scores_a['activity_ratio']:.1%} "
            f"dist={scores_a['distinctiveness_ratio']:.1%} "
            f"corum={scores_a['corum_ratio']:.1%} "
            f"chad={scores_a['chad_ratio']:.1%}"
        )

        # Score reporter B alone
        scores_b = _score_all_metrics(_prepare_for_copairs(g_b.copy()), _logger)
        _logger.info(
            f"  {name_b}: act={scores_b['activity_ratio']:.1%} "
            f"dist={scores_b['distinctiveness_ratio']:.1%} "
            f"corum={scores_b['corum_ratio']:.1%} "
            f"chad={scores_b['chad_ratio']:.1%}"
        )

        # Combine and score
        g_combined = _combine_reporters(g_a, g_b)
        if g_combined is not None:
            scores_combined = _score_all_metrics(
                _prepare_for_copairs(g_combined.copy()), _logger
            )
            _logger.info(
                f"  Combined ({len(set(g_combined.obs['perturbation']))} guides): "
                f"act={scores_combined['activity_ratio']:.1%} "
                f"dist={scores_combined['distinctiveness_ratio']:.1%} "
                f"corum={scores_combined['corum_ratio']:.1%} "
                f"chad={scores_combined['chad_ratio']:.1%}"
            )
        else:
            scores_combined = {k: math.nan for k in scores_a}

        _logger.info(f"  Step time: {time.time()-t:.0f}s")

        for reporter_label, scores in [
            (name_a, scores_a),
            (name_b, scores_b),
            (f"{name_a}+{name_b}", scores_combined),
        ]:
            rows.append({"n_cells": target, "reporter": reporter_label, **scores})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Visual style for the three reporters
_REPORTER_STYLES = [
    # (suffix to match in reporter label, color, linestyle, marker, lw)
    (0, "#2166ac", "-", "o", 2.5),  # reporter A alone
    (1, "#d6604d", "--", "s", 2.5),  # reporter B alone
    (2, "#4dac26", "-", "^", 3.5),  # combined
]

_METRIC_TITLES = {
    "activity": "Activity",
    "distinctiveness": "Distinctiveness",
    "corum": "CORUM Consistency",
    "chad": "CHAD Consistency",
}

_RATIO_YLABELS = {
    "activity": "% Active",
    "distinctiveness": "% Distinctive",
    "corum": "% CORUM Consistent",
    "chad": "% CHAD Consistent",
}


def _plot_pair_comparison(
    df: pd.DataFrame,
    name_a: str,
    name_b: str,
    output_dir: Path,
) -> None:
    """Generate 4 figures (one per mAP metric) × 3 scale variants.

    Each figure: 2 panels (ratio | mean_map), 3 lines
    (A alone, B alone, A+B combined).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reporter_labels = [name_a, name_b, f"{name_a}+{name_b}"]
    colors = ["#2166ac", "#d6604d", "#4dac26"]
    linestyles = ["-", "--", "-"]
    markers = ["o", "s", "^"]
    linewidths = [2.5, 2.5, 3.5]

    x_all = df["n_cells"].unique()

    for metric in METRICS:
        ratio_col = f"{metric}_ratio"
        map_col = f"{metric}_map_mean"
        metric_title = _METRIC_TITLES[metric]
        ratio_ylabel = _RATIO_YLABELS[metric]

        for scale in ("log10", "linear"):
            fig, (ax_ratio, ax_map) = plt.subplots(1, 2, figsize=(20, 8))

            for rep_label, color, ls, marker, lw in zip(
                reporter_labels, colors, linestyles, markers, linewidths
            ):
                sub = df[df["reporter"] == rep_label].sort_values("n_cells")
                x = sub["n_cells"].values

                if ratio_col in sub.columns:
                    ax_ratio.plot(
                        x,
                        sub[ratio_col].values * 100,
                        color=color,
                        linestyle=ls,
                        marker=marker,
                        linewidth=lw,
                        markersize=8,
                        label=rep_label,
                    )

                if map_col in sub.columns:
                    ax_map.plot(
                        x,
                        sub[map_col].values,
                        color=color,
                        linestyle=ls,
                        marker=marker,
                        linewidth=lw,
                        markersize=8,
                        label=rep_label,
                    )

            for ax, ylabel, title_suffix in [
                (ax_ratio, f"{ratio_ylabel} (%)", "% Significant"),
                (ax_map, "Mean mAP", "Mean mAP"),
            ]:
                ax.set_xlabel(f"Cells per Reporter ({scale})", fontsize=18)
                ax.set_ylabel(ylabel, fontsize=18)
                ax.set_title(f"{metric_title} — {title_suffix}", fontsize=20)
                ax.tick_params(labelsize=14)
                ax.legend(fontsize=15)
                _apply_x_scale(ax, x_all, scale, tick_fontsize=14)

            fig.suptitle(
                f"{metric_title}: {name_a} vs {name_b} vs Combined  [{scale}]",
                fontsize=22,
                fontweight="bold",
            )
            fig.tight_layout()

            stem = output_dir / f"{metric}_{scale}"
            fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
            fig.savefig(f"{stem}.svg", bbox_inches="tight")
            plt.close(fig)
            logger.info(f"  Saved {stem}.{{png,svg}}")


# ---------------------------------------------------------------------------
# SLURM job function (top-level so submitit can pickle it)
# ---------------------------------------------------------------------------


def run_pair_titration_job(
    reporter_a: str,
    reporter_b: str,
    name_a: str,
    name_b: str,
    output_dir: str,
    norm_method: str = "ntc",
    seed: int = 42,
) -> str:
    """Run the full pair titration + plotting as a single SLURM job."""
    import traceback

    _init_logger()
    try:
        path_a = Path(reporter_a)
        path_b = Path(reporter_b)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        csv_path = out / "titration_pair.csv"

        df = titrate_pair(path_a, path_b, name_a, name_b, out, norm_method, seed)
        df.to_csv(csv_path, index=False)
        _plot_pair_comparison(df, name_a, name_b, out)
        return f"OK: {len(df)} rows, plots saved to {out}"
    except Exception as e:
        traceback.print_exc()
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Titration comparison: reporter A alone vs B alone vs A+B combined.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--reporter-a", default=None, help="Path to reporter A *_cells.h5ad"
    )
    parser.add_argument(
        "--reporter-b", default=None, help="Path to reporter B *_cells.h5ad"
    )
    parser.add_argument(
        "--name-a",
        default=None,
        help="Display name for reporter A (default: stem of filename)",
    )
    parser.add_argument(
        "--name-b",
        default=None,
        help="Display name for reporter B (default: stem of filename)",
    )
    parser.add_argument(
        "--all-pairs-with",
        default=None,
        metavar="REFERENCE_H5AD",
        help="Submit one SLURM job per reporter paired with this reference "
        "(e.g. Phase_cells.h5ad). Discovers all *_cells.h5ad in --per-signal-dir.",
    )
    parser.add_argument(
        "--per-signal-dir",
        default=None,
        help="Directory containing *_cells.h5ad files for --all-pairs-with mode.",
    )
    parser.add_argument(
        "-o", "--output-dir", required=True, help="Output directory for CSVs and plots"
    )
    parser.add_argument(
        "--norm-method",
        default="ntc",
        choices=["ntc", "global"],
        help="Normalization method (default: ntc)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for downsampling (default: 42)",
    )
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Regenerate plots from existing titration_pair.csv without recomputing",
    )
    parser.add_argument(
        "--slurm", action="store_true", help="Submit as a single SLURM job"
    )
    parser.add_argument("--slurm-memory", type=str, default="200GB")
    parser.add_argument(
        "--slurm-time",
        type=int,
        default=60,
        help="SLURM time limit in minutes (default: 60)",
    )
    parser.add_argument("--slurm-cpus", type=int, default=8)
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation prompt"
    )
    args = parser.parse_args()

    _init_logger()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # --all-pairs-with: submit one job per reporter paired with reference
    # ------------------------------------------------------------------
    if args.all_pairs_with:
        from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs
        from ops_utils.data.feature_discovery import sanitize_signal_filename

        ref_path = Path(args.all_pairs_with)
        ref_name = args.name_a or ref_path.stem.replace("_cells", "")

        per_signal_dir = (
            Path(args.per_signal_dir) if args.per_signal_dir else ref_path.parent
        )
        other_files = sorted(
            f
            for f in per_signal_dir.glob("*_cells.h5ad")
            if "_sub" not in f.name and f.resolve() != ref_path.resolve()
        )

        slurm_params = {
            "timeout_min": args.slurm_time,
            "mem": args.slurm_memory,
            "cpus_per_task": args.slurm_cpus,
            "slurm_partition": "cpu,gpu",
        }

        jobs = []
        for other in other_files:
            other_name = other.stem.replace("_cells", "")
            safe = sanitize_signal_filename(other_name)[:35]
            pair_out = output_dir / safe
            jobs.append(
                {
                    "name": f"titr_{ref_name[:10]}_{safe[:20]}",
                    "func": run_pair_titration_job,
                    "kwargs": {
                        "reporter_a": str(ref_path),
                        "reporter_b": str(other),
                        "name_a": ref_name,
                        "name_b": other_name,
                        "output_dir": str(pair_out),
                        "norm_method": args.norm_method,
                        "seed": args.seed,
                    },
                }
            )

        if not args.yes:
            print(f"\nPhase-pairs SLURM submission:")
            print(f"  Reference:   {ref_name}  ({ref_path.name})")
            print(f"  Partners:    {len(jobs)} reporters from {per_signal_dir}")
            print(f"  Output root: {output_dir}")
            print(
                f"  Memory: {args.slurm_memory}  |  Time: {args.slurm_time} min  |  CPUs: {args.slurm_cpus}"
            )
            if input("\nSubmit? [y/N] ").strip().lower() != "y":
                print("Cancelled.")
                return

        result = submit_parallel_jobs(
            jobs_to_submit=jobs,
            experiment="pca_titration_phase_pairs",
            slurm_params=slurm_params,
            log_dir="pca_optimization",
            manifest_prefix="pca_titration_phase_pairs",
            wait_for_completion=True,
        )
        if result.get("failed"):
            print(f"\n{len(result['failed'])} jobs failed")
        else:
            print(f"\nAll {len(jobs)} jobs complete → {output_dir}")
        return

    # ------------------------------------------------------------------
    # Single-pair mode
    # ------------------------------------------------------------------
    if not args.reporter_a or not args.reporter_b:
        print("ERROR: provide --reporter-a and --reporter-b, or use --all-pairs-with")
        return

    path_a = Path(args.reporter_a)
    path_b = Path(args.reporter_b)
    name_a = args.name_a or path_a.stem.replace("_cells", "")
    name_b = args.name_b or path_b.stem.replace("_cells", "")

    csv_path = output_dir / "titration_pair.csv"

    if args.replot:
        if not csv_path.exists():
            print(f"ERROR: {csv_path} not found — run without --replot first")
            return
        print(f"--replot: loading {csv_path}")
        df = pd.read_csv(csv_path)
        _plot_pair_comparison(df, name_a, name_b, output_dir)
        print("Done.")
        return

    if args.slurm:
        from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

        slurm_params = {
            "timeout_min": args.slurm_time,
            "mem": args.slurm_memory,
            "cpus_per_task": args.slurm_cpus,
            "slurm_partition": "cpu,gpu",
        }
        jobs = [
            {
                "name": f"titr_pair_{name_a}_{name_b}",
                "func": run_pair_titration_job,
                "kwargs": {
                    "reporter_a": str(path_a),
                    "reporter_b": str(path_b),
                    "name_a": name_a,
                    "name_b": name_b,
                    "output_dir": str(output_dir),
                    "norm_method": args.norm_method,
                    "seed": args.seed,
                },
            }
        ]

        if not args.yes:
            print(f"\nPair Titration SLURM Job:")
            print(f"  Reporter A: {name_a}  ({path_a.name})")
            print(f"  Reporter B: {name_b}  ({path_b.name})")
            print(f"  Output:     {output_dir}")
            print(
                f"  Memory:     {args.slurm_memory}  |  Time: {args.slurm_time} min  |  CPUs: {args.slurm_cpus}"
            )
            if input("\nSubmit? [y/N] ").strip().lower() != "y":
                print("Cancelled.")
                return

        result = submit_parallel_jobs(
            jobs_to_submit=jobs,
            experiment="pca_titration_pair",
            slurm_params=slurm_params,
            log_dir="pca_optimization",
            manifest_prefix="pca_titration_pair",
            wait_for_completion=True,
        )
        if result.get("success"):
            print(f"\nJob submitted: {result.get('base_job_id')}")
        else:
            print("\nJob submission failed!")
        return

    # Local mode
    df = titrate_pair(
        cells_a_path=path_a,
        cells_b_path=path_b,
        name_a=name_a,
        name_b=name_b,
        output_dir=output_dir,
        norm_method=args.norm_method,
        random_seed=args.seed,
    )
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    print(f"Generating plots → {output_dir}")
    _plot_pair_comparison(df, name_a, name_b, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
