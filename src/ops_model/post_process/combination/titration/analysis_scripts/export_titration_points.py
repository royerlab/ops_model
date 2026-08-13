#!/usr/bin/env python
"""Export the plotted points of a titration curve as a tidy CSV (+ re-render).

Takes one ``<signal>_titration.csv`` (the aggregate written by titration.py) and
writes a CSV holding exactly the points drawn in the ``*_perguide_<scale>`` plot
for a chosen subset of metrics, plus a re-rendered figure restricted to those
metrics (same styling as the original, via ``titration._plot_titration``).

Usage::

    python -m ops_model.post_process.combination.titration.analysis_scripts.export_titration_points \\
        --csv <.../titration_guide_median/Phase/Phase_titration.csv> \\
        --metrics distinctiveness ebi
"""
import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42
import pandas as pd

from ops_model.post_process.combination.titration.titration import (
    TITRATION_MAP_LABELS,
    TITRATION_RATIO_LABELS,
    _plot_titration,
)

logger = logging.getLogger(__name__)

DEFAULT_CSV = (
    "/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3/cell_dino/"
    "zscore_per_exp/paper_v2/with_cp/with_4i/all_livecell/fixed_80%/cosine/"
    "titration_guide_median/Phase/Phase_titration.csv"
)

CONTEXT_COLS = [
    "cells_per_guide", "n_cells", "cells_per_perturbation",
    "n_guides", "n_perturbations", "n_bootstraps",
]


def export_points(csv: Path, metrics, x_col: str, outdir: Path) -> Path:
    """Write the plotted (x, y, sem) points for ``metrics`` to a tidy CSV."""
    df = pd.read_csv(csv).sort_values(x_col).reset_index(drop=True)
    out = df[[c for c in CONTEXT_COLS if c in df.columns]].copy()

    for metric in metrics:
        # Panel 1 values are plotted as percentages; panel 2 as raw mAP.
        out[f"{metric}_pct_significant"] = df[f"{metric}_ratio"] * 100
        out[f"{metric}_pct_significant_sem"] = df[f"{metric}_ratio_sem"] * 100
        out[f"{metric}_mean_map"] = df[f"{metric}_map_mean"]
        out[f"{metric}_mean_map_sem"] = df[f"{metric}_map_mean_sem"]

    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / f"{csv.stem}_{'_'.join(metrics)}_points.csv"
    out.to_csv(dest, index=False)
    logger.info(f"Wrote {len(out)} points x {len(metrics)} metrics -> {dest}")
    for metric in metrics:
        logger.info(
            f"  {TITRATION_RATIO_LABELS[metric]} / {TITRATION_MAP_LABELS[metric]}"
        )
    return dest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=DEFAULT_CSV, help="aggregate titration CSV")
    p.add_argument("--metrics", nargs="+", default=["distinctiveness", "ebi"])
    p.add_argument("--x-col", default="cells_per_guide")
    p.add_argument("-o", "--outdir", default=None,
                   help="output dir (default: <csv dir>/<metrics joined>)")
    p.add_argument("--no-plot", action="store_true", help="CSV only")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    csv = Path(args.csv)
    outdir = Path(args.outdir) if args.outdir else csv.parent / "_".join(args.metrics)

    export_points(csv, args.metrics, args.x_col, outdir)

    if not args.no_plot:
        df = pd.read_csv(csv).sort_values(args.x_col).reset_index(drop=True)
        signal = df["signal"].iloc[0]
        _plot_titration(df, signal, outdir, csv.stem.replace("_titration", ""),
                        plt, metrics=tuple(args.metrics))
        logger.info(f"Wrote figures -> {outdir}")


if __name__ == "__main__":
    main()
