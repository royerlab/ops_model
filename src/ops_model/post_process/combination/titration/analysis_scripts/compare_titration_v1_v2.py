#!/usr/bin/env python
"""Compare v1 vs v2 titration curves per marker for the 3 main metrics.

For each marker present in the paper_v1 and paper_v2 titration outputs, overlay
the v1 and v2 mean-mAP-vs-cells/guide curves for the three headline metrics
(Activity, Distinctiveness, EBI). One figure per marker (3 panels) plus a
combined grid overview. Markers present only in v2 (new to v2) are drawn v2-only.

Inspired by compare_map_scores.py but for titration *curves* rather than
scatter/slope of a single cell budget.

Usage::

    python -m ops_model.post_process.combination.titration.compare_titration_v1_v2

    # explicit dirs
    python -m ops_model.post_process.combination.titration.compare_titration_v1_v2 \\
        --v1-dir <.../paper_v1/.../titration_guide_median> \\
        --v2-dir <.../paper_v2/.../titration_guide_median> \\
        -o <output_dir>
"""
import argparse
import logging
import math
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ROOT = ("/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3/"
         "cell_dino/zscore_per_exp")
_TAIL = "with_cp/with_4i/all_livecell/fixed_80%/cosine/titration_guide_median"
DEFAULT_V1_DIR = f"{_ROOT}/paper_v1/{_TAIL}"
DEFAULT_V2_DIR = f"{_ROOT}/paper_v2/{_TAIL}"

# The 3 headline metrics (column stem, display label).
METRICS = [
    ("activity_map_mean", "Activity"),
    ("distinctiveness_map_mean", "Distinctiveness"),
    ("ebi_map_mean", "EBI"),
]
X_COL = "cells_per_guide"
V1_COLOR, V2_COLOR = "#7f7f7f", "#d62728"  # v1 gray, v2 red


def _reporter_dirs(base: Path) -> dict:
    """marker_dir_name -> per-reporter titration CSV path."""
    out = {}
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        csvs = list(d.glob("*_titration.csv"))
        if csvs:
            out[d.name] = csvs[0]
    return out


def _load(csv: Optional[Path]) -> Optional[pd.DataFrame]:
    if csv is None or not csv.exists():
        return None
    df = pd.read_csv(csv)
    return df.sort_values(X_COL) if X_COL in df.columns else None


def _draw_metric(ax, dfv1, dfv2, col, label):
    plotted = False
    for df, ver, color in ((dfv1, "v1", V1_COLOR), (dfv2, "v2", V2_COLOR)):
        if df is None or col not in df.columns:
            continue
        d = df.dropna(subset=[col])
        if d.empty:
            continue
        x, y = d[X_COL].values, d[col].values
        sem = d[f"{col}_sem"].values if f"{col}_sem" in d.columns else None
        if sem is not None and np.isfinite(sem).any():
            ax.errorbar(x, y, yerr=sem, marker="o", color=color, label=ver,
                        lw=2.5, ms=6, capsize=3, elinewidth=1.2)
        else:
            ax.plot(x, y, marker="o", color=color, label=ver, lw=2.5, ms=6)
        plotted = True
    ax.set_xscale("log")
    ax.set_xlabel("Cells / guide (log₁₀)", fontsize=12)
    ax.set_ylabel("Mean mAP", fontsize=12)
    ax.set_title(label, fontsize=14, fontweight="bold")
    ax.tick_params(labelsize=10)
    return plotted


def _plot_marker(marker, dfv1, dfv2, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    any_plotted = False
    for ax, (col, label) in zip(axes, METRICS):
        any_plotted |= _draw_metric(ax, dfv1, dfv2, col, label)
    if not any_plotted:
        plt.close(fig)
        return False
    handles, labels = axes[0].get_legend_handles_labels()
    # de-dup while preserving order
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    fig.legend(h2, l2, loc="lower center", ncol=2, fontsize=13,
               bbox_to_anchor=(0.5, -0.08))
    tag = "v1 vs v2" if dfv1 is not None else "v2 only (new marker)"
    fig.suptitle(f"Titration {tag} — {marker}", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.10, 1, 0.95])
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"{marker}_v1_v2.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_overview(markers, v1_map, v2_map, out_dir: Path, metric_col, metric_label):
    """One grid figure: a small panel per marker for a single metric (v1 vs v2)."""
    n = len(markers)
    ncols = 6
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows),
                             squeeze=False)
    for i, marker in enumerate(markers):
        ax = axes[i // ncols][i % ncols]
        dfv1 = _load(v1_map.get(marker))
        dfv2 = _load(v2_map.get(marker))
        _draw_metric(ax, dfv1, dfv2, metric_col, marker[:22])
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_title(marker[:22], fontsize=7)
        ax.tick_params(labelsize=6)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0], [0], color=V1_COLOR, marker="o", label="v1"),
                        Line2D([0], [0], color=V2_COLOR, marker="o", label="v2")],
               loc="lower center", ncol=2, fontsize=12, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle(f"v1 vs v2 titration — {metric_label} mean mAP (all markers)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"overview_{metric_col}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_highlight_new(markers, v2_map, out_dir: Path, metric_col, metric_label,
                        highlight: set):
    """All v2 markers for one metric: highlight `highlight` markers in color,
    draw the rest as gray context."""
    fig, ax = plt.subplots(figsize=(9.5, 7))
    hi_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    hi_i = 0
    for marker in markers:
        df = _load(v2_map.get(marker))
        if df is None or metric_col not in df.columns:
            continue
        d = df.dropna(subset=[metric_col]).sort_values(X_COL)
        if d.empty:
            continue
        x, y = d[X_COL].values, d[metric_col].values
        if marker in highlight:
            ax.plot(x, y, marker="o", lw=3.0, ms=7, zorder=5,
                    color=hi_colors[hi_i % 10], label=marker)
            hi_i += 1
        else:
            ax.plot(x, y, color="0.75", alpha=0.5, lw=1.2, zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("Cells / guide (log₁₀)", fontsize=13)
    ax.set_ylabel(f"{metric_label} mean mAP", fontsize=13)
    ax.set_title(f"v2 titration — all markers ({metric_label})\nnew fluorescent markers highlighted",
                 fontsize=14, fontweight="bold")
    ax.tick_params(labelsize=10)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.26), ncol=3, fontsize=11,
              title="new in v2")
    fig.tight_layout(rect=[0, 0.12, 1, 0.94])
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"all_markers_highlight_new_{metric_col}.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v1-dir", default=DEFAULT_V1_DIR)
    ap.add_argument("--v2-dir", default=DEFAULT_V2_DIR)
    ap.add_argument("-o", "--output-dir", default=None,
                    help="Default: <v2-dir>/v1_vs_v2_comparison/")
    args = ap.parse_args()

    v1_base, v2_base = Path(args.v1_dir), Path(args.v2_dir)
    out_dir = Path(args.output_dir) if args.output_dir else v2_base / "v1_vs_v2_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    v1_map = _reporter_dirs(v1_base) if v1_base.is_dir() else {}
    v2_map = _reporter_dirs(v2_base) if v2_base.is_dir() else {}
    markers = sorted(v2_map)  # every v2 marker; v1 overlaid where available
    common = sorted(set(v1_map) & set(v2_map))
    v2_only = sorted(set(v2_map) - set(v1_map))
    logger.info(f"v1 markers: {len(v1_map)} | v2 markers: {len(v2_map)} | "
                f"common: {len(common)} | v2-only: {len(v2_only)}")
    if v2_only:
        logger.info(f"  v2-only (drawn v2 alone): {v2_only}")

    n_ok = 0
    for marker in markers:
        if _plot_marker(marker, _load(v1_map.get(marker)), _load(v2_map.get(marker)), out_dir):
            n_ok += 1
    logger.info(f"Wrote {n_ok} per-marker figures to {out_dir}")

    for col, label in METRICS:
        _plot_overview(markers, v1_map, v2_map, out_dir, col, label)
    logger.info(f"Wrote 3 overview grids (activity/distinctiveness/EBI) to {out_dir}")

    # All-markers v2 plots for distinctiveness + EBI with the new fluorescent
    # markers (v2-only) highlighted against the rest in gray.
    highlight = set(v2_only)
    for col, label in (("distinctiveness_map_mean", "Distinctiveness"),
                       ("ebi_map_mean", "EBI")):
        _plot_highlight_new(markers, v2_map, out_dir, col, label, highlight)
    logger.info(f"Wrote 2 all-marker highlight plots (distinct/EBI, new={sorted(highlight)})")


if __name__ == "__main__":
    main()
