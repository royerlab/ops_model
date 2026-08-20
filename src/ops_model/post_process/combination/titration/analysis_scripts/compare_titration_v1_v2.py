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
V1_COLOR, V2_COLOR = "#7f7f7f", "#d62728"  # a=gray (baseline/older), b=red (comparison)


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


def _draw_metric(ax, dfv1, dfv2, col, label,
                  label_a="v1", label_b="v2",
                  color_a=V1_COLOR, color_b=V2_COLOR):
    plotted = False
    for df, ver, color in ((dfv1, label_a, color_a), (dfv2, label_b, color_b)):
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


def _plot_marker(marker, dfv1, dfv2, out_dir: Path,
                  label_a="v1", label_b="v2",
                  color_a=V1_COLOR, color_b=V2_COLOR,
                  file_tag="v1_v2"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    any_plotted = False
    for ax, (col, label) in zip(axes, METRICS):
        any_plotted |= _draw_metric(ax, dfv1, dfv2, col, label,
                                     label_a=label_a, label_b=label_b,
                                     color_a=color_a, color_b=color_b)
    if not any_plotted:
        plt.close(fig)
        return False
    handles, labels = axes[0].get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    fig.legend(h2, l2, loc="lower center", ncol=2, fontsize=13,
               bbox_to_anchor=(0.5, -0.08))
    tag = f"{label_a} vs {label_b}" if dfv1 is not None else f"{label_b} only (new marker)"
    fig.suptitle(f"Titration {tag} — {marker}", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.10, 1, 0.95])
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"{marker}_{file_tag}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_overview(markers, v1_map, v2_map, out_dir: Path, metric_col, metric_label,
                    label_a="v1", label_b="v2",
                    color_a=V1_COLOR, color_b=V2_COLOR):
    """One grid figure: a small panel per marker for a single metric."""
    n = len(markers)
    ncols = 6
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows),
                             squeeze=False)
    for i, marker in enumerate(markers):
        ax = axes[i // ncols][i % ncols]
        dfv1 = _load(v1_map.get(marker))
        dfv2 = _load(v2_map.get(marker))
        _draw_metric(ax, dfv1, dfv2, metric_col, marker[:22],
                     label_a=label_a, label_b=label_b,
                     color_a=color_a, color_b=color_b)
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_title(marker[:22], fontsize=7)
        ax.tick_params(labelsize=6)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0], [0], color=color_a, marker="o", label=label_a),
                        Line2D([0], [0], color=color_b, marker="o", label=label_b)],
               loc="lower center", ncol=2, fontsize=12, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle(f"{label_a} vs {label_b} titration — {metric_label} mean mAP (all markers)",
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
    ap.add_argument("--v1-dir", "--dir-a", dest="v1_dir", default=DEFAULT_V1_DIR,
                    help="Baseline / A titration_guide_median dir")
    ap.add_argument("--v2-dir", "--dir-b", dest="v2_dir", default=DEFAULT_V2_DIR,
                    help="Comparison / B titration_guide_median dir")
    ap.add_argument("--label-a", default="v1",
                    help="Legend label for A (default: v1)")
    ap.add_argument("--label-b", default="v2",
                    help="Legend label for B (default: v2)")
    ap.add_argument("--color-a", default=V1_COLOR)
    ap.add_argument("--color-b", default=V2_COLOR)
    ap.add_argument("--file-tag", default=None,
                    help="Filename tag for per-marker figures (default: <label_a>_<label_b>)")
    ap.add_argument("-o", "--output-dir", default=None,
                    help="Default: <dir-b>/<label_a>_vs_<label_b>_comparison/")
    args = ap.parse_args()

    v1_base, v2_base = Path(args.v1_dir), Path(args.v2_dir)
    label_a, label_b = args.label_a, args.label_b
    slug = f"{label_a}_vs_{label_b}".replace(" ", "_")
    file_tag = args.file_tag or f"{label_a}_{label_b}".replace(" ", "_")
    out_dir = Path(args.output_dir) if args.output_dir else v2_base / f"{slug}_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    v1_map = _reporter_dirs(v1_base) if v1_base.is_dir() else {}
    v2_map = _reporter_dirs(v2_base) if v2_base.is_dir() else {}
    markers = sorted(v2_map)  # every B marker; A overlaid where available
    common = sorted(set(v1_map) & set(v2_map))
    b_only = sorted(set(v2_map) - set(v1_map))
    logger.info(f"{label_a}: {len(v1_map)} | {label_b}: {len(v2_map)} | "
                f"common: {len(common)} | {label_b}-only: {len(b_only)}")
    if b_only:
        logger.info(f"  {label_b}-only (drawn alone): {b_only}")

    n_ok = 0
    for marker in markers:
        if _plot_marker(marker, _load(v1_map.get(marker)), _load(v2_map.get(marker)),
                         out_dir, label_a=label_a, label_b=label_b,
                         color_a=args.color_a, color_b=args.color_b,
                         file_tag=file_tag):
            n_ok += 1
    logger.info(f"Wrote {n_ok} per-marker figures to {out_dir}")

    # Overview grid only makes sense with >1 marker (else it's 1 tile + 5 blanks)
    if len(markers) > 1:
        for col, label in METRICS:
            _plot_overview(markers, v1_map, v2_map, out_dir, col, label,
                            label_a=label_a, label_b=label_b,
                            color_a=args.color_a, color_b=args.color_b)
        logger.info(f"Wrote {len(METRICS)} overview grids to {out_dir}")
    else:
        logger.info(f"Skipping overview grids (only {len(markers)} marker)")

    # Highlight plot only makes sense when B has markers A doesn't
    highlight = set(b_only)
    if highlight:
        for col, label in (("distinctiveness_map_mean", "Distinctiveness"),
                           ("ebi_map_mean", "EBI")):
            _plot_highlight_new(markers, v2_map, out_dir, col, label, highlight)
        logger.info(f"Wrote 2 highlight plots ({label_b}-only={sorted(highlight)})")
    else:
        logger.info(f"Skipping highlight plots (no {label_b}-only markers)")


if __name__ == "__main__":
    main()
