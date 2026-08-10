"""Plot expansion-sweep mAP curves from a ``bin_results.csv``.

One flexible script that replaces the family of per-head / per-layout plot
scripts we used to have. Supports three layouts:

* ``rows``    : one panel-row per head × N metric columns (e.g. 3 heads × 2
                metrics = 3×2 canvas)
* ``overlay`` : all heads overlaid on the same axes, one panel per metric
                (e.g. 1×2 with 3 curves per panel)
* ``single``  : one head, one panel per metric, drawing the requested
                directions (top / low / random) as separate lines

Colors: pick from tab10 by default, or pass ``--colors`` (one hex per head).

Examples
--------
    # v5: 3 models × 2 metrics, one row per model
    python -m ops_model.models.attention.titration.expansion.plot_sweep_curves \\
        --expansion-root /hpc/.../v5/expansion_v1 \\
        --heads gko ebionly ebifb \\
        --layout rows --metrics distinctiveness ebi_consistency \\
        --out-name v5_canvas_3x2 --title "v5 SetTransformer — top-K per gene"

    # 3 models overlaid on the same axes
    ... --layout overlay --out-name v5_overlay

    # Single head, 3 lines (top / bottom / random)
    ... --heads gko --layout single --directions top low random \\
        --out-name gko_only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib
import pandas as pd

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt

# Sensible defaults for figure-quality typography (call ``_apply_style`` to activate)
_STYLE_RC = {
    "font.size":       16,
    "axes.titlesize":  20,
    "axes.labelsize":  19,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "axes.linewidth":  1.6,
}
_DEFAULT_COLORS = ("#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e", "#17becf")

# Direction styling: (linestyle, marker, linewidth, markersize, color-override)
_DIR_STYLE = {
    "top":    ("-",  "o", 3.5, 10, None),
    "low":    ("--", "o", 2.8,  8, None),  # same color as top by default
    "random": (":",  "s", 2.5,  8, "black"),
}


def _apply_style() -> None:
    plt.rcParams.update(_STYLE_RC)


# -----------------------------------------------------------------------------
# Plot primitives
# -----------------------------------------------------------------------------

def _plot_curve(ax, sub, col, label, color, style_key):
    if sub.empty:
        return
    ls, marker, lw, ms, color_override = _DIR_STYLE[style_key]
    peak = float(sub[col].max())
    ax.plot(sub["K"], sub[col], marker=marker, linestyle=ls,
            color=color_override or color, linewidth=lw, markersize=ms,
            label=f"{label} (peak {peak:.3f})")


def _finalize_panel(ax, x_label, y_label, title, legend_loc="best"):
    ax.set_xscale("log")
    ax.set_ylim(0, 1.0)   # mAP is bounded [0, 1] — force full range so 1.0 is visible
    ax.tick_params(width=1.4, length=6)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label, fontweight="bold", fontsize=20, labelpad=2)
    if title is not None:
        ax.set_title(title, fontweight="bold", pad=6)
    ax.grid(alpha=0.3, linewidth=0.8)
    ax.legend(loc=legend_loc, frameon=True, framealpha=0.9)


# -----------------------------------------------------------------------------
# Layouts
# -----------------------------------------------------------------------------

def _plot_rows(df, heads, head_labels, colors, metrics, metric_labels,
                baseline, directions, out_dir, out_name, suptitle):
    sa = df[df["sweep"] == "A"].copy()
    rnd = sa[sa["direction"] == "random"].sort_values("K") if "random" in directions else None

    n_rows, n_cols = len(heads), len(metrics)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(6 * n_cols, 4.5 * n_rows),
                              sharex=True, sharey="col", squeeze=False)

    for i, head in enumerate(heads):
        color = colors[i % len(colors)]
        for j, (metric, m_label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[i, j]
            for d in directions:
                if d == "random":
                    _plot_curve(ax, rnd, metric, "random", color, "random")
                else:
                    sub = sa[(sa["head"] == head) & (sa["direction"] == d)].sort_values("K")
                    _plot_curve(ax, sub, metric, d, color, d)
            if baseline is not None and pd.notna(baseline.get(metric)):
                ax.axhline(baseline[metric], color="red", linestyle="--", linewidth=2)
            x_label = "K cells / gene" if i == n_rows - 1 else None
            y_label = head_labels[i] if j == 0 else None
            title = m_label if i == 0 else None
            leg_loc = "lower right" if (i == 0 and j == 0) else "best"
            _finalize_panel(ax, x_label, y_label, title, leg_loc)

    fig.tight_layout(h_pad=0.5, w_pad=0.8)
    if suptitle:
        fig.suptitle(suptitle, fontsize=22, fontweight="bold", y=1.005)
        fig.subplots_adjust(top=0.94 if n_rows > 1 else 0.88)
    _save(fig, out_dir, out_name)


def _plot_overlay(df, heads, head_labels, colors, metrics, metric_labels,
                    baseline, out_dir, out_name, suptitle):
    sa = df[df["sweep"] == "A"].copy()
    rnd = sa[sa["direction"] == "random"].sort_values("K")

    n_cols = len(metrics)
    fig, axes = plt.subplots(1, n_cols, figsize=(9 * n_cols, 6), sharex=True)
    if n_cols == 1:
        axes = [axes]
    for j, (metric, m_label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[j]
        for i, head in enumerate(heads):
            color = colors[i % len(colors)]
            sub = sa[(sa["head"] == head) & (sa["direction"] == "top")].sort_values("K")
            _plot_curve(ax, sub, metric, head_labels[i], color, "top")
        _plot_curve(ax, rnd, metric, "random", None, "random")
        if baseline is not None and pd.notna(baseline.get(metric)):
            ax.axhline(baseline[metric], color="red", linestyle="--", linewidth=2)
        _finalize_panel(
            ax, "K cells / gene", m_label, None,
            legend_loc="upper left",
        )
        # Legend outside the panel to avoid crowding curves
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  frameon=True, framealpha=0.9)

    fig.tight_layout(w_pad=2.0)
    if suptitle:
        fig.suptitle(suptitle, fontsize=22, fontweight="bold", y=1.005)
        fig.subplots_adjust(top=0.86)
    _save(fig, out_dir, out_name)


def _plot_single(df, head, head_label, color, metrics, metric_labels,
                   baseline, directions, out_dir, out_name, suptitle):
    """One head, one panel per metric — draws requested directions as separate lines."""
    sa = df[df["sweep"] == "A"].copy()
    rnd = sa[sa["direction"] == "random"].sort_values("K") if "random" in directions else None

    n_cols = len(metrics)
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5), sharex=True)
    if n_cols == 1:
        axes = [axes]
    for j, (metric, m_label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[j]
        for d in directions:
            if d == "random":
                _plot_curve(ax, rnd, metric, "random (head-agnostic)", color, "random")
            else:
                sub = sa[(sa["head"] == head) & (sa["direction"] == d)].sort_values("K")
                _plot_curve(ax, sub, metric, f"{head_label} / {d}", color, d)
        if baseline is not None and pd.notna(baseline.get(metric)):
            ax.axhline(baseline[metric], color="red", linestyle="--", linewidth=1,
                       label=f"all-cells ({baseline[metric]:.3f})")
        _finalize_panel(ax, "K cells / gene", m_label, None)

    fig.tight_layout(w_pad=1.0)
    if suptitle:
        fig.suptitle(suptitle, fontsize=20, fontweight="bold", y=1.02)
    _save(fig, out_dir, out_name)


def _save(fig, out_dir: Path, out_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"{out_name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_name}.{{png,pdf,svg}} to {out_dir}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

# Human-readable labels — extend as needed
_METRIC_LABELS = {
    "distinctiveness":     "Distinctiveness (mAP)",
    "ebi_consistency":     "EBI consistency (mAP)",
    "phenotypic_activity": "Phenotypic activity (mAP)",
    "chad_consistency":    "CHAD consistency (mAP)",
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--expansion-root", type=Path, required=True,
                   help="Path to expansion_v1 root (containing bin_results.csv and plots/)")
    p.add_argument("--heads", nargs="+", required=True,
                   help="Head names from bin_results.csv (e.g. gko ebionly ebifb)")
    p.add_argument("--head-labels", nargs="+", default=None,
                   help="Optional display labels for heads (one per --heads entry)")
    p.add_argument("--layout", choices=["rows", "overlay", "single"], required=True)
    p.add_argument("--metrics", nargs="+",
                   default=["distinctiveness", "ebi_consistency"],
                   help="Metric columns from bin_results.csv to plot")
    p.add_argument("--directions", nargs="+",
                   default=["top", "random"],
                   help="Which direction curves to draw per panel (top/low/random)")
    p.add_argument("--colors", nargs="+", default=None,
                   help="Hex colors, one per head (defaults to tab10)")
    p.add_argument("--out-name", required=True,
                   help="Filename base under expansion_root/plots/<out-name>/")
    p.add_argument("--title", default=None,
                   help="Optional figure suptitle")
    p.add_argument("--bin-csv", type=Path, default=None,
                   help="Override bin_results.csv path (default: <expansion-root>/bin_results.csv)")
    args = p.parse_args(argv)

    csv = args.bin_csv or (args.expansion_root / "bin_results.csv")
    if not csv.exists():
        raise SystemExit(f"bin_results.csv missing: {csv}")

    df = pd.read_csv(csv)
    baseline = (df[df["sweep"] == "baseline"].iloc[0]
                if (df["sweep"] == "baseline").any() else None)

    head_labels = args.head_labels or list(args.heads)
    if len(head_labels) != len(args.heads):
        raise SystemExit("--head-labels must have same length as --heads")
    # Interpret \n in labels literally (bash single-quote friendly). Users can
    # pass "geneKO model\n(per-gene)" and get a two-line label.
    head_labels = [s.replace("\\n", "\n") for s in head_labels]
    colors = args.colors or list(_DEFAULT_COLORS)
    metric_labels = [_METRIC_LABELS.get(m, m) for m in args.metrics]

    out_dir = args.expansion_root / "plots" / args.out_name

    _apply_style()

    if args.layout == "rows":
        _plot_rows(df, args.heads, head_labels, colors,
                     args.metrics, metric_labels, baseline,
                     args.directions, out_dir, args.out_name, args.title)
    elif args.layout == "overlay":
        _plot_overlay(df, args.heads, head_labels, colors,
                        args.metrics, metric_labels, baseline,
                        out_dir, args.out_name, args.title)
    elif args.layout == "single":
        if len(args.heads) != 1:
            raise SystemExit("--layout single requires exactly one --heads entry")
        _plot_single(df, args.heads[0], head_labels[0], colors[0],
                       args.metrics, metric_labels, baseline,
                       args.directions, out_dir, args.out_name, args.title)
    return 0


if __name__ == "__main__":
    sys.exit(main())
