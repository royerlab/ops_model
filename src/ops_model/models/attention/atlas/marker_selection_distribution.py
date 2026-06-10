"""Plot the distribution of markers selected by the attention-atlas mAP
matrices.

Two selection modes:
  1. `--top K`        — top-K fluor markers per row (matches the SHAP
                         atlas's default pick of K=3 markers/page). Stacks
                         color by RANK (1st pick, 2nd pick, ...).
  2. `--threshold T`  — every fluor marker with mAP ≥ T per row, no cap.
                         Stacks color by mAP MAGNITUDE BIN (0.2-0.4 etc.)
                         so a bar's segments show the strength of the
                         selections, not just their count.

The two are mutually exclusive; if both are passed, `--threshold` wins.

The SHAP atlas picks the top-K fluor markers per page by mAP:
    gene-level    → mAP DISTINCTIVENESS  (gene_reporter_distinctiveness_raw.csv)
    complex-level → mAP CONSISTENCY      (complex_reporter_chad_consistency.csv)

Plot:
  - Horizontal stacked bar, one bar per marker.
  - Bar length = total times the marker was selected.
  - Markers sorted by total selections (descending).
  - Two side-by-side panels: gene-level (left), complex-level (right).

Usage:
    # Top-K mode (atlas default — what each page actually renders)
    python ops_process/ops_analysis/napari/marker_selection_distribution.py
    python ops_process/ops_analysis/napari/marker_selection_distribution.py --top 5

    # Threshold mode — capture every marker passing a confidence bar
    python ops_process/ops_analysis/napari/marker_selection_distribution.py --threshold 0.2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Same paths as attention_atlas_shap.py — keep these literal so this
# script stays runnable without importing the (large) atlas module.
_PCA_AGG_OVERLAY = (
    Path("/home/gav.sturm/linked_folders/icd.fast.ops/organelle_attribution/")
    / "pca_optimized_v0.3" / "cell_dino" / "zscore_per_exp" / "paper_v1"
    / "with_cp" / "with_4i" / "all_livecell" / "fixed_80%" / "cosine"
    / "plots" / "marker_overlay"
)
DEFAULT_GENE_CSV = _PCA_AGG_OVERLAY / "gene_reporter_distinctiveness_raw.csv"
DEFAULT_COMPLEX_CSV = _PCA_AGG_OVERLAY / "complex_reporter_chad_consistency.csv"
DEFAULT_OUTPUT = (
    "/home/gav.sturm/linked_folders/icd.fast.ops/organelle_attribution/"
    "pca_optimized_v0.3/cell_dino/zscore_per_exp/paper_v1/with_cp/with_4i/"
    "all_livecell/fixed_80%/cosine/second_pca_consensus/plots/marker_overlay/"
    "atlas_marker_selection_dist.pdf"
)


def _top_k_per_row(df: pd.DataFrame, k: int, exclude=("Phase",)) -> list[list[str]]:
    """For each row in `df`, return the top-K column names by value.

    Sort order matches the atlas: (-value, column_name) — descending mAP
    with alphabetical tie-break. NaNs are dropped (markers with no mAP
    aren't candidates). Skips columns in `exclude` (Phase is always
    shown by the atlas independently of the top-K selection).
    """
    drop_cols = [c for c in exclude if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    out: list[list[str]] = []
    for _, row in df.iterrows():
        s = row.dropna()
        if s.empty:
            out.append([])
            continue
        ordered = sorted(s.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
        out.append([str(m) for m, _ in ordered[:k]])
    return out


def _counts_by_rank(top_lists: list[list[str]], k: int) -> dict[str, list[int]]:
    """Marker → [count at rank 1, count at rank 2, ..., count at rank k]."""
    counts: dict[str, list[int]] = {}
    for top in top_lists:
        for i, marker in enumerate(top[:k]):
            counts.setdefault(marker, [0] * k)
            counts[marker][i] += 1
    return counts


# Bin edges for the threshold-mode magnitude legend. The lowest edge
# is replaced at runtime with the user's threshold so the legend is
# never wider than the actually-included range.
_MAGNITUDE_BIN_EDGES = (0.2, 0.4, 0.6, 0.8, 1.0001)


def _select_by_threshold(df: pd.DataFrame, threshold: float,
                         exclude=("Phase",)) -> list[list[tuple[str, float]]]:
    """For each row, return the list of (marker, mAP) above `threshold`.
    No top-K cap. Phase (or any name in `exclude`) is skipped — the
    atlas always shows the Phase row independently of selection."""
    drop_cols = [c for c in exclude if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    out: list[list[tuple[str, float]]] = []
    for _, row in df.iterrows():
        s = row.dropna()
        if s.empty:
            out.append([])
            continue
        s = s[s.astype(float) >= threshold]
        if s.empty:
            out.append([])
            continue
        ordered = sorted(s.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
        out.append([(str(m), float(v)) for m, v in ordered])
    return out


def _counts_by_magnitude(
    selected_lists: list[list[tuple[str, float]]],
    bin_edges: tuple[float, ...],
) -> dict[str, list[int]]:
    """Marker → counts per magnitude bin (length = len(bin_edges) - 1).

    `bin_edges` is monotone increasing; bin i covers
    [edges[i], edges[i+1]). The last edge is treated as inclusive so
    mAP == 1.0 still lands in the top bin.
    """
    n_bins = len(bin_edges) - 1
    counts: dict[str, list[int]] = {}
    for selected in selected_lists:
        for marker, value in selected:
            # Find bin via np.searchsorted (right) - 1.
            b = int(np.searchsorted(bin_edges, value, side="right") - 1)
            b = max(0, min(b, n_bins - 1))
            counts.setdefault(marker, [0] * n_bins)
            counts[marker][b] += 1
    return counts


def _plot_panel(ax, counts: dict[str, list[int]], title: str,
                total_rows: int, colors, legend_labels: list[str],
                xlabel: str, subtitle: str) -> None:
    """Stacked horizontal-bar panel. One bar per marker; the meaning of
    the stack segments is set by the caller via `colors` + `legend_labels`
    (rank-based for top-K mode, mAP-magnitude bins for threshold mode)."""
    if not counts:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_axis_off()
        return

    n_segments = len(legend_labels)
    # Sort markers by total selections desc; alphabetical tie-break.
    ordered = sorted(
        counts.items(),
        key=lambda kv: (-sum(kv[1]), str(kv[0])),
    )
    markers = [m for m, _ in ordered]
    n = len(markers)
    y = np.arange(n)
    bottom = np.zeros(n, dtype=float)
    totals = np.array([sum(counts[m]) for m in markers], dtype=int)

    for seg_i in range(n_segments):
        vals = np.array([counts[m][seg_i] for m in markers], dtype=float)
        ax.barh(
            y, vals, left=bottom, color=colors[seg_i],
            edgecolor="white", linewidth=0.3,
            label=legend_labels[seg_i],
        )
        bottom += vals

    ax.set_yticks(y)
    ax.set_yticklabels(markers, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(f"{title}\n{subtitle}")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    # Total count annotation at the bar end.
    xmax = float(bottom.max()) if n else 0
    ax.set_xlim(0, xmax * 1.10 + 1)
    for yi, t in zip(y, totals):
        ax.text(t + xmax * 0.005 + 0.2, yi, f"{int(t):,}",
                va="center", fontsize=7, color="#333")

    ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.9)


def build_plot(gene_csv: Path, complex_csv: Path, output: Path,
               top_k: int = 3, threshold: float | None = None) -> None:
    gene_df = pd.read_csv(gene_csv, index_col=0)
    complex_df = pd.read_csv(complex_csv, index_col=0)
    print(f"Gene matrix    (mAP DISTINCTIVENESS): "
          f"{gene_df.shape[0]} rows × {gene_df.shape[1]} markers "
          f"({gene_csv.name})")
    print(f"Complex matrix (mAP CONSISTENCY):    "
          f"{complex_df.shape[0]} rows × {complex_df.shape[1]} markers "
          f"({complex_csv.name})")

    if threshold is None:
        # Top-K mode: stacks by rank (1st pick darkest → Kth lightest).
        gene_lists   = _top_k_per_row(gene_df,   k=top_k)
        complex_lists = _top_k_per_row(complex_df, k=top_k)
        gene_counts    = _counts_by_rank(gene_lists,    k=top_k)
        complex_counts = _counts_by_rank(complex_lists, k=top_k)
        colors = plt.cm.Blues(np.linspace(0.85, 0.40, top_k))
        legend_labels = [f"rank {i + 1}" for i in range(top_k)]
        xlabel_suffix = f"top-{top_k} fluor markers"
        sup_title = f"Top-{top_k} marker selection across the attention atlas"
    else:
        # Threshold mode: stacks by mAP magnitude bin.
        gene_lists   = _select_by_threshold(gene_df,   threshold=threshold)
        complex_lists = _select_by_threshold(complex_df, threshold=threshold)
        # First bin starts at the user-provided threshold so the legend
        # never advertises a range that doesn't include any data.
        edges = (threshold,) + _MAGNITUDE_BIN_EDGES[1:]
        gene_counts    = _counts_by_magnitude(gene_lists,    edges)
        complex_counts = _counts_by_magnitude(complex_lists, edges)
        n_bins = len(edges) - 1
        colors = plt.cm.Blues(np.linspace(0.40, 0.95, n_bins))
        legend_labels = [
            f"mAP {edges[i]:.2f}–{edges[i + 1] if edges[i + 1] < 1 else 1.00:.2f}"
            for i in range(n_bins)
        ]
        xlabel_suffix = f"all markers with mAP ≥ {threshold:.2f}"
        sup_title = (
            f"Markers above mAP ≥ {threshold:.2f} across the attention atlas"
        )

    print(f"Distinct markers selected (gene)   : "
          f"{len(gene_counts)} / {gene_df.shape[1]}")
    print(f"Distinct markers selected (complex): "
          f"{len(complex_counts)} / {complex_df.shape[1]}")

    # Tallest panel sets the figure height so both axes line up. Each
    # marker row gets ~0.22" of vertical space (legible @ 8pt).
    n_rows_max = max(len(gene_counts), len(complex_counts), 1)
    fig_h = max(6.0, 0.22 * n_rows_max + 1.5)
    fig, axes = plt.subplots(1, 2, figsize=(15.0, fig_h), constrained_layout=True)

    _plot_panel(
        axes[0], gene_counts,
        title="Gene KO atlas — mAP DISTINCTIVENESS",
        total_rows=len(gene_lists), colors=colors,
        legend_labels=legend_labels,
        xlabel=f"# of geneKOs selecting this marker ({xlabel_suffix})",
        subtitle=f"{len(gene_lists):,} geneKOs · {xlabel_suffix}",
    )
    _plot_panel(
        axes[1], complex_counts,
        title="CHAD complex atlas — mAP CONSISTENCY",
        total_rows=len(complex_lists), colors=colors,
        legend_labels=legend_labels,
        xlabel=f"# of complexes selecting this marker ({xlabel_suffix})",
        subtitle=f"{len(complex_lists):,} complexes · {xlabel_suffix}",
    )

    fig.suptitle(sup_title, fontsize=13, fontweight="bold")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=200)
    print(f"Wrote {output}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gene-csv", type=Path, default=DEFAULT_GENE_CSV,
                    help="Per-gene mAP distinctiveness matrix.")
    ap.add_argument("--complex-csv", type=Path, default=DEFAULT_COMPLEX_CSV,
                    help="Per-complex mAP consistency matrix.")
    ap.add_argument("--top", "--top-markers", dest="top_k", type=int, default=3,
                    help="Top-K fluor markers per row (default: 3, matches the atlas). "
                         "Ignored when --threshold is set.")
    ap.add_argument("--threshold", dest="threshold", type=float, default=None,
                    help="Selection mode: include EVERY marker with mAP ≥ threshold "
                         "per row (no top-K cap). When set, overrides --top. "
                         "Typical: --threshold 0.2.")
    ap.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    args = ap.parse_args()
    build_plot(
        args.gene_csv, args.complex_csv, args.output,
        top_k=args.top_k, threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
