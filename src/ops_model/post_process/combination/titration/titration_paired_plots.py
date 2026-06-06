"""Shared plot helpers for the Phase + fluor titration sibling scripts.

Both ``titration_phase_paired_fluor`` (single fluor marker) and
``titration_phase_paired_dual_fluor`` (two fluor markers) feed long-form
``combined`` DataFrames with one row per (entity × cells_per_guide). The entity
column is ``marker`` for single-fluor and ``pair`` for dual-fluor — pass it
via ``entity_col``. Everything else (titles, output filename prefix, legend
text) is also kwarg-driven so the two scripts share these 5 plot functions
verbatim.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_METRICS_PRETTY: List[tuple] = [
    ("activity", "Activity"),
    ("distinctiveness", "Distinctiveness"),
    ("ebi", "EBI consistency"),
]

# OPS library has 4 sgRNAs per perturbation (gene). The titration is per-sgRNA;
# display-time we multiply K by this so the x-axis reads in cells-per-pert.
GUIDES_PER_PERT = 4



# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

_LIVE_CELL_DYE_RE = re.compile(r"[_ ][Ll]ive[-_ ][Cc]ell[-_ ][Dd]ye$")
_EXCITATION_RE = re.compile(r"[_ ]\d+[_ ]excitation$", re.IGNORECASE)


def pretty_marker(name: str) -> str:
    """Strip noisy trailing tokens (``_Live_Cell_Dye``, ``_<nm>_excitation``)
    from a marker label.
    """
    s = str(name)
    s = _LIVE_CELL_DYE_RE.sub("", s)
    s = _EXCITATION_RE.sub("", s)
    return s


def pretty_pair(pair: str) -> str:
    """``A+B`` → ``A + B`` with the live-cell-dye suffix stripped from each."""
    a, b = pair.split("+", 1)
    return f"{pretty_marker(a)} + {pretty_marker(b)}"


def _setup_matplotlib():
    import matplotlib
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt
    return plt


def _fmt_K(n: float) -> str:
    if n >= 1e6: return f"{n / 1e6:.1f}M"
    if n >= 1e3: return f"{n / 1e3:.0f}K"
    return f"{int(n)}"


def bold_palette(n: int) -> list:
    """``n`` visually-distinct *saturated* colors (no pale variants like
    tab20's odd indices). Drops near-yellow shades that are illegible on
    white. Cycles past ~30.
    """
    import matplotlib.pyplot as plt
    palette = []
    for cm_name in ("tab10", "Set1", "Dark2", "Set2"):
        cm = plt.get_cmap(cm_name)
        for i in range(cm.N):
            c = cm(i)
            r, g, b = c[0], c[1], c[2]
            if (r > 0.85 and g > 0.85 and b < 0.4) or \
               (r > 0.95 and g > 0.95 and b > 0.6):
                continue
            palette.append(c)
    out = []
    while len(out) < n:
        out.extend(palette)
    return out[:n]


def _pivot_per_metric(
    combined: pd.DataFrame, metric: str, entity_col: str,
) -> pd.DataFrame:
    """Entity × cells_per_guide pivot of ``<metric>_map_mean``."""
    return (
        combined.pivot_table(
            index=entity_col, columns="cells_per_guide",
            values=f"{metric}_map_mean", aggfunc="first",
        )
        .sort_index(axis=1)
    )


def _save_all_formats(fig, out_dir: Path, basename: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"{basename}.{ext}", dpi=150, bbox_inches="tight")
    logger.info("wrote %s/%s.[png/pdf/svg]", out_dir, basename)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_winners(
    winners: pd.DataFrame,
    phase_df: pd.DataFrame,
    out_dir: Path,
    *,
    winning_col: str,        # "winning_marker" | "winning_pair"
    label_fn: Callable[[str], str],
    out_prefix: str,         # "phase_plus_fluor" | "phase_plus_dual_fluor"
    line_legend: str,
    suptitle: str,
    stack_vertical: bool = False,
    fig_w: float = 5.5,      # per-panel width (horizontal) or total width (vertical)
    fig_h: float = 5.8,      # per-panel height
    wspace: Optional[float] = None,  # subplot horizontal gap (default = mpl auto)
) -> None:
    """3-panel: Phase baseline + best-entity line, each winning point colored
    by its entity, label flowing left from the dot in the same color. Pass
    ``stack_vertical=True`` for layouts with long entity labels (e.g. dual-fluor
    pairs) — each panel gets the full figure width.
    """
    plt = _setup_matplotlib()
    from matplotlib.ticker import FuncFormatter
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert per-guide K → per-pert (×4) at display time only.
    winners = winners.copy()
    phase_df = phase_df.copy()
    winners["cells_per_guide"] = winners["cells_per_guide"] * GUIDES_PER_PERT
    phase_df["cells_per_guide"] = phase_df["cells_per_guide"] * GUIDES_PER_PERT

    unique_entities = sorted(winners[winning_col].dropna().unique().tolist())
    palette = bold_palette(len(unique_entities))
    color = {e: palette[i] for i, e in enumerate(unique_entities)}

    n = len(_METRICS_PRETTY)
    if stack_vertical:
        fig, axes = plt.subplots(n, 1, figsize=(fig_w, fig_h * n))
    else:
        fig, axes = plt.subplots(1, n, figsize=(fig_w * n, fig_h))
    x_max = float(phase_df["cells_per_guide"].max())
    for j, (m, label) in enumerate(_METRICS_PRETTY):
        ax = axes[j]
        col = f"{m}_map_mean"
        if col in phase_df.columns:
            ph = phase_df.sort_values("cells_per_guide")
            ax.plot(ph["cells_per_guide"], ph[col], marker="o", linewidth=2.5,
                    markersize=6, color="#888888", label="Phase only")
        w = winners[winners["metric"] == m].sort_values("cells_per_guide")
        if not w.empty:
            ax.plot(w["cells_per_guide"], w["paired_map_mean"],
                    linewidth=1.5, color="#555555", zorder=2, label=line_legend)
            pcolors = [color[e] for e in w[winning_col]]
            ax.scatter(w["cells_per_guide"], w["paired_map_mean"],
                       marker="s", s=55, c=pcolors,
                       edgecolor="white", linewidth=0.5, zorder=3)
            for _, row in w.iterrows():
                c = color[row[winning_col]]
                ax.annotate(
                    label_fn(row[winning_col]),
                    xy=(row["cells_per_guide"], row["paired_map_mean"]),
                    xytext=(-6, 0), textcoords="offset points",
                    fontsize=7, ha="right", va="center", color=c,
                )
        ax.set_xscale("log")
        ax.set_xlabel("cells per pert (log10)", fontsize=11)
        ax.set_ylabel("mAP (mean)", fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.grid(True, alpha=0.3)
        existing = [t for t in ax.get_xticks() if t < x_max * 0.9]
        ax.set_xticks(sorted(set(existing + [x_max])))
        ax.set_xlim(1, x_max)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _fmt_K(v)))
        ax.tick_params(axis="x", which="minor", bottom=False)
        for tl in ax.get_xticklabels():
            tl.set_rotation(30); tl.set_ha("right")
        ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle(suptitle, fontsize=13, y=1.0)
    fig.tight_layout(rect=(0, 0.0, 1, 0.96))
    if wspace is not None:
        fig.subplots_adjust(wspace=wspace)
    _save_all_formats(fig, out_dir, f"{out_prefix}_winners")
    plt.close(fig)


def plot_rank_bump(
    combined: pd.DataFrame,
    out_dir: Path,
    *,
    entity_col: str,
    label_fn: Callable[[str], str],
    out_prefix: str,
    ylabel: str,
    suptitle: str,
    top_n: int = 5,
) -> None:
    """Per-metric bump chart of rank vs cells_per_pert. Top-N highlighted in
    saturated colors with the SAME color used across panels for the same
    entity; everyone else faded gray.
    """
    plt = _setup_matplotlib()
    combined = combined.copy()
    combined["cells_per_guide"] = combined["cells_per_guide"] * GUIDES_PER_PERT
    fig, axes = plt.subplots(1, len(_METRICS_PRETTY),
                             figsize=(6 * len(_METRICS_PRETTY), 6.5))

    panel_top: Dict[str, list] = {}
    panel_meanrank: Dict[str, pd.Series] = {}
    union_top: list = []
    for m, _ in _METRICS_PRETTY:
        pivot = _pivot_per_metric(combined, m, entity_col)
        rank = pivot.rank(axis=0, ascending=False, method="min")
        mr = rank.mean(axis=1).sort_values()
        panel_meanrank[m] = mr
        order = list(mr.head(top_n).index)
        panel_top[m] = order
        for e_ in order:
            if e_ not in union_top:
                union_top.append(e_)
    palette = bold_palette(len(union_top))
    color = {e_: palette[i] for i, e_ in enumerate(union_top)}

    for j, (m, label) in enumerate(_METRICS_PRETTY):
        ax = axes[j]
        pivot = _pivot_per_metric(combined, m, entity_col)
        rank = pivot.rank(axis=0, ascending=False, method="min")
        mr = panel_meanrank[m]
        top = panel_top[m]
        for ent in rank.index:
            y = rank.loc[ent].values
            x = rank.columns.values
            if ent in top:
                ax.plot(x, y, marker="o", linewidth=2.0, markersize=5,
                        color=color[ent],
                        label=f"{label_fn(ent)} (mean rank {mr[ent]:.1f})",
                        zorder=3)
            else:
                ax.plot(x, y, color="#cccccc", linewidth=0.7, alpha=0.5, zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(left=1)
        ax.invert_yaxis()
        ax.set_xlabel("cells per pert (log10)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=7, frameon=False)
    fig.suptitle(suptitle, fontsize=13, y=1.0)
    fig.tight_layout()
    _save_all_formats(fig, out_dir, f"{out_prefix}_rank_bump")
    plt.close(fig)


def plot_delta_heatmap(
    combined: pd.DataFrame,
    phase_df: pd.DataFrame,
    out_dir: Path,
    *,
    entity_col: str,
    label_fn: Callable[[str], str],
    out_prefix: str,
    suptitle: str,
    top_n_rows: Optional[int] = None,
    label_truncate: int = 35,
    figsize_h: float = 9,
) -> None:
    """Entity × bin heatmap of Δ mAP (paired − Phase). Rows sorted by mean
    delta (best on top). If ``top_n_rows`` is set, only that many rows are
    shown — useful for the dual-fluor experiment where 230 pairs won't fit.
    """
    plt = _setup_matplotlib()
    combined = combined.copy()
    phase_df = phase_df.copy()
    combined["cells_per_guide"] = combined["cells_per_guide"] * GUIDES_PER_PERT
    phase_df["cells_per_guide"] = phase_df["cells_per_guide"] * GUIDES_PER_PERT
    fig, axes = plt.subplots(1, len(_METRICS_PRETTY),
                             figsize=(6 * len(_METRICS_PRETTY), figsize_h))
    phase_by_K = phase_df.set_index("cells_per_guide")
    for j, (m, label) in enumerate(_METRICS_PRETTY):
        ax = axes[j]
        col = f"{m}_map_mean"
        pivot = _pivot_per_metric(combined, m, entity_col)
        phase_row = phase_by_K[col].reindex(pivot.columns)
        delta = pivot.subtract(phase_row, axis=1)
        order = delta.mean(axis=1).sort_values(ascending=False).index
        if top_n_rows:
            order = order[:top_n_rows]
        delta = delta.loc[order]
        v = float(np.nanmax(np.abs(delta.values)))
        im = ax.imshow(delta.values, aspect="auto", cmap="RdBu_r",
                       vmin=-v, vmax=v)
        ax.set_xticks(range(len(delta.columns)))
        ax.set_xticklabels([_fmt_K(k) for k in delta.columns],
                           rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(delta.index)))
        pretty = [label_fn(e_) for e_ in delta.index]
        ax.set_yticklabels(
            [p[:label_truncate] + ("..." if len(p) > label_truncate else "")
             for p in pretty],
            fontsize=6,
        )
        ax.set_xlabel("cells per pert", fontsize=11)
        ax.set_title(label, fontsize=12)
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("Δ mAP (paired − Phase)", fontsize=8)
    fig.suptitle(suptitle, fontsize=13, y=1.0)
    fig.tight_layout()
    _save_all_formats(fig, out_dir, f"{out_prefix}_delta_heatmap")
    plt.close(fig)


def plot_topn_curves(
    combined: pd.DataFrame,
    phase_df: pd.DataFrame,
    out_dir: Path,
    *,
    entity_col: str,
    label_fn: Callable[[str], str],
    out_prefix: str,
    ribbon_label: str,
    suptitle: str,
    top_n: int = 5,
) -> None:
    """Phase baseline + top-N entities (by mean mAP across bins) with the
    same color used for the same entity across all panels, plus a 5–95%
    paired ribbon over the full entity population.
    """
    plt = _setup_matplotlib()
    from matplotlib.ticker import FuncFormatter
    combined = combined.copy()
    phase_df = phase_df.copy()
    combined["cells_per_guide"] = combined["cells_per_guide"] * GUIDES_PER_PERT
    phase_df["cells_per_guide"] = phase_df["cells_per_guide"] * GUIDES_PER_PERT
    fig, axes = plt.subplots(1, len(_METRICS_PRETTY),
                             figsize=(5.5 * len(_METRICS_PRETTY), 5.8))
    x_max = float(phase_df["cells_per_guide"].max())
    phase_sorted = phase_df.sort_values("cells_per_guide")

    panel_top: Dict[str, list] = {}
    union_top: list = []
    for m, _ in _METRICS_PRETTY:
        pivot = _pivot_per_metric(combined, m, entity_col)
        order = list(pivot.mean(axis=1).sort_values(ascending=False).index[:top_n])
        panel_top[m] = order
        for e_ in order:
            if e_ not in union_top:
                union_top.append(e_)
    palette = bold_palette(len(union_top))
    color = {e_: palette[i] for i, e_ in enumerate(union_top)}

    for j, (m, label) in enumerate(_METRICS_PRETTY):
        ax = axes[j]
        col = f"{m}_map_mean"
        pivot = _pivot_per_metric(combined, m, entity_col)
        order = panel_top[m]
        x = pivot.columns.values
        p5 = pivot.quantile(0.05, axis=0).values
        p95 = pivot.quantile(0.95, axis=0).values
        ax.fill_between(x, p5, p95, color="#cccccc", alpha=0.35,
                        label=ribbon_label, linewidth=0)
        ax.plot(phase_sorted["cells_per_guide"], phase_sorted[col],
                marker="o", linewidth=2.0, markersize=5, color="#000000",
                label="Phase only")
        for ent in order:
            ax.plot(x, pivot.loc[ent].values, marker="s",
                    linewidth=1.8, markersize=4, color=color[ent],
                    label=label_fn(ent), alpha=0.95)
        ax.set_xscale("log")
        ax.set_xlabel("cells per pert (log10)", fontsize=11)
        ax.set_ylabel("mAP (mean)", fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.grid(True, alpha=0.3)
        existing = [t for t in ax.get_xticks() if t < x_max * 0.9]
        ax.set_xticks(sorted(set(existing + [x_max])))
        ax.set_xlim(1, x_max)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _fmt_K(v)))
        ax.tick_params(axis="x", which="minor", bottom=False)
        for tl in ax.get_xticklabels():
            tl.set_rotation(30); tl.set_ha("right")
        ax.legend(loc="lower right", fontsize=7, frameon=False)
    fig.suptitle(suptitle, fontsize=13, y=1.0)
    fig.tight_layout()
    _save_all_formats(fig, out_dir, f"{out_prefix}_topn_curves")
    plt.close(fig)


def plot_win_share(
    combined: pd.DataFrame,
    out_dir: Path,
    *,
    entity_col: str,
    label_fn: Callable[[str], str],
    out_prefix: str,
    suptitle: str,
    drop_zero_wins: bool = False,
    top_k_rank: int = 3,
) -> None:
    """Per-entity count of (bin × metric) cells where the entity ranked
    top-K. Bars stacked by metric, sorted by total. With ``drop_zero_wins``
    (use for the 230-pair dual-fluor case) we omit pairs that never won.
    """
    plt = _setup_matplotlib()
    rows = []
    for m, label in _METRICS_PRETTY:
        pivot = _pivot_per_metric(combined, m, entity_col)
        rank = pivot.rank(axis=0, ascending=False, method="min")
        in_topk = (rank <= top_k_rank)
        counts = in_topk.sum(axis=1)
        for ent, c in counts.items():
            rows.append({"entity": ent, "metric": label, "wins": int(c)})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    pivot = df.pivot_table(index="entity", columns="metric", values="wins",
                           aggfunc="sum", fill_value=0)
    pivot["total"] = pivot.sum(axis=1)
    if drop_zero_wins:
        pivot = pivot[pivot["total"] > 0]
    pivot = pivot.sort_values("total", ascending=True).drop(columns=["total"])
    pivot.index = [label_fn(e_) for e_ in pivot.index]

    fig, ax = plt.subplots(figsize=(11, max(7, 0.25 * len(pivot))))
    palette = bold_palette(len(_METRICS_PRETTY))
    bottom = np.zeros(len(pivot))
    for i, m_label in enumerate([m[1] for m in _METRICS_PRETTY]):
        if m_label not in pivot.columns:
            continue
        vals = pivot[m_label].values
        ax.barh(pivot.index, vals, left=bottom, label=m_label, color=palette[i],
                edgecolor="white", linewidth=0.4)
        bottom += vals
    ax.set_xlabel(f"Number of bins ranked top-{top_k_rank}", fontsize=11)
    ax.set_title(suptitle, fontsize=12)
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    fig.tight_layout()
    _save_all_formats(fig, out_dir, f"{out_prefix}_win_share")
    plt.close(fig)


def plot_winner_decomposition(
    winners: pd.DataFrame,
    dual_combined: pd.DataFrame,
    single_combined: pd.DataFrame,
    phase_df: pd.DataFrame,
    out_dir: Path,
    *,
    out_prefix: str,
    suptitle: str,
    top_n_pairs: int = 4,
) -> None:
    """For the top-N winning pairs, decompose each into 4 curves per metric:
    Phase only, Phase + marker A, Phase + marker B, Phase + (A and B). Shows
    whether the dual boost is synergistic (combined exceeds the sum of parts)
    or merely additive.

    Layout: top_n_pairs rows × 3 metric columns.
    """
    plt = _setup_matplotlib()
    from matplotlib.ticker import FuncFormatter

    # Convert per-guide K → per-pert.
    dual = dual_combined.copy()
    single = single_combined.copy()
    phase = phase_df.copy()
    for df in (dual, single, phase):
        df["cells_per_guide"] = df["cells_per_guide"] * GUIDES_PER_PERT

    # Pick top-N pairs by total appearance in winners table.
    top_pairs = (
        winners["winning_pair"].value_counts().head(top_n_pairs).index.tolist()
    )
    if not top_pairs:
        return

    n_rows = len(top_pairs)
    n_cols = len(_METRICS_PRETTY)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 4.0 * n_rows),
                             squeeze=False)
    x_max = float(phase["cells_per_guide"].max())
    phase_sorted = phase.sort_values("cells_per_guide")

    # Stable colors for the 4 lines per panel
    PHASE_COLOR = "#888888"
    A_COLOR = "#1f77b4"   # blue
    B_COLOR = "#ff7f0e"   # orange
    AB_COLOR = "#d62728"  # red (the combined)

    for i, pair in enumerate(top_pairs):
        # Parse the two markers (pair is "A+B")
        a, b = pair.split("+", 1)
        single_a = single[single["marker"] == a].sort_values("cells_per_guide")
        single_b = single[single["marker"] == b].sort_values("cells_per_guide")
        dual_ab = dual[dual["pair"] == pair].sort_values("cells_per_guide")

        for j, (m, m_label) in enumerate(_METRICS_PRETTY):
            ax = axes[i, j]
            col = f"{m}_map_mean"
            # Phase only
            ax.plot(phase_sorted["cells_per_guide"], phase_sorted[col],
                    marker="o", linewidth=2.0, markersize=5, color=PHASE_COLOR,
                    label="Phase")
            # Phase + A
            if not single_a.empty:
                ax.plot(single_a["cells_per_guide"], single_a[col],
                        marker="s", linewidth=2.0, markersize=5, color=A_COLOR,
                        label=f"+ {pretty_marker(a)}")
            # Phase + B
            if not single_b.empty:
                ax.plot(single_b["cells_per_guide"], single_b[col],
                        marker="^", linewidth=2.0, markersize=5, color=B_COLOR,
                        label=f"+ {pretty_marker(b)}")
            # Phase + A + B
            if not dual_ab.empty:
                ax.plot(dual_ab["cells_per_guide"], dual_ab[col],
                        marker="D", linewidth=2.5, markersize=6, color=AB_COLOR,
                        label=f"+ both")
            ax.set_xscale("log")
            existing = [t for t in ax.get_xticks() if t < x_max * 0.9]
            ax.set_xticks(sorted(set(existing + [x_max])))
            ax.set_xlim(1, x_max)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _fmt_K(v)))
            ax.tick_params(axis="x", which="minor", bottom=False)
            for tl in ax.get_xticklabels():
                tl.set_rotation(30); tl.set_ha("right")
            ax.set_xlabel("cells per pert (log10)", fontsize=10)
            ax.set_ylabel("mAP (mean)", fontsize=10)
            if i == 0:
                ax.set_title(m_label, fontsize=12)
            ax.grid(True, alpha=0.3)
            if j == 0:
                # row label on the leftmost panel
                ax.text(-0.22, 0.5, pretty_pair(pair),
                        transform=ax.transAxes, ha="right", va="center",
                        fontsize=10, fontweight="bold", rotation=0,
                        wrap=True)
            ax.legend(loc="lower right", fontsize=8, frameon=False)

    fig.suptitle(suptitle, fontsize=13, y=1.0)
    fig.tight_layout(rect=(0.04, 0.0, 1, 0.97))
    _save_all_formats(fig, out_dir, f"{out_prefix}_winner_decomposition")
    plt.close(fig)
