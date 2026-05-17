"""Chromosome-arm overlays for pca_optimization.

Extracted from ``pca_optimization.py`` so the chromosome-coloring path
(CSV schema loader + matplotlib PDF + Plotly HTML) lives by itself and
the parent file stops accreting plot code.

Functions in this module are pure: they take their plotting backend
(``plt``) and ``_logger`` as arguments so the module has no top-level
matplotlib import and can be re-used outside the PCA-optimization run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _load_chromosome_map(csv_path: str, _logger) -> Optional[pd.DataFrame]:
    """Load symbol → chromosome / chromosome_arm CSV.

    Returns a DataFrame indexed by perturbation with a ``chr_arm`` composite
    column (e.g. ``"12q"``) used for categorical coloring, or ``None`` if the
    file is unreadable. Rows where chromosome or arm is missing get
    ``chr_arm = "unmapped"``.

    Accepts two schemas so the same path works for both the original
    chromosome panel CSVs *and* the shared ``chrom_arm_mapping.csv`` cache
    produced by the chrom-arm correction helper:

    1. ``perturbation, chromosome, chromosome_arm``  (legacy panel CSV) —
       chrom_arm = chromosome + arm (e.g. ``'12' + 'q' → '12q'``).
    2. ``symbol, chrom_arm``                          (chrom-arm cache) —
       chrom_arm values look like ``'chr12q'``; the ``chr`` prefix is
       stripped to produce ``'12q'`` so the legend matches schema (1).
    """
    sep = "\t" if str(csv_path).lower().endswith(".tsv") else ","
    try:
        df = pd.read_csv(csv_path, sep=sep)
    except Exception as exc:
        _logger.warning(f"  Chromosome CSV/TSV unreadable ({csv_path}): {exc}")
        return None
    cols = set(df.columns)
    if {"perturbation", "chromosome", "chromosome_arm"}.issubset(cols):
        df = df[["perturbation", "chromosome", "chromosome_arm"]].copy()
        df["perturbation"] = df["perturbation"].astype(str)
        chrom_str = df["chromosome"].astype(str).str.replace(r"\.0$", "", regex=True)
        arm = df["chromosome_arm"].astype(str)
        df["chr_arm"] = chrom_str + arm
        blank = chrom_str.isin({"nan", "", "None"}) | arm.isin({"nan", "", "None"})
        df.loc[blank, "chr_arm"] = "unmapped"
    elif {"gene", "updated_gene", "chrom_arm"}.issubset(cols):
        # Shared TSV: emit one row per (gene + updated_gene) so either
        # legacy or current HUGO symbol resolves to the same arm.
        df = df[["gene", "updated_gene", "chrom_arm"]].copy()
        raw = df["chrom_arm"].astype(str).str.strip()
        stripped = raw.str.replace(r"^chr", "", regex=True)
        arm = stripped.str.extract(r"([pq])$", expand=False)
        chrom_str = stripped.str.replace(r"[pq]$", "", regex=True)
        df["chromosome"] = chrom_str
        df["chromosome_arm"] = arm
        df["chr_arm"] = chrom_str + arm.fillna("")
        blank = raw.isin({"nan", "", "None", "NaN"}) | arm.isna()
        df.loc[blank, "chr_arm"] = "unmapped"
        rows_legacy = df.rename(columns={"gene": "perturbation"})[
            ["perturbation", "chromosome", "chromosome_arm", "chr_arm"]
        ]
        rows_updated = df.rename(columns={"updated_gene": "perturbation"})[
            ["perturbation", "chromosome", "chromosome_arm", "chr_arm"]
        ]
        df = (
            pd.concat([rows_legacy, rows_updated], ignore_index=True)
            .dropna(subset=["perturbation"])
        )
        df["perturbation"] = df["perturbation"].astype(str)
    elif {"symbol", "chrom_arm"}.issubset(cols):
        # legacy chrom-arm-correction cache format: split "chr<N><p|q>".
        df = df[["symbol", "chrom_arm"]].copy()
        df["perturbation"] = df["symbol"].astype(str)
        raw = df["chrom_arm"].astype(str)
        stripped = raw.str.replace(r"^chr", "", regex=True)
        arm = stripped.str.extract(r"([pq])$", expand=False)
        chrom_str = stripped.str.replace(r"[pq]$", "", regex=True)
        df["chromosome"] = chrom_str
        df["chromosome_arm"] = arm
        df["chr_arm"] = chrom_str + arm.fillna("")
        blank = raw.isin({"nan", "", "None", "NaN"}) | arm.isna()
        df.loc[blank, "chr_arm"] = "unmapped"
    else:
        _logger.warning(
            f"  Chromosome CSV/TSV {csv_path} has none of the expected schemas "
            f"(perturbation+chromosome+chromosome_arm OR "
            f"gene+updated_gene+chrom_arm OR symbol+chrom_arm); "
            f"got columns: {sorted(cols)}"
        )
        return None
    df = df.drop_duplicates("perturbation").set_index("perturbation")
    _logger.info(
        f"  Loaded chromosome map: {len(df)} perturbations, "
        f"{df['chr_arm'].nunique()} unique chr_arm categories"
    )
    return df


def _plot_chromosome_overlay(
    coords: np.ndarray,
    perts,
    chrom_df: pd.DataFrame,
    embedding_name: str,
    out_path: Path,
    plt,
    _logger,
) -> None:
    """Scatter ``coords`` colored by ``chr_arm`` from ``chrom_df``.

    Saves both PNG and SVG. Genes missing from ``chrom_df`` are drawn in light
    grey at the back so the colored layer reads cleanly on top.
    """
    perts_arr = np.asarray(perts.values if hasattr(perts, "values") else perts).astype(str)
    chr_arm = pd.Series(perts_arr).map(chrom_df["chr_arm"]).fillna("unmapped").values

    # Order categories: numerically by chromosome (so legend is "1p, 1q, 2p, ..."),
    # putting "unmapped" last.
    def _sort_key(label: str):
        if label == "unmapped":
            return (1_000, "z")
        # split into leading number + trailing arm
        i = 0
        while i < len(label) and label[i].isdigit():
            i += 1
        try:
            chrom_num = int(label[:i]) if i > 0 else 1_000
        except ValueError:
            chrom_num = 1_000
        return (chrom_num, label[i:])

    unique_labels = sorted(set(chr_arm), key=_sort_key)
    # Build a categorical palette using tab20 + tab20b + tab20c (60 distinct
    # colors), repeating only if there are >60 categories.
    import matplotlib as _mpl
    palette = (
        list(_mpl.colormaps["tab20"].colors)
        + list(_mpl.colormaps["tab20b"].colors)
        + list(_mpl.colormaps["tab20c"].colors)
    )
    color_map = {}
    color_idx = 0
    for lbl in unique_labels:
        if lbl == "unmapped":
            color_map[lbl] = (0.75, 0.75, 0.75, 0.5)  # light grey, semi-transparent
        else:
            color_map[lbl] = palette[color_idx % len(palette)]
            color_idx += 1

    fig, ax = plt.subplots(figsize=(14, 11))
    # Draw unmapped first (under), then categories in legend order.
    # Sizes doubled from the original (18→36 / 10→20) for better readability;
    # text also doubled (13→26 axes, 14→28 title, 8→16 legend).
    for lbl in ["unmapped"] + [l for l in unique_labels if l != "unmapped"]:
        mask = chr_arm == lbl
        if not mask.any():
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=36 if lbl != "unmapped" else 20,
            c=[color_map[lbl]],
            label=lbl,
            edgecolors="none",
            alpha=0.85 if lbl != "unmapped" else 0.4,
        )
    ax.set_xlabel(f"{embedding_name}1", fontsize=26)
    ax.set_ylabel(f"{embedding_name}2", fontsize=26)
    ax.tick_params(labelsize=18)
    ax.set_title(
        f"Gene-level {embedding_name} — colored by chromosome arm",
        fontsize=28,
        fontweight="bold",
    )
    # Legend off to the right, multi-column for many categories
    n_cols = max(1, int(np.ceil(len(unique_labels) / 22)))
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=16,
        ncol=n_cols,
        frameon=False,
        title="chr_arm",
        title_fontsize=18,
    )
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    _logger.info(f"  Saved {out_path.name}.{{png,svg}} ({len(unique_labels)} categories)")


def _plot_chromosome_overlay_html(
    coords: np.ndarray,
    perts,
    chrom_df: pd.DataFrame,
    embedding_name: str,
    out_path: Path,
    _logger,
) -> None:
    """Interactive Plotly version of :func:`_plot_chromosome_overlay`.

    One trace per ``chr_arm`` category so toggling the legend hides/shows each
    chromosome arm independently. Hover shows perturbation + chromosome arm.
    Saves an HTML file at ``out_path`` (``.html`` extension applied if absent).
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        _logger.warning(f"  Chromosome HTML skipped: plotly missing ({exc})")
        return

    perts_arr = np.asarray(perts.values if hasattr(perts, "values") else perts).astype(str)
    chr_arm = pd.Series(perts_arr).map(chrom_df["chr_arm"]).fillna("unmapped").values
    chrom_lookup = chrom_df["chromosome"].astype(str).to_dict()
    arm_lookup = chrom_df["chromosome_arm"].astype(str).to_dict()

    def _sort_key(label: str):
        if label == "unmapped":
            return (1_000, "z")
        i = 0
        while i < len(label) and label[i].isdigit():
            i += 1
        try:
            chrom_num = int(label[:i]) if i > 0 else 1_000
        except ValueError:
            chrom_num = 1_000
        return (chrom_num, label[i:])

    unique_labels = sorted(set(chr_arm), key=_sort_key)

    import matplotlib as _mpl
    palette_rgb = (
        list(_mpl.colormaps["tab20"].colors)
        + list(_mpl.colormaps["tab20b"].colors)
        + list(_mpl.colormaps["tab20c"].colors)
    )

    def _rgba(rgb, a):
        r, g, b = [int(round(255 * c)) for c in rgb[:3]]
        return f"rgba({r},{g},{b},{a})"

    fig = go.Figure()
    color_idx = 0
    for lbl in unique_labels:
        mask = chr_arm == lbl
        if not mask.any():
            continue
        if lbl == "unmapped":
            color = "rgba(190,190,190,0.5)"
            size = 7
        else:
            color = _rgba(palette_rgb[color_idx % len(palette_rgb)], 0.85)
            size = 9
            color_idx += 1

        sub_perts = perts_arr[mask]
        hover = [
            f"<b>{p}</b><br>chr_arm: {lbl}"
            + (f"<br>chromosome: {chrom_lookup.get(p, '')}" if chrom_lookup.get(p) else "")
            + (f"<br>arm: {arm_lookup.get(p, '')}" if arm_lookup.get(p) else "")
            for p in sub_perts
        ]
        fig.add_trace(
            go.Scattergl(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                marker=dict(size=size, color=color, line=dict(width=0)),
                name=lbl,
                text=hover,
                hoverinfo="text",
                legendgroup=lbl,
            )
        )

    fig.update_layout(
        title=dict(
            text=f"Gene-level {embedding_name} — colored by chromosome arm",
            font=dict(size=22, family="Helvetica, Arial, sans-serif"),
        ),
        xaxis=dict(title=f"{embedding_name}1", title_font=dict(size=18)),
        yaxis=dict(title=f"{embedding_name}2", title_font=dict(size=18)),
        legend=dict(
            title=dict(text="chr_arm", font=dict(size=14)),
            font=dict(size=12),
            itemsizing="constant",
            tracegroupgap=2,
        ),
        width=1300,
        height=950,
        plot_bgcolor="white",
        margin=dict(l=70, r=220, t=80, b=70),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e5e5", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e5e5e5", zeroline=False)

    if out_path.suffix.lower() != ".html":
        out_path = out_path.with_suffix(".html")
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    _logger.info(f"  Saved {out_path.name} ({len(unique_labels)} traces)")
