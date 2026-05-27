"""Violin/box comparison of distinctiveness + EBI consistency mAPs across
all paper_v1 model variants and channel/feature subsets (16 entries total:
11 cell_dino paper_v1 channel/feature combinations + 5 alternative models —
cellprofiler, dino, dynaclr, subcell, organelle_profiler).

Each (entry, pass-type) pair becomes its own violin. Pass-type is encoded
two ways so it's unambiguous:
  - x-axis label suffix: "  (1st)" vs "  (2nd)"
  - fill color: light-fill = 1st-pass, dark-fill = 2nd-pass (same hue per
    entry, so paired columns are visually grouped)

Entries without a ``second_pca_consensus/`` directory (``phase_only``, the
two downsampled variants) only contribute a 1st-pass violin. All others
contribute both.

Reads from existing ``metrics/phenotypic_distinctiveness.csv`` and
``metrics/phenotypic_consistency_ebi.csv`` — does NOT trigger backfill.

Outputs (per figure): PNG (raster, dpi=150) + PDF + SVG. SVG/PDF text is
kept as editable ``<text>`` elements (``svg.fonttype = "none"``,
``pdf.fonttype = 42``) for Illustrator / Inkscape.

Run::

    cd /hpc/mydata/gav.sturm/ops_mono && \\
      uv run python scripts/compare_model_embedding_mAP.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Keep text as editable <text> elements in SVG/PDF (Illustrator / Inkscape friendly).
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("compare_model_embedding_mAP")

# Set by main() from --font-scale; multiplied into every inline fontsize call.
_FONT_SCALE: float = 1.0


def _fs(size: float) -> float:
    """fontsize helper: returns `size * _FONT_SCALE`."""
    return size * _FONT_SCALE

ROOT = Path(
    "/home/gav.sturm/linked_folders/icd.fast.ops/organelle_attribution/pca_optimized_v0.3"
)

# (display_label, relative_path_from_ROOT, has_2nd_pass)
SUBSETS: List[Tuple[str, str, bool]] = [
    # cell_dino paper_v1 subsets
    ("all_reporters",
        "cell_dino/zscore_per_exp/paper_v1/with_cp/with_4i/all_livecell/fixed_80%/cosine", True),
    ("with_cp_with_4i_no_phase",
        "cell_dino/zscore_per_exp/paper_v1/with_cp/with_4i/no_phase/fixed_80%/cosine", True),
    ("all_livecell",
        "cell_dino/zscore_per_exp/paper_v1/all_livecell/fixed_80%/cosine", True),
    ("phase_only",
        "cell_dino/zscore_per_exp/paper_v1/phase_only/fixed_80%/cosine", False),
    ("no_phase",
        "cell_dino/zscore_per_exp/paper_v1/no_phase/fixed_80%/cosine", True),
    ("only_cp",
        "cell_dino/zscore_per_exp/paper_v1/only_cp/all_livecell/fixed_80%/cosine", True),
    ("only_4i",
        "cell_dino/zscore_per_exp/paper_v1/only_4i/all_livecell/fixed_80%/cosine", True),
    ("validation_4exp_phase_only",
        "cell_dino/zscore_per_exp/paper_v1/validation_4exp_phase_only/phase_only/fixed_80%/cosine", True),
    ("matched_livecell_cp",
        "cell_dino/zscore_per_exp/paper_v1/matched_livecell_cp/all_livecell/fixed_80%/cosine", True),
    ("matched_livecell_cp_downsampled",
        "cell_dino/zscore_per_exp/paper_v1/matched_livecell_cp_downsampled/downsampled/fixed_80%/cosine", False),
    ("cell_painting_only_downsampled",
        "cell_dino/zscore_per_exp/paper_v1/cell_painting_only_downsampled/only_cp/downsampled/fixed_80%/cosine", False),
    # other models (paper_v1/all_livecell)
    ("cellprofiler",
        "cellprofiler/zscore_per_exp/paper_v1/all_livecell/fixed_80%/cosine", True),
    ("dino",
        "dino/zscore_per_exp/paper_v1/all_livecell/fixed_80%/cosine", True),
    ("dynaclr",
        "dynaclr/zscore_per_exp/paper_v1/all_livecell/fixed_80%/cosine", True),
    ("subcell",
        "subcell/zscore_per_exp/paper_v1/all_livecell/fixed_80%/cosine", True),
    ("organelle_profiler",
        "organelle_profiler/zscore_per_exp/paper_v1/all_livecell/fixed_80%/cosine", True),
]


def _read_map_column(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if "mean_average_precision" not in df.columns:
        raise KeyError(
            f"{csv_path}: no 'mean_average_precision' column "
            f"(found {list(df.columns)})"
        )
    vals = df["mean_average_precision"].to_numpy(dtype=float)
    return vals[np.isfinite(vals)]


def _collect():
    """Returns list of dicts: {label, subset, pass_type, dist_path, ebi_path,
    dist_vals, ebi_vals}.

    ``pass_type`` is one of:
      - ``"1st"``       — root-of-subset metrics
      - ``"2nd"``       — ``second_pca_consensus/`` (ratio-sweep, historical)
      - ``"2nd-MMAP"``  — ``second_pca_consensus_MEANMAP/`` (mean-mAP sweep, new)

    The 2nd-MMAP variant is included for any subset where the MEANMAP dir
    exists on disk; otherwise it's silently skipped so legacy subsets with
    only the ratio output still work.
    """
    rows = []
    missing = []
    for subset_label, rel, has_2nd in SUBSETS:
        base = ROOT / rel
        # 1st-pass always; both 2nd-pass variants if their dirs exist.
        passes = [("1st", base)]
        if has_2nd:
            passes.append(("2nd", base / "second_pca_consensus"))
            mmap_dir = base / "second_pca_consensus_MEANMAP"
            if mmap_dir.exists():
                passes.append(("2nd-MMAP", mmap_dir))
        for pass_type, dir_path in passes:
            dist_csv = dir_path / "metrics" / "phenotypic_distinctiveness.csv"
            ebi_csv = dir_path / "metrics" / "phenotypic_consistency_ebi.csv"
            if not dist_csv.exists() or not ebi_csv.exists():
                missing.append((subset_label, pass_type, str(dir_path)))
                log.warning(
                    f"missing CSV for {subset_label}/{pass_type}: "
                    f"dist={dist_csv.exists()}, ebi={ebi_csv.exists()}"
                )
                continue
            try:
                dist_vals = _read_map_column(dist_csv)
                ebi_vals = _read_map_column(ebi_csv)
            except Exception as exc:
                log.warning(f"read failed {subset_label}/{pass_type}: {exc}")
                continue
            rows.append({
                "subset": subset_label,
                "pass_type": pass_type,
                "label": f"{subset_label}  ({pass_type})",
                "dist_path": str(dist_csv),
                "ebi_path": str(ebi_csv),
                "dist_vals": dist_vals,
                "ebi_vals": ebi_vals,
                "mean_dist": float(np.mean(dist_vals)),
                "mean_ebi": float(np.mean(ebi_vals)),
            })
    if missing:
        log.warning(f"{len(missing)} (subset, pass) groups missing CSVs — proceeding without them")
    return rows


def _violin(ax, labels, data, ylabel, title, face_colors, edge_colors):
    parts = ax.violinplot(data, showmeans=False, showmedians=False,
                          showextrema=False, widths=0.85)
    for body, fc, ec in zip(parts["bodies"], face_colors, edge_colors):
        body.set_facecolor(fc)
        body.set_edgecolor(ec)
        body.set_alpha(0.85)
        body.set_linewidth(1.5)

    xs = np.arange(1, len(data) + 1)
    qs = np.array([np.percentile(d, [25, 50, 75]) for d in data])
    means = np.array([float(np.mean(d)) for d in data])
    for x, q25, q50, q75 in zip(xs, qs[:, 0], qs[:, 1], qs[:, 2]):
        ax.plot([x - 0.18, x + 0.18], [q50, q50], color="black", linewidth=2.5, zorder=10)
        ax.plot([x, x], [q25, q75], color="black", linewidth=4, alpha=0.45,
                zorder=9, solid_capstyle="butt")
    ax.scatter(xs, means, marker="D", s=70, color="firebrick",
               edgecolor="white", linewidth=1.2, zorder=11)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=_fs(14), rotation=45, ha="right")
    ax.set_ylabel(ylabel, fontsize=_fs(22))
    ax.set_title(title, fontsize=_fs(22), pad=20)
    ax.tick_params(axis="y", labelsize=_fs(16))
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def _palette_per_subset(rows):
    """Assign each unique subset a hue (sampled from viridis), then light fill
    for 1st-pass and darker fill for 2nd-pass within that hue."""
    from matplotlib.colors import to_rgba
    unique_subsets = []
    for r in rows:
        if r["subset"] not in unique_subsets:
            unique_subsets.append(r["subset"])
    # Sample viridis evenly across the N unique subsets.
    n = max(len(unique_subsets), 1)
    cmap_pts = np.linspace(0.05, 0.95, n)
    hue_for_subset = {
        s: plt.cm.viridis(cmap_pts[i]) for i, s in enumerate(unique_subsets)
    }
    face_colors, edge_colors = [], []
    for r in rows:
        hue = hue_for_subset[r["subset"]]
        if r["pass_type"] == "1st":
            # lighter fill for 1st-pass
            rgba = to_rgba(hue, alpha=0.45)
        else:
            rgba = to_rgba(hue, alpha=0.95)
        face_colors.append(rgba)
        edge_colors.append("black")
    return face_colors, edge_colors


def _plot(rows, out_path: Path, sort_by: str = "ebi"):
    if sort_by == "ebi":
        rows = sorted(rows, key=lambda r: r["mean_ebi"])
    elif sort_by == "dist":
        rows = sorted(rows, key=lambda r: r["mean_dist"])
    elif sort_by == "subset":
        # Preserve original (subset,pass) interleave
        rows = sorted(rows, key=lambda r: (r["subset"], r["pass_type"]))

    labels = [r["label"] for r in rows]
    dist_vals = [r["dist_vals"] for r in rows]
    ebi_vals = [r["ebi_vals"] for r in rows]
    face_colors, edge_colors = _palette_per_subset(rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(32, len(rows) * 1.2), 18))
    n_dist = max(len(v) for v in dist_vals) if dist_vals else 0
    n_ebi = max(len(v) for v in ebi_vals) if ebi_vals else 0
    single_pass = len({r["pass_type"] for r in rows}) <= 1
    pass_blurb = "" if single_pass else "  ·  light = 1st-pass, dark = 2nd-pass"
    _violin(ax1, labels, dist_vals, "geneKO mAP",
            f"geneKO distinctiveness mAP (n={n_dist:,}){pass_blurb}",
            face_colors, edge_colors)
    _violin(ax2, labels, ebi_vals, "EBI mAP",
            f"EBI protein complex consistency mAP (n={n_ebi:,}){pass_blurb}",
            face_colors, edge_colors)

    # Legend — pass-type swatches dropped when only one pass is shown.
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_handles = []
    if not single_pass:
        legend_handles.extend([
            Patch(facecolor=(0.5, 0.5, 0.5, 0.45), edgecolor="black",
                  label="1st-pass (no 2nd PCA)"),
            Patch(facecolor=(0.5, 0.5, 0.5, 0.95), edgecolor="black",
                  label="2nd-pass PCA"),
        ])
    legend_handles.extend([
        Line2D([0], [0], color="black", linewidth=4, label="median"),
        Line2D([0], [0], marker="D", color="white", markerfacecolor="firebrick",
               markeredgecolor="white", markeredgewidth=1.2,
               markersize=_fs(14), linestyle="None", label="mean"),
    ])
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    fig.legend(handles=legend_handles, loc="center left",
               bbox_to_anchor=(0.89, 0.5), frameon=True, fontsize=_fs(20),
               handlelength=2.5, borderpad=1.0, labelspacing=1.2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info(f"saved {out_path}")
    if out_path.suffix.lower() == ".png":
        pdf = out_path.with_suffix(".pdf")
        fig.savefig(pdf, bbox_inches="tight")
        log.info(f"saved {pdf}")
        svg = out_path.with_suffix(".svg")
        fig.savefig(svg, bbox_inches="tight")
        log.info(f"saved {svg}")
    plt.close(fig)


# Named subset-comparison groups. Each value is a list of
# (display_label, subset_key, pass_type) tuples — the script will pull the
# matching row from _collect() and put each in one subplot.
SUBSET_COMPARISONS: dict[str, list[tuple[str, str, str]]] = {
    "CP vs matched livecell CP": [
        ("cell_painting_only (1st)", "only_cp", "1st"),
        ("cell_painting_only (2nd)", "only_cp", "2nd"),
        ("matched_livecell_cp (1st)", "matched_livecell_cp", "1st"),
        ("matched_livecell_cp (2nd)", "matched_livecell_cp", "2nd"),
    ],
    "CP vs matched livecell CP (downsampled)": [
        ("cell_painting_only_downsampled", "cell_painting_only_downsampled", "1st"),
        ("matched_livecell_cp_downsampled", "matched_livecell_cp_downsampled", "1st"),
    ],
    "Channel subsets (best pass per subset)": [
        ("phase_only", "phase_only", "1st"),
        ("all_reporters (2nd)", "all_reporters", "2nd"),
        ("all_livecell (2nd)", "all_livecell", "2nd"),
        ("all_fluorescent — with_cp/with_4i/no_phase (2nd)",
            "with_cp_with_4i_no_phase", "2nd"),
        ("all_livecell_fluorescent — no_phase (2nd)", "no_phase", "2nd"),
    ],
}


def _row_for(rows, subset: str, pass_type: str):
    for r in rows:
        if r["subset"] == subset and r["pass_type"] == pass_type:
            return r
    return None


def _plot_subset_comparisons(rows, out_path: Path) -> None:
    """Render one figure with N subplots (one per SUBSET_COMPARISONS group).
    Each subplot is a 2-panel (distinctiveness + EBI) bar chart with values
    annotated on top.
    """
    groups = list(SUBSET_COMPARISONS.items())
    n_groups = len(groups)
    fig, axes = plt.subplots(n_groups, 2, figsize=(20, 5.5 * n_groups))
    if n_groups == 1:
        axes = np.array([axes])

    for gi, (title, members) in enumerate(groups):
        ax1, ax2 = axes[gi, 0], axes[gi, 1]
        labels, dist_means, ebi_means = [], [], []
        miss = []
        for label, subset, pass_type in members:
            r = _row_for(rows, subset, pass_type)
            if r is None:
                miss.append((subset, pass_type))
                continue
            labels.append(label)
            dist_means.append(r["mean_dist"])
            ebi_means.append(r["mean_ebi"])
        if miss:
            log.warning(f"[{title}] skipped (missing data): {miss}")
        if not labels:
            for ax in (ax1, ax2):
                ax.text(0.5, 0.5, f"no data for: {title}",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=14, color="gray")
                ax.axis("off")
            continue
        palette = plt.cm.Set2.colors[: len(labels)]
        _bar(ax1, labels, dist_means, "mean geneKO mAP",
             f"{title} · distinctiveness",
             list(palette), ["black"] * len(palette))
        _bar(ax2, labels, ebi_means, "mean EBI mAP",
             f"{title} · EBI consistency",
             list(palette), ["black"] * len(palette))

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info(f"saved {out_path}")
    if out_path.suffix.lower() == ".png":
        pdf = out_path.with_suffix(".pdf")
        fig.savefig(pdf, bbox_inches="tight")
        log.info(f"saved {pdf}")
        svg = out_path.with_suffix(".svg")
        fig.savefig(svg, bbox_inches="tight")
        log.info(f"saved {svg}")
    plt.close(fig)


def _bar(ax, labels, means, ylabel, title, face_colors, edge_colors):
    xs = np.arange(len(labels))
    bars = ax.bar(xs, means, color=face_colors, edgecolor=edge_colors, linewidth=1.2)
    # Annotate the mean value on top of each bar.
    for x, m in zip(xs, means):
        ax.text(
            x, m + max(means) * 0.01,
            f"{m:.3f}",
            ha="center", va="bottom",
            fontsize=_fs(12), color="black", weight="bold",
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=_fs(14), rotation=45, ha="right")
    ax.set_ylabel(ylabel, fontsize=_fs(22))
    ax.set_title(title, fontsize=_fs(22), pad=20)
    ax.tick_params(axis="y", labelsize=_fs(16))
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_ylim(0, max(means) * 1.12)


def _plot_bar(rows, out_path: Path, sort_by: str = "ebi"):
    # Stable subset → hue map computed from the FULL row set so the same
    # subset always gets the same color regardless of which panel sorts it
    # where.
    from matplotlib.colors import to_rgba
    unique_subsets = sorted({r["subset"] for r in rows})
    n = max(len(unique_subsets), 1)
    cmap_pts = np.linspace(0.05, 0.95, n)
    hue_for_subset = {s: plt.cm.viridis(cmap_pts[i]) for i, s in enumerate(unique_subsets)}

    def colors_for(rows_local):
        face = [to_rgba(hue_for_subset[r["subset"]],
                         alpha=0.45 if r["pass_type"] == "1st" else 0.95)
                for r in rows_local]
        edge = ["black"] * len(rows_local)
        return face, edge

    # Each panel sorts INDEPENDENTLY by its own metric (lowest → highest) so
    # both columns read coherently. ``--sort-by subset`` opts back into a
    # shared alphabetical order across both panels.
    if sort_by == "subset":
        rows_dist = sorted(rows, key=lambda r: (r["subset"], r["pass_type"]))
        rows_ebi = rows_dist
    else:
        rows_dist = sorted(rows, key=lambda r: r["mean_dist"])
        rows_ebi = sorted(rows, key=lambda r: r["mean_ebi"])

    dist_labels = [r["label"] for r in rows_dist]
    dist_means = [r["mean_dist"] for r in rows_dist]
    dist_face, dist_edge = colors_for(rows_dist)

    ebi_labels = [r["label"] for r in rows_ebi]
    ebi_means = [r["mean_ebi"] for r in rows_ebi]
    ebi_face, ebi_edge = colors_for(rows_ebi)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(32, len(rows) * 1.2), 14))
    single_pass = len({r["pass_type"] for r in rows}) <= 1
    pass_blurb = "" if single_pass else "  ·  light = 1st-pass, dark = 2nd-pass"
    _bar(ax1, dist_labels, dist_means, "mean geneKO mAP",
         f"Mean geneKO distinctiveness{pass_blurb}",
         dist_face, dist_edge)
    _bar(ax2, ebi_labels, ebi_means, "mean EBI mAP",
         f"Mean EBI protein complex consistency{pass_blurb}",
         ebi_face, ebi_edge)

    fig.tight_layout(rect=[0, 0, 0.88, 1])
    if not single_pass:
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor=(0.5, 0.5, 0.5, 0.45), edgecolor="black",
                  label="1st-pass (no 2nd PCA)"),
            Patch(facecolor=(0.5, 0.5, 0.5, 0.95), edgecolor="black",
                  label="2nd-pass PCA"),
        ]
        fig.legend(handles=legend_handles, loc="center left",
                   bbox_to_anchor=(0.89, 0.5), frameon=True, fontsize=_fs(20),
                   handlelength=2.5, borderpad=1.0, labelspacing=1.2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info(f"saved {out_path}")
    if out_path.suffix.lower() == ".png":
        pdf = out_path.with_suffix(".pdf")
        fig.savefig(pdf, bbox_inches="tight")
        log.info(f"saved {pdf}")
        svg = out_path.with_suffix(".svg")
        fig.savefig(svg, bbox_inches="tight")
        log.info(f"saved {svg}")
    plt.close(fig)


def _plot_meanmap_compare(rows, out_path: Path) -> None:
    """Before/after comparison: ratio-sweep 2nd-pass vs mean-mAP-sweep 2nd-pass.

    For each subset that has BOTH ``second_pca_consensus/`` (pass_type "2nd")
    AND ``second_pca_consensus_MEANMAP/`` (pass_type "2nd-MMAP"), renders a
    paired bar comparison + a delta panel. Lets you see at a glance which
    embeddings shift (and in what direction) when the picker swaps from
    ratio to mean-mAP scoring.

    Two-row, two-column layout:
      row 1: distinctiveness mean mAP    | EBI mean mAP
      row 2: distinctiveness Δ (mmap-ratio) | EBI Δ (mmap-ratio)

    Bars sorted by subset alphabetical so each invocation is reproducible.
    """
    by_subset: dict[str, dict[str, dict]] = {}
    for r in rows:
        if r["pass_type"] not in ("2nd", "2nd-MMAP"):
            continue
        by_subset.setdefault(r["subset"], {})[r["pass_type"]] = r
    # Only keep subsets that have BOTH variants
    paired = [(sub, d["2nd"], d["2nd-MMAP"])
              for sub, d in by_subset.items()
              if "2nd" in d and "2nd-MMAP" in d]
    if not paired:
        log.warning(
            "_plot_meanmap_compare: no subsets have BOTH second_pca_consensus/ "
            "and second_pca_consensus_MEANMAP/ — skipping plot. Run the "
            "mean-mAP sweep first for at least one subset."
        )
        return
    # Each column sorts INDEPENDENTLY by its own ratio-sweep mean, ascending.
    # Distinctiveness column (dist bars + dist delta) uses one order; EBI
    # column uses its own — so both columns read "lowest → highest" on
    # their respective metric.
    dist_sorted = sorted(paired, key=lambda t: t[1]["mean_dist"])
    ebi_sorted = sorted(paired, key=lambda t: t[1]["mean_ebi"])

    dist_labels = [t[0] for t in dist_sorted]
    dist_ratio = np.array([t[1]["mean_dist"] for t in dist_sorted])
    dist_mmap = np.array([t[2]["mean_dist"] for t in dist_sorted])
    dist_delta = dist_mmap - dist_ratio

    ebi_labels = [t[0] for t in ebi_sorted]
    ebi_ratio = np.array([t[1]["mean_ebi"] for t in ebi_sorted])
    ebi_mmap = np.array([t[2]["mean_ebi"] for t in ebi_sorted])
    ebi_delta = ebi_mmap - ebi_ratio

    width = 0.38

    fig, axes = plt.subplots(2, 2, figsize=(max(20, len(paired) * 1.6), 12),
                             gridspec_kw={"height_ratios": [3, 2]})
    ax_dist, ax_ebi = axes[0]
    ax_dist_delta, ax_ebi_delta = axes[1]

    # --- Top row: paired bars ---
    def _paired_bars(ax, labels_local, ratio_vals, mmap_vals, title, ylabel):
        xs_local = np.arange(len(labels_local))
        ax.bar(xs_local - width / 2, ratio_vals, width,
               color="#9ecae1", edgecolor="black", linewidth=1.0,
               label="ratio sweep (existing)")
        ax.bar(xs_local + width / 2, mmap_vals, width,
               color="#08519c", edgecolor="black", linewidth=1.0,
               label="mean_map sweep (new)")
        for x, m in zip(xs_local - width / 2, ratio_vals):
            ax.text(x, m, f"{m:.3f}", ha="center", va="bottom",
                    fontsize=_fs(10), color="black")
        for x, m in zip(xs_local + width / 2, mmap_vals):
            ax.text(x, m, f"{m:.3f}", ha="center", va="bottom",
                    fontsize=_fs(10), color="black", weight="bold")
        ax.set_xticks(xs_local)
        ax.set_xticklabels(labels_local, rotation=45, ha="right", fontsize=_fs(12))
        ax.set_ylabel(ylabel, fontsize=_fs(16))
        ax.set_title(title, fontsize=_fs(18), pad=18)
        ax.tick_params(axis="y", labelsize=_fs(12))
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ymax = max(ratio_vals.max(), mmap_vals.max()) * 1.15
        ax.set_ylim(0, ymax)
        ax.legend(loc="upper left", fontsize=_fs(11))

    _paired_bars(ax_dist, dist_labels, dist_ratio, dist_mmap,
                 "Distinctiveness (2nd-pass): ratio vs mean_map sweep",
                 "mean per-gene mAP")
    _paired_bars(ax_ebi, ebi_labels, ebi_ratio, ebi_mmap,
                 "EBI consistency (2nd-pass): ratio vs mean_map sweep",
                 "mean per-complex mAP")

    # --- Bottom row: deltas (green=rose, red=fell) ---
    # Each delta panel mirrors its top-row sibling's ordering so the column
    # reads coherently.
    def _delta_bars(ax, labels_local, deltas, title, ylabel):
        xs_local = np.arange(len(labels_local))
        colors = ["#2ca02c" if d > 0 else "#d62728" for d in deltas]
        ax.bar(xs_local, deltas, color=colors, edgecolor="black", linewidth=1.0)
        for x, d in zip(xs_local, deltas):
            va = "bottom" if d >= 0 else "top"
            ax.text(x, d, f"{d:+.3f}",
                    ha="center", va=va, fontsize=_fs(11), weight="bold")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(xs_local)
        ax.set_xticklabels(labels_local, rotation=45, ha="right", fontsize=_fs(12))
        ax.set_ylabel(ylabel, fontsize=_fs(16))
        ax.set_title(title, fontsize=_fs(16), pad=14)
        ax.tick_params(axis="y", labelsize=_fs(12))
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        m = max(abs(deltas.min()), abs(deltas.max()), 0.02) * 1.25
        ax.set_ylim(-m, m)

    _delta_bars(ax_dist_delta, dist_labels, dist_delta,
                "Δ distinctiveness (mean_map − ratio)",
                "Δ mean mAP")
    _delta_bars(ax_ebi_delta, ebi_labels, ebi_delta,
                "Δ EBI consistency (mean_map − ratio)",
                "Δ mean mAP")

    fig.suptitle(
        f"Effect of switching 2nd-pass sweep metric from ratio → mean_map  "
        f"·  {len(paired)} subsets",
        fontsize=_fs(20), fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info(f"saved {out_path}")
    if out_path.suffix.lower() == ".png":
        pdf = out_path.with_suffix(".pdf")
        fig.savefig(pdf, bbox_inches="tight")
        log.info(f"saved {pdf}")
        svg = out_path.with_suffix(".svg")
        fig.savefig(svg, bbox_inches="tight")
        log.info(f"saved {svg}")
    plt.close(fig)


def _write_summary_csv(rows, csv_path: Path):
    df = pd.DataFrame([
        {
            "subset": r["subset"],
            "pass_type": r["pass_type"],
            "label": r["label"],
            "n_dist": int(len(r["dist_vals"])),
            "mean_dist_mAP": r["mean_dist"],
            "n_ebi": int(len(r["ebi_vals"])),
            "mean_ebi_mAP": r["mean_ebi"],
            "dist_csv": r["dist_path"],
            "ebi_csv": r["ebi_path"],
        }
        for r in rows
    ])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    log.info(f"saved {csv_path}  ({len(df)} rows)")
    return df


def _write_manifest(rows, yaml_path: Path):
    manifest = {
        "root": str(ROOT),
        "n_groups": len(rows),
        "groups": [
            {
                "subset": r["subset"],
                "pass_type": r["pass_type"],
                "dist_csv": r["dist_path"],
                "ebi_csv": r["ebi_path"],
                "n_dist": int(len(r["dist_vals"])),
                "n_ebi": int(len(r["ebi_vals"])),
                "mean_dist_mAP": r["mean_dist"],
                "mean_ebi_mAP": r["mean_ebi"],
            }
            for r in rows
        ],
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, default_flow_style=False)
    log.info(f"saved {yaml_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path,
                   default=ROOT / "violin_mAP_across_paper_v1.png",
                   help="Violin figure path (PDF mirrored alongside).")
    p.add_argument("--bar-out", type=Path,
                   default=ROOT / "bar_mAP_across_paper_v1.png",
                   help="Simple bar-plot of means with values annotated on top "
                        "(PDF mirrored alongside).")
    p.add_argument("--subset-comparisons-out", type=Path,
                   default=ROOT / "subset_comparisons_mAP.png",
                   help="Multi-panel bar plot grouping subsets into thematic "
                        "comparisons (CP vs matched, downsampled pair, "
                        "channel-subset family).")
    p.add_argument("--meanmap-compare-out", type=Path,
                   default=ROOT / "meanmap_vs_ratio_compare.png",
                   help="Before/after summary: ratio-sweep vs mean_map-sweep "
                        "2nd-pass for every subset where both runs exist on "
                        "disk. Shows paired bars + Δ panel, color-coded by "
                        "rise/fall.")
    p.add_argument("--csv-out", type=Path,
                   default=ROOT / "mAP_across_paper_v1.csv",
                   help="Per-(subset, pass) summary stats CSV.")
    p.add_argument("--yaml-out", type=Path,
                   default=ROOT / "mAP_across_paper_v1.yaml",
                   help="Manifest YAML with source CSV paths per group.")
    p.add_argument("--sort-by", type=str, default="ebi",
                   choices=["ebi", "dist", "subset"],
                   help="Sort columns left→right by mean EBI mAP (default), "
                        "mean dist mAP, or by subset name + pass_type.")
    p.add_argument("--filter", type=str, nargs="*", default=None,
                   help="If set, only render the listed subset labels (matches "
                        "the first element of each SUBSETS tuple). When passed, "
                        "the multi-panel subset-comparisons figure is skipped "
                        "and output filenames get a '_filtered' suffix.")
    p.add_argument("--filter-tag", type=str, default="filtered",
                   help="Suffix appended to output filenames when --filter is "
                        "used (e.g. 'models_only' → "
                        "violin_mAP_across_paper_v1_models_only.png).")
    p.add_argument("--filter-pass", type=str, default=None,
                   choices=["1st", "2nd"],
                   help="When --filter is set, restrict to this pass type only "
                        "(default: keep both 1st and 2nd pass of each entry). "
                        "When set, the '  (1st)' / '  (2nd)' label suffix is "
                        "dropped from each entry since all entries share the "
                        "same pass.")
    p.add_argument("--rename", type=str, nargs="*", default=None,
                   help="Display-label remaps of the form OLD=NEW (space-"
                        "separated for multiple). Useful when --filter is "
                        "active and a SUBSETS key is misleading in context "
                        "(e.g. 'all_livecell=cell_dino' for the models-only "
                        "comparison).")
    p.add_argument("--font-scale", type=float, default=1.0,
                   help="Multiplier applied to every inline fontsize in the "
                        "figures (default 1.0). Pass 2.0 to make all text 2× "
                        "bigger.")
    args = p.parse_args()

    global _FONT_SCALE
    _FONT_SCALE = args.font_scale

    rows = _collect()
    if not rows:
        log.error("No subsets with both CSVs available — nothing to plot.")
        sys.exit(1)
    log.info(f"collected {len(rows)} (subset, pass) groups")

    if args.filter:
        keep = set(args.filter)
        rows = [r for r in rows if r["subset"] in keep]
        if args.filter_pass is not None:
            rows = [r for r in rows if r["pass_type"] == args.filter_pass]
        if not rows:
            log.error(
                f"--filter matched no rows. Requested: {sorted(keep)}; "
                f"pass: {args.filter_pass}. Check SUBSETS labels."
            )
            sys.exit(1)

        rename_map: dict[str, str] = {}
        if args.rename:
            for pair in args.rename:
                if "=" not in pair:
                    log.error(f"--rename entry '{pair}' missing '=' separator.")
                    sys.exit(1)
                old, new = pair.split("=", 1)
                rename_map[old.strip()] = new.strip()

        for r in rows:
            new_subset = rename_map.get(r["subset"], r["subset"])
            r["subset"] = new_subset
            if args.filter_pass is not None:
                r["label"] = new_subset
            else:
                r["label"] = f"{new_subset}  ({r['pass_type']})"

        log.info(
            f"after --filter ({args.filter_tag}): {len(rows)} rows remain "
            f"(labels: {[r['label'] for r in rows]})"
        )
        suf = f"_{args.filter_tag}"

        def _suffix(path: Path) -> Path:
            return path.with_name(f"{path.stem}{suf}{path.suffix}")

        out = _suffix(args.out)
        bar_out = _suffix(args.bar_out)
        csv_out = _suffix(args.csv_out)
        yaml_out = _suffix(args.yaml_out)
    else:
        out = args.out
        bar_out = args.bar_out
        csv_out = args.csv_out
        yaml_out = args.yaml_out

    _write_summary_csv(rows, csv_out)
    _write_manifest(rows, yaml_out)
    _plot(rows, out, sort_by=args.sort_by)
    _plot_bar(rows, bar_out, sort_by=args.sort_by)
    if not args.filter:
        _plot_subset_comparisons(rows, args.subset_comparisons_out)
        _plot_meanmap_compare(rows, args.meanmap_compare_out)


if __name__ == "__main__":
    main()
