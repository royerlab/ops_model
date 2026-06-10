"""Plot attention-model classification accuracy vs cell count.

Two side-by-side panels (gene-KO | CHAD complex), each showing top-1
accuracy as a function of the per-gene cell-count threshold for both
phase and fluor classifiers. Mean line across genes + per-gene grey
points. Optional top-5 line as a dashed companion.

Inputs are the `cdino_eval_*.csv` files produced by Alex's per-class
attention evaluation:
  cdino_eval_phase_50.csv          gene × phase
  cdino_eval_fluorescent_50.csv    gene × fluor
  cdino_eval_phase_chad_50.csv     complex × phase
  cdino_eval_fluorescent_chad_50.csv  complex × fluor

Defaults point at `/hpc/projects/icd.fast.ops/.../cdino_eval/`. Pass
`--gene-phase-csv` etc. to override individual files; pass empty string
to disable any specific curve.

For each (top-1 / top-5) × (linear / log x-axis) variant the script
emits TWO plots: one with median + IQR (Q1–Q3) bands and one with
mean ± SEM bands, suffixed `_median-iqr` / `_mean-sem`.

Output: PDF/PNG/SVG triplets + a summary CSV with mean/std/SEM and
median/Q1/Q3 per (level, modality, n_cells).

Usage:
    python plot_eval_accuracy_curves.py
    python plot_eval_accuracy_curves.py --output /tmp/eval_curves.pdf
    python plot_eval_accuracy_curves.py --top5    # add top-5 dashed lines
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Keep text editable in Illustrator (SVG keeps <text> elements rather
# than path-tracing the glyphs; PDF uses TrueType instead of Type-3).
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

DEFAULT_DIR = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v3/attention_v3/cdino"
)
DEFAULT_OUTPUT_DIR = Path(
    "/hpc/projects/icd.fast.ops/analysis/attention_accuracy"
)
DEFAULTS = {
    "gene_phase":   DEFAULT_DIR / "cdino_eval_phase_50.csv",
    "gene_fluor":   DEFAULT_DIR / "cdino_eval_fluorescent_50.csv",
    "chad_phase":   DEFAULT_DIR / "cdino_eval_phase_chad_50.csv",
    "chad_fluor":   DEFAULT_DIR / "cdino_eval_fluorescent_chad_50.csv",
}
COLOR = {"phase": "#3A3A3A", "fluor": "#1F9B4A"}
LABEL = {"phase": "Phase classifier", "fluor": "Fluorescence classifier"}


def _load(path: Path | str | None) -> pd.DataFrame | None:
    """Load + normalize. Different eval CSVs use different identifier
    column names — pick the most specific one available and rename to
    `class_name` for downstream uniformity.

    Column priorities (most specific wins):
      label_name  → CHAD complex name (set on chad CSVs).
      gene_name   → gene symbol.
      class_name  → already-normalized name.
    """
    if path is None or str(path) == "":
        return None
    p = Path(path)
    if not p.exists():
        print(f"  [warn] missing: {p}")
        return None
    df = pd.read_csv(p)
    if "class_name" not in df.columns:
        for cand in ("label_name", "gene_name"):
            if cand in df.columns:
                df = df.rename(columns={cand: "class_name"})
                break
    need = {"n_cells", "top1_acc", "class_name"}
    miss = need - set(df.columns)
    if miss:
        print(f"  [warn] {p.name}: missing columns {miss}; skipping")
        return None
    return df


def _center_and_band(series_grouped, band: str):
    """Return (center, lo, hi) for the given pandas GroupBy.

    band="iqr"  → center=median, lo/hi = Q1/Q3
    band="sem"  → center=mean,   lo/hi = mean ∓ SEM (std / sqrt(n))
    """
    if band == "iqr":
        center = series_grouped.median()
        lo = series_grouped.quantile(0.25)
        hi = series_grouped.quantile(0.75)
    elif band == "sem":
        center = series_grouped.mean()
        sem = series_grouped.sem().fillna(0.0)
        lo = center - sem
        hi = center + sem
    else:
        raise ValueError(f"unknown band mode: {band!r}")
    return center, lo, hi


def _plot_one_panel(ax, gene_or_complex_phase_df: pd.DataFrame | None,
                      gene_or_complex_fluor_df: pd.DataFrame | None,
                      title: str, show_top5: bool = False,
                      xscale: str = "linear", band: str = "iqr"):
    """Phase + fluor curves on one axes.

    band="iqr" → median line + IQR (Q1–Q3) band per modality.
    band="sem" → mean line ± SEM band per modality.
    Top-5 is shown as a dashed companion when `show_top5`.
    """
    plotted_any = False
    all_x: set[int] = set()
    # Per-modality lateral nudge for the %-annotations: phase sits up-left,
    # fluor sits up-right of its point. Without this offset both modalities
    # stacked their labels on the same x and collided into one unreadable
    # blob — especially at the dense 10/20/50 bins.
    #
    # Log axis: points are evenly spaced in log space so adjacent ticks
    # are roughly equidistant in screen units — we don't need as much
    # nudge to avoid collisions, and large offsets push labels off the
    # bar in either direction.
    if xscale == "log":
        # Log scale: phase / fluor differ enough in y to avoid lateral
        # collisions, so labels stay centered over their x-tick. With
        # the bumped 22pt / 20pt annotation fonts the labels need more
        # vertical headroom than the linear panel needed at 16/15pt —
        # otherwise the text bottom clips into the marker.
        LATERAL_NUDGE = {"phase": 0, "fluor": 0}
        DY_TOP1 = 12
        DY_TOP5 = 6
        HA_FOR = {"phase": "center", "fluor": "center"}
    else:
        LATERAL_NUDGE = {"phase": -10, "fluor": 10}
        DY_TOP1 = 14
        DY_TOP5 = 4
        HA_FOR = {"phase": "right", "fluor": "left"}
    for modality, df in (("phase", gene_or_complex_phase_df),
                          ("fluor", gene_or_complex_fluor_df)):
        if df is None or df.empty:
            continue
        color = COLOR[modality]
        grouped = df.groupby("n_cells")["top1_acc"]
        center, lo_s, hi_s = _center_and_band(grouped, band)
        xs = np.asarray(center.index, dtype=float)
        ys = np.asarray(center.values, dtype=float)
        lo = np.asarray(lo_s.values, dtype=float)
        hi = np.asarray(hi_s.values, dtype=float)
        all_x.update(int(n) for n in center.index)

        # Faint band — IQR (Q1–Q3) or ±SEM, depending on `band`.
        ax.fill_between(xs, lo, hi, color=color, alpha=0.07,
                         linewidth=0, zorder=2)
        # Center line (median or mean) with point markers per n_cells bin.
        ax.plot(xs, ys, "-o", color=color, linewidth=4.0,
                 markersize=12, markeredgecolor="black",
                 markeredgewidth=1.0, zorder=3,
                 label=LABEL[modality])

        if show_top5 and "top5_acc" in df.columns:
            t5_grouped = df.groupby("n_cells")["top5_acc"]
            t5_center, t5_lo_s, t5_hi_s = _center_and_band(t5_grouped, band)
            t5_xs = np.asarray(t5_center.index, dtype=float)
            t5_ys = np.asarray(t5_center.values, dtype=float)
            t5_lo = np.asarray(t5_lo_s.values, dtype=float)
            t5_hi = np.asarray(t5_hi_s.values, dtype=float)
            all_x.update(int(n) for n in t5_center.index)
            ax.fill_between(t5_xs, t5_lo, t5_hi,
                             color=color, alpha=0.04, linewidth=0, zorder=2)
            ax.plot(t5_xs, t5_ys, "--s", color=color, linewidth=3.2,
                     markersize=10, markeredgecolor="black",
                     markeredgewidth=0.8, alpha=0.85, zorder=3,
                     label=f"{LABEL[modality]} (top-5)")
        plotted_any = True

    if not plotted_any:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                 ha="center", va="center", fontsize=22, color="gray",
                 style="italic")
    ax.axhline(0.0, color="gray", linewidth=0.5, alpha=0.4)
    xlabel = r"Cells/class ($\log_{10}$)" if xscale == "log" else "Cells/class"
    ax.set_xlabel(xlabel, fontsize=28)
    ax.set_ylabel("Accuracy", fontsize=28)
    ax.set_title(title, fontsize=32, fontweight="bold")
    ax.tick_params(axis="both", labelsize=24)
    # Fixed 0%–100% y-axis with tick at each 10% — the per-point %
    # annotations are gone, so the grid+axis labels now carry the
    # "how much accuracy at this cell count" read.
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    # Tiny headroom above 1.0 so the markers at ~98–100% don't clip
    # into the axis frame. Tick max stays at 100% via MultipleLocator.
    ax.set_ylim(0.0, 1.03)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v*100))}%"))
    ax.grid(True, axis="y", alpha=0.25)
    # Explicit ticks at the actual cell-count slices — readers see
    # "this number of cells, this accuracy" directly. Labels rotated so
    # adjacent tick texts don't overlap at the dense low-count bins
    # (10/20/50 sit ~10 px apart on a linear axis).
    if all_x:
        ticks = sorted(all_x)
        # On log, set the scale FIRST so the LogLocator+LogFormatter are
        # installed; then override with our fixed ticks + ScalarFormatter
        # (plain integer labels). Without this matplotlib renders the
        # default log decade ticks as `10²`/`10³` scientific notation
        # AND silently drops some of our requested ticks because the
        # major locator is the LogLocator's.
        if xscale == "log":
            from matplotlib.ticker import (
                FixedLocator, NullLocator, ScalarFormatter,
            )
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            # Suppress the LogLocator's auto minor ticks (30, 40, 60, …)
            # so only our 9 bins get labels on screen.
            ax.xaxis.set_minor_locator(NullLocator())
            fmt = ScalarFormatter()
            fmt.set_scientific(False)
            ax.xaxis.set_major_formatter(fmt)
            ax.set_xlim(min(ticks) * 0.7, max(ticks) * 1.3)
        else:
            ax.set_xticks(ticks)
            ax.set_xlim(min(ticks) - max(ticks) * 0.03,
                         max(ticks) + max(ticks) * 0.03)
        # X-tick label rotation: linear panel packs 9 ticks tightly so
        # labels overlap when horizontal — rotate 35°. Log panel
        # distributes the same ticks evenly across decades with plenty
        # of horizontal room, so keep them flat for legibility.
        rotation = 0 if xscale == "log" else 35
        ha       = "center" if xscale == "log" else "right"
        ax.set_xticklabels([str(int(t)) for t in ticks],
                             rotation=rotation, ha=ha,
                             rotation_mode="anchor")
    ax.legend(loc="lower right", fontsize=24, framealpha=0.9)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gene-phase-csv", type=Path, default=DEFAULTS["gene_phase"])
    p.add_argument("--gene-fluor-csv", type=Path, default=DEFAULTS["gene_fluor"])
    p.add_argument("--chad-phase-csv", type=Path, default=DEFAULTS["chad_phase"])
    p.add_argument("--chad-fluor-csv", type=Path, default=DEFAULTS["chad_fluor"])
    p.add_argument("--single", choices=("top1", "top1+top5"),
                    default=None,
                    help="Emit only ONE of the two variants. Default "
                         "(no flag) → emits BOTH (`<stem>.pdf` for top-1, "
                         "`<stem>_top5.pdf` for top-1+top-5).")
    p.add_argument(
        "--output", type=Path,
        default=DEFAULT_OUTPUT_DIR / "eval_accuracy_curves.pdf",
        help="Base output PDF path. Defaults to "
             "/hpc/projects/icd.fast.ops/analysis/attention_accuracy/ "
             "(auto-created). With both variants the top-5 version "
             "appends `_top5` to the stem; each variant emits both a "
             "`_median-iqr` and `_mean-sem` flavor. Also writes sibling "
             ".png, .svg previews + .summary.csv.",
    )
    args = p.parse_args()

    gp = _load(args.gene_phase_csv)
    gf = _load(args.gene_fluor_csv)
    cp = _load(args.chad_phase_csv)
    cf = _load(args.chad_fluor_csv)

    n_gene_p  = gp["class_name"].nunique() if gp is not None else 0
    n_gene_f  = gf["class_name"].nunique() if gf is not None else 0
    n_chad_p  = cp["class_name"].nunique() if cp is not None else 0
    n_chad_f  = cf["class_name"].nunique() if cf is not None else 0
    print(f"Loaded: gene-phase={n_gene_p} genes, gene-fluor={n_gene_f}, "
          f"chad-phase={n_chad_p} complexes, chad-fluor={n_chad_f}")

    # Pick which variants to render.
    if args.single == "top1":
        variants = [("", False, args.output)]
    elif args.single == "top1+top5":
        variants = [("with top-5", True, args.output)]
    else:
        variants = [
            ("", False, args.output),
            ("with top-5", True,
             args.output.with_name(f"{args.output.stem}_top5{args.output.suffix}")),
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Two x-axis flavors per top-1/top-5 variant. Linear keeps tick
    # spacing proportional to cell count (good for "look how much 200→500
    # buys you"); log compresses the high end so the early-bin gains
    # (10 → 100) are visible.
    SCALES = [("linear", ""), ("log", "_logx")]
    # Two band conventions: median+IQR (robust, nonparametric) and
    # mean±SEM (canonical for CI-style error bars). Emit both so the
    # reader can pick which framing best fits the downstream venue.
    BANDS = [("iqr", "_median-iqr"), ("sem", "_mean-sem")]
    for label, show_top5, out_path in variants:
        for xscale, scale_tag in SCALES:
            for band, band_tag in BANDS:
                stem = f"{out_path.stem}{scale_tag}{band_tag}"
                scale_out = out_path.with_name(f"{stem}{out_path.suffix}")
                fig, axes = plt.subplots(1, 2, figsize=(24, 10.0), sharey=True)
                _plot_one_panel(axes[0], gp, gf,
                                 title=f"GeneKOs (n={max(n_gene_p, n_gene_f):,})",
                                 show_top5=show_top5, xscale=xscale, band=band)
                _plot_one_panel(axes[1], cp, cf,
                                 title=f"Protein complex (n={max(n_chad_p, n_chad_f):,})",
                                 show_top5=show_top5, xscale=xscale, band=band)
                # `sharey=True` hides the right panel's y-tick labels by
                # default; re-enable so both panels are independently
                # readable. Keep y-axis label on the right panel too — saves
                # the reader from glancing left to find "Accuracy".
                axes[1].tick_params(axis="y", labelleft=True, labelsize=24)
                axes[1].set_ylabel("Accuracy", fontsize=28)
                fig.suptitle(
                    "Attention-model classification accuracy",
                    fontsize=36, fontweight="bold", y=1.00,
                )
                fig.tight_layout()
                fig.savefig(scale_out, format="pdf", dpi=200, bbox_inches="tight")
                # PNG companion at higher DPI — matches the bumped text sizes
                # and ensures the image stays sharp when embedded in slides.
                fig.savefig(scale_out.with_suffix(".png"), format="png",
                             dpi=240, bbox_inches="tight")
                # SVG companion — vector format for downstream editing
                # (Illustrator / Inkscape / Figma). Stays sharp at any zoom
                # and lets the user adjust text/colors post-hoc without
                # re-running the plot script.
                fig.savefig(scale_out.with_suffix(".svg"), format="svg",
                             bbox_inches="tight")
                plt.close(fig)
                print(f"Wrote: {scale_out}")
                print(f"       {scale_out.with_suffix('.png')}")
                print(f"       {scale_out.with_suffix('.svg')}")

    # Summary CSV — center + spread per (level, modality, n_cells).
    # Records both band conventions (mean/std/SEM and median/Q1/Q3) so the
    # CSV stays in sync with whichever plot variant the reader prefers.
    rows = []
    for level, df_p, df_f in [("gene", gp, gf), ("complex", cp, cf)]:
        for modality, df in (("phase", df_p), ("fluor", df_f)):
            if df is None:
                continue
            grouped = df.groupby("n_cells")["top1_acc"]
            for n in sorted(grouped.groups):
                vals = grouped.get_group(n).to_numpy()
                n_obs = vals.size
                std = float(np.std(vals, ddof=1)) if n_obs > 1 else 0.0
                sem = std / np.sqrt(n_obs) if n_obs > 0 else 0.0
                rows.append({
                    "level": level, "modality": modality, "n_cells": int(n),
                    "n_classes": int(df["class_name"].nunique()),
                    "top1_mean": float(np.mean(vals)),
                    "top1_std":  std,
                    "top1_sem":  float(sem),
                    "top1_median": float(np.median(vals)),
                    "top1_q25": float(np.quantile(vals, 0.25)),
                    "top1_q75": float(np.quantile(vals, 0.75)),
                })
    summary_path = args.output.with_suffix(".summary.csv")
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"       {summary_path}")


if __name__ == "__main__":
    main()
