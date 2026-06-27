"""Paired raw-vs-denoise titration plot for the fluor-denoise pipeline.

The fluor-denoise titration (run_tag=fluor_denoise) produces 14 signals —
``<marker>_raw`` and ``<marker>_denoise`` for 7 markers. The standard combined
plot colors all 14 independently; this overlays each marker's pair on shared
axes: one color per marker, raw = solid, denoise = dashed. Reuses the shared
style helpers in ``titration_paired_plots``.

Usage:
    uv run python -m ops_model.post_process.combination.titration.titration_denoise_paired
    uv run python -m ops_model.post_process.combination.titration.titration_denoise_paired --combined-csv <path>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ops_model.post_process.combination.titration.titration_paired_plots import (
    _METRICS_PRETTY, bold_palette, pretty_marker, _fmt_K, _setup_matplotlib,
    _save_all_formats)

DEFAULT_COMBINED = (
    "/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3/"
    "cell_dino/zscore_per_exp/paper_v1/fluor_denoise/all_livecell/fixed_80%/"
    "cosine/titration_guide_median/titration_combined.csv")


def _split_signal(sig: str) -> tuple[str, str]:
    """'sec61b_denoise' -> ('sec61b', 'denoise'); '..._raw' -> (..., 'raw')."""
    marker, variant = sig.rsplit("_", 1)
    return marker, variant


def plot_paired(combined_csv: str, out_dir: str | None = None) -> Path:
    plt = _setup_matplotlib()
    df = pd.read_csv(combined_csv)
    df["marker"], df["variant"] = zip(*df["signal"].map(_split_signal))
    markers = sorted(df["marker"].unique())
    colors = dict(zip(markers, bold_palette(len(markers))))
    styles = {"raw": dict(ls="-", marker="o"), "denoise": dict(ls="--", marker="s")}
    out_dir = Path(out_dir or Path(combined_csv).parent)

    fig, axes = plt.subplots(1, len(_METRICS_PRETTY),
                             figsize=(6.2 * len(_METRICS_PRETTY), 5.2), squeeze=False)
    for ax, (metric, pretty) in zip(axes[0], _METRICS_PRETTY):
        col = f"{metric}_map_mean"
        for marker in markers:
            for variant, st in styles.items():
                w = df[(df["marker"] == marker) & (df["variant"] == variant)].sort_values("cells_per_guide")
                if w.empty:
                    continue
                ax.plot(w["cells_per_guide"], w[col], color=colors[marker],
                        linestyle=st["ls"], marker=st["marker"], markersize=4,
                        linewidth=1.8, alpha=0.9)
        ax.set_xscale("log")
        ax.set_xlabel("cells per sgRNA", fontsize=11)
        ax.set_ylabel("mAP (mean)", fontsize=11)
        ax.set_title(pretty, fontsize=12)
        ax.grid(True, which="both", alpha=0.2)

    # Legend: markers (color) + raw/denoise (linestyle), built once.
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=colors[m], lw=2.4, label=pretty_marker(m)) for m in markers]
    handles += [Line2D([0], [0], color="k", ls="-", marker="o", label="raw"),
                Line2D([0], [0], color="k", ls="--", marker="s", label="denoised")]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
               fontsize=9, frameon=False)
    fig.suptitle("Fluor-marker titration: raw vs N2S-3D denoised "
                 "(color = marker, solid = raw, dashed = denoised)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.92, 0.96))
    base = "titration_denoise_paired_raw_vs_denoise"
    _save_all_formats(fig, out_dir, base)
    plt.close(fig)
    print(f"  → {out_dir}/{base}.png/.svg")
    return out_dir / f"{base}.png"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--combined-csv", default=DEFAULT_COMBINED)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()
    plot_paired(args.combined_csv, args.out_dir)


if __name__ == "__main__":
    main()
