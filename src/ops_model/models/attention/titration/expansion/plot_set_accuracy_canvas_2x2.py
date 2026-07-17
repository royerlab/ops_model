"""2x2 canvas comparing set_accuracy sliding window across BOTH models.

Rows    : geneKO model (set_accuracy), EBI model (set_accuracy_ebi)
Columns : Distinctiveness, EBI consistency
Each panel: TOP-ranked curve + head-agnostic random + all-cells baseline.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt

EXPANSION = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v4/expansion_v1"
)
CSV = EXPANSION / "bin_results.csv"
OUT_DIR = EXPANSION / "plots" / "set_accuracy_canvas_2x2"

ROWS = [
    ("set_accuracy",     "geneKO model\n(per-gene)",    "#1f77b4"),
    ("set_accuracy_ebi", "EBI model\n(per-complex)",   "#2ca02c"),
]
COLS = [
    ("distinctiveness",  "Distinctiveness (mAP)"),
    ("ebi_consistency",  "EBI consistency (mAP)"),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV)
    baseline = df[df["sweep"] == "baseline"].iloc[0] if (df["sweep"] == "baseline").any() else None
    sa = df[df["sweep"] == "A"].copy()

    # Figure-quality typography — one style block, applied globally
    plt.rcParams.update({
        "font.size":        16,
        "axes.titlesize":   20,
        "axes.labelsize":   19,
        "xtick.labelsize":  15,
        "ytick.labelsize":  15,
        "legend.fontsize":  14,
        "axes.linewidth":   1.6,
    })

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey="col")

    for i, (head, row_label, color) in enumerate(ROWS):
        top = sa[(sa["head"] == head) & (sa["direction"] == "top")].sort_values("K")
        rnd = sa[sa["direction"] == "random"].sort_values("K")

        for j, (col, col_label) in enumerate(COLS):
            ax = axes[i, j]
            if not top.empty:
                peak = float(top[col].max())
                ax.plot(top["K"], top[col], marker="o", linestyle="-",
                        color=color, linewidth=3.5, markersize=10,
                        label=f"top (peak {peak:.3f})")
            if not rnd.empty:
                peak = float(rnd[col].max())
                ax.plot(rnd["K"], rnd[col], marker="s", linestyle=":",
                        color="black", linewidth=2.5, markersize=8,
                        label=f"random (peak {peak:.3f})")
            if baseline is not None and pd.notna(baseline.get(col)):
                ax.axhline(baseline[col], color="red", linestyle="--", linewidth=2)

            ax.set_xscale("log")
            ax.tick_params(width=1.4, length=6)
            if i == 1:
                ax.set_xlabel("K cells / gene")
            if j == 0:
                ax.set_ylabel(row_label, fontweight="bold", fontsize=20,
                              labelpad=2)
            if i == 0:
                ax.set_title(col_label, fontweight="bold", pad=6)
            ax.grid(alpha=0.3, linewidth=0.8)
            # Force top-left panel (geneKO/Distinctiveness) legend into
            # the bottom-right where the curves have space; use "best"
            # elsewhere.
            leg_loc = "lower right" if (i == 0 and j == 0) else "best"
            ax.legend(loc=leg_loc, frameon=True, framealpha=0.9)

    fig.tight_layout(h_pad=0.5, w_pad=0.8)
    fig.suptitle("set_accuracy sliding window — top-K per gene",
                 fontsize=22, fontweight="bold", y=1.005)
    fig.subplots_adjust(top=0.92)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"set_accuracy_2x2.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote set_accuracy_2x2.{{png,pdf,svg}} to {OUT_DIR}")


if __name__ == "__main__":
    main()
