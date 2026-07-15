"""Plot expansion-sweep curves for set_accuracy only.

Reads ``bin_results.csv`` (produced by map_attention_expansion_v4) and renders
one figure per metric with three lines:
    - set_accuracy / top   (solid)
    - set_accuracy / low   (dashed) — "bottom-K by rank"
    - random               (head-agnostic, black dotted)
And an all-cells baseline as a red dashed horizontal line.
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
OUT_DIR = EXPANSION / "plots" / "set_accuracy_only"

METRICS = [
    ("ebi_consistency",     "EBI consistency"),
    ("distinctiveness",     "Distinctiveness"),
    ("phenotypic_activity", "Phenotypic activity"),
    ("chad_consistency",    "CHAD consistency"),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV)

    baseline = df[df["sweep"] == "baseline"].iloc[0] if (df["sweep"] == "baseline").any() else None
    sa = df[df["sweep"] == "A"].copy()

    set_top    = sa[(sa["head"] == "set_accuracy") & (sa["direction"] == "top")].sort_values("K")
    set_bottom = sa[(sa["head"] == "set_accuracy") & (sa["direction"] == "low")].sort_values("K")
    rnd        = sa[sa["direction"] == "random"].sort_values("K")

    print(f"set_accuracy/top   K rows: {len(set_top)}")
    print(f"set_accuracy/low   K rows: {len(set_bottom)}")
    print(f"random             K rows: {len(rnd)}")

    for col, label in METRICS:
        fig, ax = plt.subplots(figsize=(7, 5))
        if not set_top.empty:
            peak = float(set_top[col].max())
            ax.plot(set_top["K"], set_top[col], marker="o", linestyle="-",
                    color="#1f77b4", label=f"set_accuracy / top  (peak {peak:.3f})")
        if not set_bottom.empty:
            peak = float(set_bottom[col].max())
            ax.plot(set_bottom["K"], set_bottom[col], marker="o", linestyle="--",
                    color="#1f77b4", label=f"set_accuracy / bottom  (peak {peak:.3f})")
        if not rnd.empty:
            peak = float(rnd[col].max())
            ax.plot(rnd["K"], rnd[col], marker="s", linestyle=":",
                    color="black", label=f"random (head-agnostic, peak {peak:.3f})")
        if baseline is not None and pd.notna(baseline.get(col)):
            ax.axhline(baseline[col], color="red", linestyle="--", linewidth=1,
                       label=f"all-cells ({baseline[col]:.3f})")

        ax.set_xscale("log")
        ax.set_xlabel("K cells / gene")
        ax.set_ylabel(label)
        ax.set_title(f"set_accuracy sweep — {label}")
        ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1))
        ax.grid(alpha=0.3)
        fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(OUT_DIR / f"set_accuracy_{col}.{ext}", dpi=140, bbox_inches="tight")
        plt.close(fig)

    print(f"wrote {len(METRICS)} plots × 3 formats to {OUT_DIR}")


if __name__ == "__main__":
    main()
