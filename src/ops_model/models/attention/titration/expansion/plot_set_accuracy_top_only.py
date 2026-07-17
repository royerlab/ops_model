"""Plot expansion-sweep curves for set_accuracy — TOP + random + baseline
only (no bottom line). Output subdir: plots/set_accuracy_top_only/.

Renders both heads (set_accuracy geneKO and set_accuracy_ebi) each with
its own set of metric plots.
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

METRICS = [
    ("ebi_consistency",     "EBI consistency"),
    ("distinctiveness",     "Distinctiveness"),
    ("phenotypic_activity", "Phenotypic activity"),
    ("chad_consistency",    "CHAD consistency"),
]

HEADS = [
    ("set_accuracy",     "set_accuracy_top_only",     "#1f77b4"),
    ("set_accuracy_ebi", "set_accuracy_ebi_top_only", "#2ca02c"),
]


def render_head(head: str, subdir: str, color: str) -> None:
    out_dir = EXPANSION / "plots" / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV)
    baseline = df[df["sweep"] == "baseline"].iloc[0] if (df["sweep"] == "baseline").any() else None
    sa = df[df["sweep"] == "A"].copy()

    top = sa[(sa["head"] == head) & (sa["direction"] == "top")].sort_values("K")
    rnd = sa[sa["direction"] == "random"].sort_values("K")

    print(f"[{head}] top K rows: {len(top)}, random K rows: {len(rnd)}")

    for col, label in METRICS:
        fig, ax = plt.subplots(figsize=(7, 5))
        if not top.empty:
            peak = float(top[col].max())
            ax.plot(top["K"], top[col], marker="o", linestyle="-",
                    color=color, label=f"{head} / top  (peak {peak:.3f})")
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
        ax.set_title(f"{head} sweep — {label}")
        ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1))
        ax.grid(alpha=0.3)
        fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(out_dir / f"{head}_{col}.{ext}", dpi=140, bbox_inches="tight")
        plt.close(fig)

    print(f"  wrote {len(METRICS)} plots × 3 formats to {out_dir}")


def main() -> None:
    for head, subdir, color in HEADS:
        render_head(head, subdir, color)


if __name__ == "__main__":
    main()
