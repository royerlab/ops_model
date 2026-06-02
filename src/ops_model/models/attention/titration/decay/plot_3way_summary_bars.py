"""3-way summary bar plot: geneKO peak vs CHAD peak vs intersection.

Mirrors the layout of `plot_summary_bars` from map_attention_decay.py
(grouped bars across 3 metrics: EBI consistency, CHAD consistency,
distinctiveness), but compares the three different "peak" directions
side by side at their best K each:
  - geneKO_fast top @ K=13000   (the DIST peak)
  - chad/top      @ K=5000      (CHAD's max-K bin)
  - random_low_removed_intersection @ K=13000  (best plateau)

Reads `map_attention_expansion_strict.csv`. Plots include baseline
horizontal lines per metric.

CAVEAT on the plot title: chad's mAP is computed over ~300 genes
(Alex's CHAD classifier covers only those), not the full 1001-gene
panel that geneKO + intersection cover. Annotated on the plot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_CDINO = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v3/attention_v3/cdino"
)


def _baseline(path: Path, key: str) -> float | None:
    if not path.exists():
        return None
    try:
        return float(json.loads(path.read_text())[key])
    except Exception:
        return None


def _count_intersection_filtered(df: pd.DataFrame, K: int = 100000) -> dict:
    """Cheap count from the CSV: at large K, both `top` and the
    intersection direction keep every available cell in their respective
    pools, so `n_cells_top - n_cells_intersection` is exactly the number
    of cells the intersection mask filtered out.
    """
    top_row = df[(df["source"] == "geneKO_fast")
                  & (df["rank_type"] == "top")
                  & (df["rank_hi"] == K)]
    int_row = df[(df["source"] == "geneKO_fast")
                  & (df["rank_type"] == "random_low_removed_intersection")
                  & (df["rank_hi"] == K)]
    if top_row.empty or int_row.empty:
        return {"total": None, "filtered": None, "pool": None}
    n_top = int(top_row.iloc[0]["n_cells"])
    n_int = int(int_row.iloc[0]["n_cells"])
    return {"total": n_top, "filtered": n_top - n_int, "pool": n_int}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path,
                    default=DEFAULT_CDINO / "map_attention_expansion_strict"
                                          / "map_attention_expansion_strict.csv")
    ap.add_argument("--out", type=Path,
                    default=DEFAULT_CDINO / "map_attention_expansion_strict"
                                          / "summary_3way_peak_bars.pdf")
    ap.add_argument("--genek-k", type=int, default=13000,
                    help="K for geneKO top peak (default: 13000)")
    ap.add_argument("--chad-k", type=int, default=5000,
                    help="K for chad/top peak (default: 5000, CHAD's max)")
    ap.add_argument("--inter-k", type=int, default=13000,
                    help="K for intersection direction (default: 13000)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Cells filtered by intersection-low mask, from CSV n_cells column.
    inter_counts = _count_intersection_filtered(df, K=100000)
    if inter_counts["filtered"] is None:
        inter_label = "Intersection\n(all cells, K=100k)"
    else:
        inter_label = (
            f"Intersection\n(filters {inter_counts['filtered']:,}/"
            f"{inter_counts['total']:,} cells)"
        )

    def _pick(src, rt, K):
        sub = df[(df["source"] == src)
                  & (df["rank_type"] == rt)
                  & (df["rank_hi"] == K)]
        if sub.empty:
            return None
        r = sub.iloc[0]
        return {
            "ebi":      float(r.get("mean_map_pca", np.nan)),
            "chad":     float(r.get("mean_map_chad_pca", np.nan)),
            "distinct": float(r.get("mean_map_dist_pca", np.nan)),
            "n_genes":  int(r.get("n_genes", 0)),
        }

    # Colors encode relationships between bars:
    #   - geneKO top  → green
    #   - geneKO bottom → faded green (same family, recedes visually)
    #   - CHAD top    → orange
    #   - Intersection → green-orange blend (geneKO ⊕ CHAD avg ≈ #7C772B)
    bars = [
        ("geneKO bottom\n(low attn, K=13k)", "#A8D5B5",
         _pick("geneKO_fast", "bottom", args.genek_k)),
        ("geneKO top\n(K=13k)", "#1F9B4A",
         _pick("geneKO_fast", "top", args.genek_k)),
        ("CHAD top *\n(K=5k)", "#D9530B",
         _pick("chad", "top", args.chad_k)),
        (inter_label, "#7C772B",
         _pick("geneKO_fast", "random_low_removed_intersection", 100000)),
    ]
    bars = [(lbl, c, d) for lbl, c, d in bars if d is not None]

    # Baselines from cached jsons (next to the csv's parent's parent)
    cdino = args.csv.parent.parent
    ebi_baseline      = _baseline(cdino / "all_cells_pca_baseline.json", "mean_map")
    chad_baseline     = _baseline(cdino / "all_cells_pca_chad_baseline.json", "mean_map")
    distinct_baseline = _baseline(cdino / "all_cells_pca_distinctiveness_baseline.json",
                                    "mean_map_dist")

    # Order: Distinctiveness (left), CHAD consistency, EBI consistency (right).
    metrics = [("distinct", "Distinctiveness",   distinct_baseline),
                ("chad",     "CHAD consistency",  chad_baseline),
                ("ebi",      "EBI consistency",   ebi_baseline)]

    fig, ax = plt.subplots(figsize=(11, 6))
    n_groups = len(metrics)
    n_bars   = len(bars)
    group_w  = 0.78
    bar_w    = group_w / n_bars
    x_centers = np.arange(n_groups)

    for j, (lbl, color, d) in enumerate(bars):
        offsets = x_centers - group_w / 2 + (j + 0.5) * bar_w
        vals = [d[m[0]] for m in metrics]
        ax.bar(offsets, vals, width=bar_w, color=color,
                edgecolor="black", linewidth=0.6, label=lbl)
        for x, v in zip(offsets, vals):
            if np.isfinite(v):
                ax.text(x, v + 0.005, f"{v:.3f}", ha="center", va="bottom",
                         fontsize=8.5, rotation=0)

    # Per-metric baseline lines under each group + numeric annotation.
    for i, (_, _, base) in enumerate(metrics):
        if base is None or not np.isfinite(base):
            continue
        ax.hlines(base,
                    x_centers[i] - group_w / 2,
                    x_centers[i] + group_w / 2,
                    colors="#d62728", linestyles=":", lw=2,
                    label="All cells (PCA) baseline" if i == 0 else None,
                    zorder=5)
        ax.text(x_centers[i] + group_w / 2 + 0.01, base,
                 f"{base:.3f}", fontsize=9, color="#d62728",
                 va="center", ha="left", fontweight="bold", zorder=6)

    ax.set_xticks(x_centers)
    ax.set_xticklabels([m[1] for m in metrics], fontsize=13)
    ax.set_ylabel("mean mAP (PCA 97-d, NTC-normed)", fontsize=13)
    # Pull actual CHAD n_genes from the row data (typically ~300/1001).
    chad_d = next((d for lbl, _, d in bars if lbl.startswith("CHAD top")), None)
    chad_n = chad_d["n_genes"] if chad_d else "?"
    ax.set_title(
        "Peak-direction comparison (strict cache)"
        f"\n* CHAD computed over {chad_n}/1001 geneKOs (Alex's CHAD "
        f"classifier covers only those); others over full 1001-gene panel",
        fontsize=12.5, fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=10, framealpha=0.9, loc="upper left",
               bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

    # y-limit headroom for the labels.
    vmax_cands = []
    for _, _, d in bars:
        vmax_cands += [d["ebi"], d["chad"], d["distinct"]]
    for _, _, base in metrics:
        if base is not None and np.isfinite(base):
            vmax_cands.append(base)
    vmax = max(v for v in vmax_cands if np.isfinite(v))
    ax.set_ylim(0, max(0.65, np.ceil(vmax * 10) / 10 + 0.06))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out} (+.png/.svg)")


if __name__ == "__main__":
    main()
