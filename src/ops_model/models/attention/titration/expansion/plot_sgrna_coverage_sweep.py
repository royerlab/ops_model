"""Plot: sgRNA coverage vs # cells kept per gene (top-K by attention).

For each K in a logspace sweep, count how many of each gene's
sgRNAs are still represented in the top-K cells by attention.
Tells you: at what K do you actually start losing sgRNAs?

The curve only drops meaningfully at K ≤ ~10 — at K=1 you literally
keep one cell per gene, which has exactly one sgRNA, so retention
hits 1/n_sgrnas (=0.25 for the typical 4-sgRNA gene). Above K≈10
retention stays near 1.0.

Output:
  - `sgrna_coverage_sweep.csv`: per (gene, K) sgRNA counts
  - `sgrna_coverage_sweep.{pdf,png,svg}`: 2-panel line plot
      • mean sgRNAs retained / total sgRNAs (across genes)
      • % of genes with FULL sgRNA retention (= all sgRNAs present)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v3/attention_v3/cdino/per_experiment_filtered_all_fast"
)
DEFAULT_OUT_DIR = Path(
    "/hpc/projects/icd.fast.ops/models/alex_lin_attention/v3/attention_v3/cdino/phate_peak_groups"
)
DEFAULT_PERCENTILES = [0, 10, 25, 50, 60, 70, 75, 80, 85, 90, 92, 94, 96, 98, 99]
# K = cells kept per gene (top-K by attention). Log-spaced so the
# interesting range (K=1..10 where sgRNAs actually start dropping)
# is well-sampled while still showing the plateau at large K.
DEFAULT_KEEP_KS = [1, 2, 3, 4, 5, 7, 10, 15, 25, 50, 100, 500, 1000,
                   5000, 10000, 50000]


def _load_cache_obs(per_exp_dir: Path) -> pd.DataFrame:
    """Concat per-experiment cache obs into a single DataFrame keyed by
    true `perturbation`. Only the columns we need.
    """
    blocks = []
    for p in sorted(per_exp_dir.glob("*.h5ad")):
        if p.stat().st_size == 0:
            continue
        try:
            ob = ad.read_h5ad(p, backed="r").obs
        except Exception:
            continue
        need = ["perturbation", "sgRNA", "rank", "rank_type"]
        if not all(c in ob.columns for c in need):
            continue
        sub = ob[need].copy()
        sub = sub[sub["rank_type"].astype(str) == "top"]
        sub["perturbation"] = sub["perturbation"].astype(str)
        sub["sgRNA"]        = sub["sgRNA"].astype(str)
        sub["rank"] = pd.to_numeric(sub["rank"], errors="coerce").astype("int64")
        blocks.append(sub[["perturbation", "sgRNA", "rank"]])
    if not blocks:
        raise RuntimeError(f"No usable cache files in {per_exp_dir}")
    return pd.concat(blocks, ignore_index=True)


def _coverage_at_ks(df: pd.DataFrame, ks: list) -> pd.DataFrame:
    """For each (gene, K), count distinct sgRNAs in the top-K cells by
    attention. K capped at gene's actual cell count.
    """
    rows = []
    grouped = df.groupby("perturbation")
    for gene, sub in grouped:
        sub_sorted = sub.sort_values("rank")  # ascending: top attn first
        n = len(sub_sorted)
        total_sg = sub["sgRNA"].nunique()
        row = {"gene": gene, "n_cells": n, "total_sgrnas": total_sg}
        for K in ks:
            keep_n = min(int(K), n)
            row[f"sgrnas_top_{K}"] = (
                sub_sorted.iloc[:keep_n]["sgRNA"].nunique()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _plot(cov: pd.DataFrame, ks: list, out_stem: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.array(ks, dtype=float)
    mean_retention = []
    full_rate = []
    for K in ks:
        col = f"sgrnas_top_{K}"
        ratio = cov[col].astype(float) / cov["total_sgrnas"].astype(float)
        mean_retention.append(ratio.mean())
        full_rate.append((cov[col] == cov["total_sgrnas"]).mean() * 100)

    ax1.plot(x, mean_retention, "o-", color="#1F9B4A", lw=2.2, ms=8,
              markeredgecolor="black", markeredgewidth=0.5)
    ax1.set_xscale("log")
    ax1.set_xlabel("K = cells per gene kept (top-K by attention)", fontsize=12)
    ax1.set_ylabel("Mean sgRNA retention\n(kept / total per gene)", fontsize=12)
    ax1.set_title("Mean sgRNA retention", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3, which="both")
    ax1.axhline(1.0, color="red", linestyle=":", lw=1.2, alpha=0.6,
                 label="full retention")
    ax1.axhline(0.25, color="grey", linestyle=":", lw=1.0, alpha=0.5,
                 label="1/4 (K=1 floor for 4-sgRNA gene)")
    ax1.legend(fontsize=9, loc="lower right")

    ax2.plot(x, full_rate, "s-", color="#1F46A6", lw=2.2, ms=8,
              markeredgecolor="black", markeredgewidth=0.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("K = cells per gene kept (top-K by attention)", fontsize=12)
    ax2.set_ylabel("% of genes with FULL sgRNA retention", fontsize=12)
    ax2.set_title("Genes losing ≥1 sgRNA", fontsize=13, fontweight="bold")
    ax2.set_ylim(-2, 105)
    ax2.grid(True, alpha=0.3, which="both")
    ax2.axhline(100, color="red", linestyle=":", lw=1.2, alpha=0.6,
                 label="all genes full")
    ax2.legend(fontsize=9, loc="lower right")

    fig.suptitle(
        "Guide coverage vs top-K-by-attention cell selection "
        "(per-gene, geneKO_fast cache)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_stem}.pdf/.png/.svg")


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                    help="Per-experiment cache dir to read obs from.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KEEP_KS,
                    help="K values (cells/gene to keep, top-K by attention).")
    args = ap.parse_args()

    print(f"[load] reading obs from {args.cache_dir.name}…")
    df = _load_cache_obs(args.cache_dir)
    print(f"  {len(df):,} cells across {df['perturbation'].nunique()} genes")

    print(f"[sweep] computing sgRNA retention at {len(args.ks)} K values…")
    cov = _coverage_at_ks(df, list(args.ks))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cov_csv = args.out_dir / "sgrna_coverage_sweep.csv"
    cov.to_csv(cov_csv, index=False)
    print(f"  wrote {cov_csv}")

    print("\nPer-K summary (mean across genes):")
    for K in args.ks:
        col = f"sgrnas_top_{K}"
        ratio = cov[col].astype(float) / cov["total_sgrnas"].astype(float)
        full_rate = (cov[col] == cov["total_sgrnas"]).mean() * 100
        print(f"  K={K:>6,}: mean retention={ratio.mean():.3f}, "
              f"{full_rate:.1f}% of genes with all sgRNAs intact")

    _plot(cov, list(args.ks),
           args.out_dir / "sgrna_coverage_sweep")


if __name__ == "__main__":
    main()
