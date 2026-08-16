"""Prelim analysis: per-gene (geneKO) and per-complex (CHAD) accuracy bins,
plus coverage projection for the cell-selection runs.

For each gene/complex:
  bin_n_cells = smallest ``n_cells`` row at which top1_acc >= 0.95

Reports:
  * distribution of bin_n_cells across genes/complexes
  * how many genes/complexes never reach 0.95 (fallback needed)
  * how many genes in the v4 pool have no CHAD complex assignment
  * given the bin and the gene's actual cell count in v4, what selection rate
    each gene would face (i.e. fraction of available cells we'd keep)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

GENEKO_CSV = Path("/home/gav.sturm/linked_folders/icd.fast.ops/models/alex_lin_attention/v3/attention_v3/cdino/cdino_eval_phase_50.csv")
CHAD_CSV   = Path("/home/gav.sturm/linked_folders/icd.fast.ops/models/alex_lin_attention/v3/attention_v3/cdino/cdino_eval_phase_chad_50.csv")
SIDECAR    = Path("/hpc/projects/icd.fast.ops/models/alex_lin_attention/v4/expansion_v1/per_experiment_v4_attn.parquet")
ACC_THRESH = 0.95


def _bin_per_label(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Return df with columns [label_col, bin_n_cells, max_acc]."""
    rows = []
    for label, sub in df.groupby(label_col):
        sub = sub.sort_values("n_cells")
        meets = sub[sub["top1_acc"] >= ACC_THRESH]
        if not meets.empty:
            bn = int(meets.iloc[0]["n_cells"])
        else:
            bn = -1   # never crosses 0.95
        rows.append({label_col: label, "bin_n_cells": bn, "max_acc": float(sub["top1_acc"].max())})
    return pd.DataFrame(rows)


def main() -> int:
    gk = pd.read_csv(GENEKO_CSV)
    ch = pd.read_csv(CHAD_CSV)
    sidecar_genes = None
    try:
        sc = pd.read_parquet(SIDECAR, columns=["experiment", "well", "segmentation_id"])
        # Need gene info — use v4 per-exp h5ad obs aggregate instead.
        # For coverage estimate, just count rows = total cells in pool.
        n_pool_cells = len(sc)
    except Exception:
        n_pool_cells = None

    print("=" * 70)
    print("geneKO accuracy bins")
    print("=" * 70)
    bins_gk = _bin_per_label(gk, "gene_name")
    n_total = len(bins_gk)
    n_never = (bins_gk["bin_n_cells"] == -1).sum()
    print(f"genes total: {n_total}")
    print(f"genes that never reach acc>={ACC_THRESH}: {n_never} ({n_never/n_total:.1%})")
    print("\nbin_n_cells distribution (genes that DO reach 95%):")
    reached = bins_gk[bins_gk["bin_n_cells"] != -1]
    counts = reached["bin_n_cells"].value_counts().sort_index()
    for bn, c in counts.items():
        print(f"  n_cells={bn:>5}: {c:>4} genes ({c/n_total:.1%})")
    print(f"\nmax_acc distribution for never-reachers:")
    nr = bins_gk[bins_gk["bin_n_cells"] == -1]
    if len(nr):
        print(f"  q25={nr['max_acc'].quantile(.25):.3f}  median={nr['max_acc'].median():.3f}  "
              f"q75={nr['max_acc'].quantile(.75):.3f}  max={nr['max_acc'].max():.3f}")

    print()
    print("=" * 70)
    print("CHAD complex accuracy bins")
    print("=" * 70)
    # Use label_name (complex label) as the unit for CHAD bins
    bins_ch = _bin_per_label(ch, "label_name")
    n_total_c = len(bins_ch)
    n_never_c = (bins_ch["bin_n_cells"] == -1).sum()
    print(f"complexes total: {n_total_c}")
    print(f"complexes that never reach acc>={ACC_THRESH}: {n_never_c} ({n_never_c/n_total_c:.1%})")
    reached_c = bins_ch[bins_ch["bin_n_cells"] != -1]
    print("\nbin_n_cells distribution (complexes that DO reach 95%):")
    counts_c = reached_c["bin_n_cells"].value_counts().sort_index()
    for bn, c in counts_c.items():
        print(f"  n_cells={bn:>5}: {c:>4} complexes ({c/n_total_c:.1%})")

    # CHAD gene coverage: how many distinct genes appear in CHAD eval at all?
    n_genes_chad = ch["gene_name"].nunique()
    n_genes_geneko = gk["gene_name"].nunique()
    print(f"\nGenes with CHAD complex membership: {n_genes_chad} / {n_genes_geneko} "
          f"geneKO-pool genes ({n_genes_chad/n_genes_geneko:.1%})")
    print(f"Genes NOT in any CHAD complex: {n_genes_geneko - n_genes_chad} "
          f"({(n_genes_geneko - n_genes_chad)/n_genes_geneko:.1%}) — these get fallback (keep all cells)")

    print()
    print("=" * 70)
    print("Projected cell-selection rates per gene/complex")
    print("=" * 70)
    # Per gene/complex, bin_n_cells says "you need this many cells to be 95% accurate".
    # We use that as the per-(sgRNA, experiment) selection cap or proportional to it.
    # Approach: per-sgRNA we keep top K cells where K = bin_n_cells / n_sgRNAs_per_gene.
    # With ~4 sgRNAs per gene:
    print("If we keep top (bin_n_cells // 4) cells per sgRNA-group (~4 sgRNAs/gene):")
    for bn in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]:
        n_genes_at_bn = (reached["bin_n_cells"] == bn).sum()
        per_sgRNA = max(1, bn // 4)
        print(f"  bin={bn:>4}: {n_genes_at_bn:>4} genes, each sgRNA keeps ~{per_sgRNA} top-attn cells")

    print()
    print("Summary table — what a 'cells kept' downstream pool would look like")
    print("=" * 70)
    # Estimate total cells kept across all genes assuming ~4 sgRNAs × ~50 exps with attention
    cell_kept_geneko = 0
    for _, row in reached.iterrows():
        cell_kept_geneko += row["bin_n_cells"]    # interpreted as global cells per gene
    print(f"geneKO scope total selected cells (sum of bin_n_cells across reachable genes): "
          f"{cell_kept_geneko:,}")
    print(f"  Plus {n_never} never-reach genes — these need a fallback rule (use max bin? all cells?)")

    print()
    if n_pool_cells:
        print(f"v4 pool total cells (sidecar rows): {n_pool_cells:,}")
        print(f"geneKO scope projected retention: {cell_kept_geneko / n_pool_cells:.1%} of pool")

    return 0


if __name__ == "__main__":
    sys.exit(main())
