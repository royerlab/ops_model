"""Compute the per-marker EBI complex mAP matrix (complex × reporter) by running the paper's
`phenotypic_consistency_ebi` (copairs, EBI Complex Portal annotations) on each marker's gene
embedding. This is the CORRECT EBI metric per reporter — the `complex_reporter_chad_consistency.csv`
is a different (consistency) metric. Cached to CSV; `catalog.complex_dist()` reads it.
"""
from __future__ import annotations

import glob
import os

import anndata as ad
import pandas as pd

from ops_utils.analysis.map_scores import phenotypic_consistency_ebi

PS = ("/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3/cell_dino/"
      "zscore_per_exp/paper_v2/with_cp/with_4i/all_livecell/fixed_80%/cosine/per_signal")
PHASE_GENE = ("/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v0.3/cell_dino/"
              "zscore_per_exp/paper_v2/phase_only/fixed_80%/cosine/gene_embedding_pca_optimized.h5ad")
OUT = "/hpc/projects/icd.fast.ops/models/diffex/complex_reporter_ebi_map.csv"


def _ebi_map(h5ad):
    a = ad.read_h5ad(h5ad)
    if a.X is None or a.X.shape[1] != a.obsm["X_pca"].shape[1]:
        a.X = a.obsm["X_pca"]                                # copairs runs on adata.X
    df, _ = phenotypic_consistency_ebi(a, plot_results=False, cache_similarity=True, null_size=1000)
    # map complex_num -> name so columns/index match the rest of the app
    import yaml
    y = yaml.safe_load(open("/hpc/projects/icd.fast.ops/configs/gene_clusters/EBI_complexes_v1_updated_gene_names.yaml")) or {}
    n2n = {int(k): v["name"] for k, v in y.items() if isinstance(v, dict) and v.get("name")}
    df = df.copy(); df["name"] = df["complex_num"].map(n2n)
    return df.dropna(subset=["name"]).set_index("name")["mean_average_precision"]


def build(out=OUT):
    cols = {}
    cols["Phase"] = _ebi_map(PHASE_GENE)
    print("[ebi] Phase done")
    for h in sorted(glob.glob(f"{PS}/*_gene.h5ad")):
        sweep = h.replace("_gene.h5ad", "_sweep.csv")
        try:
            rep = pd.read_csv(sweep, usecols=["signal"])["signal"].iloc[0]   # canonical reporter name
        except Exception:
            rep = os.path.basename(h).replace("_gene.h5ad", "")
        try:
            cols[rep] = _ebi_map(h)
            print(f"[ebi] {rep}: {int((cols[rep] >= 0.01).sum())}/{len(cols[rep])} >=0.01")
        except Exception as e:
            print(f"[ebi] {rep} FAILED {e}")
    mat = pd.DataFrame(cols)
    mat.to_csv(out)
    print(f"[ebi] complex x reporter EBI mAP -> {out}  shape {mat.shape}")
    return out


if __name__ == "__main__":
    build()
