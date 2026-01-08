from ast import literal_eval
from pathlib import Path

import pandas as pd
import anndata as ad

from ops_model.data.paths import OpsPaths


def group_guides():
    from collections import defaultdict

    gene_lib = pd.read_csv(OpsPaths(experiment="doesnt_matter").other["gene_library"])
    guide_to_gene = defaultdict(list)
    for _, row in gene_lib.iterrows():
        if pd.isna(row["gene_symbol"]):
            key = "NTC"
        else:
            key = row["gene_symbol"]
        guide_to_gene[key].append(row["sgRNA"])

    return guide_to_gene


def get_gene_complexes():
    path = "/hpc/projects/intracellular_dashboard/ops/configs/annotated_gene_panel_July2025.csv"
    df = pd.read_csv(path)
    complex_df = df[["Gene.name", "In_same_complex_with"]]
    gene_list = list(complex_df["Gene.name"])
    unique_complexes = dict()
    black_set = set()
    count = 0
    for idx, row in complex_df.iterrows():
        if row["Gene.name"] in black_set:
            continue
        genes_in_complex = literal_eval(row.In_same_complex_with)
        genes_in_complex.append(row["Gene.name"])
        for g in genes_in_complex:
            to_add = [g for g in genes_in_complex if g in gene_list]
        unique_complexes[count] = to_add
        count += 1
        for g in to_add:
            black_set.add(g)
    return {k: v for k, v in unique_complexes.items() if len(v) > 2}


def load_adata(path):
    path = Path(path)
    adata_path_cells = path / "anndata_objects" / "features_processed.h5ad"
    adata_cells = ad.read_h5ad(adata_path_cells)
    adata_path_guides = path / "anndata_objects" / "guide_bulked_umap.h5ad"
    adata_guides = ad.read_h5ad(adata_path_guides)
    adata_path_genes = path / "anndata_objects" / "gene_bulked_umap.h5ad"
    adata_genes = ad.read_h5ad(adata_path_genes)

    return adata_cells, adata_guides, adata_genes
