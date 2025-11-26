from ast import literal_eval

import pandas as pd

from ops_model.data.paths import OpsPaths


def group_guides(experiment):
    from collections import defaultdict

    gene_lib = pd.read_csv(OpsPaths(experiment=experiment).gene_library)
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
