from tqdm import tqdm
from pathlib import Path
from typing import Literal, Optional
from ast import literal_eval

import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt

from ops_model.data.embeddings.utils import group_guides, get_gene_complexes
from ops_model.data.paths import OpsPaths


COLORS = [
    "lightcoral",
    "brown",
    "darkred",
    "burlywood",
    "darkgoldenrod",
]


def plot_umap_complex(
    experiment,
    adata: ad.AnnData,
    data_point_type: Literal["cell", "guide"] = "cell",
    complex_id: int = None,
):
    """
    example complex IDs:
        110: COG complex
        18: POLR complex
        48: CHMP complex
        21: PSMD complex
        547: EIF3 complex
        71: MRPL complex
        450: RPL / RPS complex

         RP_complex = a[450]
        RP_complex = [a for a in RP_complex if a.startswith('RP')]
        RPL_complex = [a for a in RP_complex if a.startswith('RPL')]
        RPS_complex = [a for a in RP_complex if a.startswith('RPS')]
        complex = RPL_complex

    """
    a = get_gene_complexes()
    gene_guide_dict = group_guides(experiment)
    complex = a[complex_id]

    umap = adata.obsm["X_umap"]
    for i, g in enumerate(complex):
        guides = gene_guide_dict.get(g, [])
        if data_point_type == "cell":
            subset = adata[adata.obs["label_str"] == g].obsm["X_umap"]
            s = 1
        else:  # guide-level
            subset = adata[adata.obs["sgRNA"].isin(guides)].obsm["X_umap"]
            s = 20

        if i == 0:
            plt.scatter(
                umap[:, 0], umap[:, 1], c="lightgrey", s=s, alpha=0.9, linewidth=0
            )
        plt.scatter(
            subset[:, 0],
            subset[:, 1],
            c=COLORS[i % len(COLORS)],
            s=s,
            alpha=0.9,
            linewidth=0,
        )
        plt.title(f"{g}_complex")
        plt.xticks([])
        plt.yticks([])
    return


def plot_umap(
    gene: str,
    adata: ad.AnnData,
    save_path: str,
    guides: Optional[list] = None,
    data_point_type: Literal["cell", "guide"] = "cell",
):

    umap = adata.obsm["X_umap"]
    if data_point_type == "cell":
        subset = adata[adata.obs["label_str"] == gene].obsm["X_umap"]
        s = 1
    else:  # guide-level
        subset = adata[adata.obs["sgRNA"].isin(guides)].obsm["X_umap"]
        s = 20

    plt.scatter(umap[:, 0], umap[:, 1], c="lightgrey", s=s, alpha=0.5, linewidth=0)
    plt.scatter(subset[:, 0], subset[:, 1], c="blue", s=s, alpha=0.5, linewidth=0)
    plt.title(gene)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(save_path, dpi=300)
    plt.figure()
    plt.close()

    # Further implementation for plotting by guide
    return


def generate_umap_plots(
    experiment,
):
    # load anndata checkpoint & check if umap exists
    path = OpsPaths(experiment).cell_profiler_out
    save_dir = path.parent / "anndata_objects"
    assert save_dir.exists(), f"Anndata objects directory does not exist: {save_dir}"
    checkpoint_path = save_dir / "features_processed.h5ad"
    adata = ad.read_h5ad(checkpoint_path)
    assert "X_umap" in adata.obsm, "UMAP embeddings not found in AnnData object."
    plots_dir = OpsPaths(experiment).embedding_plot_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    gene_guide_dict = group_guides(experiment)

    # plot UMAP with NTC cells labeled
    plot_umap(
        gene="NTC",
        adata=adata,
        save_path=plots_dir / "umap_cell_ntc.png",
        data_point_type="cell",
    )

    # Plot UMAP with NTC guides labeled
    ntc_gene = "Nontargeting"
    ntc_guides = gene_guide_dict.get(ntc_gene, [])
    guide_avg_adata = ad.read_h5ad(save_dir / "guide_bulked_umap.h5ad")
    plot_umap(
        gene=ntc_gene,
        adata=guide_avg_adata,
        save_path=plots_dir / "umap_guide_ntc.png",
        guides=ntc_guides,
        data_point_type="guide",
    )


if __name__ == "__main__":
    pass
