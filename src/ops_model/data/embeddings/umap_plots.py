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


def plot_umap(
    gene: str,
    adata: ad.AnnData,
    save_path: Optional[str] = None,
    guides: Optional[list] = None,
    data_point_type: Literal["cell", "guide", "gene"] = "cell",
):

    umap = adata.obsm["X_umap"]
    if data_point_type == "cell":
        subset = adata[adata.obs["label_str"] == gene].obsm["X_umap"]
        s = 1
        alpha = 0.5
    elif data_point_type == "gene":
        subset = adata[adata.obs["label_str"] == gene].obsm["X_umap"]
        s = 20
        alpha = 0.8
    else:  # guide-level
        subset = adata[adata.obs["sgRNA"].isin(guides)].obsm["X_umap"]
        s = 20
        alpha = 0.8

    plt.scatter(umap[:, 0], umap[:, 1], c="lightgrey", s=s, alpha=alpha, linewidth=0)
    plt.scatter(subset[:, 0], subset[:, 1], c="blue", s=s, alpha=alpha, linewidth=0)
    plt.title(gene)
    plt.xticks([])
    plt.yticks([])
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.figure()
    plt.close()

    # Further implementation for plotting by guide
    return


def plot_umap_multiple_genes(
    genes: list,
    adata: ad.AnnData,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    umap = adata.obsm["X_umap"]
    plt.scatter(umap[:, 0], umap[:, 1], c="lightgrey", s=20, alpha=0.8, linewidth=0)
    for gene in genes:
        subset = adata[adata.obs["label_str"] == gene].obsm["X_umap"]
        plt.scatter(subset[:, 0], subset[:, 1], s=20, alpha=1, linewidth=0, label=gene)
    plt.title(title if title is not None else "")
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.figure()
    plt.close()
    return


def report_umap_plot_2(
    feature_dir: str,
    adata_cells: Optional[ad.AnnData] = None,
    adata_guides: Optional[ad.AnnData] = None,
    adata_genes: Optional[ad.AnnData] = None,
    output_path: Optional[str] = None,
):
    path = Path(feature_dir)
    output_path = Path(output_path) if output_path is not None else None
    if adata_cells is None:
        adata_path_cells = path / "anndata_objects" / "features_processed.h5ad"
        adata_cells = ad.read_h5ad(adata_path_cells)
    if adata_guides is None:
        adata_path_guides = path / "anndata_objects" / "guide_bulked_umap.h5ad"
        adata_guides = ad.read_h5ad(adata_path_guides)
    if adata_genes is None:
        adata_path_genes = path / "anndata_objects" / "gene_bulked_umap.h5ad"
        adata_genes = ad.read_h5ad(adata_path_genes)
    gene_guide_dict = group_guides()

    plot_umap_multiple_genes(
        genes=[
            "RPL18",
            "RPL23",
            "RPL9",
            "RPL30",
            "RPL35",
            "RPL32",
            "RPLP2",
            "RPL27A",
            "RPL5",
            "RPL15",
            "RPL41",
            "RPL34",
            "RPL26",
            "RPL37A",
        ],
        adata=adata_genes,
        title="RPL genes UMAP",
        save_path=(
            output_path / "fig_2_umap_rpl_genes.png"
            if output_path is not None
            else None
        ),
    )
    plt.figure()
    plot_umap_multiple_genes(
        genes=["NUP54", "NUP98", "NUP214", "NUP37"],
        adata=adata_genes,
        title="NUP genes UMAP",
        save_path=(
            output_path / "fig_2_umap_nup_genes.png"
            if output_path is not None
            else None
        ),
    )
    plt.figure()
    plot_umap_multiple_genes(
        genes=["TRAPPC11", "TRAPPC4", "TRAPPC2L"],
        adata=adata_genes,
        title="TRAPPC genes UMAP",
        save_path=(
            output_path / "fig_2_umap_trappc_genes.png"
            if output_path is not None
            else None
        ),
    )
    plt.figure()
    plot_umap_multiple_genes(
        genes=["KRT18", "KRT8"],
        adata=adata_genes,
        title="KRT genes UMAP",
        save_path=(
            output_path / "fig_2_umap_krt_genes.png"
            if output_path is not None
            else None
        ),
    )
    plt.figure()

    return


def report_umap_plot_1(
    feature_dir: str,
    adata_cells: Optional[ad.AnnData] = None,
    adata_guides: Optional[ad.AnnData] = None,
    adata_genes: Optional[ad.AnnData] = None,
    output_path: Optional[str] = None,
):
    path = Path(feature_dir)
    output_path = Path(output_path) if output_path is not None else None
    if adata_cells is None:
        adata_path_cells = path / "anndata_objects" / "features_processed.h5ad"
        adata_cells = ad.read_h5ad(adata_path_cells)
    if adata_guides is None:
        adata_path_guides = path / "anndata_objects" / "guide_bulked_umap.h5ad"
        adata_guides = ad.read_h5ad(adata_path_guides)
    if adata_genes is None:
        adata_path_genes = path / "anndata_objects" / "gene_bulked_umap.h5ad"
        adata_genes = ad.read_h5ad(adata_path_genes)
    gene_guide_dict = group_guides()

    plot_umap(
        gene="NTC",
        adata=adata_cells,
        data_point_type="cell",
        save_path=(
            output_path / "fig_1_umap_cell_ntc.png" if output_path is not None else None
        ),
    )
    plt.figure()
    plot_umap(
        gene="Nontargeting",
        adata=adata_guides,
        guides=gene_guide_dict.get("Nontargeting", []),
        data_point_type="guide",
        save_path=(
            output_path / "fig_1umap_guide_ntc.png" if output_path is not None else None
        ),
    )
    plt.figure()
    plot_umap(
        gene="NTC",
        adata=adata_genes,
        data_point_type="gene",
        save_path=(
            output_path / "fig_1_umap_gene_ntc.png" if output_path is not None else None
        ),
    )
    plt.figure()

    return


def report_umap_plots(
    feature_dir: str,
    output_path: Optional[str] = None,
):
    path = Path(feature_dir)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

    adata_path_cells = path / "anndata_objects" / "features_processed.h5ad"
    adata_cells = ad.read_h5ad(adata_path_cells)
    adata_path_guides = path / "anndata_objects" / "guide_bulked_umap.h5ad"
    adata_guides = ad.read_h5ad(adata_path_guides)
    adata_path_genes = path / "anndata_objects" / "gene_bulked_umap.h5ad"
    adata_genes = ad.read_h5ad(adata_path_genes)

    report_umap_plot_1(
        feature_dir=feature_dir,
        adata_cells=adata_cells,
        adata_guides=adata_guides,
        adata_genes=adata_genes,
        output_path=output_path,
    )

    report_umap_plot_2(
        feature_dir=feature_dir,
        adata_cells=adata_cells,
        adata_guides=adata_guides,
        adata_genes=adata_genes,
        output_path=output_path,
    )

    return


if __name__ == "__main__":
    feature_dir = "/hpc/projects/intracellular_dashboard/ops/ops0031_20250424/3-assembly/dynaclr_features"
    output_path = "/hpc/projects/intracellular_dashboard/ops/ops0031_20250424/3-assembly/dynaclr_features/report_plots"
    report_umap_plots(
        feature_dir=feature_dir,
        output_path=output_path,
    )
