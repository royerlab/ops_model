import yaml
import pandas as pd
from pathlib import Path
import anndata as ad
import matplotlib.pyplot as plt
from ops_model.data.embeddings.utils import load_adata


def get_funk_clusters():
    path = Path(
        "/hpc/projects/intracellular_dashboard/ops/configs/annotated_gene_panel_July2025.csv"
    )
    df = pd.read_csv(path)

    funk_clusters = {
        "26": {
            "genes": df[df["funk_cluster"] == "26"]["Gene.name"].tolist(),
            "desc": "26 DNA Replication",
        },
        "148": {
            "genes": df[df["funk_cluster"] == "148"]["Gene.name"].tolist(),
            "desc": "148 Cell Cycle and Cytokinesis",
        },
        "14": {
            "genes": df[df["funk_cluster"] == "14"]["Gene.name"].tolist(),
            "desc": "14 Translation Initiation ",
            #  'desc': '14 Translation Initiation and tRNA Ligases'
        },
        "106": {
            "genes": df[df["funk_cluster"] == "106"]["Gene.name"].tolist(),
            "desc": "106 Proteasome 19S Regulatory Particle",
            # 'desc': '106 Proteasome 19S Regulatory Particle ATPase Subunits & Ubiquitination Factors',
        },
        "138": {
            "genes": df[df["funk_cluster"] == "138"]["Gene.name"].tolist(),
            "desc": "138 Spliceosome",
        },
        "29": {
            "genes": df[df["funk_cluster"] == "29"]["Gene.name"].tolist(),
            "desc": "29 Adhesion intracellular transport & NSL complex",
        },
        "184": {
            "genes": df[df["funk_cluster"] == "184"]["Gene.name"].tolist(),
            "desc": "184 Actin Cytoskeletion & nuclear transport",
        },
        "201": {
            "genes": df[df["funk_cluster"] == "201"]["Gene.name"].tolist(),
            "desc": "201 Golgi-ER transport",
        },
        "13": {
            "genes": df[df["funk_cluster"] == "13"]["Gene.name"].tolist(),
            "desc": "13 DNA damage",
        },
        "199": {
            "genes": df[df["funk_cluster"] == "199"]["Gene.name"].tolist(),
            "desc": "199 RNA Polymerase II",
        },
        "200": {
            "genes": df[df["funk_cluster"] == "200"]["Gene.name"].tolist(),
            "desc": "200 COP9 signalosome",
        },
        "52": {
            "genes": df[df["funk_cluster"] == "52"]["Gene.name"].tolist(),
            "desc": "52 Spliceosome",
        },
        "155": {
            "genes": df[df["funk_cluster"] == "155"]["Gene.name"].tolist(),
            "desc": "155 RNA Polymerase I",
        },
        # '82': {
        #     'genes': df[df['funk_cluster'] == '82']['Gene.name'].tolist(),
        #     'desc': '82 Mitochondrial Ribosome',
        # },
        "3": {
            "genes": df[df["funk_cluster"] == "3"]["Gene.name"].tolist(),
            "desc": "3 DNA Damage",
        },
        "46": {
            "genes": df[df["funk_cluster"] == "46"]["Gene.name"].tolist(),
            "desc": "46 Cell Cycle",
        },
        "66": {
            "genes": df[df["funk_cluster"] == "66"]["Gene.name"].tolist(),
            "desc": "66 Ribosome 40S subunit",
        },
        "212": {
            "genes": df[df["funk_cluster"] == "212"]["Gene.name"].tolist(),
            "desc": "212 Chaperonin TCP-1",
        },
        "136": {
            "genes": df[df["funk_cluster"] == "136"]["Gene.name"].tolist(),
            "desc": "136 40S Ribosome Biogenesis",
        },
        "110": {
            "genes": df[df["funk_cluster"] == "110"]["Gene.name"].tolist(),
            "desc": "110 Spliceosome",
        },
        "214": {
            "genes": df[df["funk_cluster"] == "214"]["Gene.name"].tolist(),
            "desc": "214 Augmin complex",
        },
        "21": {
            "genes": df[df["funk_cluster"] == "21"]["Gene.name"].tolist(),
            "desc": "21 Ribosome Biogenesis",
        },
        "15": {
            "genes": df[df["funk_cluster"] == "15"]["Gene.name"].tolist(),
            "desc": "15 60S Ribosome Biogenesis",
        },
        "23": {
            "genes": df[df["funk_cluster"] == "23"]["Gene.name"].tolist(),
            "desc": "23 Ribosome 60S subunit",
        },
        "203": {
            "genes": df[df["funk_cluster"] == "203"]["Gene.name"].tolist(),
            "desc": "203 Ribosome Biogenesis",
        },
        "216": {
            "genes": df[df["funk_cluster"] == "216"]["Gene.name"].tolist(),
            "desc": "216 Ribosome Biogenesis",
        },
        "179": {
            "genes": df[df["funk_cluster"] == "179"]["Gene.name"].tolist(),
            "desc": "179 DNA Damage",
        },
    }
    return funk_clusters


def print_cluster_info(funk_clusters):
    for k, v in funk_clusters.items():
        print(f"Cluster {k} ({v['desc']}): {len(v['genes'])} genes")

    return


def plot_funk_clusters(
    adata, funk_clusters, save_path=None, report_dir=None, filename="funk_clusters.png"
):
    """
    Plot UMAP colored by Funk functional clusters.

    Creates a 7×4 grid showing all 28 functional clusters (DNA replication,
    ribosome, spliceosome, etc.) colored on UMAP.

    Args:
        adata: AnnData object with UMAP coordinates and gene labels
        funk_clusters: Dictionary of functional cluster definitions
        save_path: Legacy parameter - direct path to save file
        report_dir: Path to report directory (preferred over save_path)
        filename: Filename to use when saving to report_dir
    """
    fig, axs = plt.subplots(nrows=7, ncols=4, figsize=(20, 28))
    axs = axs.flatten()
    for i, (cluster_num, cluster_info) in enumerate(funk_clusters.items()):
        genes = cluster_info["genes"]
        desc = cluster_info["desc"]
        plt.sca(axs[i])
        umap = adata.obsm["X_umap"]
        plt.scatter(umap[:, 0], umap[:, 1], c="lightgrey", s=20, alpha=0.8, linewidth=0)
        for gene in genes:
            subset = adata[adata.obs["label_str"] == gene].obsm["X_umap"]
            plt.scatter(
                subset[:, 0], subset[:, 1], s=40, alpha=1, linewidth=0, label=gene
            )
        plt.title(desc)
        plt.xticks([])
        plt.yticks([])
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="10")
        plt.tight_layout()

    # Determine save path
    if report_dir is not None:
        save_path = Path(report_dir) / "plots" / filename

    if save_path is not None:
        print(f"Saving funk cluster UMAP plot to {save_path}")
        plt.savefig(save_path, dpi=300)

    return


def check_cluster_genes(cluster_list_path: str, gene_list_path: str):
    """
    Validate that all genes in cluster definitions exist in the gene panel.

    1) Load the yaml found at cluster_list_path and the csv found at gene_list_path
    2) Check that each gene in each cluster is found in the gene list
    3) Raise an error if any genes are missing, otherwise print success message

    Args:
        cluster_list_path: Path to YAML file containing cluster definitions
        gene_list_path: Path to CSV file containing valid gene names

    Raises:
        ValueError: If any genes in the clusters are not found in the gene list
    """
    # Load cluster definitions from YAML
    with open(cluster_list_path, "r") as f:
        clusters = yaml.safe_load(f)

    # Load gene list from CSV
    gene_df = pd.read_csv(gene_list_path)
    valid_genes = set(gene_df["Gene.name"].tolist())

    # Check each cluster for missing genes
    missing_genes_by_cluster = {}
    cluster_info = []

    for cluster_id, cluster_data in clusters.items():
        cluster_name = cluster_data["name"]
        cluster_genes = cluster_data["genes"]

        # Find missing genes
        missing = [gene for gene in cluster_genes if gene not in valid_genes]

        if missing:
            missing_genes_by_cluster[cluster_id] = {
                "name": cluster_name,
                "missing_genes": missing,
            }

        cluster_info.append((cluster_id, cluster_name, len(cluster_genes)))

    # Report results
    if missing_genes_by_cluster:
        error_msg = "\nValidation FAILED: The following genes are missing from the gene panel:\n\n"
        for cluster_id, info in missing_genes_by_cluster.items():
            error_msg += f"Cluster {cluster_id} ({info['name']}):\n"
            for gene in info["missing_genes"]:
                error_msg += f"  - {gene}\n"
            error_msg += "\n"
        raise ValueError(error_msg)
    else:
        print("\n✓ Validation PASSED: All genes found in gene panel!\n")
        print("Cluster summary:")
        for cluster_id, cluster_name, num_genes in cluster_info:
            print(f"  Cluster {cluster_id} ({cluster_name}): {num_genes} genes")
        print()

    return


def plot_clusters(adata, cluster_list_path: str, save_path: str = None):
    """
    Plot UMAP colored by gene clusters from YAML configuration.

    Takes as input a yaml file containing cluster names and lists of gene KOs,
    creates plots similar to plot_funk_clusters but for each cluster in the yaml,
    and saves plots to disk at save_path.

    Args:
        adata: AnnData object with UMAP coordinates and gene labels
        cluster_list_path: Path to YAML file containing cluster definitions
        save_path: Path where the plot PNG will be saved

    Raises:
        ValueError: If any gene in the clusters is not found in adata.obs['label_str']
    """
    # Load cluster definitions from YAML
    with open(cluster_list_path, "r") as f:
        clusters = yaml.safe_load(f)

    # Calculate grid dimensions (5 columns max)
    num_clusters = len(clusters)
    ncols = min(5, num_clusters)
    nrows = (num_clusters + ncols - 1) // ncols  # Ceiling division

    # Create figure with subplots
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))

    # Handle case where there's only one subplot (axs won't be an array)
    if num_clusters == 1:
        axs = [axs]
    else:
        axs = axs.flatten()

    # Get all available genes in adata
    available_genes = set(adata.obs["label_str"].unique())

    # Plot each cluster
    umap = adata.obsm["X_umap"]
    for i, (cluster_id, cluster_data) in enumerate(clusters.items()):
        cluster_name = cluster_data["name"]
        genes = cluster_data["genes"]

        # Check if all genes exist in adata
        missing_genes = [gene for gene in genes if gene not in available_genes]
        if missing_genes:
            pass
            # raise ValueError(
            #     f"Cluster {cluster_id} ({cluster_name}) contains genes not found in adata: {missing_genes}"
            # )

        # Plot this cluster
        plt.sca(axs[i])

        # Background: all points in grey
        plt.scatter(umap[:, 0], umap[:, 1], c="lightgrey", s=20, alpha=0.8, linewidth=0)

        # Highlight genes in this cluster
        for gene in genes:
            subset = adata[adata.obs["label_str"] == gene].obsm["X_umap"]
            plt.scatter(
                subset[:, 0], subset[:, 1], s=40, alpha=1, linewidth=0, label=gene
            )

        # Format subplot
        plt.title(f"{cluster_id}: {cluster_name}")
        plt.xticks([])
        plt.yticks([])
        if cluster_name == "NTCs":
            continue
        else:
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="10")

    # Hide any unused subplots
    for i in range(num_clusters, len(axs)):
        axs[i].axis("off")

    plt.tight_layout()

    if save_path is not None:
        print(f"Saving cluster UMAP plot to {save_path}")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    return


if __name__ == "__main__":
    pass
