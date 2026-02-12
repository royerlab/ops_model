import ast
import yaml
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import anndata as ad

from copairs import map as copairs_map
import matplotlib.pyplot as plt


def adata_to_copairs_df(adata) -> pd.DataFrame:
    """
    Convert AnnData to DataFrame format expected by copairs.

    Parameters
    ----------
    adata : AnnData
        AnnData object with features in .X and metadata in .obs

    Returns
    -------
    df : DataFrame
        DataFrame with metadata and feature columns
    """
    # Get features as DataFrame
    if hasattr(adata.X, "toarray"):
        features_df = pd.DataFrame(
            adata.X.toarray(), index=adata.obs_names, columns=adata.var_names
        )
    else:
        features_df = pd.DataFrame(
            adata.X, index=adata.obs_names, columns=adata.var_names
        )

    # Combine with metadata
    df = pd.concat([adata.obs, features_df], axis=1)

    return df


def phenotypic_activity_assesment(adata: ad.AnnData, plot_results=True):
    """
    Following the example in:
    https://github.com/cytomining/copairs/blob/v0.5.1/examples/phenotypic_activity.ipynb

    adata: guide level AnnData object
    """
    df = adata_to_copairs_df(adata)
    df["is_NTC"] = df["perturbation"].apply(lambda x: x == "NTC")
    meta_cols = ["sgRNA", "n_cells", "perturbation", "is_NTC"]
    feat_cols = [col for col in df.columns if col not in meta_cols]
    meta = df[meta_cols]
    feats = df[feat_cols]
    results = copairs_map.average_precision(
        meta,
        np.asarray(feats),
        pos_sameby=["perturbation"],
        pos_diffby=["sgRNA"],
        neg_sameby=[],
        neg_diffby=["is_NTC"],
    )
    activity_map = copairs_map.mean_average_precision(
        results, sameby=["perturbation"], null_size=1000000, threshold=0.05, seed=0
    )
    activity_map["-log10(p-value)"] = -activity_map["corrected_p_value"].apply(np.log10)
    active_ratio = activity_map.below_corrected_p.mean()

    if plot_results:
        import matplotlib.pyplot as plt

        active_ratio = activity_map.below_corrected_p.mean()

        plt.scatter(
            data=activity_map,
            x="mean_average_precision",
            y="-log10(p-value)",
            c="below_corrected_p",
            cmap="tab10",
            s=10,
        )
        plt.title("Phenotypic activity assesement")
        plt.xlabel("mAP")
        plt.ylabel("-log10(p-value)")
        plt.axhline(-np.log10(0.05), color="black", linestyle="--")
        plt.text(
            0.65,
            1.5,
            f"Phenotypically active = {100 * active_ratio:.2f}%",
            va="center",
            ha="left",
        )
        plt.show()

    return activity_map, active_ratio


def phenotypic_distinctivness(
    adata: ad.AnnData, activity_map: pd.DataFrame, plot_results=True
):
    """
    adata: guide level AnnData object
    """
    # 1) Identify active perturbations
    active_perturbations = activity_map[activity_map["below_corrected_p"] == True][
        "perturbation"
    ].tolist()
    print(f"Number of active perturbations: {len(active_perturbations)}")

    # 2) Filter adata to only include active perturbations (exclude NTC)
    adata_filtered = adata[adata.obs["perturbation"].isin(active_perturbations)].copy()
    print(
        f"Filtered to {adata_filtered.n_obs} observations from {len(active_perturbations)} active perturbations"
    )

    # Convert to copairs format
    df = adata_to_copairs_df(adata_filtered)

    # Prepare metadata and feature columns
    meta_cols = ["sgRNA", "n_cells", "perturbation"]
    # Only keep meta_cols that exist in df
    meta_cols = [col for col in meta_cols if col in df.columns]
    feat_cols = [col for col in df.columns if col not in meta_cols]

    meta = df[meta_cols]
    feats = df[feat_cols]

    # 3) Calculate average precision
    # Positive pairs: same perturbation, different sgRNA
    # Negative pairs: different perturbation
    results = copairs_map.average_precision(
        meta,
        np.asarray(feats),
        pos_sameby=["perturbation"],
        pos_diffby=["sgRNA"],
        neg_sameby=[],
        neg_diffby=["perturbation"],
    )

    # 4) Calculate mean average precision for each gene with null distribution
    distinctiveness_map = copairs_map.mean_average_precision(
        results, sameby=["perturbation"], null_size=1000000, threshold=0.05, seed=0
    )

    # Add -log10(p-value) for easier interpretation
    distinctiveness_map["-log10(p-value)"] = -distinctiveness_map[
        "corrected_p_value"
    ].apply(np.log10)

    # Calculate ratio of distinctive perturbations
    distinctive_ratio = distinctiveness_map.below_corrected_p.mean()
    print(
        f"Proportion of phenotypically distinctive perturbations: {100 * distinctive_ratio:.2f}%"
    )

    if plot_results:
        plt.scatter(
            data=distinctiveness_map,
            x="mean_average_precision",
            y="-log10(p-value)",
            c="below_corrected_p",
            cmap="tab10",
            s=10,
        )
        plt.title("Phenotypic distinctiveness")
        plt.xlabel("mAP")
        plt.ylabel("-log10(p-value)")
        plt.axhline(-np.log10(0.05), color="black", linestyle="--")
        plt.text(
            0.65,
            1.5,
            f"Phenotypically distinct = {100 * distinctive_ratio:.2f}%",
            va="center",
            ha="left",
        )
        plt.show()

    return distinctiveness_map, distinctive_ratio


def phenotypic_consistency_corum(
    adata: ad.AnnData, activity_map: pd.DataFrame, plot_results=True
):
    """
    adata: Gene level AnnData object
    activity_map: DataFrame with columns 'perturbation', 'mean_average_precision', 'corrected_p_value', 'below_corrected_p'
    """
    path = "/hpc/projects/intracellular_dashboard/ops/configs/annotated_gene_panel_July2025.csv"
    gene_panel = pd.read_csv(path)

    active_genes = activity_map[activity_map["below_corrected_p"]][
        "perturbation"
    ].tolist()
    active_genes = [gene for gene in active_genes if gene != "NTC"]
    df = adata_to_copairs_df(adata)
    df = df[df["perturbation"].isin(active_genes)]

    metadata_cols = [
        "perturbation",
        "n_cells",
        "guides",
        "reporter",
        "experiment",
        "in_complex",
        "is_NTC",
        "n_experiments",
    ]
    complex_map_list = []
    for p in df["perturbation"].tolist():
        complex_col = gene_panel.loc[
            gene_panel["Gene.name"] == p, "In_same_complex_with"
        ]
        complex_str = complex_col.iloc[0] if not complex_col.empty else []
        complex_list = ast.literal_eval(complex_str) if complex_str else []
        complex_list.append(p)
        # filter complex list to only include phenotypically active genes
        complex_list = [gene for gene in complex_list if gene in active_genes]
        if len(complex_list) > 1:
            df["in_complex"] = df["perturbation"].apply(
                lambda x: any([gene in x for gene in complex_list])
            )
            metadata_cols = [col for col in metadata_cols if col in df.columns]
            feature_cols = [col for col in df.columns if col not in metadata_cols]
            meta = df[metadata_cols]
            feats = df[feature_cols]
            try:
                results_complex = copairs_map.average_precision(
                    meta,
                    np.asarray(feats),
                    pos_sameby=["in_complex"],
                    pos_diffby=["perturbation"],
                    neg_sameby=[],
                    neg_diffby=["in_complex"],
                )
                map = copairs_map.mean_average_precision(
                    results_complex,
                    sameby=["in_complex"],
                    null_size=1000000,
                    threshold=0.05,
                    seed=0,
                )
                complex_map = map[map["in_complex"] == True].copy()
                complex_map["complex_id"] = p
                complex_map.drop(columns=["in_complex", "indices"], inplace=True)
                complex_map_list.append(complex_map)
            except:
                continue
    all_complex_results_df = pd.concat(complex_map_list, ignore_index=True)
    all_complex_results_df["-log10(p-value)"] = -all_complex_results_df[
        "corrected_p_value"
    ].apply(np.log10)

    consistency_corum_ratio = all_complex_results_df.below_corrected_p.mean()
    if plot_results:
        plt.scatter(
            data=all_complex_results_df,
            x="mean_average_precision",
            y="-log10(p-value)",
            c="below_corrected_p",
            cmap="tab10",
            s=10,
        )
        plt.title("Phenotypic consistency (CORUM)")
        plt.xlabel("mAP")
        plt.ylabel("-log10(p-value)")
        plt.axhline(-np.log10(0.05), color="black", linestyle="--")
        plt.text(
            0.65,
            1.5,
            f"Phenotypically distinct = {100 * consistency_corum_ratio:.2f}%",
            va="center",
            ha="left",
        )
        plt.show()

    return all_complex_results_df, consistency_corum_ratio


def phenotypic_consistency_manual_annotation(
    adata: ad.AnnData, activity_map: pd.DataFrame, plot_results=True
):
    """
    adata: Gene level AnnData object
    activity_map: DataFrame with columns 'perturbation', 'mean_average_precision', 'corrected_p_value', 'below_corrected_p'
    """
    path = "/hpc/projects/icd.ops/configs/gene_clusters/chad_positive_controls_v4.yml"
    with open(path, "r") as f:
        gene_clusters = yaml.safe_load(f)

    active_genes = activity_map[activity_map["below_corrected_p"]][
        "perturbation"
    ].tolist()
    active_genes = [gene for gene in active_genes if gene != "NTC"]
    df = adata_to_copairs_df(adata)
    df = df[df["perturbation"].isin(active_genes)]

    metadata_cols = [
        "perturbation",
        "n_cells",
        "guides",
        "reporter",
        "experiment",
        "in_complex",
        "is_NTC",
        "n_experiments",
    ]

    complex_map_list = []

    for k, v in gene_clusters.items():
        complex = v["genes"]
        complex_list = [gene for gene in complex if gene in active_genes]
        if len(complex_list) > 1:
            df["in_complex"] = df["perturbation"].apply(
                lambda x: any([gene in x for gene in complex_list])
            )
            metadata_cols = [col for col in metadata_cols if col in df.columns]
            feature_cols = [col for col in df.columns if col not in metadata_cols]
            meta = df[metadata_cols]
            feats = df[feature_cols]
            try:
                results_complex = copairs_map.average_precision(
                    meta,
                    np.asarray(feats),
                    pos_sameby=["in_complex"],
                    pos_diffby=["perturbation"],
                    neg_sameby=[],
                    neg_diffby=["in_complex"],
                )
                map = copairs_map.mean_average_precision(
                    results_complex,
                    sameby=["in_complex"],
                    null_size=1000000,
                    threshold=0.05,
                    seed=0,
                )
                complex_map = map[map["in_complex"] == True].copy()
                complex_map["complex_num"] = k
                complex_map.drop(columns=["in_complex", "indices"], inplace=True)
                complex_map_list.append(complex_map)
            except:
                continue

    all_complex_results_df = pd.concat(complex_map_list, ignore_index=True)
    all_complex_results_df["-log10(p-value)"] = -all_complex_results_df[
        "corrected_p_value"
    ].apply(np.log10)

    phenotypic_consistency_ratio = all_complex_results_df.below_corrected_p.mean()
    if plot_results:
        plt.scatter(
            data=all_complex_results_df,
            x="mean_average_precision",
            y="-log10(p-value)",
            c="below_corrected_p",
            cmap="tab10",
            s=10,
        )
        plt.title("Phenotypic consistency (Manual)")
        plt.xlabel("mAP")
        plt.ylabel("-log10(p-value)")
        plt.axhline(-np.log10(0.05), color="black", linestyle="--")
        plt.text(
            0.65,
            1.5,
            f"Phenotypically distinct = {100 * phenotypic_consistency_ratio:.2f}%",
            va="center",
            ha="left",
        )
        plt.show()

    return all_complex_results_df, phenotypic_consistency_ratio


def metric_umap(adata, metric_map, metric_name="activity", umap_key="X_umap_cuml"):
    """
    Generic function to visualize phenotypic metrics on UMAP.

    Parameters
    ----------
    adata : AnnData
        AnnData object at perturbation level with UMAP coordinates
    metric_map : pd.DataFrame
        DataFrame with columns: 'perturbation', 'mean_average_precision',
        '-log10(p-value)', 'below_corrected_p'
    metric_name : str
        Name of the metric ('activity' or 'distinctiveness') for labeling
    umap_key : str
        Key in adata.obsm containing UMAP coordinates (default: 'X_umap_cuml')

    Returns
    -------
    None
        Modifies adata.obs in place and displays plots

    Notes
    -----
    1) Aligns the mean_average_precision and -log10(p-value) from metric_map to adata.obs
    2) Plots UMAP colored by mAP and -log10(p-value)
    3) Colors only points below the corrected p-value threshold, others shown in grey
    """
    # Check if UMAP coordinates exist
    if umap_key not in adata.obsm.keys():
        raise ValueError(
            f"UMAP coordinates not found in adata.obsm['{umap_key}']. Please compute UMAP first."
        )

    # 1) Create mapping from metric_map and align to adata.obs
    metric_dict = metric_map.set_index("perturbation")[
        ["mean_average_precision", "-log10(p-value)", "below_corrected_p"]
    ].to_dict("index")

    # Create column names based on metric_name
    map_col = f"{metric_name}_mAP"
    log10p_col = f"{metric_name}_log10p"
    significance_col = f"is_{metric_name}"

    # Map to adata.obs
    adata.obs[map_col] = adata.obs["perturbation"].map(
        lambda x: metric_dict.get(x, {}).get("mean_average_precision", np.nan)
    )
    adata.obs[log10p_col] = adata.obs["perturbation"].map(
        lambda x: metric_dict.get(x, {}).get("-log10(p-value)", np.nan)
    )
    adata.obs[significance_col] = adata.obs["perturbation"].map(
        lambda x: metric_dict.get(x, {}).get("below_corrected_p", False)
    )

    # Mark NTC guides
    adata.obs["is_NTC"] = adata.obs["perturbation"] == "NTC"

    # Get UMAP coordinates
    umap_coords = adata.obsm[umap_key]

    # 2-3) Create two vertically stacked plots
    fig, axes = plt.subplots(2, 1, figsize=(10, 16))

    # Separate data into significant, non-significant, and NTC
    significant_mask = (adata.obs[significance_col] == True) & (~adata.obs["is_NTC"])
    nonsignificant_mask = (adata.obs[significance_col] == False) & (
        ~adata.obs["is_NTC"]
    )
    ntc_mask = adata.obs["is_NTC"]

    # Customize labels based on metric_name
    if metric_name == "activity":
        title_prefix = "Phenotypic Activity"
        legend_label = "Phenotypically active"
        grey_label = "Not active / NTC"
        summary_label = "Active"
    elif metric_name == "distinctiveness":
        title_prefix = "Phenotypic Distinctiveness"
        legend_label = "Phenotypically distinctive"
        grey_label = "Not distinctive / NTC"
        summary_label = "Distinctive"
    else:
        title_prefix = f"Phenotypic {metric_name.capitalize()}"
        legend_label = f"Phenotypically {metric_name}"
        grey_label = f"Not {metric_name} / NTC"
        summary_label = metric_name.capitalize()

    # Plot 1: Colored by mAP
    ax = axes[0]

    # Plot grey points first (non-significant and NTC)
    grey_mask = nonsignificant_mask | ntc_mask
    if grey_mask.any():
        ax.scatter(
            umap_coords[grey_mask, 0],
            umap_coords[grey_mask, 1],
            c="lightgrey",
            s=20,
            alpha=0.5,
            label=grey_label,
        )

    # Plot significant points colored by mAP
    if significant_mask.any():
        scatter1 = ax.scatter(
            umap_coords[significant_mask, 0],
            umap_coords[significant_mask, 1],
            c=adata.obs.loc[significant_mask, map_col],
            s=30,
            alpha=0.8,
            cmap="viridis",
            label=legend_label,
        )
        plt.colorbar(scatter1, ax=ax, label="Mean Average Precision")

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(f"{title_prefix}: Colored by mAP", fontsize=14, fontweight="bold")
    ax.legend(markerscale=1.5, loc="best")

    # Plot 2: Colored by -log10(p-value)
    ax = axes[1]

    # Plot grey points first (non-significant and NTC)
    if grey_mask.any():
        ax.scatter(
            umap_coords[grey_mask, 0],
            umap_coords[grey_mask, 1],
            c="lightgrey",
            s=20,
            alpha=0.5,
            label=grey_label,
        )

    # Plot significant points colored by -log10(p-value)
    if significant_mask.any():
        scatter2 = ax.scatter(
            umap_coords[significant_mask, 0],
            umap_coords[significant_mask, 1],
            c=adata.obs.loc[significant_mask, log10p_col],
            s=30,
            alpha=0.8,
            cmap="plasma",
            label=legend_label,
        )
        plt.colorbar(scatter2, ax=ax, label="-log10(p-value)")

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(
        f"{title_prefix}: Colored by -log10(p-value)", fontsize=14, fontweight="bold"
    )
    ax.legend(markerscale=1.5, loc="best")

    plt.tight_layout()
    plt.show()

    # Print summary
    n_significant = significant_mask.sum()
    n_nonsignificant = nonsignificant_mask.sum()
    n_ntc = ntc_mask.sum()
    print(f"\nUMAP Summary ({metric_name}):")
    print(
        f"  {summary_label} perturbations: {n_significant} ({100*n_significant/len(adata):.1f}%)"
    )
    print(
        f"  Non-{metric_name} perturbations: {n_nonsignificant} ({100*n_nonsignificant/len(adata):.1f}%)"
    )
    print(f"  NTC: {n_ntc} ({100*n_ntc/len(adata):.1f}%)")

    return


def map_main(
    adata_guide_path: Optional[str],
    adata_guide: Optional[ad.AnnData],
    adata_gene_path: Optional[str],
    adata_gene: Optional[ad.AnnData],
    save_dir: str,
):
    """
    1) If paths provided load objects, if objects provided continue, if either missing error
    2) run phenotypic_activity_assesment
    3) run phenotypic_distinctivness
    4) run phenotypic_consistency_corum
    5) run phenotypic_consistency_manual_annotation
    After running each step save the resulting csv in save_dir

    """
    # 1) Load or validate AnnData objects
    if adata_guide is None and adata_guide_path is None:
        raise ValueError("Either adata_guide or adata_guide_path must be provided")
    if adata_gene is None and adata_gene_path is None:
        raise ValueError("Either adata_gene or adata_gene_path must be provided")

    # Load guide-level AnnData if path provided
    if adata_guide is None:
        print(f"Loading guide-level AnnData from {adata_guide_path}")
        adata_guide = ad.read_h5ad(adata_guide_path)

    # Load gene-level AnnData if path provided
    if adata_gene is None:
        print(f"Loading gene-level AnnData from {adata_gene_path}")
        adata_gene = ad.read_h5ad(adata_gene_path)

    # Create save directory if it doesn't exist
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 2) Run phenotypic activity assessment
    print("\nRunning phenotypic activity assessment...")
    activity_map, active_ratio = phenotypic_activity_assesment(
        adata_guide, plot_results=False
    )
    activity_csv_path = save_path / "phenotypic_activity.csv"
    activity_map.to_csv(activity_csv_path, index=False)
    print(f"Saved activity results to {activity_csv_path}")
    print(f"Active ratio: {100 * active_ratio:.2f}%")

    # 3) Run phenotypic distinctiveness
    print("\nRunning phenotypic distinctiveness...")
    distinctiveness_map, distinctive_ratio = phenotypic_distinctivness(
        adata_guide, activity_map, plot_results=False
    )
    distinctiveness_csv_path = save_path / "phenotypic_distinctiveness.csv"
    distinctiveness_map.to_csv(distinctiveness_csv_path, index=False)
    print(f"Saved distinctiveness results to {distinctiveness_csv_path}")
    print(f"Distinctive ratio: {100 * distinctive_ratio:.2f}%")

    # 4) Run phenotypic consistency (CORUM)
    print("\nRunning phenotypic consistency (CORUM)...")
    consistency_corum_map, consistency_corum_ratio = phenotypic_consistency_corum(
        adata_gene, activity_map, plot_results=False
    )
    consistency_corum_csv_path = save_path / "phenotypic_consistency_corum.csv"
    consistency_corum_map.to_csv(consistency_corum_csv_path, index=False)
    print(f"Saved CORUM consistency results to {consistency_corum_csv_path}")
    print(f"CORUM consistency ratio: {100 * consistency_corum_ratio:.2f}%")

    # 5) Run phenotypic consistency (manual annotation)
    print("\nRunning phenotypic consistency (manual annotation)...")
    consistency_manual_map, consistency_manual_ratio = (
        phenotypic_consistency_manual_annotation(
            adata_gene, activity_map, plot_results=False
        )
    )
    consistency_manual_csv_path = save_path / "phenotypic_consistency_manual.csv"
    consistency_manual_map.to_csv(consistency_manual_csv_path, index=False)
    print(f"Saved manual consistency results to {consistency_manual_csv_path}")
    print(f"Manual consistency ratio: {100 * consistency_manual_ratio:.2f}%")

    print(f"\nAll results saved to {save_dir}")

    return {
        "activity_map": activity_map,
        "active_ratio": active_ratio,
        "distinctiveness_map": distinctiveness_map,
        "distinctive_ratio": distinctive_ratio,
        "consistency_corum_map": consistency_corum_map,
        "consistency_corum_ratio": consistency_corum_ratio,
        "consistency_manual_map": consistency_manual_map,
        "consistency_manual_ratio": consistency_manual_ratio,
    }
