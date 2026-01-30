#!/usr/bin/env python
"""
Project cells from multiple experiments into a shared UMAP embedding space.

This script enables quality control and batch effect analysis by:
1. Sampling cells from multiple experiments (stratified by gene)
2. Computing a reference UMAP on the sampled cells
3. Projecting additional cells from each experiment onto the shared embedding
4. Generating QC visualizations

Use cases:
- Visualize batch effects across experiments
- Identify outlier experiments or wells
- Quality control for multi-experiment screens

Usage:
    python scripts/project_shared_umap.py --config configs/umap_projection/example.yml
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import warnings

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from sklearn.decomposition import PCA
import umap

from ops_model.data.feature_metadata import FeatureMetadata


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary with defaults applied
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Apply defaults
    defaults = {
        "base_dir": "/hpc/projects/intracellular_dashboard/ops",
        "channel": "Phase2D",
        "reference_sampling": {"cells_per_experiment": 1000, "random_seed": 42},
        "projection": {"cells_per_experiment": 5000},
        "embedding": {
            "n_pca_components": 128,
            "n_umap_neighbors": 15,
            "umap_min_dist": 0.1,
            "metric": "cosine",
        },
        "output": {
            "output_dir": None,  # Will be auto-generated if not specified
            "generate_plots": True,
        },
    }

    # Merge defaults
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                if subkey not in config[key]:
                    config[key][subkey] = subvalue

    # Generate output directory if not specified
    if config["output"]["output_dir"] is None:
        feature_type = config["feature_type"]
        channel = config["channel"]
        config["output"]["output_dir"] = (
            f"{config['base_dir']}/umap_projections/"
            f"{feature_type}_{channel}_projection"
        )

    return config


def validate_config(config: dict) -> bool:
    """
    Validate configuration parameters.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, raises ValueError otherwise
    """
    # Check required fields
    required = ["experiments", "feature_type"]
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")

    # Validate experiments list
    if not isinstance(config["experiments"], list) or len(config["experiments"]) == 0:
        raise ValueError("experiments must be a non-empty list")

    # Validate feature_type
    valid_feature_types = ["dino", "cellprofiler"]
    if config["feature_type"] not in valid_feature_types:
        raise ValueError(
            f"feature_type must be one of {valid_feature_types}, "
            f"got: {config['feature_type']}"
        )

    print("✓ Config validation passed")
    return True


def load_experiment_cells(
    experiment: str,
    channel: str,
    feature_type: str,
    base_dir: Union[str, Path],
    max_cells: Optional[int] = None,
    verbose: bool = True,
) -> Optional[ad.AnnData]:
    """
    Load cell-level data for a single experiment/channel.

    Args:
        experiment: Experiment ID (e.g., "ops0089_20251119")
        channel: Channel name (e.g., "Phase2D", "GFP")
        feature_type: Feature type ("dino", "cellprofiler")
        base_dir: Base directory for OPS data
        max_cells: Maximum cells to load (random subsample if exceeded)
        verbose: Print progress

    Returns:
        AnnData object with cell-level features, or None if file not found
    """
    base_dir = Path(base_dir)
    exp_short = experiment.split("_")[0]

    # Find experiment directory (may have date suffix)
    exp_dirs = list(base_dir.glob(f"{exp_short}*"))
    if not exp_dirs:
        if verbose:
            print(f"  ✗ Experiment directory not found: {base_dir}/{exp_short}*")
        return None

    exp_dir = exp_dirs[0]

    # Build file path based on feature type
    if feature_type == "cellprofiler":
        file_path = (
            exp_dir
            / "3-assembly"
            / "cell-profiler"
            / "anndata_objects"
            / f"features_processed_{channel}.h5ad"
        )
    else:  # dino
        file_path = (
            exp_dir
            / "3-assembly"
            / f"{feature_type}_features"
            / "anndata_objects"
            / f"features_processed_{channel}.h5ad"
        )

    if not file_path.exists():
        if verbose:
            print(f"  ✗ File not found: {file_path}")
        return None

    # Load AnnData
    if verbose:
        print(f"  Loading {experiment}...")
    adata = ad.read_h5ad(file_path)

    # Add experiment column if not present
    if "experiment" not in adata.obs.columns:
        adata.obs["experiment"] = experiment

    # Subsample if needed
    if max_cells is not None and adata.shape[0] > max_cells:
        if verbose:
            print(f"    Subsampling {max_cells} cells from {adata.shape[0]}")
        sc.pp.subsample(adata, n_obs=max_cells, random_state=42)

    if verbose:
        print(f"    Loaded: {adata.shape[0]} cells × {adata.shape[1]} features")

    return adata


def sample_cells_stratified_by_gene(
    adata: ad.AnnData,
    n_cells: int,
    random_seed: int = 42,
) -> ad.AnnData:
    """
    Sample cells stratified by gene to ensure representation.

    Args:
        adata: AnnData object
        n_cells: Target number of cells to sample
        random_seed: Random seed for reproducibility

    Returns:
        Subsampled AnnData
    """
    if "label_str" not in adata.obs.columns:
        # Fall back to random sampling if no gene labels
        warnings.warn("label_str column not found, using random sampling")
        sc.pp.subsample(
            adata, n_obs=min(n_cells, adata.shape[0]), random_state=random_seed
        )
        return adata

    # Get unique genes and their counts
    gene_counts = adata.obs["label_str"].value_counts()
    n_genes = len(gene_counts)

    if n_cells >= adata.shape[0]:
        # Return all cells if requesting more than available
        return adata

    # Calculate cells per gene (approximately equal)
    cells_per_gene = max(1, n_cells // n_genes)

    sampled_indices = []
    np.random.seed(random_seed)

    for gene in gene_counts.index:
        gene_cells = adata.obs[adata.obs["label_str"] == gene].index.tolist()
        n_sample = min(cells_per_gene, len(gene_cells))
        sampled = np.random.choice(gene_cells, size=n_sample, replace=False)
        sampled_indices.extend(sampled)

    # If we didn't reach target, sample more from all genes
    if len(sampled_indices) < n_cells:
        remaining = n_cells - len(sampled_indices)
        available = [idx for idx in adata.obs.index if idx not in sampled_indices]
        if len(available) > 0:
            additional = np.random.choice(
                available, size=min(remaining, len(available)), replace=False
            )
            sampled_indices.extend(additional)

    return adata[sampled_indices, :].copy()


def sample_cells_from_experiments(
    experiments: List[str],
    channel: str,
    feature_type: str,
    base_dir: Union[str, Path],
    cells_per_experiment: int,
    random_seed: int = 42,
    verbose: bool = True,
) -> ad.AnnData:
    """
    Sample cells from multiple experiments to create reference dataset.

    Uses stratified sampling by gene to ensure representation across all genes
    in each experiment.

    Args:
        experiments: List of experiment IDs
        channel: Channel name (same for all experiments)
        feature_type: Feature type
        base_dir: Base directory
        cells_per_experiment: Number of cells to sample per experiment
        random_seed: Random seed for reproducibility
        verbose: Print progress

    Returns:
        AnnData object with sampled cells from all experiments
    """
    if verbose:
        print("\n" + "=" * 80)
        print("SAMPLING CELLS FOR REFERENCE EMBEDDING")
        print("=" * 80)
        print(f"Target: {cells_per_experiment} cells per experiment")
        print(f"Strategy: Stratified by gene")
        print()

    sampled_adatas = []

    for exp in experiments:
        if verbose:
            print(f"Experiment: {exp}")

        # Load cell-level data
        adata = load_experiment_cells(
            experiment=exp,
            channel=channel,
            feature_type=feature_type,
            base_dir=base_dir,
            max_cells=None,  # Load all for proper stratification
            verbose=verbose,
        )

        if adata is None:
            continue

        # Check if we have enough cells
        if adata.shape[0] < cells_per_experiment:
            warnings.warn(
                f"Experiment {exp} has only {adata.shape[0]} cells, "
                f"requested {cells_per_experiment}. Using all cells."
            )
            sampled = adata.copy()
        else:
            # Stratified sampling by gene
            sampled = sample_cells_stratified_by_gene(
                adata, n_cells=cells_per_experiment, random_seed=random_seed
            )

        if verbose:
            print(f"    Sampled: {sampled.shape[0]} cells")
            if "label_str" in sampled.obs.columns:
                n_genes = sampled.obs["label_str"].nunique()
                print(f"    Genes represented: {n_genes}")

        sampled_adatas.append(sampled)

    if len(sampled_adatas) == 0:
        raise ValueError("No data loaded from any experiment")

    # Concatenate all sampled data
    if verbose:
        print(f"\nCombining {len(sampled_adatas)} experiments...")

    adata_combined = ad.concat(
        sampled_adatas,
        axis=0,
        join="inner",
        merge="same",
        label="batch",
        keys=[f"batch_{i}" for i in range(len(sampled_adatas))],
    )

    # Mark as reference cells
    adata_combined.obs["is_reference"] = True

    if verbose:
        print(
            f"✓ Reference dataset: {adata_combined.shape[0]} cells × {adata_combined.shape[1]} features"
        )
        print("=" * 80)

    return adata_combined


def compute_reference_embedding(
    adata_reference: ad.AnnData,
    n_pca_components: int = 128,
    n_umap_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    metric: str = "cosine",
    random_seed: int = 42,
    verbose: bool = True,
) -> Tuple[ad.AnnData, PCA, umap.UMAP]:
    """
    Compute PCA and UMAP on reference dataset.

    Uses sklearn PCA and umap-learn for explicit models that can transform new data.

    Args:
        adata_reference: AnnData with sampled cells
        n_pca_components: Number of PCA components
        n_umap_neighbors: Number of neighbors for UMAP
        umap_min_dist: UMAP min_dist parameter
        metric: Distance metric
        random_seed: Random seed
        verbose: Print progress

    Returns:
        Tuple of (adata with embeddings, PCA model, UMAP model)
    """
    if verbose:
        print("\n" + "=" * 80)
        print("COMPUTING REFERENCE EMBEDDING")
        print("=" * 80)

    n_cells = adata_reference.shape[0]
    n_features = adata_reference.shape[1]

    # Adjust PCA components if needed
    max_components = min(n_cells - 1, n_features - 1)
    n_pca_components = min(n_pca_components, max_components)

    # Compute PCA
    if verbose:
        print(f"Computing PCA ({n_pca_components} components)...")

    pca = PCA(n_components=n_pca_components, random_state=random_seed)
    X_pca = pca.fit_transform(adata_reference.X)
    adata_reference.obsm["X_pca"] = X_pca

    if verbose:
        variance_explained = pca.explained_variance_ratio_.sum()
        print(f"  ✓ PCA complete: {variance_explained:.2%} variance explained")

    # Compute UMAP
    if verbose:
        print(
            f"Computing UMAP (n_neighbors={n_umap_neighbors}, min_dist={umap_min_dist})..."
        )

    # Adjust neighbors if needed
    n_neighbors = min(n_umap_neighbors, n_cells - 1)

    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=umap_min_dist,
        n_components=2,
        metric=metric,
        random_state=random_seed,
    )
    X_umap = umap_model.fit_transform(X_pca)
    adata_reference.obsm["X_umap"] = X_umap

    if verbose:
        print(f"  ✓ UMAP complete")
        print("=" * 80)

    return adata_reference, pca, umap_model


def project_cells_onto_reference(
    adata_source: ad.AnnData,
    pca_model: PCA,
    umap_model: umap.UMAP,
    verbose: bool = True,
) -> ad.AnnData:
    """
    Project cells from source experiment onto reference embedding.

    Args:
        adata_source: AnnData with cells to project
        pca_model: Fitted PCA model from reference
        umap_model: Fitted UMAP model from reference
        verbose: Print progress

    Returns:
        AnnData with projected coordinates
    """
    # Transform to PCA space
    X_pca = pca_model.transform(adata_source.X)
    adata_source.obsm["X_pca"] = X_pca

    # Transform to UMAP space
    X_umap = umap_model.transform(X_pca)
    adata_source.obsm["X_umap"] = X_umap

    # Mark as projected cells
    adata_source.obs["is_reference"] = False

    return adata_source


def project_all_experiments(
    experiments: List[str],
    channel: str,
    feature_type: str,
    base_dir: Union[str, Path],
    pca_model: PCA,
    umap_model: umap.UMAP,
    cells_per_experiment: int,
    random_seed: int = 42,
    verbose: bool = True,
) -> List[ad.AnnData]:
    """
    Project cells from all experiments onto reference embedding.

    Args:
        experiments: List of experiment IDs
        channel: Channel name
        feature_type: Feature type
        base_dir: Base directory
        pca_model: Fitted PCA model
        umap_model: Fitted UMAP model
        cells_per_experiment: Max cells to project per experiment
        random_seed: Random seed
        verbose: Print progress

    Returns:
        List of projected AnnData objects
    """
    if verbose:
        print("\n" + "=" * 80)
        print("PROJECTING EXPERIMENTS")
        print("=" * 80)
        print(f"Projecting up to {cells_per_experiment} cells per experiment")
        print()

    projected_adatas = []

    for exp in experiments:
        if verbose:
            print(f"Projecting: {exp}")

        # Load cells
        adata = load_experiment_cells(
            experiment=exp,
            channel=channel,
            feature_type=feature_type,
            base_dir=base_dir,
            max_cells=cells_per_experiment,
            verbose=verbose,
        )

        if adata is None:
            continue

        # Project onto reference
        adata_projected = project_cells_onto_reference(
            adata, pca_model, umap_model, verbose=False
        )

        if verbose:
            print(f"    ✓ Projected {adata_projected.shape[0]} cells")

        projected_adatas.append(adata_projected)

    if verbose:
        print("=" * 80)

    return projected_adatas


def combine_all_data(
    adata_reference: ad.AnnData,
    projected_adatas: List[ad.AnnData],
    verbose: bool = True,
) -> ad.AnnData:
    """
    Combine reference and projected data into single AnnData.

    Args:
        adata_reference: Reference AnnData with sampled cells
        projected_adatas: List of projected AnnData objects
        verbose: Print progress

    Returns:
        Combined AnnData with all cells
    """
    if verbose:
        print("\n" + "=" * 80)
        print("COMBINING DATA")
        print("=" * 80)

    all_adatas = [adata_reference] + projected_adatas

    adata_combined = ad.concat(all_adatas, axis=0, join="inner", merge="same")

    # Ensure required metadata columns exist
    required_cols = ["experiment", "well", "guide", "label_str", "is_reference"]
    for col in required_cols:
        if col not in adata_combined.obs.columns:
            if col == "is_reference":
                adata_combined.obs[col] = False
            else:
                adata_combined.obs[col] = "unknown"

    if verbose:
        n_ref = adata_combined.obs["is_reference"].sum()
        n_proj = (~adata_combined.obs["is_reference"]).sum()
        print(f"Combined: {adata_combined.shape[0]} cells total")
        print(f"  Reference: {n_ref} cells")
        print(f"  Projected: {n_proj} cells")
        print(f"  Experiments: {adata_combined.obs['experiment'].nunique()}")
        print("=" * 80)

    return adata_combined


def generate_qc_plots(
    adata_combined: ad.AnnData,
    output_dir: Path,
    verbose: bool = True,
) -> None:
    """
    Generate standard QC plots for batch effect visualization.

    Args:
        adata_combined: Combined AnnData with all projections
        output_dir: Directory to save plots
        verbose: Print progress
    """
    if verbose:
        print("\n" + "=" * 80)
        print("GENERATING QC PLOTS")
        print("=" * 80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: UMAP colored by experiment
    if verbose:
        print("Generating UMAP by experiment...")

    fig, ax = plt.subplots(figsize=(12, 10))

    experiments = adata_combined.obs["experiment"].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(experiments)))

    for i, exp in enumerate(sorted(experiments)):
        mask = adata_combined.obs["experiment"] == exp
        umap_coords = adata_combined.obsm["X_umap"][mask]
        ax.scatter(
            umap_coords[:, 0],
            umap_coords[:, 1],
            c=[colors[i]],
            label=exp,
            alpha=0.6,
            s=5,
            rasterized=True,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Shared UMAP Projection - Colored by Experiment")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    plot_path = output_dir / "umap_by_experiment.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    if verbose:
        print(f"  ✓ Saved: {plot_path}")
        print("=" * 80)


def save_outputs(
    adata_combined: ad.AnnData,
    output_dir: Path,
    verbose: bool = True,
) -> None:
    """
    Save combined AnnData with projections.

    Args:
        adata_combined: Combined AnnData with all cells
        output_dir: Output directory
        verbose: Print progress
    """
    if verbose:
        print("\n" + "=" * 80)
        print("SAVING OUTPUTS")
        print("=" * 80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save combined AnnData
    output_path = output_dir / "projected_cells.h5ad"
    adata_combined.write_h5ad(output_path)

    if verbose:
        print(f"✓ Saved AnnData: {output_path}")
        print(f"  Shape: {adata_combined.shape}")
        print(f"  Experiments: {adata_combined.obs['experiment'].nunique()}")
        print("=" * 80)


def main():
    """Main entry point for shared UMAP projection."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Project cells from multiple experiments into shared UMAP space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/project_shared_umap.py --config configs/umap_projection/phase2d_qc.yml
        """,
    )

    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )

    args = parser.parse_args()

    try:
        # Load and validate config
        print(f"Loading config: {args.config}")
        config = load_config(args.config)

        print("\nConfiguration:")
        print(f"  Experiments: {len(config['experiments'])}")
        print(f"  Feature type: {config['feature_type']}")
        print(f"  Channel: {config['channel']}")
        print(
            f"  Reference samples: {config['reference_sampling']['cells_per_experiment']} cells/experiment"
        )
        print(
            f"  Projection: {config['projection']['cells_per_experiment']} cells/experiment"
        )
        print(f"  Output: {config['output']['output_dir']}")

        validate_config(config)

        # Sample cells for reference
        adata_reference = sample_cells_from_experiments(
            experiments=config["experiments"],
            channel=config["channel"],
            feature_type=config["feature_type"],
            base_dir=config["base_dir"],
            cells_per_experiment=config["reference_sampling"]["cells_per_experiment"],
            random_seed=config["reference_sampling"]["random_seed"],
            verbose=True,
        )

        # Compute reference embedding
        adata_reference, pca_model, umap_model = compute_reference_embedding(
            adata_reference=adata_reference,
            n_pca_components=config["embedding"]["n_pca_components"],
            n_umap_neighbors=config["embedding"]["n_umap_neighbors"],
            umap_min_dist=config["embedding"]["umap_min_dist"],
            metric=config["embedding"]["metric"],
            random_seed=config["reference_sampling"]["random_seed"],
            verbose=True,
        )

        # Project all experiments
        projected_adatas = project_all_experiments(
            experiments=config["experiments"],
            channel=config["channel"],
            feature_type=config["feature_type"],
            base_dir=config["base_dir"],
            pca_model=pca_model,
            umap_model=umap_model,
            cells_per_experiment=config["projection"]["cells_per_experiment"],
            random_seed=config["reference_sampling"]["random_seed"],
            verbose=True,
        )

        # Combine all data
        adata_combined = combine_all_data(
            adata_reference=adata_reference,
            projected_adatas=projected_adatas,
            verbose=True,
        )

        # Save outputs
        save_outputs(
            adata_combined=adata_combined,
            output_dir=config["output"]["output_dir"],
            verbose=True,
        )

        # Generate QC plots
        if config["output"]["generate_plots"]:
            generate_qc_plots(
                adata_combined=adata_combined,
                output_dir=config["output"]["output_dir"],
                verbose=True,
            )

        print("\n" + "=" * 80)
        print("PROJECTION COMPLETE")
        print("=" * 80)
        print(f"Output directory: {config['output']['output_dir']}")
        print("  - projected_cells.h5ad")
        if config["output"]["generate_plots"]:
            print("  - umap_by_experiment.png")
            print("  - umap_reference_vs_projected.png")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
